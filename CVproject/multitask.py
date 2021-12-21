import torch
from torch import optim
from torch.utils.data import DataLoader
from foodDataset import FoodDataset
from model import ViT_Ingre, BEiT_Food, ViT_MultiTask, ViT_FuseMultiTask
from tqdm import tqdm
import copy
import os
import gc
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='PyTorch ViT Training')
parser.add_argument('--model_name_or_path',default='google/vit-base-patch16-224',help='path to dataset')
parser.add_argument('--num_labels',default=172)
parser.add_argument("--epoch", default=40)
parser.add_argument("--model", default="vit-base")
parser.add_argument("--dataset", default="vrieo")
parser.add_argument("--multigpu", action='store_true', help='enable/disable using multigpu and multiprocess')
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--device", default=0)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--tag", default="")
parser.add_argument("--loadingre", action="store_true")
parser.add_argument("--batchaccum", default=1, type=int)
parser.add_argument("--num_ingres", default=353, type=int)
parser.add_argument("--cache_dir", default = "/home/dell/.cache/huggingface/transformers")

args = parser.parse_args()
#DDP Config
if args.multigpu:
    maindevice = torch.device("cuda", 0)

    local_rank = int(args.local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    device = torch.device("cuda", int(local_rank))
else:
    args.device = int(args.device)
    maindevice = torch.device("cuda", args.device)
    device = torch.device("cuda",args.device)


def computeMetric(data, rndigit = -1):
    acc = 0
    for item in data:
            if item[0] == item[1]:
                acc += 1

    acc = acc/(len(data) + 1e-6)
    if rndigit != -1:
        acc = round(acc, rndigit)
    return acc

def test(model, data, bestacc):
    topk = 5
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        predResult = list()
        for batch, (images, label, ingres) in enumerate(data):
            images, label, ingres = images.to(device), label.to(device), ingres.to(device)
            output = model(images, label, ingres)
            predict = output.foodlogits
            predlabel = predict.softmax(dim=1).argmax(dim=1)
            _, topkpred = predict.topk(topk, 1, True, True)
            predResult += zip(list(predlabel), list(label))
            topklabel = torch.stack([label]*topk, dim=1).to(device)
            correct += torch.eq(topkpred, topklabel).sum().float().item()
            total += label.size(0)
            del images, label, ingres, output, predict, predlabel

        top5acc = round(correct / (total+1e-6), 3)
        top1acc = computeMetric(predResult, rndigit = 3)
        del predResult
        print("Top1 Acc:{}, Top5 Acc:{}".format(str(top1acc), str(top5acc)))
    update = False
    if top1acc > bestacc[0]:
        bestacc = [top1acc, top5acc]
        update = True
    return bestacc, update


Datasets = {
    "foodnet":{
        "train":"./foodnet/train_list.txt",
        "test":"./foodnet/test_list.txt",
        "val":"./foodnet/val_list.txt"
    },
    "vrieo":{
        "train":"./multilabeldata/train.txt",
        "test":"./multilabeldata/test.txt",
        "val":"./multilabeldata/dev.txt"
    }
}

Models = {
    "vit-large":{
        "path":"google/vit-large-patch16-224"
    },
    "vit-base":{
        "path":'google/vit-base-patch16-224'
    },
    "vit-base-384":{
        "path":'google/vit-base-patch16-384'
    },
    "vit-base-384-p32":{
        "path":'google/vit-base-patch32-384'
    },
    "vit-large-384":{
        "path":'google/vit-large-patch16-384'
    },
    "vit-base-2":{
        "path":'google/vit-base-patch16-224-in21k'
    },
    "beit-base":{
        "path":"beit-base-patch16-224-pt22k-ft22k"
    },
    "beit-large":{
        "path":"microsoft/beit-large-patch16-224-pt22k-ft22k"
    }
}

def lossBalancer(loss1:torch.FloatTensor, loss2:torch.FloatTensor, factor:float) -> torch.FloatTensor:
    loss1_val = loss1.item()
    loss2_val = loss2.item()
    newloss = loss1_val * loss1 + factor * loss2_val * loss2
    #newloss /= loss1_val + factor * loss2_val
    return newloss

if __name__ == "__main__":
    args.model_name_or_path = Models[args.model]["path"]
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    
    if device == maindevice:
        print("loading data...")
    args.batch_size = int(args.batch_size)
    dataset = Datasets[args.dataset]
    trainDataset = FoodDataset(dataset["train"], args)
    if args.multigpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        traindata = DataLoader(trainDataset, batch_size = args.batch_size, num_workers = 16, sampler = train_sampler)
    else:
        traindata = DataLoader(trainDataset, batch_size = args.batch_size, num_workers = 16, shuffle=True)
    testDataset = FoodDataset(dataset["test"], args)
    testdata = DataLoader(testDataset, batch_size = args.batch_size, num_workers = 16, shuffle=True)
    validDataset = FoodDataset(dataset["val"], args)
    validdata = DataLoader(validDataset, batch_size = args.batch_size, num_workers = 16, shuffle=True)
    
    if device == maindevice:
        print("loading model...")
    if args.model == "beit":
        model = BEiT_Food(args=args).to(device).train()
    else:
        model = ViT_FuseMultiTask(args=args).to(device).train()
    # print(model)
    # exit()
    
    if args.finetune:
        model.load_state_dict(torch.load("checkpoint/vit_foodnet.pth"))
        args.num_labels = 172
        model.num_labels = args.num_labels
        model.vit.classifier = torch.nn.Linear(model.vit.config.hidden_size, args.num_labels)
        model = model.to(device).train()
    
    if args.multigpu:
        model = DDP(model, find_unused_parameters=False, device_ids=[local_rank],  output_device=local_rank)
    
    trainableParams = filter(lambda p: p.requires_grad, model.parameters())
    lr = args.lr * args.batch_size * args.batchaccum/32
    optimizer = torch.optim.Adam(trainableParams, lr=lr)
    
    bestacc = [0,0]
    epochloss = list()

    for epoch in range(args.epoch):
        if device == maindevice:
            print("Epoch:",epoch+1)
            bar = tqdm(total = len(traindata), ncols=100)
        model = model.train()
        combfactor = 1.0
        if args.multigpu:
            traindata.sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        #optimizer.zero_grad()
        for batch, (images, label, ingres) in enumerate(traindata):
            images, label, ingres = images.to(device), label.to(device), ingres.to(device)
            output = model(images, label, ingres)
            #outputloss = 1/mulfactor * output.foodloss + combfactor/mulfactor * output.ingreloss
            outputloss = output.foodloss + combfactor * output.ingreloss
            #outputloss = lossBalancer(output.foodloss, output.ingreloss, combfactor)
            if args.batchaccum == 1:
                optimizer.zero_grad()
                outputloss.backward()
                optimizer.step()
            else:
                loss = outputloss/args.batchaccum
                loss.backward()
                if (batch + 1) % args.batchaccum == 0 or batch == len(traindata)-1:
                    optimizer.step()
                    optimizer.zero_grad()
                    
            epochloss.append(output.foodloss.item())
            if device == maindevice:
                if batch % 5 == 0:
                    bar.set_description_str("Loss:{}".format(str(round(sum(epochloss)/len(epochloss), 3))))
                bar.update(1)
            del output, images, label, ingres, outputloss
        
        if device == maindevice:
            bar.close()
            print("Evaluating on Valid set...")
            test(model.eval(), validdata, bestacc)
            print("Evaluating on Test set...")
            bestacc, update = test(model.eval(), testdata, bestacc)
            if update:
                print("New Best!")
                bestmodel = copy.deepcopy(model)
                bestmodel.to("cpu")

            modelname = "mt_" + args.model + "_" + args.dataset
            if args.tag != "":
                modelname += "_" + args.tag
            model_path = str(os.path.join("checkpoint", modelname + ".pth"))
            if (epoch + 1) % 5 == 0:
                if args.multigpu:
                    torch.save(bestmodel.module.state_dict(), model_path)
                else:
                    torch.save(bestmodel.state_dict(), model_path)
                print("[BEST] Top1 Acc:{}, Top5 Acc{}".format(str(bestacc[0]), str(bestacc[1])))
        torch.cuda.empty_cache()
        gc.collect()

    print("[BEST] Top1 Acc:{}, Top5 Acc{}".format(str(bestacc[0]), str(bestacc[1])))
    if args.multigpu:
        torch.save(bestmodel.module.state_dict(), model_path)
    else:
        torch.save(bestmodel.state_dict(), model_path)