import torch
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, BeitFeatureExtractor
img_to_tensor = transforms.ToTensor()


def loaddatavireo(textfile, loadingre):
    path = "ready_chinese_food"
    textfile = open(textfile, "r", encoding="utf-8").readlines()
    if loadingre:
        data = list()
        for line in textfile:
            line = line[:-1].split("\t")
            imgpath = path + line[0]
            label = int((line[0].split("/")[-2]))
            ingres = line[1].split(" ")
            data.append((imgpath, label, ingres))
        #data = [(path + line[:-1].split("\t")[0], int(line[:-1].split("\t")[0].split("/")[-2]), line[:-1].split("\t")[1].split(" ")) for line in textfile]
    else:
        data = [(path + line[:-1], int(line[:-1].split("/")[-2])) for line in textfile]
    random.shuffle(data)
    return data

def loaddatafoodnet(textfile):
    path = "foodnet"
    datatype = textfile[9:-9] + "/"
    textfile = open(textfile, "r", encoding="utf-8").readlines()
    data = [(path+datatype+line[:-1].split(" ")[0], int(line[:-1].split(" ")[-1])) for line in textfile]
    return data

class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, args):
        self.args = args
        if args.dataset == "foodnet":
            self.data = loaddatafoodnet(filepath)
        else:
            self.data = loaddatavireo(filepath, args.loadingre)
        if args.model == "beit":
            self.feature_extractor = BeitFeatureExtractor.from_pretrained(args.model_name_or_path)
        else:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name_or_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        image = Image.open(data[0])
        image = image.convert("RGB")
        imgTensor = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        label = data[1]
        
        if self.args.dataset != "foodnet":
            label -= 1
        if not self.args.loadingre:
            return (imgTensor, label)
        else:
            ingres = [int(item) for item in data[2]]
            ingrearray = np.zeros(353)
            for pos in ingres:
                ingrearray[pos] = 1
            ingreTensor = torch.Tensor(ingrearray)
            return (imgTensor, label, ingreTensor)
