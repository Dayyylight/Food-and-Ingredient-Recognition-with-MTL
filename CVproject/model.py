import copy
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, BeitForImageClassification, ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ViT_Ingre(nn.Module):
    def __init__(self, args):
        super(ViT_Ingre, self).__init__()
        if args.model == "vit-base-2":
            self.vit = ViTModel.from_pretrained(args.model_name_or_path)
        else:
            self.vit = ViTForImageClassification.from_pretrained(args.model_name_or_path)
        self.vit.classifier = nn.Linear(self.vit.config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = args.num_labels
        self.args = args
        self.vit.config.hidden_dropout_prob = 0.1
        self.vit.config.attention_probs_dropout_prob = 0.1

    #define a forward pass through that architecture + loss computation
    def forward(self, pixel_values, labels):
        if self.args.model == "vit-base-2":
            feature = self.vit(pixel_values=pixel_values).pooler_output#[:,0]
            feature = self.dropout(feature)
            logits = self.vit.classifier(feature)
        else:
            logits = self.vit(pixel_values=pixel_values).logits
        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

class MultitaskOutput():
    def __init__(self, foodloss:torch.FloatTensor, ingreloss:torch.FloatTensor, foodlogits:torch.FloatTensor, ingrelogits:torch.FloatTensor):
        self.foodloss = foodloss
        self.ingreloss = ingreloss
        self.foodlogits = foodlogits
        self.ingrelogits = ingrelogits

class ViT_MultiTask(nn.Module):
    def __init__(self, args):
        super(ViT_MultiTask, self).__init__()
        if "384" in args.model:
            self.VitModel = ViTForImageClassification.from_pretrained(args.model_name_or_path)
        else:
            self.VitModel = ViTForImageClassification.from_pretrained(args.model_name_or_path,  cache_dir=args.cache_dir)

        shared_hiddensize = self.VitModel.config.hidden_size
        private_hiddensize = int(shared_hiddensize/2)
        self.VitModel.classifier = nn.Linear(self.VitModel.config.hidden_size, shared_hiddensize)
        self.foodfc = nn.Linear(shared_hiddensize, private_hiddensize)
        self.ingrefc = nn.Linear(shared_hiddensize, private_hiddensize)
        self.foodclassifier = nn.Linear(private_hiddensize, args.num_labels)
        self.ingreclassifier = nn.Linear(private_hiddensize, args.num_ingres)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.Tanh()
        self.shapredbn = nn.BatchNorm1d(shared_hiddensize)
        self.privatebn = nn.BatchNorm1d(private_hiddensize)
        self.VitModel.config.hidden_dropout_prob = 0.1
        self.VitModel.config.attention_probs_dropout_prob = 0.1
        self.num_labels = args.num_labels
        self.num_ingres = args.num_ingres
        self.args = args

    #define a forward pass through that architecture + loss computation
    def forward(self, pixel_values, labels, ingres):
        shareoutput = self.VitModel(pixel_values=pixel_values).logits
        shareoutput = self.dropout(self.act(self.shapredbn(shareoutput)))
        foodhidden = self.foodfc(shareoutput)
        ingrehidden = self.ingrefc(shareoutput)
        foodhidden = self.dropout(self.act(self.privatebn(foodhidden)))
        ingrehidden = self.dropout(self.act(self.privatebn(ingrehidden)))
        foodlogits = self.foodclassifier(foodhidden)
        ingrelogits = self.ingreclassifier(ingrehidden)

        foodloss = None
        if labels is not None:
            foodloss_fct = nn.CrossEntropyLoss()
            ingreloss_fct = nn.BCEWithLogitsLoss()
            foodloss = foodloss_fct(foodlogits.view(-1, self.num_labels), labels.view(-1))
            ingreloss = ingreloss_fct(ingrelogits, ingres)

        return MultitaskOutput(
            foodloss = foodloss,
            ingreloss = ingreloss,
            foodlogits = foodlogits,
            ingrelogits = ingrelogits,
            )

class ViT_FuseMultiTask(nn.Module):
    def __init__(self, args):
        super(ViT_FuseMultiTask, self).__init__()
        sharedDepth = 4
        if "384" in args.model:
            VitModel = ViTForImageClassification.from_pretrained(args.model_name_or_path)
        else:
            VitModel = ViTForImageClassification.from_pretrained(args.model_name_or_path,  cache_dir=args.cache_dir)
        self.sharedEmbedding = copy.deepcopy(VitModel.vit.embeddings)
        self.sharedEncoder = copy.deepcopy(VitModel.vit.encoder)
        self.sharedEncoder.layer = self.sharedEncoder.layer[:sharedDepth]

        self.FoodEncoder = copy.deepcopy(VitModel.vit.encoder)
        self.FoodEncoder.layer = self.FoodEncoder.layer[sharedDepth:]
        self.IngreEncoder = copy.deepcopy(self.FoodEncoder)
        self.layernorm = copy.deepcopy(VitModel.vit.layernorm)
        
        self.imageclassifier = nn.Linear(VitModel.config.hidden_size * 2, args.num_labels)
        self.ingreclassifier = nn.Linear(VitModel.config.hidden_size, args.num_ingres)
        
        self.num_labels = args.num_labels
        del VitModel

    def forward(self, images, foodlabels, ingrelabels):
        image_embedding = self.sharedEmbedding(images)
        image_feature = self.sharedEncoder(image_embedding).last_hidden_state
        
        food_encoder_outputs = self.FoodEncoder(image_feature).last_hidden_state
        ingre_encoder_outputs = self.IngreEncoder(image_feature).last_hidden_state
        food_sequence_output = food_encoder_outputs[:, 0, :]
        ingre_sequence_output = ingre_encoder_outputs[:, 0, :]
        food_sequence_output = self.layernorm(food_sequence_output)
        ingre_sequence_output = self.layernorm(ingre_sequence_output)
        food_logits = self.imageclassifier(torch.cat((food_sequence_output, ingre_sequence_output),dim=-1))
        ingre_logits = self.ingreclassifier(ingre_sequence_output)

        foodloss_fct = nn.CrossEntropyLoss()
        ingreloss_fct = nn.BCEWithLogitsLoss()
        foodloss = foodloss_fct(food_logits.view(-1, self.num_labels), foodlabels.view(-1))
        ingreloss = ingreloss_fct(ingre_logits, ingrelabels)
        return MultitaskOutput(
            foodloss = foodloss,
            ingreloss = ingreloss,
            foodlogits = food_logits,
            ingrelogits = ingre_logits,
            )

class BEiT_Food(nn.Module):
    def __init__(self, args):
        super(BEiT_Food, self).__init__()
        self.beit = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
        self.beit.classifier = nn.Linear(self.beit.config.hidden_size, args.num_labels)
        self.num_labels = args.num_labels

    #define a forward pass through that architecture + loss computation
    def forward(self, pixel_values, labels):
        logits = self.beit(pixel_values=pixel_values).logits

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )