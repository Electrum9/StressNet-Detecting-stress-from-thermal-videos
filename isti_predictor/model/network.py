        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 2 13:11:19 2020

@author: satish
"""
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torchvision
from torchvision import models
import torch.nn.functional as F
from model.pep_detector import pep_detector
print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Dropout

print("Loading ResNet Model")
start_time = time.time()
# resnet_model = torchvision.models.resnet50(pretrained=True, progress=True).float()

from facenet_pytorch import MTCNN, InceptionResnetV1
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path

torch.cuda.empty_cache()

sam_path = Path.cwd() / "model/sam_vit_b_01ec64.pth"
sam =  sam_model_registry["vit_b"](checkpoint=sam_path)

class ConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))

    def forward(self, x):
        preprocessed = [torch.stack([self.preprocess(f) for f in s], axis=0)  for s in x] # preprocess all frames in each snippet for the given batch
        out = [self.conv_layer(s) for s in preprocessed]

        return torch.cat(out)

    def preprocess(self, x):
        """
        Analogue of preprocess method for Sam class.
        Just resizes the input, does not normalize with respect to mean
        and stddev, as that is intended for RGB images (not thermal).
        """
        # Pad
        h, w = x.shape[-2:]
        
        padh = 1024 - h
        padw = 1024 - w

        x = F.pad(x, (0, padw, 0, padh)).to(torch.half)
        print(x.dtype)
        return x

class FeatureExtractor(nn.Module):
        def __init__(self):
                super().__init__()

                # self.path = Path.cwd() / "model/sam_vit_b_01ec64.pth"
                # self.sam =  sam_model_registry["vit_b"](checkpoint=self.path)
                # self.enc_model = SamPredictor(
                breakpoint()
                self.image_encoder = sam.image_encoder
                children = list(self.image_encoder.children())[1:]
                self.first = nn.Sequential(*children[0]) # extract out layers in ModuleList, cascade them
                self.rest = nn.Sequential(self.first, children[1])
                
        def forward(self, x):
            breakpoint()
            embeddings = []

            for s in x:
                embed = self.rest(s)
                embeddings.append(embed)

            embeddings = torch.cat(embeddings)
            return embeddings


class Classifier(nn.Module):
        def __init__(self, pred_isti, scale=0.5):
                super().__init__()
                self.representation_size = int(pred_isti*scale)

                #self.Attn = Attention(pred_isti)
                self.fc1  = nn.Linear(pred_isti, self.representation_size)
                self.fc2  = nn.Linear(self.representation_size, 1)
                self.sig  = nn.Sigmoid()

        def forward(self, x1):
                x1 = torch.squeeze(x1)
                x1 = F.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                x1 = self.sig(x1)

                return x1


class Merge_LSTM(nn.Module):
        def __init__(self, in_dim, h_dim, num_l, frame_rate, fps):
                super().__init__()
                self.in_dim             = in_dim
                self.h_dim              = h_dim
                self.num_l              = num_l
                self.frame_rate = frame_rate
                self.embedding = FeatureExtractor() #initialize SpatialBackbone
                self.lstm_layer   = nn.LSTM(self.in_dim, self.h_dim, self.num_l, batch_first=True)
                self.detected_pep = pep_detector(30, 4) #initialize linear layers
                self.stress               = Classifier(fps)

        def forward(self, x):
                #breakpoint()
                batch_size, timesteps, C, H, W = x.size()
                x = self.embedding(x)
                breakpoint()
                #timestamp/15 as frame rate is 15 fps. we will push 1 second info to lstm as 1 seq
                x = x.view(batch_size, timesteps//self.frame_rate, -1)
                x_out, (h_o, c_o) = self.lstm_layer(x)
                x_out = x_out[-1].view(batch_size, timesteps, -1).squeeze()
                x_out = self.detected_pep(x_out)
                x_out2 = self.stress(x_out)

                return x_out, x_out2

class model_parallel(nn.Module):
    def __init__(self, in_dim, h_dim, num_l, frame_rate, fps):
        super().__init__()
        self.sub_network1 = ConvBackbone()
        self.sub_network2 = Merge_LSTM(in_dim, h_dim, num_l, frame_rate, fps)
        breakpoint()

        self.sub_network1.cuda(1).half()
        self.sub_network2.cuda(0).half()

    def forward(self, x):
        breakpoint()
        x = x.cuda(1)
        x = self.sub_network1(x)
        breakpoint()
        x = x.cuda(0).unsqueeze(0)
        x = self.sub_network2(x)
        return x
        
if __name__ == '__main__':

        lstm = Merge_LSTM(256, 6, 3).cuda()
        print(lstm)
        inputs = torch.rand(1,499,3,240,200).float().cuda()
        #import pdb; pdb.set_trace()
        out = lstm(inputs)

                
