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

class SpatialBackbone(nn.Module):
        def __init__(self):
                super().__init__()

                self.path = Path.cwd() / "model/sam_vit_b_01ec64.pth"
                self.sam =  sam_model_registry["vit_b"](checkpoint=self.path)
                # self.enc_model = SamPredictor(self.sam)
                breakpoint()
                # self.sam.image_encoder.patch_embed.proj = nn.Conv2d(1, 1280, kernel_size=(16, 16), stride=(16, 16))
                self.sam.cuda()

        def forward(self, x):
                # sam.pixel_mean.to(device=x.device)
                # sam.pixel_std.to(device=x.device)

                breakpoint()
                snippets = [snippet.repeat(1, 3, 1, 1) for snippet in x] # grab individual snippets, duplicate along channel dimension
                # snippets = [snippet for snippet in x] # grab individual snippets, duplicate along channel dimension
                # breakpoint()
                #x = self.convs(torch.cat(frames))
                #breakpoint()
                #self.final_fc.to(x.device)
                #x = x.squeeze()
                #x = self.final_fc(x)
                # breakpoint()
                # x = resnet_model.last_linear(x)
                # x = self.avg_p(x)
                breakpoint()

                embeddings = []

                # for s in snippets:
                #     for f in s:
                #         p = self.preprocess(f).unsqueeze(0)
                #         e = self.sam.image_encoder(p)
                #         embeddings.append(e)
                # breakpoint()

                breakpoint()
                preprocessed = [torch.stack([self.sam.preprocess(f) for f in s], axis=0)  for s in snippets]
                # preprocessed = snippets
                embeddings = [self.sam.image_encoder(p) for p in preprocessed]

                # # for s in snippets:
                #     breakpoint()
                #     # enc_model.set_image(s)
                #     e = sam.get_image_embedding()

                #     embeddings.append(e)
                #     # x = x.view(x.shape[0], -1)
                    #x.shape is frames x flat_feature_vector

                x = torch.cat(embeddings)
                    
                return x

        def preprocess(self, x):
            """
            Analogue of preprocess method for Sam class.
            Just resizes the input, does not normalize with respect to mean
            and stddev, as that is intended for RGB images (not thermal).
            """
            # Pad
            h, w = x.shape[-2:]
            print(f"{self.sam.image_encoder.img_size=}")
            padh = self.sam.image_encoder.img_size - h
            padw = self.sam.image_encoder.img_size - w

            x = F.pad(x, (0, padw, 0, padh))
            return x

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
                self.cnn                  = SpatialBackbone() #initialize SpatialBackbone
                self.lstm_layer   = nn.LSTM(self.in_dim, self.h_dim, self.num_l, batch_first=True)
                self.detected_pep = pep_detector(30, 4) #initialize linear layers
                self.stress               = Classifier(fps)

        def forward(self, x):
                #breakpoint()
                batch_size, timesteps, C, H, W = x.size()
                x = self.cnn(x)
                #timestamp/15 as frame rate is 15 fps. we will push 1 second info to lstm as 1 seq
                x = x.view(batch_size, timesteps//self.frame_rate, -1)
                x_out, (h_o, c_o) = self.lstm_layer(x)
                x_out = x_out[-1].view(batch_size, timesteps, -1).squeeze()
                x_out = self.detected_pep(x_out)
                x_out2 = self.stress(x_out)

                return x_out, x_out2
        
if __name__ == '__main__':

        lstm = Merge_LSTM(256, 6, 3).cuda()
        print(lstm)
        inputs = torch.rand(1,499,3,240,200).float().cuda()
        #import pdb; pdb.set_trace()
        out = lstm(inputs)

                
