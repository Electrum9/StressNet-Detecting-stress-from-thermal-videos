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

print("Loading ResNet Model")
start_time = time.time()
# resnet_model = torchvision.models.resnet50(pretrained=True, progress=True).float()

from facenet_pytorch import MTCNN, InceptionResnetV1

resnet_model = InceptionResnetV1(pretrained='vggface2').float()

class CNN(nn.Module):
        def __init__(self):
                super().__init__()
                #loading blocks of ResNet
                resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # blocks           = list(resnet_model.children())[0:8]
                blocks     = list(resnet_model.children())[:-3]
                resnet_model.last_linear = nn.Linear(in_features=1792, out_features=2048, bias=False)
                self.convs = nn.Sequential(*blocks)     
                # self.avg_p = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        def forward(self, x):
                frames = [frame for frame in x]
                x = self.convs(torch.cat(frames))
                x = resnet_model.last_linear(x)
                # x = self.avg_p(x)
                x = x.view(x.shape[0], -1)
                #x.shape is frames x flat_feature_vector
                
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
                self.cnn                  = CNN() #initialize CNN
                self.lstm_layer   = nn.LSTM(self.in_dim, self.h_dim, self.num_l, batch_first=True)
                self.detected_pep = pep_detector(30, 4) #initialize linear layers
                self.stress               = Classifier(fps)

        def forward(self, x):
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

                
