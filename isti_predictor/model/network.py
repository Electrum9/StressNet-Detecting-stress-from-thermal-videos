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
resnet_model = torchvision.models.resnet50(pretrained=True, progress=True).float()

class CNN(nn.Module):
        def __init__(self):
                super().__init__()
                #loading blocks of ResNet
                resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                blocks     = list(resnet_model.children())[0:8]
                self.convs = nn.Sequential(*blocks)     
                self.avg_p = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        def forward(self, x):
                frames = [frame for frame in x]
                x = self.convs(torch.cat(frames))
                x = self.avg_p(x)
                x = x.view(x.shape[0], -1)
                #x.shape is frames x flat_feature_vector
                
                return x

class OutputLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=1000,
        representation_size=None,
        cls_head=False,
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes
        modules = []
        if representation_size:
            modules.append(nn.Linear(embedding_dim, representation_size))
            modules.append(nn.Tanh())
            modules.append(nn.Linear(representation_size, num_classes))
        else:
            modules.append(nn.Linear(embedding_dim, num_classes))

        self.net = nn.Sequential(*modules)

        if cls_head:
            self.to_cls_token = nn.Identity()

        self.cls_head = cls_head
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                if module.weight.shape[0] == self.num_classes:
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        breakpoint()
        if self.cls_head:
            x = self.to_cls_token(x[0,:])
        else:
            """
            Scaling Vision Transformer: https://arxiv.org/abs/2106.04560
            """
            x = torch.mean(x, dim=1)

        return self.net(x)

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

class MyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        self.transformer_model = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                                dropout, custom_encoder=transformer_encoder, custom_decoder=transformer_decoder)


    def forward(self, src):
        out = self.transformer_model.encoder(src)
        return out

class Merge_LSTM(nn.Module):
        def __init__(self, in_dim, h_dim, num_l, frame_rate, fps):
                super().__init__()
                self.in_dim             = in_dim
                self.h_dim              = h_dim
                self.num_l              = num_l
                self.frame_rate = frame_rate
                self.cnn                  = CNN() #initialize CNN
                self.transformer = MyTransformer(d_model=2048)
                #self.lstm_layer   = nn.LSTM(self.in_dim, self.h_dim, self.num_l, batch_first=True)
                self.detected_pep = pep_detector(30, 4) #initialize linear layers
                self.stress               = Classifier(fps)

                self.post_transformer_ln = nn.LayerNorm(2048)

                self.cls_layer = OutputLayer(2048, num_classes=2, representation_size=1024, cls_head=True)

        def forward(self, x):
                batch_size, timesteps, C, H, W = x.size()
                x = self.cnn(x)
                #timestamp/15 as frame rate is 15 fps. we will push 1 second info to lstm as 1 seq
                curr_device = x.device
                cls_token = nn.Parameter(torch.zeros(1, 2048)).to(curr_device)
                pos_embed = nn.Parameter(torch.zeros(46, 2048)).to(curr_device)
                x = torch.cat((cls_token, x), dim=0)
                x += pos_embed

                breakpoint()
                x = self.transformer(x)
                x = self.post_transformer_ln(x)
                x = self.cls_layer(x)

                #x = x.view(batch_size, timesteps//self.frame_rate, -1)
                #x_out, (h_o, c_o) = self.lstm_layer(x)
                #x_out = x_out[-1].view(batch_size, timesteps, -1).squeeze()
                #x_out = self.detected_pep(x_out)
                #x_out2 = self.stress(x_out)
                
                return x_out, x_out2
        
if __name__ == '__main__':

        lstm = Merge_LSTM(256, 6, 3).cuda()
        print(lstm)
        inputs = torch.rand(1,499,3,240,200).float().cuda()
        #import pdb; pdb.set_trace()
        out = lstm(inputs)

                
