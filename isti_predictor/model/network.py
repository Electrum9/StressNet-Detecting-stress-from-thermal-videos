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
# from model.pep_detector import pep_detector
print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Dropout

print("Loading ResNet Model")
start_time = time.time()
# resnet_model = torchvision.models.resnet50(pretrained=True, progress=True).float()
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
# breakpoint()

class NestedTensor(object):
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        # cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class Merge_LSTM(nn.Module):
    def __init__(self, *args):
        super().__init__()

        self.spatial_backbone = model.backbone
        self.encoder = model.transformer.encoder
        self.proj = model.input_proj
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))

        self.classifier = nn.Sequential(nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1)
                                       )

        for name, module in self.classifier.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                nn.init.normal_(module.bias)

    
    def forward(self, x):
        # breakpoint()
        x = x.squeeze(0)
        masks = torch.ones(x.shape).squeeze(1).to(x.device)
        x = x.repeat(1,3,1,1)
        nested = NestedTensor(x, masks)
        # frames = [NestedTensor(frame.repeat(1, 1, 3, 1, 1), torch.ones(frame.shape[-2:]).to(frame.device))  for frame in x]
        features, pos_embeddings = self.spatial_backbone(nested)
        src, masks = features[-1].decompose()
        src = model.input_proj(src) # condense down to 256 channels
        src = self.avgpool(src) # condense further spatially
        masks = self.maxpool(masks.to(dtype=src.dtype))
        src = src.flatten(2).permute(0,2,1)

        cls_token = nn.Parameter(torch.zeros((1, 1, src.shape[-1]))).to(src.device)
        src = torch.cat([cls_token, src], dim=0)
        masks = torch.cat([torch.ones(1,1,1).to(masks.device), masks], dim=0)
        pos_embeddings = nn.Parameter(torch.zeros(src.shape)).to(src.device) # generate our own positional embedding
        # pos_embeddings = pos_embeddings[-1].flatten(2).permute(0,2,1)
        # pos_embeddings = torch.mean(pos_embeddings, dim=1).unsqueeze(1)
        masks = masks.flatten(1).permute(1,0)

        src = self.encoder(src, src_key_padding_mask=masks, pos=pos_embeddings)
        # breakpoint()
        final = self.classifier(src[0,:]) # only operate on very first token

        return final

#class Merge_LSTM(nn.Module):
#        def __init__(self, in_dim, h_dim, num_l, frame_rate, fps):
#                super().__init__()
#                self.in_dim             = in_dim
#                self.h_dim              = h_dim
#                self.num_l              = num_l
#                self.frame_rate = frame_rate
#                self.cnn                  = DETR_Backbone() #initialize CNN
#                # self.transformer = MyTransformerEncoder(d_model=2048)
#                #self.lstm_layer   = nn.LSTM(self.in_dim, self.h_dim, self.num_l, batch_first=True)
#                # self.detected_pep = pep_detector(30, 4) #initialize linear layers

#                self.post_transformer_ln = nn.LayerNorm(2048)

#                self.cls_layer = OutputLayer(2048, num_classes=2, representation_size=1024, cls_head=True)

#        def forward(self, x):
#                #breakpoint()
#                batch_size, timesteps, C, H, W = x.size()
#                x = self.cnn(x)
#                #timestamp/15 as frame rate is 15 fps. we will push 1 second info to lstm as 1 seq
#                curr_device = x.device
#                cls_token = nn.Parameter(torch.zeros(1, 2048)).to(curr_device)
#                # pos_embed = nn.Parameter(torch.zeros(46, 2048)).to(curr_device)

#                PE = torch.zeros((46, 2048))

#                pos = torch.arange(46).unsqueeze(1)
#                div_term = torch.exp(torch.arange(0, 2048, 2) * (-np.log(10000.0) / 2048))

#                PE[:, 0::2] = torch.sin(pos * div_term)
#                PE[:, 1::2] = torch.cos(pos * div_term)

#                pos_embed = nn.Parameter(PE).to(curr_device)


#                x = torch.cat((cls_token, x), dim=0)
#                x += pos_embed

#                #breakpoint()
#                # x = self.transformer(x)
#                x = self.post_transformer_ln(x)
#                x = self.cls_layer(x)

#                #x = x.view(batch_size, timesteps//self.frame_rate, -1)
#                #x_out, (h_o, c_o) = self.lstm_layer(x)
#                #x_out = x_out[-1].view(batch_size, timesteps, -1).squeeze()
#                #x_out = self.detected_pep(x_out) # info for generating isti signal
#                #x_out2 = self.stress(x_out) # stress class
                
#                return x
        
if __name__ == '__main__':

        lstm = Merge_LSTM(256, 6, 3).cuda()
        print(lstm)
        inputs = torch.rand(1,499,3,240,200).float().cuda()
        #import pdb; pdb.set_trace()
        out = lstm(inputs)

                
