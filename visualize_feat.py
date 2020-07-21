#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:06:14 2020

@author: aj611
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from skimage import io
from PIL import Image
from skimage.util import random_noise
from skimage.transform import rotate
from skimage import exposure

from torchvision import transforms
from torchvision import models as torch_models

from models import BaselineNet

ANGLE = 180
VARIANCE = 0.0025
ADAPT_HIST_CLIP_VAL = 0.017
LOWER = 0.05
UPPER = 99.05

out_c = 3
resnet_layers = 18
train_res4 = False

# load the model
#model_path = '/freespace/local/aj611/code_miccai_lit_review/experiments_ct_baseline/fib/baseline_ct_1/checkpoint_best.pth.tar'

#model = BaselineNet(out_c, resnet_layers, train_res4)
#model.cuda()

# loading trained model
#best_checkpoint = torch.load(model_path)
#model.state_dict_load(best_checkpoint['state_dict'])

model = torch_models.resnet18(pretrained=True)
print(model)

model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list

# get all the model children as list
model_children = list(model.children())

# counter to keep count of the conv layers
counter = 0 

# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")

# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    
    # visualize the first conv layer filters
plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig('./output/filter.png')
plt.show()

# read and visualize an image
#img = cv.imread(f"../input/{args['image']}")
#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.show()

#img = io.imread('krishna.jpg')
img = io.imread('12.png')

'''
img_aug = rotate(img, ANGLE)
plt.imshow(img_aug, cmap = 'gray')
plt.show()

img_aug = random_noise(img, var=VARIANCE)
plt.imshow(img_aug, cmap = 'gray')
plt.show()

img_aug = exposure.equalize_hist(img)
plt.imshow(img_aug, cmap = 'gray')
plt.show()

img_aug = exposure.equalize_adapthist(img, clip_limit=ADAPT_HIST_CLIP_VAL)
plt.imshow(img_aug, cmap = 'gray')
plt.show()

p1, p2 = np.percentile(img, (LOWER, UPPER))
img_aug = exposure.rescale_intensity(img, in_range=(p1, p2))
plt.imshow(img_aug, cmap = 'gray')
plt.show()

img_aug = exposure.adjust_gamma(img, 2)
plt.imshow(img_aug, cmap = 'gray')
plt.show()

img_aug = exposure.adjust_log(img, 1)
plt.imshow(img_aug, cmap = 'gray')
plt.show()
'''

# define the transforms
transform = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

#img = Image.fromarray(img.astype('uint8'), 'L').convert('RGB')
img = Image.fromarray(img.astype('uint8'), 'L').convert('RGB')
#img = np.array(img)
# apply the transforms
img = transform(img)
print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())

# pass the image through all the layers
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))

# make a copy of the `results`
outputs = results

# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"./output/layer_{num_layer}.png")
    # plt.show()
    plt.close()
