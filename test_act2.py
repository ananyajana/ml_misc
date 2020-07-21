#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:22:33 2020

@author: aj611
"""
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import cv2

def returnCAM(feature_conv, weight_softmax, class_idx):
    #generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
        
def get_cam(net, feature_blobs, img_pil, root_img):
    # these are for inception model
    #params = list(net.parameters())
    #weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    
    weight_softmax_params = list(net._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    
    normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )
    
    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
            ])
    
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)
    
    h_x = F.softmax(logit, dim=1).data.squeeze()
    print('type of h_x: ', type(h_x))
    #print('h_x: ', h_x)
    probs, idx = h_x.sort(0, True)
    print('idx: ', idx)
    #print('probs: ', probs)
    print('[idx[0].item()]', [idx[0].item()])
    CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
    if CAMs is None:
        print('CAMS not found')
        return
    
    # render the CAM and output
    print('output CAM.jpg for the top1 prediction')
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap*0.3 + img*0.5
    cv2.imwrite('cam.jpg', result)

def hook_feature(module, input, output):
    feature_blobs.append(output.data.cpu().numpy())

net = models.resnet18(pretrained = True)
net.cuda()
net.eval()
final_conv = 'layer4'
net._modules.get(final_conv).register_forward_hook(hook_feature)


feature_blobs = [] 
root = 'cat.jpg'
img = Image.open(root)
get_cam(net, feature_blobs, img, root)