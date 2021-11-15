import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import random

import os
import GPUtil
import numpy as np
import random
import torch.backends.cudnn as cudnn

from cams.Taylor_cam import *
from cams.Multi_cam import *
from cams.calibration_util import *

from PIL import Image, ImageDraw
from models.relevance import vgg,resnet


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

testset = torchvision.datasets.ImageNet('./Data/imagenet', 
                            split='val', download=True, transform=transform)

test_model = 'vgg'

if test_model == 'vgg':
    model = vgg.vgg16_bn(pretrained = True).cuda()
else:
    model = resnet.resnet50(pretrained = True).cuda()
    
input_path = './samples/'

output_path = './Results/'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

files = os.listdir(input_path)
    
model.eval()

for imgname in files:
    if imgname.endswith('jpg'):
        input_img = input_path + imgname
        
    img = Image.open(input_img).convert("RGB")
    
    img_tensor = img.copy()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_tensor)
    img_tensor = img_tensor.unsqueeze(axis=0)

    output = F.softmax(model(img_tensor.cuda()))
    label = output.argmax()

    target_confi = output.T[label].detach().cpu().numpy().squeeze()

    taylor_cam = Taylor_cam(test_model,model,img_tensor,label,target_confi,testset)
    taylor_cam_best,taylor_inc_list = get_best_threshold(taylor_cam,model,output,label,img_tensor)
    
    grad_cam,grad_cam_pp,relevance_cam,score_cam = multi_CAM(test_model,model,img_tensor)
    
    img_tensor2 = img.copy()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    target_img = transform(img_tensor2).numpy().transpose(1,2,0)
    
    save_path = output_path + imgname[:-4]
    
    fig = plt.figure(figsize=(15, 15))
    plt.subplots_adjust(bottom=0.01)

    plt.subplot(2, 7, 1)
    plt.imshow(target_img)
    plt.axis("off")
    plt.title('Original')
    
    plt.subplot(2, 7, 2)
    plt.imshow((grad_cam),cmap='seismic')
    plt.imshow(target_img,alpha = 0.3)
    plt.axis("off")
    plt.title('Grad CAM')
    
    plt.subplot(2, 7, 3)
    plt.imshow((grad_cam_pp),cmap='seismic')
    plt.imshow(target_img,alpha = 0.3)
    plt.axis("off")
    plt.title('Grad CAM ++')
    
    plt.subplot(2, 7, 4)
    plt.imshow((score_cam),cmap='seismic')
    plt.imshow(target_img,alpha = 0.3)
    plt.axis("off")
    plt.title('Score CAM')
    
    plt.subplot(2,7,5)
    plt.imshow((relevance_cam),cmap='seismic')
    plt.imshow(target_img,alpha = 0.3)
    plt.axis("off")
    plt.title('Relevance CAM')
    
    plt.subplot(2,7,6)
    plt.imshow((taylor_cam),cmap='seismic')
    plt.imshow(target_img,alpha = 0.3)
    plt.axis("off")
    plt.title('Taylor CAM')

    plt.subplot(2,7,7)
    plt.imshow((taylor_cam_best),cmap='seismic')
    plt.imshow(target_img,alpha = 0.3)
    plt.axis("off")
    plt.title('Calibrated Taylor CAM')
    plt.tight_layout()
    
    plt.draw()
    plt.savefig((save_path +'.jpg'))
    plt.clf()
    plt.close()

#    plt.savefig(save_path,dpi=200,transparent=True, bbox_inches='tight',padding=0) 