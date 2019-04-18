from __future__ import absolute_import, division, print_function

## Import Python packages
import sys, os
sys.path.append(os.path.abspath('../'))

import numpy as np

## Import pytorch packages

import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from utlis import recreate_image

from model.cifar import *

class AdvSolver(object):
    def __init__ (self, net, eps, criterion):
        self.net = net
        self.eps = eps
        self.criterion = criterion
        
    def fgsm(self, x_adv, y, device, target=False, x_val_min = -1., x_val_max = 1.):
        '''
        Implementation of Fast Gradient Sign Method 
        x_adv: input image data
        y: labels (target) of the input data
        target: boolen function to indicate whether we 
        x_val_min: the lower bound of the input data
        x_val_max: the upper bound of the input data
        '''
               
        x_adv.requires_grad = True
        y_adv = self.net(x_adv)
        
        # Calculate the loss
        if target:
            loss = self.criterion(y_adv, y)
        else:
            loss = - self.criterion(y_adv, y)
        
        # Zero all existing gradients
        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
            
        # Calculate gradients of model in backward pass
        loss.backward()
        
        # Collect datagrad
        x_adv_grad = x_adv.grad.data
        
        # FGSM step: only one gradient ascent step
        x_adv = x_adv - self.eps * x_adv_grad.sign()
        # x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        
        y_adv = self.net(x_adv)
        y_adv_pred = y_adv.max(1, keepdim=True)[1] # get the index of the max log-probability

        return x_adv, y_adv_pred
        
    def i_fgsm(self, x_adv, y, device, target=False, alpha=1., iteration=1, x_val_min = -1., x_val_max = 1.):
        x = x_adv.clone()
        x_adv.requires_grad = True
        for i in range(iteration):
            y_adv = self.net(x_adv)
            if target:
                loss = self.criterion(y_adv, y)
            else:
                loss = - self.criterion(y_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward()

            # x_adv.grad.sign_()
            # x_adv = x_adv + alpha * x_adv.grad
            
            # Collect datagrad
            x_adv_grad = x_adv.grad.data
            
            x_adv = x_adv - alpha * x_adv_grad.sign()
            x_adv = where(x_adv > x+self.eps, x+self.eps, x_adv)
            x_adv = where(x_adv < x-self.eps, x-self.eps, x_adv)
            # x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            
            x_adv.requires_grad = True

        y_adv = self.net(x_adv)
        y_adv_pred = y_adv.max(1, keepdim=True)[1] # get the index of the max log-probability

        return x_adv, y_adv_pred
        

class GenAdv(object):
    def __init__(self, net, device, criterion, eps=0.01, adv_iter=100, method='fgsm'):
        self.net = net
        self.net.eval()
        self.device = device
        self.adv_iter = adv_iter
        self.method = method
        self.eps = eps
        
        # define criterion function, e.g. cross_entropy
        self.criterion = criterion
        
    def generate_adv(self, data, y, target=False):
        '''
        data: the inpout initial data
        y:  the targets (labels) of the input data
        '''
        if self.method == 'fgsm':
            x_adv, y_adv = AdvSolver(self.net, self.eps, self.criterion).fgsm(data, y, self.device, target=target)
        elif self.method == 'i_fgsm':
            x_adv, y_adv = AdvSolver(self.net, self.eps, self.criterion).i_fgsm(data, y, self.device, target=target)
        
        if target:
            return x_adv, y_adv, y
        else: 
            return x_adv, y_adv
 
    def aggregate_adv_noise(self, x, adv_noise, method='uniform'):
        if method == 'uniform'：
            x_adv_aggregate = x + torch.mean(adv_noise)
            x_adv_aggregate = where(x_adv_aggregate > x+self.eps, x+self.eps, x_adv_aggregate)
            x_adv_aggregate = where(x_adv_aggregate < x-self.eps, x-self.eps, x_adv_aggregate)
        
        return x_adv_aggregate
    
       
# main code
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Preparing data, decompose the features and labels.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

    for _, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        break
    
    print(y)
    dir_name = 'model/cifar/pretrained/checkpoint'

    # Load DenseNet
    densenet = DenseNet121().to(device)
    dense_checkpoint = torch.load(dir_name + '/DenseNet.t7')
    densenet.load_state_dict(dense_checkpoint['net'])
    
    # Load ResNet
    resnet = ResNet18().to(device)
    res_checkpoint = torch.load(dir_name + '/DenseNet.t7')
    resnet.load_state_dict(res_checkpoint['net'])
    
    # Load MobileNet
    mobilenet = MobileNet().to(device)
    mobile_checkpoint = torch.load(dir_name + '/DenseNet.t7')
    mobilenet.load_state_dict(mobile_checkpoint['net'])
    
    nets = [densenet, resnet, mobilenet]
    adv_noise = torch.tensor([], device=device)
    for i in range(nets):
        net = nets[i]
        criterion = F.cross_entropy
        Generate_Adv = GenAdv(net, device, criterion)
        x_adv, y_adv = Generate_Adv.generate_adv(x, y)
        
        print(y_adv)
        adv_noise = torch.cat([adv_noise, x_adv - x])
    
    x_adv_aggregate = Generate_Adv.aggregate_adv_noise(x, adv_noise)
    recreated_image = recreate_image(x_adv_aggregate)
    
    noise_image = recreated_image - x
    
    if not os.path.exists('../generated_images'):
            os.makedirs('../generated_images')
    
    for i in range(len(recreated_image)):
        cv2.imwrite('../generated_images/noise_image_' + str(i) + '.jpg', noise_image[i])
        cv2.imwrite('../generated_images/recreated_image_' + str(i) + '.jpg', recreated_image[i])
    
