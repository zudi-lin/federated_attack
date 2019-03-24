from __future__ import absolute_import, division, print_function

## Import Python packages
import sys, os
sys.path.append(os.path.abspath('../'))

import numpy as np

## Import pytorch packages

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from model.cifar import VGG

class AdvSolver(object):
    def __init__ (self, net, eps, criterion):
        self.net = net
        self.eps = eps
        self.criterion = criterion
        
    def fgsm(self, x_adv, target, device, x_val_min = -1., x_val_max = 1.):
        '''
        x_adv: input image data
        target: labels (target) of the input data
        x_val_min: the lower bound of the input data
        x_val_max: the upper bound of the input data
        '''
               
        x_adv.requires_grad = True
        y_adv = self.net(x_adv)
        
        # Calculate the loss
        loss = self.criterion(y_adv, target)
        
        # Zero all existing gradients
        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
            
        # Calculate gradients of model in backward pass
        loss.backward()
        
        # Collect datagrad
        x_adv_grad = x_adv.grad.data
        
        # FGSM step: only one gradient ascent step
        x_adv = x_adv + self.eps * x_adv_grad.sign()
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        
        y_adv = self.net(x_adv)
        y_adv_pred = y_adv.max(1, keepdim=True)[1] # get the index of the max log-probability

        return x_adv, y_adv_pred
    

class GenAdv(object):
    def __init__(self, net, device, criterion, adv_iter=100, method='fgsm'):
        self.net = net
        self.device = device
        self.adv_iter = adv_iter
        self.method = method
        
        # define criterion function, e.g. cross_entropy
        self.criterion = criterion
        
    def generate_adv(self, data, target, eps=0.001):
        '''
        eps: the learning rate to generate adversary example
        data: the inpout initial data
        target:  the targets (labels) of the input data
        '''
        if self.method == 'fgsm':
            data_adv, target_adv = AdvSolver(self.net, eps, self.criterion).fgsm(data, target, self.device)
            
            
        return data_adv, target_adv
            
       
### Debugging
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VGG('VGG11').to(device)
    x = torch.randn(2,3,32,32).to(device)
    y = net(x).to(device)
    print(y.size())
    y = y.max(1, keepdim=False)[1]
    print(y.size())
    criterion = F.cross_entropy
    Generate_Adv = GenAdv(net, device, criterion)
    data_adv, _ = Generate_Adv.generate_adv(x, y)
    # print(x)
    print(torch.sum(torch.abs(x-data_adv)))
  


