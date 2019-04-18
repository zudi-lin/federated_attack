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
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        
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
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            
            x_adv.requires_grad = True

        y_adv = self.net(x_adv)
        y_adv_pred = y_adv.max(1, keepdim=True)[1] # get the index of the max log-probability

        return x_adv, y_adv_pred
        

class GenAdv(object):
    def __init__(self, net, device, criterion, adv_iter=100, method='fgsm'):
        self.net = net
        self.net.eval()
        self.device = device
        self.adv_iter = adv_iter
        self.method = method
        
        # define criterion function, e.g. cross_entropy
        self.criterion = criterion
        
    def generate_adv(self, data, y, target=False, eps=0.01):
        '''
        eps: the learning rate to generate adversary example
        data: the inpout initial data
        y:  the targets (labels) of the input data
        '''
        if self.method == 'fgsm':
            x_adv, y_adv = AdvSolver(self.net, eps, self.criterion).fgsm(data, y, target=target, self.device)
        elif self.method == 'i_fgsm':
            x_adv, y_adv = AdvSolver(self.net, eps, self.criterion).i_fgsm(data, y, target=target, self.device)
            
        return x_adv, y_adv
 
def aggregate_adv_noise(x, adv_noise, method='uniform'):
    if method = 'uniform'
        x_adv_aggregate = x + torch.mean(adv_noise)
    
    return x_adv_aggregate
    
    
    
       
### Debugging
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Preparing data, decompose the features and labels.
    # x, y = 
    
    nets = [densenet_cifar.to(device), resnet34.to(device), VGG('VGG11').to(device)]
    adv_noise = torch.tensor([], device=device)
    for i in range(nets):
        net = nets[i]
        criterion = F.cross_entropy
        Generate_Adv = GenAdv(net, device, criterion)
        x_adv, _ = Generate_Adv.generate_adv(x, y)
        
        adv_noise = torch.cat([adv_noise, x_adv - x])
    
    x_adv_aggregate = aggregate_adv_noise(x, adv_noise)
    
    


