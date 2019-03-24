from __future__ import absolute_import, division, print_function

## Import Python packages
import sys
import numpy as np

## Import pytorch packages

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

class AdvSolver(object):
    def __init__ (self, net, eps, loss):
        self.net = net
        self.eps = eps
        self.loss = loss
        
    def fgsm(self, x, target, device, x_val_min = -1., x_val_max = 1.):
        '''
        x: input image data
        target: labels (target) of the input data
        val_min: the lower bound of the input data
        val_max: the upper bound of the input data
        '''
        x, target = x.to(device), target.to(device)
                
        x_adv = Variable(x.data, requires_grad=True)
        y_adv = self.net(x_adv)
        
        # Calculate the loss
        cost = self.loss(y_adv, target)
        
        # Zero all existing gradients
        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
            
        # Calculate gradients of model in backward pass
        cost.backward()
        
        # Collect datagrad
        x_adv_grad = x_adv.grad.data
        
        # FGSM step: only one gradient ascent step
        x_adv = x_adv + self.eps * x_adv_grad.sign()
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        
        y_adv = self.net(x_adv)
        y_adv_pred = y_adv.max(1, keepdim=True)[1] # get the index of the max log-probability

        return x_adv, y_adv_pred
    

class GenAdv(object):
    def __init__(self, net, device, loss, adv_iter=100, method='fgsm'):
        self.net = net
        self.device = device
        self.adv_iter = adv_iter
        self.method = method
        
        # define loss function, e.g. cross_entropy
        self.loss = loss
        
    def generate_adv(self, eps, data, target):
        '''
        eps: the learning rate to generate adversary example
        data: the inpout initial data
        target:  the targets (labels) of the input data
        '''
        if self.method = 'fgsm':
            data_adv, target_adv = AdvSolver(self.net, eps, self.loss).fgsm(data, target, self.device)
            
            
        return data_adv, target_adv
            
       