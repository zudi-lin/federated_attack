'''Prepare Target Images from CIFAR10'''
from __future__ import print_function

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sys
sys.path.append(os.path.abspath('../'))
from datasets import FederatedCIFAR10


adv_dataset = FederatedCIFAR10(root='../data', train=False, download=True, transform=None, sample_type='random')
adv_data = adv_dataset.export_data()
print(adv_data.keys())
print('data shape: ', adv_data['image'].shape, adv_data['label'].shape)

fl = h5py.File('adv_data.h5', 'w')
fl.create_dataset('image', data=adv_data['image'], compression='gzip')
fl.create_dataset('label', data=adv_data['label'], compression='gzip')
fl.close()