'''Analysis of model weights'''
from __future__ import print_function
import os, sys
sys.path.append(os.path.abspath('../'))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse

def intra_feature_correlation(X):
    # correlation between feature maps at a conv layer
    pass

def inter_feature_correlation(X,Y):
    # correlation between feature maps from different model
    pass

def visualize_layer(X):
    X_tsne = TSNE(n_components=2).fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.1)
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.1)
    plt.show()

def statistics(X):
    mean = np.mean(X)
    mean_filter = np.mean(X, 0) # mean of each 3x3 filter
    range_of_value = np.ptp(X, 0)

def similarity(X1, X2):
    len1 = X1.shape[0]
    len2 = X2.shape[0]
    print(len1, len2)
    X1_expand = np.stack([X1 for _ in range(len2)], 1)
    X2_expand = np.stack([X2 for _ in range(len1)], 0)
    print(X1_expand.shape, X2_expand.shape)
    assert X1_expand.shape == X2_expand.shape
    diff = X1_expand - X2_expand
    diff_abs = np.abs(diff).sum(2)
    
    plt.figure(figsize=(20,20))
    plt.imshow(diff_abs[:20,:20])
    plt.show()

def get_weights(model):
    conv_weights = []
    for param_tensor in model.state_dict():
        if 'conv' in str(param_tensor):
            tensor_size = model.state_dict()[param_tensor].size()
            if tensor_size[2]==3:
                print(param_tensor, '\t', tensor_size)
                weight0 = model.state_dict()[param_tensor].numpy()
                weight1 = np.reshape(weight0, (tensor_size[0]*tensor_size[1],9))
                assert (weight0[0,0].reshape(9)-weight1[0]).sum()==0
                conv_weights.append(weight1)

    print('number of 3x3 conv layer: ', len(conv_weights))
    all_weights = np.concatenate(conv_weights, 0)
    print(all_weights.shape)
    print(all_weights.mean())
    print(np.abs(all_weights).mean())
    print((np.abs(all_weights)>0.001).sum()/all_weights.shape[0])

if __name__ == '__main__':
    from model.imagenet import *
    model = resnet34(pretrained=True)
    print('model type: ', model.__class__.__name__)

    get_weights(model)
    