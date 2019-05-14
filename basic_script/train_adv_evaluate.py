from __future__ import absolute_import, division, print_function

## Import Python packages
import sys, os, glob
sys.path.append(os.path.abspath('../'))

import numpy as np

## Import pytorch packages

import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image
import torchvision
import torchvision.transforms as transforms

from utils.utils import recreate_image

from model.cifar import *


def aggregate_adv_noise(x, adv_noise, eps=0.05, method='majority_voting'):
    if method == 'majority_voting':
        x_adv_aggregate = x + eps * torch.mean(adv_noise, dim=0).sign()
        x_adv_aggregate = torch.where(x_adv_aggregate > x+eps, x+eps, x_adv_aggregate)
        x_adv_aggregate = torch.where(x_adv_aggregate < x-eps, x-eps, x_adv_aggregate)
    elif method == 'mean_aggregation':
        x_adv_aggregate = x + torch.mean(adv_noise, dim=0)
        x_adv_aggregate = torch.where(x_adv_aggregate > x+eps, x+eps, x_adv_aggregate)
        x_adv_aggregate = torch.where(x_adv_aggregate < x-eps, x-eps, x_adv_aggregate)

    return x_adv_aggregate

def compute_AAR(x, y, x_adv, nets, target=False):
    '''
    compute average attack rate
    '''
    aar = 0.
    for i in range(len(nets)):
        net = nets[i]
        y_adv = net(x_adv)
        y_adv_pred = y_adv.max(1, keepdim=True)[1].squeeze()
        if target:
            aar += torch.mean(torch.tensor(y == y_adv_pred, dtype=torch.float32))
        else:
            aar += torch.mean(torch.tensor(y != y_adv_pred, dtype=torch.float32))

    return aar/len(nets)

class AdvSolver(object):
    def __init__ (self, net, eps, criterion):
        self.net = net
        self.eps = eps
        self.criterion = criterion

    def fgsm(self, x, y, device, target=False, x_val_min = -1., x_val_max = 1.):
        '''
        Implementation of Fast Gradient Sign Method
        x_adv: input image data
        y: labels (target) of the input data
        target: boolen function to indicate whether we
        x_val_min: the lower bound of the input data
        x_val_max: the upper bound of the input data
        '''

        x_adv = Variable(x.data, requires_grad=True)
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

    def i_fgsm(self, x, y, device, target=False, alpha=1., iteration=10, x_val_min = -1., x_val_max = 1.):
        x_adv = Variable(x.data, requires_grad=True)
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
            x_adv = torch.where(x_adv > x+self.eps, x+self.eps, x_adv)
            x_adv = torch.where(x_adv < x-self.eps, x-self.eps, x_adv)
            # x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)
            # x_adv.requires_grad = True

        y_adv = self.net(x_adv)
        y_adv_pred = y_adv.max(1, keepdim=True)[1] # get the index of the max log-probability

        return x_adv, y_adv_pred


class GenAdv(object):
    def __init__(self, net, device, criterion, eps=0.05, adv_iter=100, method='i_fgsm'):
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
            return x_adv, y_adv
        else:
            return x_adv, y_adv


# main code
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # Preparing data, decompose the features and labels.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    for _, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        break

    print(str(y)+'\n')
    dir_name = '../outputs/checkpoints'

    # # Load ResNet
    # resnet = ResNet18().to(device)
    # resnet = torch.nn.DataParallel(resnet)
    # res_checkpoint = torch.load(dir_name + '/ResNet.t7')
    # resnet.load_state_dict(res_checkpoint['net'])

    # # Load DenseNet
    # densenet = DenseNet121().to(device)
    # densenet = torch.nn.DataParallel(densenet)
    # dense_checkpoint = torch.load(dir_name + '/DenseNet.t7')
    # densenet.load_state_dict(dense_checkpoint['net'])

    # # Load MobileNet
    # mobilenet = MobileNet().to(device)
    # mobilenet = torch.nn.DataParallel(mobilenet)
    # mobile_checkpoint = torch.load(dir_name + '/MobileNet.t7')
    # mobilenet.load_state_dict(mobile_checkpoint['net'])

    # # To do: load more models
    # nets = [densenet, resnet, mobilenet]

    fl = open('result.txt', 'a+')

    #nets = [ResNet18().to(device) for _ in range(10)]

    nets = [ResNet18().to(device),
            ResNet18().to(device),
            VGG('VGG19').to(device),
            VGG('VGG19').to(device),
            DenseNet121().to(device),
            DenseNet121().to(device),
            GoogLeNet().to(device),
            GoogLeNet().to(device),
            ShuffleNetV2(1).to(device),
            ShuffleNetV2(1).to(device)]

    for i in range(10):
        nets[i] = torch.nn.DataParallel(nets[i], device_ids=range(2))
        net_name = glob.glob(dir_name + '/*_3%d.t7' % i)
        print(net_name[0])
        net_checkpoint = torch.load(net_name[0])
        nets[i].load_state_dict(net_checkpoint['net'])

    adv_noise = []
    for i in range(len(nets)):
        net = nets[i]
        criterion = F.cross_entropy
        Generate_Adv = GenAdv(net, device, criterion)
        # x_adv, _ = Generate_Adv.generate_adv(x, y)
        pseudo_target = torch.ones_like(y)
        x_adv, y_adv = Generate_Adv.generate_adv(x, pseudo_target, target=True)
        #print((y_adv==1).sum())

        print('=== Prediction results for adversarial examples of net %d' % i)
        # aar = compute_AAR(x, y, x_adv, nets, target=False)
        aar = compute_AAR(x, pseudo_target, x_adv, nets, target=True)
        print(aar.item())
        fl.write('%.4f\t' % (aar.item()))

        # for j in range(len(nets)):
        #     net = nets[j]
        #     y_adv = net(x_adv)
        #     print('=== Prediction results for adversarial examples of net %d' % j)
        #     print(y_adv.max(1, keepdim=True)[1].squeeze())

        adv_noise.append(x_adv - x)

    adv_noise = torch.stack(adv_noise)
    print(adv_noise.size())
    x_adv_aggregate = aggregate_adv_noise(x, adv_noise)

    print('=== Prediction results for aggregated adversarial examples')
    # aar_aggregate = compute_AAR(x, y, x_adv_aggregate, nets, target=False)
    aar_aggregate = compute_AAR(x, pseudo_target, x_adv_aggregate, nets, target=True)
    print(aar_aggregate.item())
    fl.write('%.4f\n' % (aar_aggregate.item()))
    fl.close()

    # for i in range(len(nets)):
    #     net = nets[i]
    #     y_adv_agg = net(x_adv_aggregate)
    #     y_adv_agg_pred = y_adv_agg.max(1, keepdim=True)[1].squeeze()
    #     print(y_adv_agg_pred)

    # if not os.path.exists('../generated_images'):
    #     os.makedirs('../generated_images')

    # for i in range(len(y)):
    #     recreated_image = recreate_image(x_adv_aggregate[i])
    #     original_image = recreate_image(x[i])
    #     noise_image = recreated_image - original_image
    #     cv2.imwrite('../generated_images/original_image_' + str(i) + '.jpg', original_image)
    #     cv2.imwrite('../generated_images/noise_image_' + str(i) + '.jpg', noise_image)
    #     cv2.imwrite('../generated_images/recreated_image_' + str(i) + '.jpg', recreated_image)
