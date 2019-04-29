'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

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

from model.cifar import *
from datasets import FederatedCIFAR10
from utils import progress_bar

MODEL_MAP = {'resnet18': ResNet18(),
             'vgg19': VGG('VGG19'),
             'densenet121': DenseNet121(),
             'googlenet': GoogLeNet(),
             'shufflenet': ShuffleNetV2(1)}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--client', type=int, help='index of local client')
parser.add_argument('--sample', type=str, help='data sampling approach')
parser.add_argument('--model', type=str, help='model architecture')
args = parser.parse_args()

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
best_acc = 0 # best train accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('Index of local client: ', args.client)
assert args.sample in ['random', 'class']
print('Data sampling type: '+args.sample)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = FederatedCIFAR10(root='../data', train=True, download=True, transform=transform_train, sample_type=args.sample)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
assert args.model in MODEL_MAP.keys()
net = MODEL_MAP[args.model]
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
model_name = net.__class__.__name__
print('model: ', model_name)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

# Training
def train(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('../outputs/checkpoint/'):
            os.makedirs('../outputs/checkpoint/')
        torch.save(state, '../outputs/checkpoint/'+str(model_name)+'_'+str(args.client)+'.t7')
        best_acc = acc
    
    if best_acc > 99.5: exit()

for epoch in range(start_epoch, start_epoch+150):
    scheduler.step()
    train(epoch)

# sbatch train_federated.sh 10 class resnet18
# sbatch train_federated.sh 11 class resnet18
# sbatch train_federated.sh 12 class resnet18
# sbatch train_federated.sh 13 class resnet18
# sbatch train_federated.sh 14 class resnet18
# sbatch train_federated.sh 15 class resnet18
# sbatch train_federated.sh 16 class resnet18
# sbatch train_federated.sh 17 class resnet18
# sbatch train_federated.sh 18 class resnet18
# sbatch train_federated.sh 19 class resnet18

# sbatch train_federated.sh 20 random resnet18
# sbatch train_federated.sh 21 random resnet18
# sbatch train_federated.sh 22 random vgg19
# sbatch train_federated.sh 23 random vgg19
# sbatch train_federated.sh 24 random densenet121
# sbatch train_federated.sh 25 random densenet121
# sbatch train_federated.sh 26 random googlenet
# sbatch train_federated.sh 27 random googlenet
# sbatch train_federated.sh 28 random shufflenet
# sbatch train_federated.sh 29 random shufflenet

# sbatch train_federated.sh 30 class resnet18
# sbatch train_federated.sh 31 class resnet18
# sbatch train_federated.sh 32 class vgg19
# sbatch train_federated.sh 33 class vgg19
# sbatch train_federated.sh 34 class densenet121
# sbatch train_federated.sh 35 class densenet121
# sbatch train_federated.sh 36 class googlenet
# sbatch train_federated.sh 37 class googlenet
# sbatch train_federated.sh 38 class shufflenet
# sbatch train_federated.sh 39 class shufflenet