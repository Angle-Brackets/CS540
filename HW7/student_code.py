# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms
import numpy as np


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        
        # Stage 1
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.r1 = nn.ReLU()
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.r2 = nn.ReLU()
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3
        self.f1 = nn.Flatten()

        # Stage 4
        self.l1 = nn.Linear(in_features=400, out_features=256)
        self.r3 = nn.ReLU()

        # Stage 5
        self.l2 = nn.Linear(in_features=256, out_features=128)
        self.r4 = nn.ReLU()

        # Stage 6
        self.l3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        shape_dict = {}
        # certain operations
        out = self.m1(self.r1(self.c1(x)))
        shape_dict[1] = list(out.shape)
        
        out = self.m2(self.r2(self.c2(out)))
        shape_dict[2] = list(out.shape)
        
        out = self.f1(out)
        shape_dict[3] = list(out.shape)

        out = self.r3(self.l1(out))
        shape_dict[4] = list(out.shape)

        out = self.r4(self.l2(out))
        shape_dict[5] = list(out.shape)

        out = self.l3(out)
        shape_dict[6] = list(out.shape)
        
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_params = sum([np.prod(p.size()) for p in model.parameters()])

    return model_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
