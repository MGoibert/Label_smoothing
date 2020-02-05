#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:12:50 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import os
from operator import itemgetter
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import urllib.request
import scipy

from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from joblib import delayed, Parallel


# ----- Dataset

batch_size = 100
test_batch_size = 100
device = torch.device("cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(1)
np.random.seed(1)

root = './data'
trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root=root, train=True,
                           transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False,
                          transform=trans, download=True)

val_data = []
test = []
for i, x in enumerate(test_set):
    if i < 1000:
        val_data.append(x)
    else:
        test.append(x)

    # Limit values for X
lims = -0.5, 0.5


train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=True,
                                          batch_size=test_batch_size)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=len(val_data),
                                         shuffle=True)
# Convert tensors into test_loader into double tensors
#test_loader.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
#            test_loader.dataset)), map(itemgetter(1), test_loader.dataset)))


# ------ Model

class MNISTnet(nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10, bias=False)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.double()
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.soft(x)
        return x


# ------ Train model
        
loss_func = smooth_cross_entropy
num_epochs = 15
model = MNISTnet()
smoothing_methods = ["adversarial"]
smoothing_method = "adversarial"

alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.0]
#alpha = 0

#alphas = [0]
dict_gap = {}
dict_gradient = {}
for alpha in alphas:
    dict_gap[alpha] = []
    dict_gradient[alpha] = []
    net, loss_history, acc_tr = train_model_smooth(model, train_loader, val_loader, loss_func, num_epochs, alpha=alpha,
                       smoothing_method=smoothing_method, num_classes=10)
    print("test accuracy =", test_model(net, test_loader))

    for i, p in enumerate(net.parameters()):
        theta = p.data

    for i in range(10):
        dict_gradient[alpha].insert(0, np.linalg.norm(theta[i], ord=2))
        for j in range(10):
            if i != j:
                dict_gap[alpha].insert(0, np.linalg.norm(theta[i] - theta[j], ord=1))

#sns.palplot(sns.color_palette("GnBu_d", 8))
#col = sns.color_palette("GnBu_d", 8) +  [(0.7561707035755478, 0.21038062283737025, 0.22352941176470587)]
#plt.style.use('ggplot')
#for i, alpha in enumerate(dict_gap.keys()):
#    sns.kdeplot(dict_gap[alpha], bw=5, color=col[i], lw=2)
#plt.xlabel(r'BLS Temp. 0.1 0.01 0.001 0.0001 Adv. train. Natural: $0.4 \; 0.5$')
#plt.xlabel(r'Gradient gap distribution: $|| \theta_i - \theta_j ||_1, \forall \; i \neq j$')
#plt.ylabel(r'Density')
#plt.title("Gradient gap for ALS vs natural classifiers")
#plt.savefig("/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/gradient_gap", dpi=500)
#plt.show()


sns.palplot(sns.color_palette("GnBu_d", 8))
col = sns.color_palette("GnBu_d", 8) +  [(0.7561707035755478, 0.21038062283737025, 0.22352941176470587)]
plt.style.use('ggplot')
for i, alpha in enumerate(dict_gap.keys()):
    sns.kdeplot(dict_gradient[alpha], bw=5, color=col[i], lw=2)
#plt.xlabel(r'BLS Temp. 0.1 0.01 0.001 0.0001 Adv. train. Natural: $0.4 \; 0.5$')
plt.xlabel(r'Gradient distribution: $|| \theta_i||_2, \forall \; i$')
plt.ylabel(r'Density')
plt.title("Gradient for ALS vs natural classifiers")
plt.savefig("/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/gradient", dpi=500)
plt.show()




