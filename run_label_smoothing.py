#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:05:17 2019

@author: m.goibert,
         Elvis Dohmatob <gmdopp@gmail.com>
"""


"""
Libraries
"""

from operator import itemgetter
import time
import random
import logging

from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import delayed, Parallel


from Train_test_label_smoothing import (smooth_CE, smooth_label, one_hot,
                                        train_model_smooth, test_model,
                                        attack_fgsm, attack_BIM, DeepFool,
                                        CW_attack, run_attack, device)
from utils import parse_cmdline_args
from lenet import LeNet, ResNet18


# Change precision tensor and set seed
torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(1)
np.random.seed(1)


"""
Models
"""

# Linear model for MNIST


class MLPNet(nn.Module):

    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.soft(x)
        return x

    def __repr__(self):
        return "MLP"


# LeNet model for CIFAR10

class LeNetCifar(nn.Module):

    def __init__(self):
        super(NetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


"""
Environment and datastets
"""

# Device
device = device
print("device run = ", device)

# Parse command-line arguments
args = parse_cmdline_args()
dataset = args.dataset
batch_size = args.batch_size
test_batch_size = args.test_batch_size

# Dataset

if dataset == "MNIST":

    # -------------- Import MNIST

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

elif dataset == "CIFAR10":

    # ---------------- Import CIFAR10

    root = './data'
    batch_size = args.batch_size
    trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = dset.CIFAR10(root=root, train=True,
                             transform=trans, download=True)
    test_set = dset.CIFAR10(root=root, train=False,
                            transform=trans, download=True)

    val_data = []
    test = []
    for i, x in enumerate(test_set):
        if i < 1000:
            val_data.append(x)
        else:
            test.append(x)

    # Limit values for X
    lims = -1, 1
    # Name of classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=True,
                                          batch_size=test_batch_size)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=100,
                                         shuffle=True)
# Convert tensors into test_loader into double tensors
test_loader.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
            test_loader.dataset)), map(itemgetter(1), test_loader.dataset)))


# Parameters

num_jobs = args.num_jobs        # for parallelisation
loss_func = smooth_CE
num_classes = 10
num_epsilons = args.num_epsilons
alphas = np.linspace(0, 1, num=args.num_alphas)
num_epochs = args.num_epochs
epsilons = np.linspace(0, args.max_epsilon, num=args.num_epsilons)
experiment_name = args.experiment_name
if experiment_name == "temperature":
    temperatures = np.logspace(-4, 3, num=8)
else:
    temperatures = [0.1]
model = args.model
attack_method = args.attack_method

if attack_method == "DeepFool":
    epsilons = [1]

# define what device we are using
#cuda = torch.cuda.is_available()
#logging.info("CUDA Available: {}".format(cuda))
#device = torch.device("cuda" if cuda else "cpu")


"""
Running
"""


def run_experiment(alpha, kind, epsilons, temperature=None):
    if dataset + "_" + model == "MNIST_LeNet":
        net0 = LeNet()
        # load the pretrained net
        pretrained_net = "lenet_mnist_model.pth"
        net0.load_state_dict(torch.load(pretrained_net, map_location='cpu'))
    elif dataset + "_" + model == "MNIST_Linear":
        net0 = MLPNet()
    elif dataset + "_" + model == "CIFAR10_LeNet":
        net0 = LeNetCifar()
    elif dataset + "_" + model == "CIFAR10_ResNet":
        net0 = ResNet18()
    net0 = net0.to(device)

    print("ls Kind = {} \n".format(kind))
    print("alpha = {}".format(alpha))

    if False:
        net = net0
    else:
        net, loss_history, acc_tr = train_model_smooth(
            net0, train_loader, val_loader, loss_func, num_epochs, alpha=alpha,
            kind=kind, num_classes=num_classes, temperature=temperature)
        acc_test = test_model(net, test_loader)

        print("Accuracy (training) = %g " % acc_tr)
        print("Accuracy (Test) = {} ".format(acc_test))

    # run attack (possibly massively in parallel over test data and epsilons)
    accuracy_adv = []
    t0 = time.time()
    accs_adv, exs_adv = run_attack(net, test_loader, alpha, kind, temperature,
                                   epsilons, loss_func, num_classes, lims=lims,
                                   attack_method=attack_method)
    delta_time = time.time() - t0

    for epsilon in epsilons:
        acc_adv = accs_adv[epsilon]
        # ex_adv = exs_adv[epsilon]
        # print("epsilon = %s" % epsilon)
        accuracy_adv.append(acc_adv)
    print("Execution time = %.2f sec" % delta_time)
    return (net, alpha, kind, temperature, loss_history, acc_tr, acc_test,
            accuracy_adv, delta_time)

# run experiments in parallel with joblib
# XXX You need to instal joblib version 0.11 like so
# XXX conda install -c conda-forge joblib=0.11
# XXX Newer versions produce the error:
# XXX RuntimeError: Expected object of scalar type Float but got scalar type
# XXX Double for argument #4 'mat1'

df = []
jobs = [(alpha, "boltzmann", temperature) for alpha in alphas
        for temperature in temperatures]
if experiment_name != "temperature":
    jobs += [(alpha, kind, None) for alpha in alphas
             for kind in ["standard", "adversarial", "second_best"]]
if num_jobs > 1:
    print("Using joblib...")
    results = Parallel(n_jobs=num_jobs)(
        delayed(run_experiment)(alpha, kind, epsilons, temperature=temperature)
        for alpha, kind, temperature in jobs)
else:
    results = [run_experiment(alpha, kind, epsilons, temperature=temperature)
               for alpha, kind, temperature in jobs]
for _, alpha, kind, temperature, _, _, acc_test, accs, _ in results:
    for epsilon, acc in zip(epsilons, accs):
        df.append(dict(alpha=alpha, epsilon=epsilon, acc_test=acc_test, acc=acc,
                       kind=kind, temperature=temperature))
df = pd.DataFrame(df)
results_file = "%s_%s_results_%s_experiment_%s_attack.csv" % (
    dataset, model, experiment_name, attack_method)
df.to_csv(results_file, sep=",")
print("Results written to file: %s" % results_file)
