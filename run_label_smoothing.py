#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:05:17 2019

@author: m.goibert
"""





"""
Libraries
"""

from operator import itemgetter
import os
import time
import random

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from joblib import delayed, Parallel

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# os.chdir("/Users/m.goibert/Documents/Criteo/Code/LS_good_version")
# os.getcwd()
from Train_test_label_smoothing import (smooth_CE, smooth_label, one_hot,
                                        train_model_smooth, test_model,
                                        attack_fgsm, run_fgsm)
from utils import parse_cmdline_args

# Change precision tensor
torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(1)
np.random.seed(1)


"""
Model
"""

# Model
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.soft = nn.Softmax(dim = 1)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.soft(x)
        return x

    def __repr__(self):
        return "MLP"

### Parse command-line arguments
args = parse_cmdline_args()
### Dataset
# Import MNIST
root = './data'
batch_size = args.batch_size
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

val_data = []
test = []
for i, x in enumerate(test_set):
    if i < 1000:
        val_data.append(x)
    else:
        test.append(x)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=1,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=1000,
                                          shuffle=True)


# Convert tensors into test_loader into double tensors
test_loader.dataset = tuple(zip( map( lambda x: x.double(), map(itemgetter(0), test_loader.dataset)),
                          map(itemgetter(1), test_loader.dataset) ))

# Running the experiement
num_jobs = args.num_jobs
loss_func = smooth_CE
num_classes = 10
num_epsilons = args.num_epsilons
alphas = np.linspace(0, 1, num=args.num_alphas)
num_epochs = args.num_epochs
epsilons = np.linspace(0, args.max_epsilon, num=args.num_epsilons)
experiment_name = args.experiment_name
if experiment_name == "temperature":
    temperatures = np.logspace(-4, 1, num=5)
else:
    temperatures = [0.1]


"""
Running
"""

def run_experiment(alpha, kind, epsilons, temperature=None):
    net = MLPNet()

    print("alpha = ", alpha)
    #optimizer = optim.SGD(model.parameters(), lr=1.75) 
    net, loss_history, acc_tr = train_model_smooth(
        net, train_loader, val_loader, loss_func, num_epochs, alpha = alpha,
        kind = kind, num_classes = num_classes, temperature = temperature)

    print("Accuracy (training) = ", acc_tr)
    print("Accuracy (Test) = ", test_model(net, test_loader))

    accuracy_adv = []
    for epsilon in epsilons:
        print("epsilon = ", epsilon)
        start_time = time.time()
        acc_adv, ex_adv = run_fgsm(net, test_loader, alpha, kind, temperature,
                                   epsilon, loss_func, num_classes)
        accuracy_adv.append(acc_adv)
        end_time = time.time()
        delta_time = (end_time - start_time)
        print("Execution time = %.2f sec" % delta_time)

    return (net, alpha, kind, temperature, loss_history, acc_tr, accuracy_adv,
            delta_time)

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
             for kind in ["standard", "adversarial"]]
for _, alpha, kind, temperature, _, _, accs, _ in Parallel(n_jobs=num_jobs)(
        delayed(run_experiment)(alpha, kind, epsilons, temperature=temperature)
        for alpha, kind, temperature in jobs):
    for epsilon, acc in zip(epsilons, accs):
        df.append(dict(alpha=alpha, epsilon=epsilon, acc=acc, kind=kind,
                       temperature=temperature))
df = pd.DataFrame(df)
results_file = "results_%s_experiment.pkl" % experiment_name
df.to_pickle(results_file)
print("Results written to file: %s" % results_file)
df.to_pickle("results.pkl")
