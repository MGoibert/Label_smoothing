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
import time
import random
import logging

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd

from joblib import delayed, Parallel

# os.chdir("/Users/m.goibert/Documents/Criteo/Code/LS_good_version")
# os.getcwd()
from Train_test_label_smoothing import (smooth_CE, smooth_label, one_hot,
                                        train_model_smooth, test_model,
                                        attack_fgsm, run_fgsm)
from utils import parse_cmdline_args
from lenet import LeNet

# Change precision tensor
torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(1)
np.random.seed(1)


def generate_overlap(num_samples, gamma=.3, random_state=None):
    from sklearn.utils import check_random_state
    rng = check_random_state(random_state)
    x = 2 * rng.rand(num_samples) - 1
    y = np.sign(x)
    mask = np.abs(x) <= gamma
    y[mask] = rng.choice([-1, 1], size=mask.sum())
    return x[:, None], (y + 1) / 2


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

# Running the experiement
num_jobs = args.num_jobs
loss_func = smooth_CE
num_classes = 10
num_epsilons = args.num_epsilons
alphas = np.linspace(0, 1, num=args.num_alphas)
num_epochs = args.num_epochs
epsilons = np.linspace(0, args.max_epsilon, num=args.num_epsilons)
experiment_name = args.experiment_name
use_cnn = args.use_cnn
if experiment_name == "temperature":
    temperatures = np.logspace(-4, 3, num=8)
else:
    temperatures = [0.1]

lims = 0, 1

# define what device we are using
cuda = torch.cuda.is_available()
logging.info("CUDA Available: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")


# XXX for ML Big Days presentation
presentation_mode = True

if presentation_mode:
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=1,
                                          shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=1000,
                                             shuffle=True)
else:
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    from mlp import MLP as MLPNet
    X, y = generate_overlap(10000)
    lims = -1, 1
    train_size = .6
    num_classes = 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=.6)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, train_size=.6)
    X_train = torch.DoubleTensor(X_train)
    y_train = torch.LongTensor(y_train.astype(int))
    X_val = torch.DoubleTensor(X_val)
    y_val = torch.LongTensor(y_val.astype(int))
    X_test = torch.DoubleTensor(X_test)
    y_test = torch.LongTensor(y_test.astype(int))
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=1, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=1000, shuffle=True)


# Convert tensors into test_loader into double tensors
test_loader.dataset = tuple(zip( map( lambda x: x.double(), map(itemgetter(0), test_loader.dataset)),
                          map(itemgetter(1), test_loader.dataset) ))

"""
Running
"""

def run_experiment(alpha, kind, epsilons, temperature=None):
    if use_cnn:
        net = LeNet()
        # load the pretrained net
        pretrained_net = "lenet_mnist_model.pth"
        net.load_state_dict(torch.load(pretrained_net, map_location='cpu'))
    else:
        net = MLPNet()
    net = net.to(device)

    logging.info("alpha = {}".format(alpha))
    #optimizer = optim.SGD(model.parameters(), lr=1.75) 
    net, loss_history, acc_tr = train_model_smooth(
        net, train_loader, val_loader, loss_func, num_epochs, alpha = alpha,
        kind = kind, num_classes = num_classes, temperature = temperature)

    logging.info("Accuracy (training) = %g " % acc_tr)
    logging.info("Accuracy (Test) = {} ".format(
        test_model(net, test_loader)))

    accuracy_adv = []
    for epsilon in epsilons:
        logging.info("epsilon = %s" % epsilon)
        start_time = time.time()
        acc_adv, ex_adv = run_fgsm(net, test_loader, alpha, kind, temperature,
                                   epsilon, loss_func, num_classes, lims=lims)
        accuracy_adv.append(acc_adv)
        end_time = time.time()
        delta_time = (end_time - start_time)
        logging.info("Execution time = %.2f sec" % delta_time)

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
results_file = "_results_%s_experiment.csv" % experiment_name
df.to_pickle(results_file)
logging.info("Results written to file: %s" % results_file)
