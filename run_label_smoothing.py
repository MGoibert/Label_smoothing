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
import os
from operator import itemgetter
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from joblib import delayed, Parallel


from label_smoothing.functional import (
    smooth_cross_entropy, smooth_label, one_hot, train_model_smooth, test_model,
    run_attack, device)
from label_smoothing.utils import parse_cmdline_args
from label_smoothing.mlp import MNISTMLP
from label_smoothing.lenet import LeNet, LeNetCIFAR10
from label_smoothing.resnet import ResNet18

# Change precision tensor and set seed
torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(1)
np.random.seed(1)


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
        elif i>1000 and i<=1500:
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
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=len(val_data),
                                         shuffle=True)
# Convert tensors into test_loader into double tensors
test_loader.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
            test_loader.dataset)), map(itemgetter(1), test_loader.dataset)))


# Parameters

num_jobs = args.num_jobs        # for parallelisation
loss_func = smooth_cross_entropy
num_classes = 10
alphas = np.linspace(args.min_alpha, args.max_alpha, num=args.num_alphas)
alphas = [0.0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
num_epochs = args.num_epochs
num_iter_attack = args.num_iter_attack
epsilons = np.append(np.linspace(args.min_epsilon, args.max_epsilon,
                                 num=args.num_epsilons),
                     [5, 10, 100, 1000, 10000])
epsilons = [1]
epsilons = np.unique(epsilons)
temperatures = np.logspace(-4, -1, num=4)
model = args.model
attack_methods = args.attack_method
smoothing_methods = args.smoothing_method
if type(attack_methods)==str:
    attack_methods = [attack_methods]

if attack_methods == "DeepFool":
    epsilons = [1]

to_save_model = args.to_save_model
use_saved_model = args.use_saved_model

# define what device we are using
#cuda = torch.cuda.is_available()
#logging.info("CUDA Available: {}".format(cuda))
#device = torch.device("cuda" if cuda else "cpu")


"""
Running
"""


def run_experiment(alpha, smoothing_method, epsilons, temperature=None):
    if dataset + "_" + model == "MNIST_LeNet":
        net0 = LeNet()
        net = LeNet()
        # load the pretrained net
        pretrained_net = "lenet_mnist_model.pth"
        net0.load_state_dict(torch.load(pretrained_net, map_location='cpu'))
    elif dataset + "_" + model == "MNIST_Linear":
        net0 = MNISTMLP()
        net = MNISTMLP()
    elif dataset + "_" + model == "CIFAR10_LeNet":
        net0 = LeNetCIFAR10()
        net = LeNetCIFAR10()
    elif dataset + "_" + model == "CIFAR10_ResNet":
        net0 = ResNet18()
        net = ResNet18()

    net0 = net0.to(device)
    net = net.to(device)

    print(net0)
    print("label-smoothing method = {} \n".format(smoothing_method))
    print("alpha = %.4f" % alpha)

    if not os.path.exists("model_dict/"):
        os.makedirs("model_dict/")
    
    file_dict = "model_dict/%s.pt" % (dataset + "_" + model)
    model_specifications = str(smoothing_method) + "_" + str(alpha) + "_" + str(temperature)
    print("model spe.:", model_specifications)
    
    if  os.path.exists(file_dict):
        checkpoint = torch.load(file_dict, map_location='cpu')
        if use_saved_model == True and model_specifications in checkpoint.keys():
            to_train = False
            net.load_state_dict(checkpoint[model_specifications])
            loss_history = checkpoint["loss_%s"%(model_specifications)]
            acc_tr = checkpoint["acc_tr_%s"%(model_specifications)]
            print("Trained model %s with spe. %s loaded successfully" %(file_dict, model_specifications))
        else:
            print("No saved model (specifications)")
            to_train = True
    else:
        checkpoint = {}
        print("No saved model (whole architecture)")
        to_train = True

    if to_train == True:
        net, loss_history, acc_tr = train_model_smooth(
            net0, train_loader, val_loader, loss_func, num_epochs, alpha=alpha,
            smoothing_method=smoothing_method, num_classes=num_classes,
            temperature=temperature)

    if to_save_model == True:
        checkpoint.update({
        model_specifications:net.state_dict(),
        "loss_%s"%(model_specifications):loss_history,
        "acc_tr_%s"%(model_specifications):acc_tr
        })
        torch.save(checkpoint, file_dict)
        print("Model saved in %s"%file_dict)

    acc_test = test_model(net, test_loader)

    print("Accuracy (training) = %g " % acc_tr)
    print("Accuracy (Test) = {} ".format(acc_test))

    # run attack (possibly massively in parallel over test data and epsilons)
    adv_accs = {}
    t0 = time.time()
    df = []
    for attack_method in attack_methods:
        adv_accs[attack_method] = []
        accs, _ = run_attack(net, test_loader, loss_func, epsilons,
                                 attack_method=attack_method, alpha=alpha,
                                 num_classes=num_classes,
                                 smoothing_method=smoothing_method,
                                 temperature=temperature, lims=lims,
                                 num_iter=num_iter_attack)
        delta_time = time.time() - t0
        print("Time elapsed: %.4f" % delta_time)

        for epsilon, eps_accs in accs.items():
            for label, acc in enumerate(eps_accs):
                df.append(dict(alpha=alpha, attack_method=attack_method,
                               temperature=temperature, acc=acc,
                               epsilon=epsilon, label=label,
                               smoothing_method=smoothing_method))

    pid = os.getpid()
    filename = "res_dataframes/tmp/pid=%i.csv" % pid
    df = pd.DataFrame(df)
    df.to_csv(filename)
    print(filename)

    return (net, alpha, smoothing_method, temperature, loss_history, acc_tr,
            acc_test, df, delta_time)

# run experiments in parallel with joblib
# XXX You need to instal joblib version 0.11 like so
# XXX conda install -c conda-forge joblib=0.11
# XXX Newer versions produce the error:
# XXX RuntimeError: Expected object of scalar type Float but got scalar type
# XXX Double for argument #4 'mat1'

if not os.path.exists("res_dataframes/tmp"):
    os.makedirs("res_dataframes/tmp")

df = []
jobs = []
for smoothing_method in smoothing_methods:
    if smoothing_method == "boltzmann":
        jobs += [(alpha, "boltzmann", temperature) for alpha in alphas
                 for temperature in temperatures]
    else:
        jobs += [(alpha, smoothing_method, None) for alpha in alphas]
if num_jobs > 1:
    print("Using joblib...")
    results = Parallel(n_jobs=num_jobs)(
        delayed(run_experiment)(alpha, smoothing_method, epsilons,
                                temperature=temperature)
        for alpha, smoothing_method, temperature in jobs)
else:
    results = [run_experiment(alpha, smoothing_method, epsilons,
                              temperature=temperature)
               for alpha, smoothing_method, temperature in jobs]
df = pd.concat(list(map(itemgetter(-2), results)))
results_file = "res_dataframes/%s_%s_smoothing=%s_attacks=%s.csv" % (
    dataset, model, "+".join(smoothing_methods),
    "+".join(attack_methods))

df.to_csv(results_file, sep=",")
print("Results written to file: %s" % results_file)
