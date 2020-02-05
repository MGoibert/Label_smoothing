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
from scipy.io import loadmat
import urllib.request

from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

from joblib import delayed, Parallel


from label_smoothing.functional import (
    smooth_cross_entropy, smooth_label, one_hot, train_model_smooth, test_model,
    run_attack, device, run_attack_transferred)
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
learning_rate = args.learning_rate

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
        elif i >= 1000 and i < 3000:
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
        if i < 5000:
            val_data.append(x)
        elif i >= 5000 and i <8000:
            test.append(x)
    train_data = []
    for i, x in enumerate(train_set):
        if i < 7000:
            train_data.append(x)

    # Limit values for X
    lims = -1, 1
    # Name of classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #os.remove(root+ "/cifar-10-python.tar.gz")


elif dataset == "SVHN":

    # ---------------- Import SVHN

    root = './data'
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = dset.SVHN(root=root, split="train",
                           transform=transform, download=True)
    test_set = dset.SVHN(root=root, split="test",
                           transform=transform, download=True)
    print("HEY")

    val_data = []
    test = []
    for i, x in enumerate(test_set):
        if i < 5000:
            val_data.append(x)
        elif i >= 5000 and i < 5300:
            test.append(x)
    train_data = []
    for i, x in enumerate(train_set):
        if i < 5000:
            train_data.append(x)

    # Limit values for X
    lims = -1, 1

    # Remove the big files downloaded
    #os.remove(root+ "/train_32x32.mat")
    #os.remove(root+ "/test_32x32.mat")


train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=True,
                                          batch_size=test_batch_size)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=len(val_data),
                                         shuffle=True)

# Convert tensors into test_loader into double tensors
#test_loader.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
#            test_loader.dataset)), map(itemgetter(1), test_loader.dataset)))



# ------------------ Parameters

# Model parameters
model = args.model
num_jobs = args.num_jobs
loss_func = smooth_cross_entropy
num_classes = 10
num_epochs = args.num_epochs

# Attack parameters
attack_methods = args.attack_method
if type(attack_methods)==str:
    attack_methods = [attack_methods]
num_iter_attack = args.num_iter_attack

# Label Smoothing parameters
smoothing_methods = args.smoothing_method
if type(smoothing_methods)==str:
    smoothing_methods = [smoothing_methods]
temperatures = np.logspace(-4, -1, num=4)
alphas = np.linspace(args.min_alpha, args.max_alpha, num=args.num_alphas)

epsilons = np.linspace(args.min_epsilon, args.max_epsilon,
                                 num=args.num_epsilons)
print("Epsilons = ", epsilons)
#epsilons = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

# Baseline (adversarial training) parameters
adv_message = ""
adv_training = args.adv_training
if adv_training:
    print("Adversarial training enabled")
adv_training_param = args.adv_training_param
adv_training_reg_param = args.adv_training_reg_param
if adv_training:
    alphas = [0.0]
    adv_message = "_PGD_adv_training"

# Saving mode parameters
to_save_model = args.to_save_model
use_saved_model = args.use_saved_model

# Defensive distillation ?
defensive_distillation = args.defensive_distillation
defensive_distill_message = ""
teacher_message = ""
if defensive_distillation:
    defensive_distill_message = "_defensive"
    teacher_message= "_teacher"

# define what device we are using
#cuda = torch.cuda.is_available()
#logging.info("CUDA Available: {}".format(cuda))
#device = torch.device("cuda" if cuda else "cpu")


"""
Running
"""


def run_experiment(alpha, smoothing_method, epsilons, temperature=None):
    # Loading the correct NN architecture
    if dataset + "_" + model == "MNIST_LeNet":
        net0 = LeNet()
        net = LeNet()
        teacher_net = LeNet()
        pretrained_net = "lenet_mnist_model.pth"
        net0.load_state_dict(torch.load(pretrained_net, map_location='cpu'))
    elif dataset + "_" + model == "MNIST_Linear":
        net0 = MNISTMLP()
        net = MNISTMLP()
        teacher_net = MNISTMLP()
    elif dataset + "_" + model == "CIFAR10_LeNet" or dataset + "_" + model == "SVHN_LeNet":
        net0 = LeNetCIFAR10()
        net = LeNetCIFAR10()
        teacher_net = LeNetCIFAR10()
    elif dataset + "_" + model == "CIFAR10_ResNet" or dataset + "_" + model == "SVHN_ResNet":
        net0 = ResNet18()
        net = ResNet18()
        teacher_net = ResNet18()

    net0 = net0.to(device)
    net = net.to(device)
    teacher_net = teacher_net.to(device)

    print(net0)
    print("label-smoothing method = {} \n".format(smoothing_method))
    print("alpha = %.4f" % alpha)

    if not os.path.exists("model_dict/"):
        os.makedirs("model_dict/")
    
    file_dict = "model_dict/%s.pt" % (dataset + "_" + model)
    model_specifications = str(smoothing_method) + "_" + str(alpha) + "_" + str(temperature) + str(adv_message)
    print("model spe.:", model_specifications)
       
    if  os.path.exists(file_dict):
        checkpoint = torch.load(file_dict)
        if use_saved_model == True and model_specifications in checkpoint.keys():
            # Defensive distillation
            if defensive_distillation and (model_specifications + str(defensive_distill_message) in checkpoint.keys()):
                print("Loading the defensive distillation model!")
                to_train = False
                net.load_state_dict(checkpoint[model_specifications + str(defensive_distill_message)])
                loss_history = checkpoint["loss_%s"%(model_specifications + str(defensive_distill_message))]
                acc_tr = checkpoint["acc_tr_%s"%(model_specifications + str(defensive_distill_message))]
                print("Trained model %s with spe. %s loaded successfully" %(file_dict, model_specifications + str(defensive_distill_message)))
            elif defensive_distillation and (model_specifications + str(teacher_message) in checkpoint.keys()):
                print("Loading teacher net for defensive distillation ! (spe %s)" %(model_specifications + str(teacher_message) ))
                teacher_net.load_state_dict(checkpoint[model_specifications + str(teacher_message)])
                print("Teacher test acc =", test_model(teacher_net, test_loader))
                to_train = True
            elif defensive_distillation:
                print("Training the teacher net !")
                to_train = True
                teacher_net, teacher_loss, teacher_acc = train_model_smooth(
                    net0, train_loader, val_loader, loss_func, num_epochs,learning_rate=learning_rate, alpha=alpha,
                    smoothing_method=smoothing_method, num_classes=num_classes,
                    temperature=temperature, adv_training=adv_training,
                    adv_training_param=adv_training_param, adv_training_reg_param=adv_training_reg_param,
                    defensive_distillation=defensive_distillation)
                print("Teacher validation acc =", teacher_acc)
                print("Teacher test acc =", test_model(teacher_net, test_loader))
                model_specifications_teacher = model_specifications + str(teacher_message)
                checkpoint.update({model_specifications_teacher:teacher_net.state_dict()})
                torch.save(checkpoint, file_dict)
                print("Teacher model saved in %s with specifi. %s" %(file_dict, model_specifications_teacher))
            # Load an already trained model
            elif not defensive_distillation:
                to_train = False
                teacher_net = False
                net.load_state_dict(checkpoint[model_specifications])
                loss_history = checkpoint["loss_%s"%(model_specifications)]
                acc_tr = checkpoint["acc_tr_%s"%(model_specifications)]
                print("Trained model %s with spe. %s loaded successfully" %(file_dict, model_specifications))
        else:
            to_train = True
    else:
        checkpoint = {}
        print("No saved model (whole architecture)")
        to_train = True

    if to_train == True:
        # Train you model (if not loaded before)
        net, loss_history, acc_tr = train_model_smooth(
            net0, train_loader, val_loader, loss_func, num_epochs,learning_rate=learning_rate, alpha=alpha,
            smoothing_method=smoothing_method, num_classes=num_classes,
            temperature=temperature, adv_training=adv_training,
            adv_training_param=adv_training_param, adv_training_reg_param=adv_training_reg_param,
            defensive_distillation=defensive_distillation, teacher_model=teacher_net)

    if to_save_model == True:
        # Save the trained model
        model_specifications = model_specifications + str(defensive_distill_message)
        checkpoint.update({
        model_specifications:net.state_dict(),
        "loss_%s"%(model_specifications):loss_history,
        "acc_tr_%s"%(model_specifications):acc_tr
        })
        torch.save(checkpoint, file_dict)
        print("Model saved in %s with specifi %s"%(file_dict, model_specifications))

    acc_test = test_model(net, test_loader)

    print("Accuracy (training) = %g " % acc_tr)
    print("Accuracy (Test) = {} ".format(acc_test))

    # run attack (possibly massively in parallel over test data and epsilons)
    adv_accs = {}
    t0 = time.time()
    df = []
    for attack_method in attack_methods:
        # Run the attack and outputs adversarial accuracy
        transferred = 0
        if transferred > 0:
            acc_tfr = run_attack_transferred(net)
            print("TRANSFERRED ATTACK ACCURACY =", acc_tfr)
            break
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

    # Save progressively
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

# Put the results in a dataframe
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
# Save your final results dataframe
df = pd.concat(list(map(itemgetter(-2), results)))
results_file = "res_dataframes/%s_%s%s_smoothing=%s_attacks=%s%s.csv" % (
    dataset, model, defensive_distill_message, "+".join(smoothing_methods),
    "+".join(attack_methods), adv_message)

df.to_csv(results_file, sep=",")
print("Results written to file: %s" % results_file)
