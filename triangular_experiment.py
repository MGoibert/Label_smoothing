#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:46:31 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from operator import itemgetter

from Train_test_label_smoothing import (smooth_CE, smooth_label, one_hot,
                                        train_model_smooth, test_model,
                                        attack_fgsm, attack_triangular,
                                        run_fgsm)
from utils import parse_cmdline_args

# Change precision tensor
torch.set_default_tensor_type(torch.DoubleTensor)
np.random.seed(1)


"""
Simple linear model
"""


class lin(nn.Module):

    def __init__(self):
        super(lin, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.soft(x)
        return x

"""
Generating the triengular example
"""


def generate_overlap(num_samples, gamma=.6):
    """
    Y is a binomial random variable. X is a triangular law (of different type
    knowing Y). Gamma is the parameter for Y. Gamma must be between 1/2 and 2/3.
    """
    assert 1. / 2. <= gamma <= 2. / 3.
    y = np.random.choice([0, 1], size=num_samples, p=[1 - gamma, gamma])
    x = [np.random.triangular(-1, 0, 1) * (y[i] == 1) - abs(np.random.triangular(
        -1, 0, 1)) * (y[i] == 0) for i in range(num_samples)]
    return [[elem] for elem in x], y

# Arguments/Parameters
args = parse_cmdline_args()


def run_experiment(net0, alpha, kind, epsilons, temperature=None,
                   define_net=False):
    """
    Specific function to run the triangular experiement.
    alpha is the strenght of label smoothing
    kind = "standard", "aversarial", "boltzmann" or "second_best"
    epsilons = strenght of adversarial attacks
    temperature = parameter if kind = "boltzmann"
    define_net = True or False. If True, uses the net given by net0
    without training.
    """
    print("alpha = ", alpha)
    net_lin = lin()
    param = [torch.tensor([[0.], [1.]]), torch.tensor([0., 0.5])]
    for i, p in enumerate(net_lin.parameters()):
        p.data = param[i]
        if i == 0:
            p.requires_grad = False

    if define_net == False:
        net, loss_history, acc_tr = train_model_smooth(
            net_lin, train_loader, val_loader, loss_func, num_epochs,
            alpha=alpha, kind=kind, num_classes=None,
            temperature=temperature)
    else:
        net = net0
        loss_history = 0

    print("Accuracy (Test) = ", test_model(net, test_loader))

    accuracy_adv = []
    for epsilon in epsilons:
        print("epsilon = ", epsilon)
        acc_adv, ex_adv = run_fgsm(net, test_loader, alpha, kind, temperature,
                                   epsilon, loss_func, num_classes=None,
                                   method_attack="triangular")
        accuracy_adv.append(acc_adv)

    return (net, alpha, kind, temperature, loss_history, accuracy_adv)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ---------------------- Loop on gamma


gammas = np.linspace(1. / 2., 2. / 3., args.num_gammas)
num_samples = args.num_samples
loss_func = smooth_CE
epsilons = np.linspace(0., 1., args.num_epsilons)
alphas = np.linspace(0., 1., args.num_alphas)
num_epochs = args.num_epochs

dict_result0 = {}
dict_results = {}

for ind_gamma, gamma in enumerate(gammas):

    # ------------- Generating the samples

    X, y = generate_overlap(num_samples, gamma=gamma)
    train_size = .6
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
                              batch_size=num_samples, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=1, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=round(0.4 * 0.4 * num_samples),
                            shuffle=True)
    test_loader.dataset = tuple(zip(map(lambda x: x.double(),
                                        map(itemgetter(0), test_loader.dataset)),
                                    map(itemgetter(1), test_loader.dataset)))

    # --------------- Bayes classifier

    net0 = lin()
    param = [torch.tensor([[0.], [1.]]), torch.tensor([0., 0])]
    for i, p in enumerate(net0.parameters()):
        p.data = param[i]
    dict_result0["{0}".format(ind_gamma)] = run_experiment(net0, alpha=0,
                        kind="adversarial", epsilons=epsilons, define_net=True)

    # --------------- Adversarial LS

    net_fin = lin()
    dict_results["{0}".format(ind_gamma)] = [run_experiment(net_fin,
        alpha=alpha, kind="adversarial", epsilons=epsilons) for alpha in alphas]
    theta = [0, 0]
    for i, al in enumerate(alphas):
        plt.plot(dict_results["{0}".format(ind_gamma)][i][4])
        plot_name = "loss_plot/loss_" + \
            str(round(gamma, 2)) + "_" + str(round(al, 2)) + ".png"
        plt.savefig(plot_name, dpi=100)
        plt.clf()
        for j, p in enumerate(dict_results["{0}".format(ind_gamma)][i][0].
                              parameters()):
            print(p.data)
            theta[j] = p.data[1] - p.data[0]
            print("b = ", theta[1] / theta[0])

    # ---------------- Saving the results

    df = []
    for alpha, (_, _, _, _, _, accs) in zip(alphas,  dict_results["{0}".
                                                        format(ind_gamma)]):
        for epsilon, acc in zip(epsilons, accs):
            df.append(dict(alpha=alpha, epsilon=epsilon, acc=acc,
                           kind="adversarial", temperature=0))
    df = pd.DataFrame(df)

    df.loc[df['alpha'] == 0, 'acc'] = dict_result0["{0}".format(ind_gamma)][5]

    file_name = "dataframes_outputs/df_gamma_" + str(round(gamma, 2)) + ".csv"
    df.to_csv(file_name, sep=",")
