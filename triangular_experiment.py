#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:46:31 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from operator import itemgetter
from Train_test_label_smoothing import (smooth_CE, smooth_label, one_hot,
                                        train_model_smooth, test_model,
                                        attack_fgsm, attack_triangular,
                                        run_fgsm)


class lin(nn.Module):
    def __init__(self):
        super(lin, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.soft = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.soft(x)
        return x


def generate_overlap(num_samples, gamma=.6, random_state=None):
    y = np.random.choice([0,1], size = num_samples, p = [1 - gamma, gamma])
    x = [ np.random.triangular(-1,0,1)*(y[i]==1) - abs(np.random.triangular(-1,0,1))*(y[i]==0) for i in range(num_samples) ]
    return [[elem] for elem in x], y


num_samples = 20000
X, y = generate_overlap(num_samples, gamma= 2/3)
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
                              batch_size=50000, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=1, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=8000, shuffle=True)
test_loader.dataset = tuple(zip( map( lambda x: x.double(), map(itemgetter(0), test_loader.dataset)),
                          map(itemgetter(1), test_loader.dataset) ))




def run_experiment(net0, alpha, kind, epsilons, temperature=None, define_net=False):
    print("alpha = ", alpha)
    net_lin = lin()
    param = [torch.tensor([[0.],[1.]]), torch.tensor([0.,0.5])]
    for i, p in enumerate(net_lin.parameters()):
        p.data = param[i]
        if i == 0:
            p.requires_grad=False

    if define_net==False:
        net, loss_history, acc_tr = train_model_smooth(
                net_lin, train_loader, val_loader, loss_func, num_epochs, alpha = alpha,
                kind = kind, num_classes=None, temperature = temperature)
    else:
        net = net0
        loss_history=0

    print("Accuracy (Test) = ", test_model(net, test_loader))

    accuracy_adv = []
    for epsilon in epsilons:
        print("epsilon = ", epsilon)
        acc_adv, ex_adv = run_fgsm(net, test_loader, alpha, kind, temperature,
                                   epsilon, loss_func, num_classes=None, method_attack="triangular")
        accuracy_adv.append(acc_adv)

    return (net, alpha, kind, temperature, loss_history, accuracy_adv)
    


# ----------------------------------------------------------------------------
# ---- Running the experiement for the Bayes classifier

net0 = lin()
param = [torch.tensor([[0.],[1.]]), torch.tensor([0.,0])]
for i, p in enumerate(net0.parameters()):
    print(p.data)
    p.data = param[i]


loss_func = smooth_CE  
epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

result0 = run_experiment(net0, alpha=0, kind="adversarial", epsilons=epsilons, define_net=True)


for j, p in enumerate(result0[0].parameters()):
    print(p.data)

# ----------------------------------------------------------------------------
# ---- Running the experiement for LS classifier (several alpha)


net_fin = lin()

loss_func = smooth_CE  
epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
num_epochs = 15

results = [run_experiment(net_fin, alpha=alpha, kind="adversarial", epsilons=epsilons) for alpha in alphas]

theta = [0,0]
for i, al in enumerate(alphas):
    plt.plot(results[i][4])
    plt.show()
    for j, p in enumerate(results[i][0].parameters()):
        print(p.data)
        theta[j] = p.data[1]-p.data[0]
    print("b = ", theta[1]/theta[0])


# ----------------------------------------------------------------------------
# ---- Plotting the results

df = []
for alpha, (_, _, _, _, _, accs) in zip(alphas,  results):
    for epsilon, acc in zip(epsilons, accs):
        df.append(dict(alpha=alpha, epsilon=epsilon, acc=acc, kind="adversarial",
                       temperature=0))
df = pd.DataFrame(df)

df.loc[ df['alpha']==0, 'acc' ] = result0[5]


df.to_csv("/Users/m.goibert/Documents/df_066.csv", sep=",")






# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ---------------------- Boucle pour gamma


gammas = [0.5, 0.6, 0.65, 2/3]
num_samples = 20000
loss_func = smooth_CE
epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
num_epochs = 15

dict_result0 = {}
dict_results = {}

for ind_gamma, gamma in enumerate(gammas):
    # globals()['string%s' % x] = 'Hello'

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
                            batch_size=round(0.4*0.4*num_samples),
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
                         alpha=alpha, kind="adversarial", epsilons=epsilons)
                         for alpha in alphas]
    theta = [0, 0]
    for i, al in enumerate(alphas):
        plt.plot(dict_results["{0}".format(ind_gamma)][i][4])
        plt.show()
        for j, p in enumerate(dict_results["{0}".format(ind_gamma)][i][0].parameters()):
            print(p.data)
            theta[j] = p.data[1]-p.data[0]
            print("b = ", theta[1]/theta[0])

    # ---------------- Saving the results

    df = []
    for alpha, (_, _, _, _, _, accs) in zip(alphas,  dict_results["{0}".format(ind_gamma)]):
        for epsilon, acc in zip(epsilons, accs):
            df.append(dict(alpha=alpha, epsilon=epsilon, acc=acc,
                           kind="adversarial", temperature=0))
    df = pd.DataFrame(df)

    df.loc[df['alpha'] == 0, 'acc'] = dict_result0["{0}".format(ind_gamma)][5]

    file_name = "df_gamma_" + str(round(gamma,2))
    df.to_csv(file_name, sep=",")




