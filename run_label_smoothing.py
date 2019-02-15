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
alphas = np.concatenate((np.linspace(0, 0.1, 3),
                         np.linspace(0.1,1, 10)[1:]))
num_epochs = args.num_epochs  # XXX TODO rm hack
kind = "boltzmann"
temperature = 0.1
epsilons = np.linspace(0,0.3, args.num_epsilons)



"""
Running
"""

net = MLPNet()


def run_experiment(net, alpha, kind, epsilons):
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

    return net, loss_history, acc_tr, accuracy_adv, delta_time


# run experiments in parallel
results = Parallel(n_jobs=num_jobs)(delayed(run_experiment)(net, alpha, kind,
                                                            epsilons)
                                    for alpha in alphas)

# form dataframe with results
df = []
_, axes = plt.figure(1, len(alphas), figsize=(3 * len(alphas), 4),
                     sharex=True, sharey=True)
axes = axes.ravel()
for alpha, ax, (_, loss_history, _, accs, _) in zip(alphas, axes, results):
    ax.plot(loss_history)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    for epsilon, acc in zip(epsilons, accs):
        df.append(dict(alpha=alpha, epsilon=epsilon, acc=acc, kind=kind,
                       temperature=temperature))
plt.tight_layout()
df = pd.DataFrame(df)
df.to_pickle("results.pkl")
plt.show()
assert 0


"""
Analysis
"""


for i, alph in enumerate(alphas):
    if i < 10:
        name = "a =" + str(alph)
        plt.plot( epsilons[0:3], accuracies_adv[i][0:3], label = name, alpha = 0.6 )
plt.legend(loc=(1.04,0))
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Adv. accurarcy for different values of epsilon (Boltzmann method)")

plt.savefig('/Users/m.goibert/Documents/Criteo/Code/LS_good_version/adv_acc_bolt2.png', dpi = 300)
plt.show()

# -----

plt.plot( epsilons, accuracies_adv[0], label = "alpha = 0", alpha = 0.6 )
plt.plot( epsilons, accuracies_adv[2], label = "alpha = 0.1", alpha = 0.6 )
plt.plot( epsilons, accuracies_adv[4], label = "alpha = 0.3", alpha = 0.6 )
plt.legend()
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Adv. accurarcy for different values of epsilon (Boltzmann method)")
plt.savefig('/Users/m.goibert/Documents/Criteo/Code/LS_good_version/adv_acc_bolt3.png', dpi = 300)

plt.show()

# -----
d = {'acc_adv':[], 'epsilon':[], 'alpha': [], 'method': []}
df = pd.DataFrame(data = d)

for i in range(len(accuracies_adv)):
    alph = np.repeat(alphas[i], len(accuracies_adv[i]))
    meth = np.repeat("boltzmann", len(accuracies_adv[i]))
    d = { 'acc_adv': accuracies_adv[i], 'epsilon': epsilons, 'alpha': alph, 'method':meth }
    df_temp = pd.DataFrame(data = d)
    df = df.append(df_temp, ignore_index=True)
    
    
    
sns.set()
sns.relplot(x="epsilon", y="acc_adv", hue="alpha", data=df, kind = "line");

# ---------------------------------
# ---------------------------------


"""
Normal model : test to check the functions
"""
model = MLPNet()
#optimizer = optim.SGD(model.parameters(), lr=1)
model_norm, loss_history, acc = train_model_smooth(model, train_loader, val_loader,
                       loss_func, num_epochs=num_epochs, alpha = 0, kind = kind,
                       num_classes = num_classes, temperature = 0.1)
plt.plot(loss_history)
plt.show()
acc

test_acc = test_model(model_norm, test_loader)

accuracies_norm = []
examples_norm = []
# Run test for each epsilon
for eps in epsilons:
    print("eps= ", eps)
    acc, ex = run_fgsm(model_norm,  test_loader, eps, loss_func)
    accuracies_norm.append(acc)
    examples_norm.append(ex)

plt.plot(epsilons, accuracies_norm)
plt.show()



# ---



model = MLPNet()
#optimizer = optim.SGD(model.parameters(), lr=1)
model_a1, loss_history, acc = train_model_smooth(model, train_loader, val_loader,
                       loss_func, num_epochs=num_epochs, alpha = 0.1, kind = kind,
                       num_classes = num_classes, temperature = 0.1)
plt.plot(loss_history)
plt.show()
acc

test_acc = test_model(model_a1, test_loader)

accuracies_1= []
examples_1 = []
# Run test for each epsilon
for eps in epsilons:
    print("eps= ", eps)
    acc, ex = run_fgsm(model_a1,  test_loader, eps, loss_func)
    accuracies_1.append(acc)
    examples_1.append(ex)

plt.plot(epsilons, accuracies_1)
plt.show()







