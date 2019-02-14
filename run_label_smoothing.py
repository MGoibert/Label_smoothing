#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:05:17 2019

@author: m.goibert
"""




"""
Libraries
"""

import os
#os.chdir("/Users/m.goibert/Downloads/safe_ml-master/code")
os.getcwd()
import torch
import torch.nn as nn
#from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.data import DataLoader
#from torch.utils.data import TensorDataset
#import torch.nn.functional as F

#from mlp import MLP
#from utils import (train_net, get_prediction_uncertainty, logging,
#                   parse_cmdline_args)
#from datasets import generate_tsipras18_data
#import fgsm_attack
#from plotting import plot_adversarial_results

#from tqdm import tqdm
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Change precision tensor
torch.set_default_tensor_type(torch.DoubleTensor)




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
    
    
"""
Dataset
"""

# Import MNIST
root = './data'
batch_size = 500
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




    
"""
Training
"""

# All parameters / Initialization
loss_func = smooth_CE
num_classes = 10
alphas = np.concatenate( (np.linspace(0,0.1, 11), np.linspace(0.1,1, 10)[1:]) )
num_epoch = 8
kind = "adversarial"
temperature = 0.1


train_res = []
for alpha in alphas:
    model = MLPNet()
    optimizer = optim.SGD(model.parameters(), lr=2)
    
    model, loss_history, acc = train_model_smooth(model, train_loader, val_loader,
                       optimizer, loss_func, num_epoch,
                       alpha = alpha, kind = kind, num_classes = num_classes,
                       temperature = 0.1)
    train_res.append( [model, loss_history, acc] )

i = 0
for i, alpha in enumerate(alphas):
    loss_history = train_res[i][1]
    #loss_history = torch.stack( loss_history )
    plt.plot(loss_history)#.detach().numpy())
    plt.show()
    
print( list(map(itemgetter(2), train_res)) )
    
plt.plot(alphas ,list(map(itemgetter(2), train_res)))
plt.xlabel('Alpha')
plt.ylabel('Train accuracy')
plt.title("Standard accuracy on the train set (standard smoothing)")
plt.show()






"""
Testing
"""

test_accs = []
for i, (model, loss, acc) in enumerate(train_res):
    test_acc = test_model(model, test_loader)
    test_accs.append(test_acc)
    
plt.plot(alphas, test_accs)
plt.xlabel('Alpha')
plt.ylabel('Standard accuracy')
plt.title("Standard accuracy (standard smoothing)")
plt.show()




"""
Adversarial training and testing
"""


epsilons = np.linspace(0,0.1, 11)

accuracies = []

for epsilon in epsilons:
    print("epsilon", epsilon)
    accuracy = []
    c = 0
    for model, loss, acc in train_res:
        start_time = time.time()
        c +=1
        print("Model n°", c)
        acc, ex = run_fgsm(model, test_loader, epsilon, loss_func)
        accuracy.append(acc)
        end_time = time.time()
        print("Execution time = %.2f sec" % (end_time - start_time))
    accuracies.append(accuracy)
        
        
for i, epsilon in enumerate(epsilons):
    name = "epsilon =" + str(epsilon)
    plt.plot( alphas, accuracies[i], label = name )
plt.legend()
plt.xlabel("Alpha (Boltzmann method)")
plt.ylabel("Accuracy")
plt.title("Adv. accurarcy for different values of epsilon (Boltzmann method)")
plt.show()




d = {'acc_adv':[], 'epsilon':[], 'alpha': [], 'method': []}
df = pd.DataFrame(data = d)

for i in range(len(accuracies)):
    eps = np.repeat(epsilons[i], len(accuracies[i]))
    meth = np.repeat("boltzmann", len(accuracies[i]))
    d = { 'acc_adv': accuracies[i], 'epsilon': eps, 'alpha': alphas, 'method':meth }
    df_temp = pd.DataFrame(data = d)
    df = df.append(df_temp, ignore_index=True)
    
file_name = '/Users/m.goibert/Documents/Criteo/Code/LS_good_version/Dataframes_outputs/acc_adv_boltzmann.csv'
df.to_csv(file_name, sep = ',')


