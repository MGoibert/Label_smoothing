#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:45:23 2019

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
from tqdm import tqdm
import torch.optim as optim
import seaborn as sns
torch.set_default_tensor_type(torch.DoubleTensor)

#from Train_test_label_smoothing import (smooth_CE, smooth_label, one_hot,
#                                        train_model_smooth, test_model,
#                                        attack_fgsm, attack_triangular,
#                                        run_fgsm)



# --------------------
# --- Useful functions
#---------------------

# Generate dataset
def generate_dataset(num_samples, d, random_state=None):
    Y = (np.random.randint(0,1+1, num_samples) - 1/2)*2
    
    sigma = [1- (i)/(d) for i in range(d)]
    X2 = np.transpose(np.random.normal(np.array(list(Y)*d).reshape(d, num_samples)*np.repeat([s**2 for s in sigma], num_samples).reshape(d, num_samples), 
                          np.repeat(sigma, num_samples).reshape(d,num_samples), 
                          [d, num_samples])).tolist()
    return X2, Y

def one_hot(y, num_classes=None):
    """
    One hot encoding
    """
    if num_classes is None:
        classes, _ = y.max(0)
        num_classes = (classes.max() + 1).item()
    if y.dim() > 0:
        y_ = torch.zeros(len(y), num_classes, device=y.device)
    else:
        y_ = torch.zeros(1, num_classes)
    y_.scatter_(1, y.unsqueeze(-1), 1)
    return y_
    
def smooth_label_exp(y, alpha, y_pred):
    y_ = (1 - alpha) * one_hot(y, num_classes=num_classes)
    a = torch.zeros(len(y), num_classes)
    a.scatter_(1, (y_pred < 0.5).long(), 1)
    y_ = y_ + alpha*a
    return y_


def smooth_CE_exp(outputs, labels):
    size = len(outputs)
    if outputs[0].dim() == 0:
        for i in range(size):
            outputs[i] = outputs[i].unsqueeze(-1)
    if labels[0].dim() == 0:
        for i in range(size):
            labels[i] = labels[i].unsqueeze(-1)

    res = 1/size * sum( [ torch.log(1-outputs[i] +10**(-7)*(outputs[i] == 1).double())*labels[i][0] + 
                         torch.log(outputs[i] +10**(-7)*(outputs[i] == 0).double())*labels[i][1] for i in range(size) ] )
    return -res


# Convert dataset to pytorch data
def generate(num_samples, d):
    X, y = generate_dataset(num_samples, d)
    y[ y == -1] = 0
    train_size = .6
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size)
    X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, train_size=train_size)
    X_train = torch.DoubleTensor(X_train)
    y_train = torch.LongTensor(y_train.astype(int))
    X_val = torch.DoubleTensor(X_val)
    y_val = torch.LongTensor(y_val.astype(int))
    X_test = torch.DoubleTensor(X_test)
    y_test = torch.LongTensor(y_test.astype(int))
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=len(X_train), shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=1, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                                  batch_size=len(X_val), shuffle=False)
    test_loader.dataset = tuple(zip( map( lambda x: x.double(), map(itemgetter(0), test_loader.dataset)),
                                    map(itemgetter(1), test_loader.dataset) ))
    return train_loader, test_loader, val_loader

alpha = 0
# Train
def train_model(model, num_epochs, train_loader, val_loader):
    model = lin()
    for p in model.parameters():
        p[0] = torch.tensor( np.repeat(1.0,d), requires_grad=True )
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.75)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True,factor=0.3)
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.double()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            smoothed_y_batch = smooth_label_exp(y_batch, alpha, y_pred=y_pred)
            loss = loss_func(y_pred, smoothed_y_batch)
            loss.backward()
            for p in model.parameters():
                print(p[0][d-1].grad.data)
            
            optimizer.step()
            print(loss)
            
        model.eval()
        for x_val, y_val in val_loader:
            x_val = x_val.double()
        y_val_pred = model(x_val)
        smoothed_y_val = smooth_label_exp(y_val, alpha, y_pred=y_val_pred)
        val_loss = loss_func(y_val_pred, smoothed_y_val)
        loss_history.append(val_loss.item())
        scheduler.step(val_loss)
    
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data = data.double()
            output = (model(data) > 0.5).long().squeeze(-1)
            correct += sum( (output == target).double() ).item()
    acc = correct / len(val_loader.dataset)
    return model, loss_history, acc


def adv_attack(data, epsilon, y):
    #if epsilon > 1:
    #    eps = np.insert(np.repeat(epsilon, d), 0, 2)
    #    perturbed_data = data - torch.tensor(eps, dtype=torch.double)*np.sign(data).double()
    #else:
    #    eps = np.insert(np.repeat(epsilon, d), 0, 0)
    #    perturbed_data = data - torch.tensor(eps, dtype=torch.double)*np.sign(data).double()
    perturbed_data = data - epsilon*np.sign(y)#.double()
    return perturbed_data

def run_attack(model, test_loader, epsilon):
    correct = 0
    for data, target in test_loader:
        if (model(data)>0.5).double().item() != target.item():
            continue
        
        perturbed_data = adv_attack(data, epsilon, target.item())
        pred_perturbed = (model(perturbed_data) > 0.5).double().item()
        if pred_perturbed == target:
            correct +=1
    adv_acc = correct/float(len(test_loader))
    return adv_acc
    
# --------------------
# --- Experiments
#---------------------


# ------------------
# --- ALS classifier
#-------------------



# Parameters
d = 10
alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]#, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#alphas = [0]

num_samples = 10000
num_classes = 2
loss_func = smooth_CE_exp
num_epochs = 12
#alpha = 0.05

#epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
epsilons = [0]

num_exp = 15
dict_tot = {}

dict_param = {}
dict_param2 = {}

# Architecture
class lin(nn.Module):
    def __init__(self):
        super(lin, self).__init__()
        self.fc1 = nn.Linear(d, 1, bias=False)
        self.soft = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.soft(x)
        return x
    
for alpha in alphas:
    print("\n \n Alpha =", alpha, "\n \n")
    dict_param[alpha] = {}
    dict_param2[alpha] = {}
    for j in range(d-1):
        dict_param[alpha][j] = []
        dict_param2[alpha][j] = []
    dict_param2[alpha][d-1] = []

    dict_acc = {}
    keys = [0] + epsilons
    for key in keys:
        dict_acc[key] = []
    
    for i in range(num_exp):
        model = lin()
    
        # Generate dataset
        train_loader, test_loader, val_loader = generate(num_samples, d)

        # Train the dataset
        model, loss_history, acc_val = train_model(model, num_epochs,
                                           train_loader, val_loader)
        plt.plot(loss_history)
        plt.show()
        #print(acc_val)
        
        for i, p in enumerate(model.parameters()):
            for j in range(len(p.data[i])-1):
                dict_param2[alpha][j].append( (p.data[i][j]).item() )
                dict_param[alpha][j].append( (p.data[i][j]/p.data[i][len(p.data[i])-1]).item() )
            dict_param2[alpha][len(p.data[i])-1].append( (p.data[i][len(p.data[i])-1]).item() )
                
        # Test standard accuracy
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.double()
                output = (model(data) > 0.5).long().squeeze(-1)
                correct += sum( (output == target).double() ).item()
        acc = correct / len(test_loader.dataset)
        print("Standard accuracy (test) =", acc)
        dict_acc[0].append(acc)
    
    
        # Adversarial attack accuracy
        accuracy_adv = []
        for epsilon in epsilons:
            print("epsilon =", epsilon)
            acc_adv = run_attack(model, test_loader, epsilon)
            dict_acc[epsilon].append(acc_adv)
    
    dict_tot[alpha] = dict_acc

dict_param2

plt.plot(keys, [dict_acc[key] for key in keys])

# Model -- Value for w

for alpha in dict_param.keys():
    for j in range(d-1):
        val = np.asarray(dict_param[alpha][j])
        med = np.around(np.median(val), decimals = 2)
        q_0 = np.around(np.quantile(val, 0.25), decimals = 2)
        q_1 = np.around(np.quantile(val, 0.75), decimals = 2)
        
        #val = np.asarray(dict_param[alpha][j])[ np.asarray(np.abs(dict_param[alpha][j]) < np.std(dict_param[alpha][j])) ]
        #val = val[ val > 0 ]
        m = np.around(np.mean(val), decimals = 2)
        sd = np.around(np.std(val), decimals = 2)
        print("Alpha =", alpha, "j=", j, "mean=", np.mean(val), "std=", np.std(val), "Median=", med, "q0=", q_0, "q1=", q_1)

for i, p in enumerate(model.parameters()):
    print(p.data)
    for j in range(len(p.data[i])-1):
        print(p.data[i][j])
        print(i)
        print((p.data[i][0]/p.data[i][1]).item())
#        p02 = p.data
#        para = p.data

#p0 - p005
#p0 - p02



# -----------------------------
# --- Natural linear classifier
#------------------------------



# Parameters
#param = torch.tensor([np.insert(list(np.repeat(1/d, d)), 0, 0)] )
#param = torch.tensor([list(np.repeat(1/d, d))] )
param = torch.tensor([list(np.repeat(1.0, d))] )
#param = torch.tensor(list([[1.0, 1.0]]) )
#param2 = torch.tensor([list(np.repeat(0, np.int(d-3)))+list(np.repeat(1/(d-np.int(d-3)), d-np.int(d-3)))] )


# Architecture
class lin(nn.Module):
    def __init__(self):
        super(lin, self).__init__()
        self.fc1 = nn.Linear(d, 1, bias=False)
        self.soft = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.soft(x)
        return x

dict_acc = {}
keys = [0] + epsilons
for key in keys:
    dict_acc[key] = []

for i in range(num_exp):
    model = lin()
    
    # Generate dataset
    train_loader, test_loader, val_loader = generate(num_samples, d)
    
    for i, p in enumerate(model.parameters()):
        p.data = param

    # Test standard accuracy
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.double()
            output = (model(data) > 0.5).long().squeeze(-1)
            correct += sum( (output == target).double() ).item()
    acc = correct / len(test_loader.dataset)
    print("Standard accuracy (test) =", acc)
    dict_acc[0].append(acc)
    
    
    # Adversarial attack accuracy
    accuracy_adv = []
    for epsilon in epsilons:
        print("epsilon =", epsilon)
        acc_adv = run_attack(model, test_loader, epsilon)
        dict_acc[epsilon].append(acc_adv)

dict_tot[1] = dict_acc
dict_tot={}

col =  sns.color_palette("GnBu_d", 6) + [(0.7561707035755478, 0.21038062283737025, 0.22352941176470587)]
#sns.palplot([(0.7561707035755478, 0.21038062283737025, 0.22352941176470587)])
plt.style.use('ggplot')
for i, alpha in enumerate(dict_tot.keys()):
    if i < 6 or i == len(dict_tot.keys())-1:
        if i == len(dict_tot.keys())-1:
            i = 6
        m = [ np.mean(dict_tot[alpha][key]) for key in keys ]
        q_low = [ np.quantile(dict_tot[alpha][key], 0.1) for key in keys ]
        q_high = [ np.quantile(dict_tot[alpha][key], 0.9) for key in keys ]
        offsets = [np.abs(np.array(q_low) - np.array(m)), np.abs(np.array(q_high) - np.array(m))]
        plt.errorbar(keys, m, yerr=offsets, fmt='o-', color=col[i], ecolor=col[i], elinewidth=3, capsize=0)
plt.xlabel(r'Perturbation $\varepsilon$')
plt.ylabel("Accuracy")
plt.title("Accuracies for ALS and Bayes model")
plt.savefig("/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/plot_d="+str(d)+".png", dpi=500)
plt.show()

sns.palplot(sns.color_palette("GnBu_d", 10)[9])
sns.palplot(sns.color_palette("YlOrRd_r", 13)[:10])
sns.palplot(sns.dark_palette("red"))
col = [(0.21084711008586443, 0.2813533256439831, 0.30823529411764705),(0.6254722542611817, 0.8427579136229655, 0.7432167115212098)]
col_t = [(0.21084711008586443, 0.2813533256439831, 0.30823529411764705, 0.5), (0.6254722542611817, 0.8427579136229655, 0.7432167115212098, 0.5)]
col = sns.color_palette("YlOrRd_r", 13)[:10]
col_t = sns.color_palette("YlOrRd_r", 13)[:10]

plt.style.use('ggplot')
#alpha = 0.05
for w in range(d):
    #print(dict_param[alpha])
    m = [ np.median(dict_param2[alpha][w]) for alpha in dict_param2.keys() ]
    q_low = [ np.quantile(dict_param2[alpha][w], 0.25) for alpha in dict_param2.keys()  ]
    q_high = [ np.quantile(dict_param2[alpha][w], 0.75) for alpha in dict_param2.keys()  ]
    offsets = [np.abs(np.array(q_low) - np.array(m)), np.abs(np.array(q_high) - np.array(m))]
    plt.errorbar(dict_param.keys(), m, yerr=offsets, fmt='o-', color=col[w], ecolor=col_t[w], elinewidth=3, capsize=0)
    #plt.plot(dict_param.keys(), m, 'go-', color=col[w])
plt.xlabel(r'ALS parameter $\alpha$')
plt.ylabel(r"Parameter value")
plt.title("Parameters value for ALS models (d=3)")
#plt.savefig("/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/parameter_val_d="+str(d)+".png", dpi=500)
plt.show()

plt.style.use('ggplot')
#alpha = 0.05
for w in range(d-1):
    #print(dict_param[alpha])
    m = [ np.median([dict_param2[alpha][w][i]/dict_param2[alpha][d-1][i] for i, elem in enumerate(dict_param2[alpha][w])] ) for alpha in dict_param.keys() ]
    q_low = [ np.quantile(np.abs([dict_param2[alpha][w][i]/dict_param2[alpha][d-1][i] for i, elem in enumerate(dict_param2[alpha][w])]), 0.35) for alpha in dict_param.keys()  ]
    q_high = [ np.quantile(np.abs([dict_param2[alpha][w][i]/dict_param2[alpha][d-1][i] for i, elem in enumerate(dict_param2[alpha][w])]), 0.65) for alpha in dict_param.keys()  ]
    offsets = [np.abs(np.array(q_low) - np.array(m)), np.abs(np.array(q_high) - np.array(m))]
    plt.errorbar(dict_param.keys(), m, yerr=offsets, fmt='o-', color=col[w], ecolor=col_t[w], elinewidth=3, capsize=0)
    #plt.plot(dict_param.keys(), m, 'go-', color=col[w])
plt.xlabel(r'ALS parameter $\alpha$, $i=3 i=4 i = 5 i = 6 i = 7 i = 8 i = 9$')
plt.ylabel(r"Ratio $w_i / w_d$")
plt.title("Relative parameters value for ALS models (d=2)")
#plt.savefig("/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/parameter_plot_d="+str(d)+".png", dpi=500)
plt.show()

plt.style.use('ggplot')
#alpha = 0.05
for w in range(d-1):
    #print(dict_param[alpha])
    m = [ np.median(dict_param[alpha][w]) for alpha in dict_param.keys() ]
    q_low = [ np.quantile(dict_param[alpha][w], 0.4) for alpha in dict_param.keys()  ]
    q_high = [ np.quantile(dict_param[alpha][w], 0.6) for alpha in dict_param.keys()  ]
    offsets = [np.abs(np.array(q_low) - np.array(m)), np.abs(np.array(q_high) - np.array(m))]
    plt.errorbar(dict_param.keys(), m, yerr=offsets, fmt='o-', color=col[w], ecolor=col_t[w], elinewidth=3, capsize=0)
    #plt.plot(dict_param.keys(), m, 'go-', color=col[w])
plt.xlabel(r'ALS parameter $\alpha$')
plt.ylabel(r"Parameter value")
plt.title("Parameters value for ALS models (d=2)")
#plt.savefig("/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/parameter_val_d="+str(d)+".png", dpi=500)
plt.show()

# ----------------------
# Save results as a file
# ----------------------
import pickle
file_name = "/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/" + "new_toy_ex_d=" + str(d) + ".pkl"
with open(file_name, 'wb') as f:
    pickle.dump(dict_tot, f, pickle.HIGHEST_PROTOCOL)

file_name_w = "/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/" + "w_val_d=" + str(d) + "_v2.pkl"
with open(file_name_w, 'wb') as f:
    pickle.dump(dict_param2, f, pickle.HIGHEST_PROTOCOL)

d = 2
file_name_w = "/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/" + "w_val_d=" + str(d) + ".pkl"
with open(file_name_w, 'rb') as f:
    dict_param = pickle.load(f)
    
alpha = 0.01
for key in dict_param[alpha].keys():
    print(np.mean(dict_param[alpha][key]))

# -----------------------------
# --- plot dict final
#------------------------------

m = [ np.mean(dict_final[0.1][key]) for key in keys ]
q_low = [ np.quantile(dict_final[0.1][key], 0.1) for key in keys ]
q_high = [ np.quantile(dict_final[0.1][key], 0.9) for key in keys ]
offsets = [np.abs(np.array(q_low) - np.array(m)), np.abs(np.array(q_high) - np.array(m))]

m2 = [ np.mean(dict_final[0.05][key]) for key in keys ]
q_low2 = [ np.quantile(dict_final[0.05][key], 0.1) for key in keys ]
q_high2 = [ np.quantile(dict_final[0.05][key], 0.9) for key in keys ]
offsets2 = [np.abs(np.array(q_low2) - np.array(m2)), np.abs(np.array(q_high2) - np.array(m2))]

m3 = [ np.mean(dict_final[0.2][key]) for key in keys ]
q_low3 = [ np.quantile(dict_final[0.2][key], 0.1) for key in keys ]
q_high3 = [ np.quantile(dict_final[0.2][key], 0.9) for key in keys ]
offsets3 = [np.abs(np.array(q_low3) - np.array(m3)), np.abs(np.array(q_high3) - np.array(m3))]

m4 = [ np.mean(dict_final[10][key]) for key in keys ]
q_low4 = [ np.quantile(dict_final[10][key], 0.1) for key in keys ]
q_high4 = [ np.quantile(dict_final[10][key], 0.9) for key in keys ]
offsets4 = [np.abs(np.array(q_low4) - np.array(m4)), np.abs(np.array(q_high4) - np.array(m4))]

#plt.errorbar(keys, m, yerr=offsets, fmt='o-', color='lightcoral', ecolor='gray', elinewidth=3, capsize=0)
plt.errorbar(keys, m2, yerr=offsets2, fmt='o-', color='skyblue', ecolor='gray', elinewidth=3, capsize=0)
#plt.errorbar(keys, m3, yerr=offsets3, fmt='o-', color='turquoise', ecolor='gray', elinewidth=3, capsize=0)
plt.errorbar(keys, m4, yerr=offsets4, fmt='o-', color='coral', ecolor='gray', elinewidth=3, capsize=0)
plt.show()



# -----------------------------------
# --- Density dataset plot for d = 2
#------------------------------------


# Generate dataset
data = generate_dataset(5000, 2)
idx1 = [i for i, e in enumerate(data[1]) if e == 1]
X_class1 = [data[0][ind] for ind in idx1]
idx0 = [i for i, e in enumerate(data[1]) if e == -1]
X_class0 = [data[0][ind] for ind in idx0]

X1_class1 = list(map(itemgetter(0), X_class1))
X2_class1 = list(map(itemgetter(1), X_class1))
#sns.kdeplot(X1_class1, X2_class1, cmap="Reds", shade=True, bw=.25)

X1_class0 = list(map(itemgetter(0), X_class0))
X2_class0 = list(map(itemgetter(1), X_class0))
#sns.kdeplot(X1_class0, X2_class0, cmap="Reds", shade=True, bw=.25)

X1 = list(map(itemgetter(0), data[0]))
X2 = list(map(itemgetter(1), data[0]))
#sns.kdeplot(X1, X2, cmap="Reds", shade=True, bw=.15)

df = pd.DataFrame({"x1": X1_class1 + X1_class0, "x2": X2_class1 + X2_class0, "y": [data[1][ind] for ind in idx1] + [data[1][ind] for ind in idx0]})


sns.kdeplot(X1_class1, X2_class1, bw=0.5, label="class 1")
sns.kdeplot(X1_class0, X2_class0, bw=0.5, label="class -1")
plt.plot([-2,2], [2,-2], color="red")
plt.plot([0,0], [2,-2], color="red", linestyle='--')
plt.plot([-2/3,2/3], [2,-2], color="red", linestyle=':')
plt.xlim(-3.4, 3.4)
plt.ylim(-3.4, 3.4)
plt.legend()


sns.kdeplot(X1, X2, bw=1, cmap="Reds", shade=True)
plt.plot([-2,2], [2,-2], color="red")
#plt.plot([0,0], [2,-2], color="red", linestyle='--')
#plt.plot([-1/2,1/2], [2,-2], color="red", linestyle=':')
#plt.plot([4,-4], [1,-1], color="black", linestyle=':')
plt.plot([1,-1], [1,-1], color="black", linestyle=':')
plt.plot([1,-1], [4,-4], color="black", linestyle=':')
plt.plot([-1,1], [0.33,-0.33], color="red", linestyle=':')
#plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.title("KDE plot for d=2")
plt.savefig("/Users/m.goibert/Documents/Criteo/Project_1-Label_Smoothing/New_toy_example/kde_variability.png", dpi=500)
plt.show()

sns.kdeplot(X1_class1, X2_class1, bw=0.5, cmap="Reds", shade=True)
plt.plot([-2,2], [2,-2], color="red")
plt.plot([0,0], [2,-2], color="red", linestyle='--')
plt.plot([-1/2,1/2], [2,-2], color="red", linestyle=':')
plt.plot([4,-4], [1,-1], color="black", linestyle=':')
#plt.axis('equal')
plt.xlim(-2, 3.4)
plt.ylim(-2, 3.4)
plt.show()

plt.plot([-1/4,1/4], [1,-1], color="red", linestyle=':')
plt.plot([1,-1], [1/4,-1/4], color="black", linestyle=':')
plt.axis('equal')
plt.show()





# ---- Compute w Monte Carlo

alpha = 0
d = 2
w_list = [[elem, 1] for elem in np.linspace(-1, 2, 41)]
w1 = np.linspace(-1, 2, 41)

d_w = []
for w in w_list:
    print("w = ", w)

    num = 50000
    X, Y = generate_dataset(num, d)
    dec = [ np.dot(w,x) for x in X ]
    p1 = [1/(1+np.exp(- np.dot(w, x) )) for x in X]
    p0 = [1 - p1_ for p1_ in p1]

    i = 0
    count = 0
    for i in range(num):
        if dec[i]>0 and Y[i]==1:
            count += alpha*np.log(p0[i]) + (1-alpha)*np.log(p1[i])
        if dec[i]<0 and Y[i]==1:
            count += np.log(p1[i])
        if dec[i]>0 and Y[i]==-1:
            count += np.log(p0[i])
        if dec[i]<0 and Y[i]==-1:
            count += (1-alpha)*np.log(p0[i]) + alpha*np.log(p1[i])
    count = -count/num
    d_w.append(count)

df = pd.DataFrame({"w1":w1, "loss":d_w})
plt.plot(w1, d_w)



