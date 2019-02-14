#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:19:48 2019

@author: m.goibert
"""

"""
Creation des fonctions pour faire tourner les algos sur MNIST
"""



import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm




# -----------------------



def smooth_CE(outputs, labels):
    
    """
    Loss function for smoothed labeled.
    Generalization of the cross-entropy loss. Needs a softmax as the last layer
    of the model
    """

    size = len(outputs)
    if outputs[0].dim() == 0:
        for i in range(size):
            outputs[i] = outputs[i].unsqueeze(-1)
    if labels[0].dim() == 0:
        for i in range(size):
            labels[i] = labels[i].unsqueeze(-1)
            
    res = 1/size * sum( [ torch.dot(outputs[i], labels[i]) for i in range(size) ] )
    return -res



# -----------------------
    

def one_hot(y, num_classes=None):
    """
    One hot encoding
    """
    if num_classes is None:
        num_classes = y.max()
    if y.dim()>0:
        y_ = torch.zeros(len(y), num_classes)
    else :
        y_ = torch.zeros(1, num_classes)
    y_.scatter_(1, y.unsqueeze(-1), 1)
    return y_


# -----------------------
    


def smooth_label(y, alpha, num_classes=None, y_pred=None, kind="standard",
                 temperature=.1):
    """
    Implements label-smoothing, both the standard and adversarial flavors.
    """

    y_ = (1 - alpha) * one_hot(y, num_classes=10)
    if alpha > 0.:
        if kind == "standard":
            salt = torch.ones_like(y_)
            salt = salt / (salt.sum(-1)-1).unsqueeze(-1)
        elif kind == "adversarial":
            bad_values, _ = y_pred.min(dim=-1)
            salt = (y_pred == bad_values.unsqueeze(-1)).double()
            salt = salt / salt.sum(-1).unsqueeze(-1)
        elif kind == "boltzmann":
            salt = F.softmax(-y_pred / temperature, dim=-1)
        else:
            raise NotImplementedError(kind)
        salt = salt * alpha
        y_ = y_ + salt
    return y_



# -----------------------



def train_model_smooth(model, train_loader, val_loader, optimizer, loss_func, num_epoch,
                       alpha = 0, kind = "standard", num_classes = None,
                       temperature = 0.1):
    """
    Training of a model using label smoothing.
    alpha is the parameter calibrating the strenght of the label smoothing
    kind = "standrard", "boltzmann" or "adversarial" is the type of label
        smoothing
    temperature is useful for kind = "boltzmann"
    
    Output :
        - the trained model
        - the loss function after each iteration
        - the accuracy on the training set
    """
    
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, verbose=True,
            factor=0.8)
    
    loss_history = []
    
    for epoch in range(num_epoch):
        
        model.train()
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.double()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            smoothed_y_batch = smooth_label(y_batch, alpha, y_pred=y_pred, kind=kind,
                                            num_classes=num_classes, temperature = temperature)
            loss = loss_func(y_pred, smoothed_y_batch)
            loss.backward()
            optimizer.step()
            
            #loss_history.append(loss)
            
            # eval
        #if val_loader is not None:
        model.eval()
        for x_val, y_val in val_loader:
            x_val = x_val.double()
        y_val_pred = model(x_val)
        smoothed_y_val = smooth_label(y_val, alpha, y_pred=y_val_pred, kind=kind,
                                      num_classes=num_classes, temperature = temperature)
        val_loss = loss_func(y_val_pred, smoothed_y_val)
        loss_history.append(val_loss.item())
            #print("Epoch {} / {}: val loss = {}".format(epoch + 1, num_epochs,
            #                                            val_loss))
        scheduler.step(val_loss)
        
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data = data.double()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(val_loader.dataset)
            
    return model, loss_history, acc


# ----------------------
    

def test_model(model, test_loader):
    
    """
    Run the model on the test set. Outputs the test set standard accuracy
    """
    
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.double()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_loader.dataset)
    return acc


# -----------------------

def attack_fgsm(data, epsilon, data_grad):
    
    """
    Run the FGSM method attack on a single data point using espilon.
    Returns the perturbated data.
    """
    
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon*sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data



# -----------------------
    

def run_fgsm(model, test_loader, epsilon, loss_func):
    
    """
    Run the fgsm attack on the whole test set.
    Outputs = adversarial accuracy and adversarial examples
    """
    
    correct = 0
    adv_examples = []
    
    for data, target in test_loader:
        
        data.requires_grad = True
        output = model(data)
        target_smooth = smooth_label(target, alpha, y_pred=output, kind=kind,
                                            num_classes=num_classes, temperature = temperature)
        init_pred = output.max(1, keepdim=True)[1] # Prediction (original data)
        
        if init_pred.item() != target.item():
            continue # If the model is already wrong, continue
        
        # Compute the gradient of the loss with respect to the data
        loss = loss_func(output, target_smooth)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        perturbed_data = attack_fgsm(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # Prediction (perturbated data)
        
        # Check for success
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        
        
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
    
    



# -----------------------
    
x = [2,1,2]
y = [1,1,1]
x+y


    
    
    