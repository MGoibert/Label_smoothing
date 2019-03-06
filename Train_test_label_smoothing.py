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
    res = 1. / size * \
        sum([torch.dot(torch.log(outputs[i]), labels[i]) for i in range(size)])
    return -res


# -----------------------


def one_hot(y, num_classes=None):
    """
    One hot encoding
    """
    if num_classes is None:
        classes, _ = y.max(0)
        num_classes = (classes.max() + 1).item()
    if y.dim() > 0:
        y_ = torch.zeros(len(y), num_classes)
    else:
        y_ = torch.zeros(1, num_classes)
    y_.scatter_(1, y.unsqueeze(-1), 1)
    return y_


# -----------------------


def smooth_label(y, alpha, num_classes=None, y_pred=None, kind="standard",
                 temperature=.1):
    """
    Implements label-smoothing. Methods:
        - Standard: uniform weights for all non-true classes
        - Adversarial: weight only on the true class and the smallest logit
            classe(s)
        - Boltzmann: warm adversarial using boltzmann distribution (with
            parameter temperature)
        - Second_best: weight only on the true class and the highest
            non-true logit class
    For each method, the true class receive weight at least 1-alpha
    """

    y_ = (1 - alpha) * one_hot(y, num_classes=num_classes)
    if alpha > 0.:
        if kind == "standard":
            salt = torch.ones_like(y_)
            salt = (1 - one_hot(y, num_classes=num_classes)) * \
                salt / (salt.sum(-1) - 1).unsqueeze(-1)
        elif kind == "adversarial":
            bad_values, _ = y_pred.min(dim=-1)
            salt = (y_pred == bad_values.unsqueeze(-1)).double()
            salt = salt / salt.sum(-1).unsqueeze(-1)
        elif kind == "boltzmann":
            a = torch.gather(y_pred, 1, y.unsqueeze(-1))
            b = (y_pred != a).double() * y_pred
            b[b == 0] = float('inf')
            salt = F.softmax(-b / temperature, dim=-1)
        elif kind == "second_best":
            bad_values = y_pred.max(dim=-1)[0] * \
                ( (y_pred.max(dim=-1)[1] != y).double() ) + \
                (y_pred * ( (y_pred != y_pred.max(-1)[0].unsqueeze(-1)).double() )). \
                max(dim=-1)[0] * ((y_pred.max(dim=-1)[1] == y).double())
            salt = (y_pred == bad_values.unsqueeze(-1)).double()
            salt = salt / salt.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError(kind)
        salt = salt * alpha
        y_ = y_ + salt
    return y_


# -----------------------


def train_model_smooth(model, train_loader, val_loader, loss_func, num_epochs,
                       alpha=0, kind="standard", num_classes=None,
                       temperature=0.1):
    """
    Training of a model using label smoothing.
    alpha is the parameter calibrating the strenght of the label smoothing
    kind = "standrard", "adversarial", "boltzmann" or "second_best" 
        is the type of label smoothing
    temperature is useful for kind = "boltzmann"

    Output :
        - the trained model
        - the loss function after each iteration
        - the accuracy on the validation set
    """

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, verbose=True,
            factor=0.8)

    loss_history = []

    for epoch in range(num_epochs):

        model.train()
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.double()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            smoothed_y_batch = smooth_label(y_batch, alpha, y_pred=y_pred,
                kind=kind, num_classes=num_classes, temperature=temperature)
            loss = loss_func(y_pred, smoothed_y_batch)
            loss.backward()
            optimizer.step()

            # loss_history.append(loss)

            # eval
        # if val_loader is not None:
        model.eval()
        for x_val, y_val in val_loader:
            x_val = x_val.double()
        y_val_pred = model(x_val)
        smoothed_y_val = smooth_label(y_val, alpha, y_pred=y_val_pred, kind=kind,
                            num_classes=num_classes, temperature=temperature)
        val_loss = loss_func(y_val_pred, smoothed_y_val)
        loss_history.append(val_loss.item())
        # print("Epoch {} / {}: val loss = {}".format(epoch + 1, num_epochs,
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

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.double()
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_loader.dataset)
    return acc


# -----------------------

def attack_fgsm(data, epsilon, data_grad, lims=(-1, 1)):
    """
    Run the FGSM method attack on a single data point using espilon.
    Returns the perturbated data.
    """
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, *lims)

    return perturbed_data


# -----------------------


def attack_triangular(data, epsilon, r):
    """
    Run the optimal attack on the Triangular example (linear model)
    """

    perturbed_data = data - epsilon * \
        (data.item() >= -r.item()) + epsilon * (data.item() < -r.item())
    perturbed_data = torch.clamp(perturbed_data, -1, 1)

    return perturbed_data


# -----------------------


def run_fgsm(model, test_loader, alpha, kind, temperature,
             epsilon, loss_func, num_classes=None, lims=(0, 1),
             method_attack=None):
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
                            num_classes=num_classes, temperature=temperature)
        # Prediction (original data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue  # If the model is already wrong, continue

        if method_attack == None:
            loss = loss_func(output, target_smooth)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = attack_fgsm(data, epsilon, data_grad, lims=lims)

        elif method_attack == "triangular":
            theta = [0, 0]
            for i, p in enumerate(model.parameters()):
                theta[i] = p.data[1] - p.data[0]
            r = theta[1] / theta[0]
            perturbed_data = attack_triangular(data, epsilon, r)

        output = model(perturbed_data)
        # Prediction (perturbated data)
        final_pred = output.max(1, keepdim=True)[1]

        # Check for success
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon,
                                        correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


# -----------------------
