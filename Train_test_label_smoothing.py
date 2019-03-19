#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:19:48 2019

@author: m.goibert,
         Elvis Dohmatob <gmdopp@gmail.com>
"""

"""
Creation des fonctions pour faire tourner les algos sur MNIST
"""

import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from attacks import FGSM, BIM, DeepFool, CW

cuda = torch.cuda.is_available()
logging.info("CUDA Available: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
print("device train = ", device)


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
            salt = torch.ones_like(y_, device=y_.device)
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
            optimizer, mode='min', patience=3, verbose=True,
            factor=0.5)

    loss_history = []

    for epoch in range(num_epochs):

        model.train()
        for x_batch, y_batch in tqdm(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
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
            x_val, y_val = x_val.to(device), y_val.to(device)
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
            data, target = data.to(device), target.to(device)
            data = data.double()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(val_loader.dataset)

    return model, loss_history, acc


def test_model(model, test_loader):
    """
    Run the model on the test set. Outputs the test set standard accuracy
    """

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.double()
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_loader.dataset)
    return acc


def attack_triangular(data, epsilon, r, lims=(-1, 1)):
    """
    Run the optimal attack on the Triangular example (linear model)
    """

    perturbed_data = data - epsilon * \
        (data.item() >= -r.item()) + epsilon * (data.item() < -r.item())
    perturbed_data = torch.clamp(perturbed_data, *lims)
    return perturbed_data


def run_attack(model, test_loader, alpha, kind, temperature,
               epsilons, loss_func, num_classes=None, lims=(0, 1),
               attack_method=None, num_iter=100):
    """
    Run the fgsm attack on the whole test set.
    Outputs = adversarial accuracy and adversarial examples

    Parameters
    ----------
    epsilons: list-like
        For CW attacks, this is interpreted as a list of confidences
    """
    model.eval()
    correct = {}
    num_test = 0
    adv_examples = {}

    print("Running attack")
    for batch_idx, (data, target) in enumerate(test_loader):
        num_test += len(data)
        data, target = data.to(device), target.to(device)
        output = model(data)

        # Prediction (original data)
        init_pred = output.argmax(1)

        # XXX really ?
        ok_mask = init_pred == target
        if ok_mask.sum() == 0:
            continue
        data = data[ok_mask]
        data.requires_grad = True
        target = target[ok_mask]
        output = model(data)

        if attack_method in ["FGSM", "BIM"]:
            target_smooth = smooth_label(
                target, alpha, y_pred=output, num_classes=num_classes,
                kind=kind, temperature=temperature)
            # target_smooth = target_smooth[ok_mask]
        if attack_method == "FGSM":
            attacker = FGSM(model, loss_func, lims=lims)
            perturbed_data = [attacker(data, target_smooth, epsilon)
                              for epsilon in epsilons]
        elif attack_method == "BIM":
            attacker = BIM(model, loss_func, lims=lims, num_iter=num_iter)
            perturbed_data = [attacker(data, target_smooth, epsilon)
                              for epsilon in epsilons]
        elif attack_method == "DeepFool":
            attacker = DeepFool(model, lims=lims, num_classes=num_classes,
                                num_iter=num_iter)
            perturbed_data = attacker(data, target)
            perturbed_data = [perturbed_data] * len(epsilons)  # XXX hack
        elif attack_method == "CW":
            attacker = CW(model, targeted=False, num_classes=num_classes,
                          cuda=cuda, lims=lims, num_iter=num_iter)
            perturbed_data = [attacker.run(data, target, epsilon,
                                           batch_idx=batch_idx)
                              for epsilon in epsilons]

        elif attack_method == "triangular":
            theta = [0, 0]
            for i, p in enumerate(model.parameters()):
                theta[i] = p.data[1] - p.data[0]
            r = theta[1] / theta[0]
            epsilons_ = torch.from_numpy(epsilons)
            perturbed_data = list(attack_triangular(data, epsilons_, r).t())

        perturbed_data = torch.stack(perturbed_data, dim=0)
        output = model(perturbed_data.view(-1, *list(data[0].size())))
        output = output.view(len(epsilons), len(data), *output.shape[1:])

        output = output
        target = target

        # Check for success
        for epsilon, pdata, o in zip(epsilons, perturbed_data, output):
            # Prediction (perturbated data)
            correct[epsilon] = correct.get(epsilon, 0)
            # final_pred = o.argmax(1, keepdim=True)
            final_pred = o.argmax(1)
            correct[epsilon] += (final_pred == target).sum().item()

            # XXX uncomment
            # if epsilon not in adv_examples:
            #     adv_examples[epsilon] = []
            # if final_pred.item() == target.item():
            #     correct[epsilon] += 1
            #     # Special case for saving 0 epsilon examples
            #     if (epsilon == 0) and (len(adv_examples) < 5):
            #         adv_ex = pdata.squeeze().detach().cpu().numpy()
            #         adv_examples[epsilon].append(
            #             (init_pred.item(), final_pred.item(), adv_ex))
            # else:
            #     # Save some adv examples for visualization later
            #     if len(adv_examples[epsilon]) < 5:
            #         adv_ex = pdata.squeeze().detach().cpu().numpy()
            #         adv_examples[epsilon].append(
            #             (init_pred.item(), final_pred.item(), adv_ex))

    final_acc = {}
    for epsilon in epsilons:
        final_acc[epsilon] = correct[epsilon] / float(num_test)
        print("Epsilon: %.3f\tTest Accuracy = %i / %i = %f" % (
            epsilon, correct[epsilon], num_test,
            final_acc[epsilon]))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
