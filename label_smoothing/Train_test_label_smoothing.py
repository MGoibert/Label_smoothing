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

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .attacks import FGSM, BIM, DeepFool, CW, TriangularAttack

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


def _has_converged(history, convergence_threshold=1e-4, window_size=5):
    """
    Checks whether training has converged
    """
    history = history[-window_size:]
    window_size = len(history)
    relative_change = np.mean(
        np.abs(np.diff(history)) / np.abs(history[:-1]))

    if relative_change <= convergence_threshold:
        logging.info(("Convergence reached; average absrel change in loglik in "
               "%i past iterations: %g" % (window_size, relative_change)))
        return True
    else:
        return False


def train_model_smooth(model, train_loader, val_loader, loss_func, num_epochs,
                       learning_rate=0.1, verbose=1, alpha=0, kind="standard",
                       num_classes=None, temperature=0.1, use_lbfgs=False,
                       enable_early_stopping=False, compute_scores=True):
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
    # configure optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if use_lbfgs:
        optimizer = optim.LBFGS(parameters, lr=learning_rate)
    else:
        optimizer = optim.SGD(parameters, lr=learning_rate)
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, verbose=True,
            factor=0.5)

    # main learning loop
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        if verbose:
            train_loader = tqdm(train_loader)
        for x_batch, y_batch in train_loader:
            # prepare mini-batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.double()

            def closure():
                optimizer.zero_grad()
                y_pred = model(x_batch)
                smoothed_y_batch = smooth_label(y_batch, alpha, y_pred=y_pred,
                        kind=kind, num_classes=num_classes, temperature=temperature)
                loss = loss_func(y_pred, smoothed_y_batch)
                loss.backward()
                return loss

            # gradient step
            if use_lbfgs:
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

        # validation stuff
        if val_loader is not None:
            model.eval()
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_val = x_val.double()
            y_val_pred = model(x_val)
            smoothed_y_val = smooth_label(y_val, alpha, y_pred=y_val_pred, kind=kind,
                                    num_classes=num_classes, temperature=temperature)
            val_loss = loss_func(y_val_pred, smoothed_y_val)
            loss_history.append(val_loss.item())
        scheduler.step(val_loss)

        # check convergence
        if enable_early_stopping and val_loader is not None:
            if _has_converged(loss_history):
                print("Converged after %i / %i" % (epoch + 1, num_epochs))
                break

    # compute accuracy
    if compute_scores:
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
    else:
        return model, loss_history


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


def run_attack(model, test_loader, loss_func, epsilons, attack_method=None,
               alpha=None, kind="adversarial", temperature=None, num_classes=None,
               lims=(0, 1), num_iter=100):
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
    num_test = np.zeros(num_classes + 1)
    adv_examples = {}

    print("Running attack")
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        num_test[-1] += len(data)
        for label, counts in zip(*np.unique(target.cpu().data.numpy(),
                                        return_counts=True)):
            num_test[label] += counts

        # initial forward pass
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)

        # Prediction (original data)
        init_pred = output.argmax(1)
        ok_mask = init_pred == target
        num_ok = ok_mask.sum()
        if num_ok == 0:
            continue

        # maybe smooth labels
        if attack_method in ["FGSM", "BIM"] and alpha is not None:
            target_smooth = smooth_label(
                target, alpha, y_pred=output, num_classes=num_classes,
                kind=kind, temperature=temperature)
        else:
            target_smooth = target

        # instantiate attacker
        kwargs = {}
        if attack_method == "DeepFool":
            attacker = DeepFool(model, lims=lims, num_classes=num_classes,
                                num_iter=num_iter)
        elif attack_method == "triangular":
            attacker = TriangularAttack(model)
        elif hasattr(attack_method, "__call__"):
            attacker = attack_method
        elif attack_method == "FGSM":
            kwargs["pred"] = output
            attacker = FGSM(model, loss_func, lims=lims)
        elif attack_method == "BIM":
            attacker = BIM(model, loss_func, lims=lims, num_iter=num_iter)
        elif attack_method == "CW":
            attacker = CW(model, targeted=False, num_classes=num_classes,
                          cuda=cuda, lims=lims, num_iter=num_iter)
            kwargs["batch_idx"] = batch_idx
        else:
            raise NotImplementedError(attack_method)

        # run attacker
        if attack_method == "DeepFool":
            perturbed_data = attacker(data, target_smooth, **kwargs)
            perturbed_data = [perturbed_data] * len(epsilons)  # XXX hack
        elif attack_method == "triangular":
            perturbed_data = attacker(data, target_smooth, epsilons, **kwargs)
        else:
            perturbed_data = [attacker(data, target_smooth, epsilon, **kwargs)
                              for epsilon in epsilons]

        # reshape perturbed data
        perturbed_data = torch.stack(perturbed_data, dim=0)
        output = model(perturbed_data.view(-1, *list(data[0].size())))
        output = output.view(len(epsilons), len(data), *output.shape[1:])

        # apply ok_mask
        perturbed_data = perturbed_data[:, ok_mask]
        output = output[:, ok_mask]
        target = target[ok_mask]

        # Check for success
        target = target.cpu().data.numpy()
        output = output.cpu().data.numpy()
        for epsilon, _, o in zip(epsilons, perturbed_data, output):
            if epsilon not in correct:
                correct[epsilon] = np.zeros(num_classes + 1)

            # Prediction (perturbated data)
            correct[epsilon] = correct.get(epsilon, 0)
            final_pred = o.argmax(1)
            correct[epsilon][-1] += (final_pred == target).sum().item()
            for label in np.unique(target):
                mask = (target == label).astype(bool)
                correct[epsilon][label] += (final_pred[mask] == target[mask]).sum().item()
                assert correct[epsilon][label] <= num_test[label]

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
        final_acc[epsilon] = correct[epsilon] / num_test.astype(float)
        print("Epsilon: %.3f" % epsilon)
        for t in range(num_classes):
            print("\tClass=%i, Test Accuracy = %i / %i = %f" % (
                t, correct[epsilon][t], num_test[t],
                final_acc[epsilon][t]))
        print("\tClass=any, Test Accuracy = %i / %i = %f" % (
            correct[epsilon][-1], num_test[-1],
            final_acc[epsilon][-1]))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
