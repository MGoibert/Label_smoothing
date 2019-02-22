"""
Author: Elvis Dohmatob <gmdopp@gmail.com>
"""

import sys
import logging
import argparse

from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

# config
torch.set_default_tensor_type(torch.DoubleTensor)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def parse_cmdline_args():
    """
    Parses command-line arguments / options for this software.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments like a boss!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--experiment_name', type=str, default="tsipras18",
        choices=["tsipras18", "mnist", "spheres"],
        help="name of experiment to run")
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help="batch size for SGD")
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help="number of passes to make over data")
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help="learning rate for SGD")
    parser.add_argument(
        '--dropout', type=float, default=0.,
        help="amount of dropout to use in training")
    parser.add_argument(
        '--use_cuda', action="store_true",
        help="whether to use cuda or not")
    parser.add_argument(
        '--num_epsilons', type=int, default=20,
        help='number of epislon values to consider in attack')
    parser.add_argument(
        '--num_jobs', type=int, default=1,
        help="number of jobs to spawn for the experiment")

    return parser.parse_args()


def one_hot(y, num_classes=None):
    """
    One hot encoding
    """
    if num_classes is None:
        num_classes = y.max() + 1
    y_ = torch.zeros(len(y), num_classes)
    y_.scatter_(1, y.unsqueeze(-1), 1)
    return y_


def smooth_label(y, alpha, num_classes=None, y_pred=None, kind="standard",
                 temperature=.1):
    """
    Implements label-smoothing, both the standard and adversarial flavors.
    """
    assert len(y_pred.shape) == 2
    assert 0 <= alpha <= 1.
    if num_classes is None:
        assert y_pred is not None
        num_classes = y_pred.size(-1)
    batch_size = len(y)

    y_ = (1 - alpha) * one_hot(y, num_classes)
    if alpha > 0.:
        if kind == "standard":
            salt = torch.ones_like(y_)
            salt = salt / salt.sum(-1).unsqueeze(-1)
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



def smooth_cross_entropy(outputs, labels, alpha, kind="standard",
                         temperature=.1):
    """
    Loss function for smoothed labeled.
    Generalization of the cross-entropy loss. Needs a softmax as the last layer
    of the model
    """
    if alpha:
        outputs = smooth_label(outputs, alpha, kind=kind,
                               temperature=temperature)

    size = len(outputs)
    if outputs[0].dim() == 0:
        for i in range(size):
            outputs[i] = outputs[i].unsqueeze(-1)
    if labels[0].dim() == 0:
        for i in range(size):
            labels[i] = labels[i].unsqueeze(-1)
    res = 1. /size * sum([torch.dot(outputs[i], labels[i])
                          for i in range(size)])
    return -res



def _has_converged(history, convergence_threshold=1e-4, window_size=5):
    """
    Checks whether training has converged
    """
    history = history[-window_size:]
    window_size = len(history)
    relative_change = np.mean(
        np.abs(np.diff(history)) / np.abs(history[:-1]))

    if relative_change <= convergence_threshold:
        logging.info(("Convergence reached; average absrel change in loglik in"
                      " %i past iterations: %g" % (window_size,
                                                   relative_change)))
        return True
    else:
        return False


def train_net(net, train_loader, val_data=None, val_loss_func=None,
              learning_rate=0.01, num_epochs=50, patience=5,
              reduce_lr_factor=.75, loss_func=F.cross_entropy,
              enable_early_stopping=True):
    """
    Train a neural network
    """
    # prepare the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    logging.info("Optimizer:")
    print(optimizer)
    if val_data is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, verbose=True,
            factor=reduce_lr_factor)
        print(scheduler)
        if val_loss_func is None:
            val_loss_func = loss_func

    # training loop
    history = {"train": [], "val": []}
    for epoch in range(num_epochs):
        net.train()
        for x_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            y_pred = net(x_batch)
            loss = loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            history["train"].append(loss.item())

        # eval
        if val_data is not None:
            net.eval()
            x_val, y_val = val_data
            y_val_pred = net(x_val)
            val_loss = val_loss_func(y_val_pred, y_val)
            history["val"].append(val_loss.item())
            print("Epoch {} / {}: val loss = {}".format(epoch + 1, num_epochs,
                                                        val_loss))
            scheduler.step(val_loss)

            if enable_early_stopping:
                if _has_converged(history["val"]):
                    break
    return net, history
