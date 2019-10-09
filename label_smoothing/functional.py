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

from .attacks import FGSM, BIM, DeepFool, CW, CWBis, TriangularAttack

cuda = torch.cuda.is_available()
logging.info("CUDA Available: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
print("device train = ", device)


def smooth_cross_entropy(outputs, labels):
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
    res = 1. / size * sum([torch.dot(torch.log(outputs[i]), labels[i])
                           for i in range(size)])
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


def smooth_label(y, alpha, num_classes=None, y_pred=None,
                 smoothing_method="standard", temperature=.1):
    """
    Implements label-smoothing.

    Parameters
    ----------
    smoothing_method: string, Technology to use
        - standard: uniform weights for all non-true classes
        - sdversarial: weight only on the true class and the smallest logit
            classe(s)
        - boltzmann: warm adversarial using boltzmann distribution (with
            parameter temperature)
        - second_best: weight only on the true class and the highest
            non-true logit class

    For each method, the true class receive weight at least 1 - alpha
    """
    oh = one_hot(y, num_classes=num_classes)
    if alpha == 0:
        return oh
    num_classes = oh.size(1) if num_classes is None else num_classes
    y_ = (1 - alpha) * oh

    if smoothing_method == "standard":
        salt = torch.ones_like(y_, device=y_.device)
        salt = (1. - oh) / (num_classes - 1)
    elif smoothing_method == "adversarial":
        bad_values, _ = y_pred.min(dim=-1)
        salt = (y_pred == bad_values.unsqueeze(-1)).double()
        salt = salt / salt.sum(-1).unsqueeze(-1)
    elif smoothing_method == "boltzmann":
        a = torch.gather(y_pred, 1, y.unsqueeze(-1))
        b = (y_pred != a).double() * y_pred
        b[b == 0] = float('inf')
        salt = F.softmax(-b / temperature, dim=-1)
    elif smoothing_method == "second_best":
        bad_values = y_pred.max(dim=-1)[0] * \
            ( (y_pred.max(dim=-1)[1] != y).double() ) + \
            (y_pred * ( (y_pred != y_pred.max(-1)[0].unsqueeze(-1)).double() )). \
            max(dim=-1)[0] * ((y_pred.max(dim=-1)[1] == y).double())
        salt = (y_pred == bad_values.unsqueeze(-1)).double()
        salt = salt / salt.sum(-1).unsqueeze(-1)
    else:
        raise NotImplementedError(smoothing_method)
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

## Adversarial training dependencies
class Attack(object):
    """
    An abstract class representing attacks.
    Arguments:
        name (string): name of the attack.
        model (nn.Module): a model to attack.
    .. note:: device("cpu" or "cuda") will be automatically determined by a given model.
    
    """
    def __init__(self, name, model):
        self.attack = name
        self.mode = 'float'
        
        self.model = model
        self.training = model.training
        self.model_name = str(model).split("(")[0]
        self.device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
                
    # Whole structure of the model will be NOT displayed for print pretty.        
    def __str__(self):
        info = self.__dict__.copy()
        del info['model']
        del info['attack']
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"
    
    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()
        
        if self.mode == 'int' :
            images = self._to_uint(images)
            
        return images
    
    def _to_uint(self, images):
        return (images*255).type(torch.uint8)
    
    # It changes model to the original eval/train.
    def _switch_model(self):
        if self.training :
            self.model.train()
        else :
            self.model.eval()
    
    # It Defines the computation performed at every call.
    # Should be overridden by all subclasses.
    def forward(self, *input):
        raise NotImplementedError
    
    # Determine return all adversarial images as 'int' OR 'float'.
    def set_mode(self, mode):
        if mode == 'float' :
            self.mode = 'float'
        elif mode == 'int' :
            self.mode = 'int'
        else :
            raise ValueError(mode + " is not valid")
    
    def update_model(self, model) :
        self.model = model
        self.training = model.training
    
    # Save image data as torch tensor from data_loader.
    # If you want to reduce the space of dataset, set 'to_unit8' as True.
    # If you don't want to know about accuaracy of the model, set accuracy as False.
    def save(self, file_name, data_loader, to_uint8 = True, accuracy = True):
        
        self.model.eval()
        
        image_list = []
        label_list = []
        
        correct = 0
        total = 0
        
        total_batch = len(data_loader)
        
        for step, (images, labels) in enumerate(data_loader) :
            adv_images = self.__call__(images, labels)
          
            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())
            
            if accuracy :
                outputs = self.model(adv_images.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

            print('- Save Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')
        
        if accuracy :
            acc = 100 * float(correct) / total
            print('\n- Accuracy of the model : %f %%' % (acc), end='')
        
        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), file_name)
        print('\n- Save Complete!')
        
        self._switch_model()
        
    # Load image data as torch dataset
    # When scale=True it automatically tansforms images to [0, 1]
    def load(self, file_name, scale = True) :
        adv_images, adv_labels = torch.load(file_name)
        
        if scale :
            adv_data = torch.utils.data.TensorDataset(adv_images.float() / adv_images.max(), adv_labels)
        else :
            adv_data = torch.utils.data.TensorDataset(adv_images.float(), adv_labels)
            
        return adv_data


class PGD(Attack):
    """
    PGD attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): max iterations. (DEFALUT : 40)
        
    """
    def __init__(self, model, loss_func, eps=0.3, alpha=2/255, iters=40):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.loss_func = loss_func
    
    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = self.loss_func

        ori_images = images.data
        
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=-1, max=1).detach_()

        adv_images = images
        
        return adv_images


def train_model_smooth(model, train_loader, val_loader, loss_func, num_epochs,
                       learning_rate=0.1, verbose=1, alpha=0,
                       smoothing_method="standard", num_classes=None,
                       temperature=0.1, use_lbfgs=False,  compute_scores=True,
                       enable_early_stopping=False, adv_training=False, adv_training_param=0.2, 
                       adv_training_reg_param=0.75):
    """
    Training of a model using label smoothing.
    alpha is the parameter calibrating the strenght of the label smoothing
    smoothing_method = "standrard", "adversarial", "boltzmann" or "second_best" 
        is the type of label smoothing
    temperature is useful for smoothing_method = "boltzmann"

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
            optimizer, mode='min', patience=4, verbose=True,
            factor=0.1)
    if adv_training:
        pgd_attack = PGD(model, loss_func, eps = adv_training_param, alpha = adv_training_reg_param, iters = 3)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

    # main learning loop
    loss_history = []
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            print("Epoch:", epoch, " and learning_rate:", current_lr)
        model.train()
        if verbose:
            train_loader = tqdm(train_loader)
        for x_batch, y_batch in train_loader:
            # prepare mini-batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.double()
            if adv_training:
                x_batch.requires_grad = True
                
            def closure(): 
                optimizer.zero_grad()
                y_pred = model(x_batch)
                smoothed_y_batch = smooth_label(
                    y_batch, alpha, y_pred=y_pred,
                    smoothing_method=smoothing_method,
                    num_classes=num_classes,
                    temperature=temperature)
                loss = loss_func(y_pred, smoothed_y_batch)
                #loss.backward(retain_graph=True)
                if adv_training and epoch >= 20:
                    #x_adv_batch = PGD_training(model, x_batch, smoothed_y_batch, loss_func,
                    #    eps=adv_training_param, alpha=adv_training_reg_param, iters=20).double()
                    x_adv_batch = pgd_attack(x_batch, smoothed_y_batch)
                    y_pred_adv = model(x_adv_batch)
                    loss_adv = loss_func(y_pred_adv, smoothed_y_batch)
                    loss_tot = 0.5*loss + 0.5*loss_adv
                    loss_tot.backward(retain_graph=True)

                    #x_batch_perturbated = x_batch + adv_training_param*x_batch.grad.data.sign()
                    #model.zero_grad()
                    #y_pred_perturbated = model(x_batch_perturbated)
                    #loss_perturbated = loss_func(y_pred_perturbated, smoothed_y_batch)
                    #loss_tot = adv_training_reg_param*loss + (1-adv_training_reg_param)*loss_perturbated
                    #loss_tot.backward(retain_graph=True)
                    return loss_tot
                else:
                    loss.backward(retain_graph=True)
                    return loss
                #return loss

            # gradient step
            if use_lbfgs:
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

        # validation stuff
        if val_loader is not None and epoch >= 20:
            model.eval()
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_val = x_val.double()
            y_val_pred = model(x_val)
            smoothed_y_val = smooth_label(y_val, alpha, y_pred=y_val_pred,
                                          smoothing_method=smoothing_method,
                                          num_classes=num_classes,
                                          temperature=temperature)
            val_loss = loss_func(y_val_pred, smoothed_y_val)
            loss_history.append(val_loss.item())
            print("loss =", val_loss.item())
            scheduler.step(val_loss)
        #scheduler.step()

        # check convergence
        if enable_early_stopping and val_loader is not None:
            if _has_converged(loss_history):
                print("Converged after %i / %i epochs" % (
                    epoch + 1, num_epochs))
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
               alpha=None, smoothing_method="adversarial", temperature=None,
               num_classes=None, lims=(0, 1), num_iter=100):
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

    if len(np.unique(epsilons)) != len(epsilons):
        raise ValueError("The epsilons must be unique!")

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
        attacker = FGSM(model, loss_func, lims=lims)
    elif attack_method == "BIM":
        attacker = BIM(model, loss_func, lims=lims, num_iter=num_iter)
    elif attack_method == "CW":
        attacker = CW(model, lims=lims, num_iter=num_iter)
    elif attack_method == "CWBis":
        attacker = CWBis(model, targeted=False, num_classes=num_classes,
                         cuda=cuda, lims=lims, num_iter=num_iter)
    else:
        raise NotImplementedError(attack_method)
    print("Num iter = ", num_iter)

    print("Running %s attack" % attack_method)
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        batch_size = len(data)
        print("batch %i (%i examples)" % (batch_idx + 1, batch_size))
        if attack_method == "CWBis":
            kwargs["batch_idx"] = batch_idx

        # count number of examples per class
        for label, counts in zip(*np.unique(target.cpu().data.numpy(),
                                            return_counts=True)):
            num_test[label] += counts
        num_test[-1] += batch_size  # total

        # initial forward pass
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        if attack_method == "FGSM":
            kwargs["pred"] = output

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
                smoothing_method=smoothing_method, temperature=temperature)
        else:
            target_smooth = target

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

            # count correct predictions on perturbed data
            final_pred = o.argmax(1)
            correct[epsilon][-1] += (final_pred == target).sum().item()
            for label in np.unique(target):
                mask = (target == label).astype(bool)
                num_hits = (final_pred[mask] == target[mask]).sum().item()
                correct[epsilon][label] += num_hits
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
