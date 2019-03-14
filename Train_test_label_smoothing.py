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
import copy
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm


def where(cond, x, y):
    """
    code from :
    https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.double()
    return (cond * x) + ((1 - cond) * y)


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


def attack_BIM(data, target_smooth, model, loss_func, epsilon, eps_iter=None,
               num_iter=4, lims=(-1, 1)):
    """
    BIM (or PGD, or I-FGSM) method: iterative algorithm based on FGSM
    """
    if eps_iter == None:
        eps_iter = epsilon / num_iter

    x_adv = Variable(data.data, requires_grad=True)
    target_smooth = target_smooth.detach()
    for i in range(num_iter):
        x_adv = Variable(x_adv.data, requires_grad=True)
        h_adv = model(x_adv)
        cost = loss_func(h_adv, target_smooth)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + eps_iter * x_adv.grad
        x_adv = where(x_adv > data + epsilon, data + epsilon, x_adv)
        x_adv = where(x_adv < data - epsilon, data - epsilon, x_adv)
        x_adv = Variable(x_adv.data.detach(), requires_grad=True)

    x_adv = torch.clamp(x_adv, *lims)
    del h_adv
    return x_adv


# -----------------------


def DeepFool(image, true_label, model, maxiter=50, lims=(-1, 1), num_classes=10):
    """  
    Our VGG model accepts images with dimension [B,C,H,W] and also we have
    trained the model with the images normalized with mean and std.
    Therefore the image input to this function is mean ans std normalized.
    min_val and max_val is used to clamp the final image
    """

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        model = model.cuda()
        image = image.cuda()

    model.train(False)
    f = model(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f)).argsort()[::-1]
    label = I[0]  # Image label

    input_shape = image.detach().cpu().numpy().shape

    # pert_image stores the perturbed image
    pert_image = image.detach().cpu().numpy().copy()
    w = np.zeros(input_shape)                #
    r_tot = np.zeros(input_shape)   # r_tot stores the total perturbation

    pert_image = torch.from_numpy(pert_image)

    if is_cuda:
        pert_image = pert_image.cuda()

    x = Variable(pert_image, requires_grad=True)
    fs = model(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]

    k_i = label  # k_i stores the label of the ith iteration
    loop_i = 0

    while loop_i < maxiter and k_i == label:
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        # Gradient wrt to the predicted label
        grad_khat_x0 = copy.deepcopy(x.grad.data.cpu().numpy())

        for k in range(1, num_classes):
            if x.grad is not None:
                x.grad.data.fill_(0)
            fs[0, I[k]].backward(retain_graph=True)
            grad_k = x.grad.data.cpu().numpy()

            w_k = grad_k - grad_khat_x0
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / (np.linalg.norm(w_k.flatten()))

            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert) * (w / np.linalg.norm(w.flatten()))
        r_tot = np.float64(r_tot + r_i.squeeze())

        if is_cuda:
            pert_image += (torch.from_numpy(r_tot)).cuda()
        else:
            pert_image += torch.from_numpy(r_tot).double()

        x = Variable(pert_image, requires_grad=True)
        fs = model(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
    pert_image = torch.clamp(pert_image, *lims)

    return pert_image



# -----------------------


def _to_attack_space(x, lims=(0, 1)):
    """
    For C&W attack: transform an input from the model space (]min, max[,
    depending on the data) into  an input of the attack space (]-inf, inf[).
    Take a torch tensor as input.
    """
    # map from [min, max] to [-1, +1]
    a = (lims[0] + lims[1]) / 2
    b = (lims[1] - lims[0]) / 2
    x = (x - a) / b

    # from [-1, +1] to approx. (-1, +1)
    x = x * 0.999999999999999

    # from (-1, +1) to (-inf, +inf)
    x = 1. / 2. * torch.log((1 + x) / (1 - x))

    return x


def _to_model_space(x, lims=(0, 1)):
    """
    For C&W attack: transform an input in the attack space (]-inf, inf[) into
    an input of the model space (]min, max[, depending on the data).
    Take a torch tensor as input.
    """

    # from (-inf, +inf) to (-1, +1)
    x = (1 - torch.exp(-2 * x * 0.999)) / (1 + torch.exp(
        -2 * x * 0.999))

    # map from (-1, +1) to (min, max)
    a = (lims[0] + lims[1]) / 2
    b = (lims[1] - lims[0]) / 2
    x = x * b + a

    return x

# ----------


def _soft_to_logit(softmax_list):
    """
    Maths: if p_i is the softmax corresponding to the logit z_i, then
    z_i = log(p_i) + const. This has not a unique solution, we choose const=0.

    XXX To check
    """
    return torch.log(softmax_list)

# ----------


def _fct_to_min(adv_x, reconstruct_data, target, logits, c, confidence=0, lims=(0, 1)):
    """
    C&W attack: Objective function to minimize. Untargeted implementation.
    Parameters
    ---------
    adv_x: adversarial exemple (original data in the attack space + perturbation)
    reconstruct_data: almost original data (original after to_attack_space and
                     to_model_space)
    target: tensor, true target class
    logits: logits computed by the model with adv_x as input
    c: constant value tunning the strenght of having the adv. example
        missclassified wrt having an adv example close to the original data.
    confidence: parameter tunning the level of confidence for the adv exemple.

    Technical
    ---------
    We compute the general objective function to minimize in order to obtain a
    "good" adversarial example (misclassified by the model, but close to the
    original data). This function has two part:
        - the distance between the original data and the adv. data: it is the l2
        distance in this setup.
        - the constraint that the adv. example must be misclassified. The model
        does not predict the original/true class iff the the logits of the true
        class is smaller than the largest logit of the other classes, i.e. the
        function computed in is_adv_loss is less than zero.
    Note the confidence value (default=0) tunne how smaller the logit of the
    class must be compared to the highest logit.

    Please refer to Toward Evaluating the Robustness of Neural Networks, Carlini
    and Wagner, 2017 for more information.
    """

    # Index of original class
    c_min = target.item()

    # We need the index of the largest logits that does not correspond to the
    # original class
    # index of all the logits exept the original class
    ind = np.array(
        [i for i in range(len(logits.squeeze())) if i != target.item()])
    # index of the largest logit in the list of all logits exept the one of
    # the original class. The list of logits has been modified, so the values
    # of the index too.
    c_max = logits.squeeze()[ind].argmax(-1, keepdim=True).item()
    # the indices after the original class have been shifted one place to the
    # left. We need to shift back to the right the value of c_max iff c_max
    # is superior or equal to the index of the original class (c_min)
    c_max = c_max * (c_max < c_min) + (c_max + 1) * (c_max >= c_min)

    # Constraint part of the objective function: corresponds to the constraint
    # that the classifier must misclassified the adversarial example
    is_adv_loss = max(
        logits.squeeze()[c_min] - logits.squeeze()[c_max] + confidence, 0)

    # Perturbation size part of the objective function: corresponds to the
    # minimization of the distance between the "true" data and the adv. data.
    l2_dist = torch.sum((adv_x - reconstruct_data)**2) / (lims[1] - lims[0])**2

    # Objective function
    tot_dist = l2_dist + c * is_adv_loss
    return tot_dist

# ----------


def CW_attack(data, target, model, binary_search_steps=5, max_iterations=200,
              confidence=0, learning_rate=0.05, initial_c=1e-2, lims=(0, 1)):
    """
    Carlini & Wagner attack.
    Untargeted implementation, L2 setup.
    """

    att_original = _to_attack_space(data.detach(), lims=lims)
    reconstruct_original = _to_model_space(att_original, lims=lims)

    c = initial_c
    lower_bound = 0
    upper_bound = np.inf
    best_x = data

    for binary_search_step in range(binary_search_steps):
        perturb = torch.zeros_like(att_original, requires_grad=True)
        optimizer_CW = torch.optim.Adam([perturb], lr=learning_rate)
        found_adv = False

        for iteration in range(max_iterations):
            x = _to_model_space(att_original + perturb, lims=lims)
            logits = _soft_to_logit(model(x))
            cost = _fct_to_min(x, reconstruct_original, target, logits, c,
                               confidence, lims=lims)
            optimizer_CW.zero_grad()
            cost.backward()
            optimizer_CW.step()

            if logits.squeeze().argmax(-1, keepdim=True) != target:
                found_adv = True
            else:
                found_adv = False

        if found_adv == True:
            upper_bound = c
            best_x = x
        else:
            lower_bound = c
        if upper_bound == np.inf:
            c = 10 * c
        else:
            c = (lower_bound + upper_bound) / 2

    return best_x


# -----------------------


def attack_triangular(data, epsilon, r, lims=(-1, 1)):
    """
    Run the optimal attack on the Triangular example (linear model)
    """

    perturbed_data = data - epsilon * \
        (data.item() >= -r.item()) + epsilon * (data.item() < -r.item())
    perturbed_data = torch.clamp(perturbed_data, *lims)

    return perturbed_data


# -----------------------


def run_attack(model, test_loader, alpha, kind, temperature,
               epsilon, loss_func, num_classes=None, lims=(0, 1),
               attack_method=None):
    """
    Run the fgsm attack on the whole test set.
    Outputs = adversarial accuracy and adversarial examples
    """

    model.eval()
    correct = 0
    adv_examples = []

    for data, target in test_loader:

        data.requires_grad = True
        output = model(data)
        target_smooth = smooth_label(target, alpha, y_pred=output, kind=kind,
                            num_classes=num_classes, temperature=temperature)
        # Prediction (original data)
        init_pred = output.argmax(1, keepdim=True)

        if init_pred.item() != target.item():
            continue  # If the model is already wrong, continue

        if attack_method == "FGSM":
            loss = loss_func(output, target_smooth)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = attack_fgsm(data, epsilon, data_grad, lims=lims)

        elif attack_method == "BIM":
            model.zero_grad()
            data.requires_grad = True
            perturbed_data = attack_BIM(data, target_smooth, model, loss_func,
                                epsilon, eps_iter=None, num_iter=4, lims=lims)

        elif attack_method == "DeepFool":
            perturbed_data = DeepFool(data, target, model, maxiter=50, lims=lims,
                                      num_classes=num_classes)

        elif attack_method == "CW":
            perturbed_data = CW_attack(data, target, model, binary_search_steps=5,
            max_iterations=200, confidence=0, learning_rate=0.05, initial_c=1e-2,
            lims=lims)

        elif attack_method == "triangular":
            theta = [0, 0]
            for i, p in enumerate(model.parameters()):
                theta[i] = p.data[1] - p.data[0]
            r = theta[1] / theta[0]
            perturbed_data = attack_triangular(data, epsilon, r)

        output = model(perturbed_data)
        # Prediction (perturbated data)
        final_pred = output.argmax(1, keepdim=True)

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
