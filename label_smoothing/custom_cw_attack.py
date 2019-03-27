import numpy as np
import torch


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
    if False:
        adv_x = adv_x[:1]
        reconstruct_data = reconstruct_data[:1]
        logits = logits[:1]
    c_min = target.data.numpy()  # .item()

    # c_min = target.data.numpy()

    # We need the index of the largest logits that does not correspond to the
    # original class
    # index of all the logits exept the original class
    ind = np.array(
        [[i for i in range(len(logits[t].squeeze())) if i != target[t].item()]
         for t in range(target.shape[0])])
    # index of the largest logit in the list of all logits exept the one of
    # the original class. The list of logits has been modified, so the values
    # of the index too.
    c_max = [l.squeeze()[i].argmax(-1, keepdim=True).item()
             for l, i in zip(logits, ind)]
    c_max = np.array(c_max)
    # the indices after the original class have been shifted one place to the
    # left. We need to shift back to the right the value of c_max iff c_max
    # is superior or equal to the index of the original class (c_min)
    c_max = c_max * (c_max < c_min) + (c_max + 1) * (c_max >= c_min)

    # Constraint part of the objective function: corresponds to the constraint
    # that the classifier must misclassified the adversarial example
    i = range(len(logits))
    is_adv_loss = np.maximum(
        (logits[i, c_min] - logits[i, c_max] + confidence).data.numpy(), 0)
    is_adv_loss = torch.from_numpy(is_adv_loss).to(adv_x.device)

    # Perturbation size part of the objective function: corresponds to the
    # minimization of the distance between the "true" data and the adv. data.
    scale = (lims[1] - lims[0]) ** 2
    l2_dist = ((adv_x - reconstruct_data) **2).sum(1).sum(1).sum(1) / scale

    # Objective function
    tot_dist = l2_dist + c * is_adv_loss
    return tot_dist

# ----------


def CW_attack(data, target, model, binary_search_steps=5, num_iter=200,
              confidence=0, learning_rate=0.05, initial_c=1e-2, lims=(0, 1)):
    """
    Carlini & Wagner attack.
    Untargeted implementation, L2 setup.
    """
    batch_size = len(data)
    att_original = _to_attack_space(data.detach(), lims=lims)
    reconstruct_original = _to_model_space(att_original, lims=lims)

    c = torch.ones(batch_size) * initial_c
    lower_bound = np.zeros(batch_size)
    upper_bound = np.ones(batch_size) * np.inf
    best_x = data

    for binary_search_step in range(binary_search_steps):
        perturb = [torch.zeros_like(att_original[t], requires_grad=True)
                   for t in range(batch_size)]
        optimizer_CW = [torch.optim.Adam([perturb[t]], lr=learning_rate)
                        for t in range(batch_size)]
        perturb = torch.cat([perturb_.unsqueeze(0) for perturb_ in perturb])
        found_adv = torch.zeros(batch_size).byte()

        for iteration in range(num_iter):
            x = _to_model_space(att_original + perturb, lims=lims)
            logits = _soft_to_logit(model(x))
            cost = _fct_to_min(x, reconstruct_original, target, logits, c,
                               confidence, lims=lims)
            for t in range(batch_size):
                optimizer_CW[t].zero_grad()
                cost[t].backward(retain_graph=True)
                optimizer_CW[t].step()
                if logits[t].squeeze().argmax(-1, keepdim=True) != target[t]:
                    found_adv[t] = 1
                else:
                    found_adv[t] = 0

        for t in range(batch_size):
            if found_adv[t]:
                upper_bound[t] = c[t]
                best_x[t] = x[t]
            else:
                lower_bound[t] = c[t]
            if upper_bound[t] == np.inf:
                c[t] = 10 * c[t]
            else:
                c[t] = (lower_bound[t] + upper_bound[t]) / 2

    return best_x
