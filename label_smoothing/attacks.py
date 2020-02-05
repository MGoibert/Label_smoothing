"""
Synopsis: Various adversarial attacks
Author: Morgane Goibert,
        Elvis Dohmatob <gmdopp@gmail.com>
"""
import copy

import numpy as np

import torch
from torch.autograd import Variable

from .externals.rwightman.attacks import AttackCarliniWagnerL2
from .custom_cw_attack import CW_attack


def where(cond, x, y):
    """
    code from :
    https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.double()
    return (cond * x) + ((1 - cond) * y)



class _BaseAttack(object):
    def __init__(self, model, lims=(-1, 1), num_iter=None):
        self.model = model
        self.lims = lims
        self.num_iter = num_iter

    def run(data, target, epsilon):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def clamp(self, data):
        if self.lims is None:
            return data
        if hasattr(self.lims, "__call__"):
            return self.lims(data)
        else:
            return torch.clamp(data, *self.lims)


class FGSM(_BaseAttack):
    """
    Fast Gradient Sign Method
    """
    def __init__(self, model, loss_func, lims=(-1, 1)):
        super(FGSM, self).__init__(model, lims=lims)
        self.loss_func = loss_func

    def run(self, data, target, epsilon, pred=None, retain_graph=True):
        """
        XXX `retain_graph=True` is needed in case caller is calling this
        function in a loop, etc.
        """
        if pred is None:
            pred = self.model(data)
        loss = self.loss_func(pred, target)
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        return self.clamp(data + epsilon * data.grad.data.sign())


class BIM(_BaseAttack):
    """
    BIM (Basic Iterative Method) method: iterative algorithm based on FGSM
    """
    def __init__(self, model, loss_func, lims=(-1, 1), num_iter=4):
        super(BIM, self).__init__(model, lims=lims, num_iter=num_iter)
        self.loss_func = loss_func

    def run(self, data, target, epsilon, epsilon_iter=None):
        target = target.detach()
        if epsilon_iter is None:
            epsilon_iter = epsilon / self.num_iter

        x_adv = data
        for _ in range(self.num_iter):
            x_adv = Variable(x_adv.data, requires_grad=True)

            # forward pass
            h_adv = self.model(x_adv)
            loss = self.loss_func(h_adv, target)

            # backward pass
            self.model.zero_grad()
            loss.backward(retain_graph=True)

            # single-step of FGSM: data <-- x_adv
            x_adv.grad.sign_()  # x_adv.grad <-- x_adv.grad.sign()
            x_adv = x_adv + epsilon_iter * x_adv.grad
            x_adv = where(x_adv > data + epsilon, data + epsilon, x_adv)
            x_adv = where(x_adv < data - epsilon, data - epsilon, x_adv)

        return self.clamp(x_adv)


class CW(_BaseAttack):
    """
    Carlini-Wagner Method
    """
    def __init__(self, model, lims=(-1, 1), binary_search_steps=5,
                 num_iter=10):
        _BaseAttack.__init__(self, model, lims=lims)
        self.binary_search_steps = binary_search_steps
        self.num_iter = num_iter

    def run(self, data, target, confidence, **kwargs):
        perturbed_data = CW_attack(
            data, target, self.model, num_iter=self.num_iter,
            binary_search_steps=self.binary_search_steps, lims=self.lims, **kwargs)
        return self.clamp(perturbed_data)


class CWBis(_BaseAttack, AttackCarliniWagnerL2):
    """
    Carlini-Wagner Method
    """
    def __init__(self, model, lims=(-1, 1), targeted=False,
                 binary_search_steps=15, num_iter=20, cuda=None,
                 debug=False, num_classes=1000):
        _BaseAttack.__init__(self, model, lims)
        AttackCarliniWagnerL2.__init__(
            self, model, targeted=targeted, search_steps=binary_search_steps,
            max_steps=num_iter, cuda=cuda, debug=debug, clamp_min=lims[0],
            clamp_max=lims[1], clamp_fn=None, num_classes=num_classes)

    def run(self, data, target, confidence, **kwargs):
        # XXX Doesn't seem to work for confidence > 0
        return AttackCarliniWagnerL2.run(self, data.detach(), target,
                                         confidence, **kwargs)


class DeepFool(_BaseAttack):
    def __init__(self, model, num_classes, lims=(-1, 1), num_iter=2000):
        super(DeepFool, self).__init__(model, lims=lims, num_iter=num_iter)
        self.num_classes = num_classes

    def run(self, image, true_label, epsilon=None):
        """
        Our VGG model accepts images with dimension [B,C,H,W] and also we have
        trained the model with the images normalized with mean and std.
        Therefore the image input to this function is mean ans std normalized.
        min_val and max_val is used to clamp the final image
        """
        if image.size(0) > 1:
            raise NotImplementedError("Minibatch version")

        self.model.eval()

        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.model = self.model.cuda()
            image = image.cuda()

        f = self.model(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
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
        fs = self.model(x)
        fs_list = [fs[0, I[k]] for k in range(self.num_classes)]

        k_i = label  # k_i stores the label of the ith iteration
        loop_i = 0

        while loop_i < self.num_iter and k_i == label:
            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            # Gradient wrt to the predicted label
            grad_khat_x0 = copy.deepcopy(x.grad.data.cpu().numpy())

            for k in range(1, self.num_classes):
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
            fs = self.model(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        return self.clamp(pert_image)


class TriangularAttack(_BaseAttack):
    """
    Run the optimal attack on the Triangular example (linear model)
    """
    def __init__(self, model, lims=(-1, 1)):
        super(TriangularAttack, self).__init__(model, lims=lims)

    def _attack(self, data, epsilon, r):
        perturbed_data = data - epsilon * (data.item() >= -r.item()
        ) + epsilon * (data.item() < -r.item())
        return self.clamp(perturbed_data)
        return perturbed_data

    def run(self, data, target, epsilon):
        theta = [0, 0]
        for i, p in enumerate(self.model.parameters()):
            theta[i] = p.data[1] - p.data[0]
        r = theta[1] / theta[0]
        epsilon_ = torch.from_numpy(epsilon)
        perturbed_data = self._attack(data, epsilon_, r)
        if hasattr(epsilon, "__len__"):
            perturbed_data = list(perturbed_data.t())
        return perturbed_data
