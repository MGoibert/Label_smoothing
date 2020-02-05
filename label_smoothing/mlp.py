"""
Author: Elvis Dohmatob <gmdopp@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)


class MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, hidden_dims=[], dropout=0,
                 final_activation=None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.final_activation = final_activation

        layers = []
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for l, _ in enumerate(layer_dims[:-1]):
            layers.append(nn.Linear(layer_dims[l], layer_dims[l + 1]))
            if l < len(layer_dims) - 2:
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(p=self.dropout))
        # layers.append(nn.LogSoftmax())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)
        if self.final_activation is not None:
            z = self.final_activation(z, dim=-1)
        return z


class MNISTMLP(nn.Module):
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, temp=False):
        x = x.view(-1, 28 * 28)
        x = x.double()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if temp:
            return self.soft(x/temp)
        else:
            x = self.soft(x)
            return x
