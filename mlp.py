import torch.nn as nn


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
