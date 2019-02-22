"""
"""
import sys
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed

from utils import parse_cmdline_args
from _utils import train_net, smooth_label, one_hot
from mlp import MLP


def generate_overlap(num_samples, gamma=.5, beta=.5, random_state=None):
    assert 0. <= gamma <= 1.
    assert 0. < beta < 1.
    rng = check_random_state(random_state)
    x = 2 * rng.rand(num_samples) - 1
    y = (np.sign(x) + 1) / 2
    mask = np.abs(x) <= gamma
    y[mask] = rng.choice(range(2), size=mask.sum(), p=[beta, 1. - beta])
    return x[:, None], y


def run_experiment(alpha, kind, train_loader):
    net = MLP(n_features, output_dim=2)
    loss_func = lambda y_pred, y: F.binary_cross_entropy_with_logits(
        y_pred, smooth_label(y, alpha, y_pred=y_pred, kind=kind))
    net, _ = train_net(net, train_loader, learning_rate=learning_rate,
                       loss_func=loss_func)
    net = net.eval()
    preds = net(X_test).argmax(-1).data.numpy()
    # acc = roc_auc_score(preds, y_test.data.numpy().ravel())
    acc = np.mean(preds == y_test.data.numpy().ravel())
    return alpha, kind, net, acc


# misc
random_state = 42
args = parse_cmdline_args()
n_features = 1
num_train_samples = 10000
num_test_samples = 10000
batch_size = args.batch_size
learning_rate = args.learning_rate
gamma = .3
beta = .25
alphas = np.linspace(0, 1, num=args.num_alphas)
num_jobs = args.num_jobs
kinds = ["adversarial", "standard", "boltzmann"]

# prepare data
X_train, y_train = generate_overlap(num_train_samples, gamma=gamma, beta=beta,
                                    random_state=random_state)
X_test, y_test = generate_overlap(num_test_samples, gamma=0.,
                                  random_state=random_state)
X_train = torch.DoubleTensor(X_train)
y_train = torch.LongTensor(y_train.astype(int))
X_test = torch.DoubleTensor(X_test)
y_test = torch.LongTensor(y_test.astype(int))
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test),
                         batch_size=1, shuffle=True)


# run experiments in parallel
results = Parallel(n_jobs=num_jobs)(delayed(run_experiment)(alpha, kind,
                                                            train_loader)
                                    for alpha in alphas for kind in kinds)

# gather results
df = []
xs = np.linspace(-1, 1, num=1000)
for alpha, kind, net, acc in results:
    df.append(dict(alpha=alpha, kind=kind, acc=acc))
    logging.info("(%g, %s) ==> %g" % (alpha, kind, acc))
    preds = net(torch.tensor(xs[:, None]))
    # plt.figure()
    # plt.plot(xs, 2 * preds.argmax(-1).data.numpy() - 1,
    #          label="$\\alpha=%g$" % alpha)
    # plt.title("decision boundary: kind=%s, $\\alpha=%g$" % (kind, alpha))
    # plt.tight_layout()

# save / plot results
df = pd.DataFrame(df)
out_file = "understanding.csv"
df.to_csv(out_file)
logging.info("Dataframe of results written to file: %s" % out_file)
if False:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    df.alpha = round(df.alpha, 3)
    sns.factorplot(data=df, x="alpha", y="acc", hue="kind")
    plt.tight_layout()
    plt.show()
