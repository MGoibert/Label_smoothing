#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:52:47 2019

@author: m.goibert
"""

# Plot the results using seaborn

import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

results_file = "results.pkl"
if len(sys.argv) > 1:
    results_file = sys.argv[1]
if results_file.endswith(".csv"):
    df_results = pd.read_csv(results_file)
else:
    assert results_file.endswith(".pkl")
    df_results = pd.read_pickle(results_file)
df_results = df_results.loc[df_results.kind.isin(["boltzmann", "standard"])]
df_results['alpha'] = round(df_results['alpha'], 3)

# # XXX rm hack
# df = df_results.loc[df_results.alpha.isin(
#     df_results.alpha.unique()[::2])]
df = df_results
df.epsilon = round(df.epsilon, 2)

# Plot 1 : advresarial accuracy as a function of epsilon for several models
# (alphas), for different methods (kind)
p2 = sns.factorplot(data=df, x="epsilon", y="acc", hue="alpha",
                    kind="point", col="kind", legend="full")
p2.set(xlabel = "$\\epsilon$ (adv. strenght)", ylabel = "Adversarial accuracy")
# plt.suptitle('Adv accuracy for different values of alpha')
plt.tight_layout()
plt.savefig('adv_acc_methods.png', dpi=300, bbox_inches='tight')

# Plot 2 : advresarial accuracy as a function of epsilon for several models
# (alphas)
# df = []
# for (kind, epsilon), subdf in df_results.groupby(["kind", "epsilon"]):
#     df.append(dict(kind=kind, epsilon=epsilon, acc=subdf["acc"].max()))
# df_results = pd.DataFrame(df)
# XXX rm hack
# df = df_results.loc[df_results.alpha.isin(
#     df_results.alpha.unique()[::2])]
df = df.loc[df.epsilon.isin(df.epsilon.unique()[::2])]
df.alpha = round(df.alpha, 2)
df.epsilon = round(df.epsilon, 2)
df = df.loc[df.epsilon < .6]
p1 = sns.factorplot(data=df, x="alpha", y="acc",  # reducer=np.max,
                    kind="point", legend="full", hue="kind", col="epsilon",
                    col_wrap=3)
p1.set(xlabel = "$\\alpha$", ylabel = "Adversarial accuracy")
# plt.title('Adv accuracy for different values of alpha (methods = ")')
plt.tight_layout()
plt.savefig('adv_acc_all.png', dpi=300, bbox_inches="tight")
plt.show()
