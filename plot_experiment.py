#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:52:47 2019

@author: m.goibert
"""

# Plot the results using seaborn

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df_results = pd.read_pickle("results.pkl")
df_results['alpha'] = round(df_results['alpha'], 3)


# Plot 1 : advresarial accuracy as a function of epsilon for several models
# (alphas)
p1 = sns.relplot(x="epsilon", y="acc", hue="alpha", data=df_results,
            kind="line", legend="full")
p1.set(xlabel = "Epsilon (adv. strenght)", ylabel = "Adversarial accuracy")
plt.title('Adv accuracy for different values of alpha (method = boltzmann)')
plt.savefig('adv_acc_boltzmann', dpi = 300)
plt.show()


# Plot 2 : advresarial accuracy as a function of epsilon for several models
# (alphas), for different methods (kind)
p2 = sns.relplot(x="epsilon", y="acc", hue="alpha", data=df_results,
            kind="line", col = "kind", legend="full")
p2.set(xlabel = "Epsilon (adv. strenght)", ylabel = "Adversarial accuracy")
plt.title('Adv accuracy for different values of alpha')
#plt.savefig('adv_acc_methods', dpi = 300)
plt.show()








