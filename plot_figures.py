"""
Plot experimental results from .csv files generate by run_label_smoothing.py

"""
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")  # XXX use better style, e.g "ggplot" ?

# read files
df = pd.concat(list(map(pd.read_csv, sys.argv[1:])))
del df["Unnamed: 0"]

# only look at average accuracy over all classes
df = df.loc[df.label == df.label.max()]
print(df.head())

# avoid long floats in plots
df.alpha = round(df.alpha, 2)
df.epsilon = round(df.epsilon, 2)

# choose color palette
palette = sns.color_palette("YlOrRd", 15)

# do the plotting
for x in ["alpha", "epsilon"]:
    if x == "alpha":
        hue = "epsilon"
    else:
        hue = "alpha"
    sns.factorplot(data=df, x=x, y="acc", hue=hue, col="smoothing_method",
                   palette=palette)

# last sip
plt.tight_layout()
plt.show()
