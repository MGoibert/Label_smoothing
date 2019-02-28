import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


dfs = []
for df in glob.glob("archive/27-02-2019/df_0*"):
    aux = df.split("_")[1][:-4]
    aux = aux + "0" * (3 - len(aux))
    gamma = float(aux) / 100
    df = pd.read_csv(df)
    df["gamma"] = gamma
    dfs.append(df)
df = pd.concat(dfs)
df["$\\alpha$"] = df.pop("alpha")
df["$\\gamma$"] = df.pop("gamma")
df["$\\epsilon$"] = df.pop("epsilon")
df["$acc_\\epsilon$"] = df.pop("acc")
sns.factorplot(data=df, x="$\\alpha$", y="$acc_\\epsilon$", hue="$\\epsilon$",
               col="$\\gamma$", col_wrap=2)
out_file = "fig.pdf"
plt.savefig(out_file, dpi=200, bbox_inches="tight")
print("fig written to file: %s" % out_file)
plt.show()
