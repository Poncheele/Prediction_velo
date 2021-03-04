#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact  # widget manipulation

pd.options.display.max_rows = 8
from download import download
# %%
""" url = "https://docs.google.com/spreadsheets/d/1ssxsl9AIobDofXFohvwxqCPF0tn6dgXpixhiDzus0iE/edit#gid=59478853"
path_target = "./velo.csv"
download(url, path_target, replace=True)
 """
# df: data frame
velo_raw = pd.read_csv("velo.csv")
velo=velo_raw.iloc[1:,0:5]
plt.figure(figsize=(50, 50))
velo.iloc[:,4]
plt.figure()
plt.plot(velo.iloc[1:8,4])
plt.show()

# %%
