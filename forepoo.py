#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact  # widget manipulation

pd.options.display.max_rows = 50
from download import download
import datetime
from copy import deepcopy
#%%
#to see if variables are well correled
pd.plotting.lag_plot(df)
# %%
fig, ax = plt.subplots(figsize=(10,5))
ax = plt.gca(xlim=(1, len(df)), ylim=(-1.0, 1.0))
pd.plotting.autocorrelation_plot(df)
# %%
df=deepcopy(df4)
#%%
df = df.to_frame()
# %%
df4.corr(df4.shift(1))
# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(df, model='additive')
x = decomposed.plot()
# %%
df['Statio'] = df.diff()
# %%
df
# %%
df['Statio'].plot()
# %%
decomposed = seasonal_decompose(df['Statio'].dropna(), model='additive')
x = decomposed.plot()
# %%
# %%
toto.corr().plot()
# %%
sns.heatmap(toto.corr())
# %%
df3[1]
# %%
df4 = deepcopy(df_day)
# %%
df.plot()
# %%
df
# %%
df.plot()
# %%