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
import prediction_module as pm 
df = pm.Load_db_predict().save_as_df()
df_day = pm.df_a_day(df)
df1 =pm.formatedweek(df_day)
#df1 = df1.drop(54)
df1 = df1.drop(0)
#%%
df = deepcopy(df1)
df
#%%
toto = df[df['weekday']!=6]
df_week = toto[toto['weekday']!=5]
#%%
df_fri = df[df['weekday']==4]
#%%
df_week
#%%
days = ['Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday']
#%%
#to see if variables are well correled
pd.plotting.lag_plot(df_week)
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
sns.set_palette("Paired", n_colors=7)
fig, ax = plt.subplots(figsize=(10,5))
df.groupby(['weekday'])[0].plot()
ax.legend(labels=days)
ax.set_xticks = df1.index
plt.show()
# %%
data = df.groupby(['weekday'])[0]
# %%
liste = data.apply(list)
# %%
len(liste[0])==len(liste[1])
# %%
sns.set_palette("Paired", n_colors=7)
fig, ax = plt.subplots(figsize=(10,5))
df_week.groupby(['weekday'])[0].plot()
ax.legend(labels=days)
ax.set_xticks = df1.index
plt.show()
# %%
data
#%%
data = df.groupby(['weekday'])[0]
data = data.apply(list)
# %%
data_cor = pd.DataFrame()
# %%
for i in range(len(days)-2):
       data_cor[days[i]] =  data[i]

# %%
data_cor.corr()
# %%
sns.heatmap(data_cor.iloc[:,:5].corr(), annot=True)
# %%
import numpy as np
from sklearn.linear_model import LinearRegression
X = data_cor.iloc[:53,1:4]
y = np.array(data_cor.iloc[:53,4])
reg = LinearRegression().fit(X, y)
reg.score(X, y)

#%%
X = np.concatenate((l.astype('int'),m.astype('int')),axis = 1)
# %%
data_cor.iloc[:,5:7]
# %%
np.concatenate(np.array(data_cor.iloc[:,0:4]),np.array(data_cor.iloc[:,5:7]))

# %%
np.concatenate((l.astype('int'),m.astype('int')),axis = 1)

# %%
np.array(data_cor.iloc[:,4])
# %%
data_cor
#%%
p = data_cor.iloc[:,1:4]
# %%
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array(p)
y = np.array(data_cor.iloc[:,4])
reg = LinearRegression().fit(X, y)
reg.score(X, y)
# %%

# %%
yo = np.array([[1839.0,	1906.0,	1970.0]])
reg.predict(yo)
# %%
data_cor
# %%

# %%
i = deepcopy(data_cor.iloc[:,6])
# %%
del i[0]
# %%
i[53] = 0
# %%
i = i.to_frame()
i.index = (np.arange(len(i)))
# %%
data_cor_pr = deepcopy(data_cor)
# %%
data_cor_pr['Sunday'] = i
# %%
datata = data_cor_pr.iloc[:52,]
# %%

# %%
np.concatenate(np.array(data_cor.iloc[:,0:4]),np.array(data_cor.iloc[:,5:7]))# %%

# %%
l = np.array(data_cor.iloc[:-1,0:4]).astype('int')
m = np.array(data_cor.iloc[:-1,5:7]).astype('int')
t = np.concatenate((l,m),axis = 1)
# %%
import numpy as np
from sklearn.linear_model import LinearRegression
X = t
y = np.array(data_cor.iloc[:-1,4])
reg = LinearRegression().fit(X, y)
reg.score(X, y)
# %%
sns.heatmap(data_cor_pr.corr(), annot=True)

# %%
yo = np.array([[1438.0,	1727.0,	1826.0,	1713.0,1018.0	,853.0]])
reg.predict(yo)
# %%
data_cor_pr
# %%

# %%
