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
# %%
'''
url = "https://docs.google.com/spreadsheets/d/1ssxsl9AIobDofXFohvwxqCPF0tn6dgXpixhiDzus0iE/edit#gid=59478853"
path_target = "./velo.csv"
download(url, path_target, replace=True)
'''
#%%
# df: data frame
velo_raw = pd.read_csv("velo.csv")
velo=velo_raw.iloc[2:,0:4]
plt.figure(figsize=(50, 50))
v=velo.iloc[:,3]
plt.figure()
plt.plot(velo.iloc[2:55,3])
plt.show()
#%%
velo
#%%
v=velo.iloc[:,3]
def firstdigit(v2):
    for i in range(2,len(v2)+2):
        if v2[i] > 100:
            v2[i] = int(v2[i]%100)
        if v2[i] > 10:
            v2[i] = int(v2[i]%10)
    return v2
    
df = firstdigit(v)

# %%
l=np.arange(1,10)
l2=[]
for i in l:
    l2.append(list(df).count(i))
l2=np.array(l2)/sum(l2)
l2
#%%
l3=[]
for i in range (1,10):
    l3.append(np.log10(1+1/i))
l3
plt.plot(l3)
# %%
plt.plot(l2,'.',label='velo')
plt.plot(l3,'.',color='m',label='benford')
# %%
plt.bar(l,l3)
plt.bar(l,l2)

# %%
velo=velo_raw.iloc[2:,0:4]
velo
# %%
width=0.35
fig, ax = plt.subplots()
rects1 = ax.bar(l+width/2,l3, width, label='Benford')
rects2 = ax.bar(l-width/2,l2, width, label='Velo')
ax.legend()
fig.tight_layout()
plt.show()
# %%
velo
#%% 
time_improved = pd.to_datetime(velo['Date'] +
                               ' ' + velo['Heure / Time'],
                               format='%d/%m/%Y %H:%M:%S')

# Where d = day, m=month, Y=year, H=hour, M=minutes
time_improved
#%%
from copy import deepcopy
velo_ts = deepcopy(velo)
# %%

velo_ts['Date']=time_improved
del velo_ts['Heure / Time']

# %%
del velo["Vélos depuis le 1er janvier / Grand total"]
velo.groupby(['Date']).max()
# %%
velo_ts = velo_ts.dropna()
velo_ts =velo_ts.sort_values(['Date'])
# %%
velo_ts = velo_ts.set_index(['Date'])

#%%
velo_raw = pd.read_csv("velo.csv")
velo=velo_raw.iloc[2:,0:4]
velo
# %%
velo =velo.drop_duplicates('Date')

time_improved = pd.to_datetime(velo['Date'],format='%d/%m/%Y')
velo['Date'] = time_improved
# %%
del velo['Heure / Time']
velo
# %%
velo_ts = velo.set_index(np.arange(len(velo['Date'])))
# %
#%%
df_a_day(velo)
# %%
import prediction_module as pm 
df = pm.Load_db().save_as_df()
df_day = pm.df_a_day(df)
df1 =pm.formatedweek(df_day)
#%%
days = ['Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday']

# %%
df = deepcopy (df1)
# %%
sns.set_palette("Paired", n_colors=7)
fig, ax = plt.subplots(figsize=(10,5))
df1.groupby(['weekday'])[0].plot()
ax.legend(labels=days)
ax.set_xticks = df1.index
plt.show()
# %%
df1 = df1.drop(54)
# %%
df1 = df1.drop(0)
#%%
df1[df1['weekday']==3]
# %%
df_day
# %%
df1 =pm.formatedweek(df_day)
# %%