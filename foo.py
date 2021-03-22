#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact  # widget manipulation

pd.options.display.max_rows = 50
from download import download
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
# %%
velo_ts

# %%
df1 = velo_ts['Vélos depuis le 1er janvier / Grand total'] - velo_ts["""Vélos ce jour / Today's total"""]

# %%
del df1[0]
df1.index = np.arange(len(df1))
df1
#%%
from copy import deepcopy
df2 = deepcopy(df1)
#%%
df2
#%%
i=1
while df2[i]>0:
    df2[i]=df2[i]-df2[0:i].sum()
    i+=1
#%%
df1 = deepcopy(df2)
#%%
df1
#%%
j=i+1
df1[j]
#%%
df1[i]=566
#%%
print(i)
df1[i]
#%%
from copy import deepcopy
df =deepcopy(df1)
df
#%%
df.iloc[290:300,]
#%%
j+=1
for k in range(j,len(df)):
    df[k]=df[k]-df[j-1:k].sum()
#%%
df[len(df-1)]=velo
# %%
velo_ts
# %%
df
# %%
velo_ts.iloc[-1,2]
# %%
def df_a_day(velo):
    last_count = velo.iloc[-1,3]
    velo =velo.drop_duplicates('Date')
    time_improved = pd.to_datetime(velo['Date'],format='%d/%m/%Y')
    velo['Date'] = time_improved
    del velo['Heure / Time']
    velo_ts = velo.set_index(np.arange(len(velo['Date'])))
    df1 = velo_ts['Vélos depuis le 1er janvier / Grand total'] - velo_ts["""Vélos ce jour / Today's total"""]
    del df1[0]
    df1.index = np.arange(len(df1))
    i = 1
    while df1[i] > 0:
        df1[i] = df1[i] - df1[0:i].sum()
        i += 1
    df1[i]=566 #use the real count at 2020-12-31
    j= i+2
    for k in range(j,len(df1)):
        df1[k] = df1[k] - df1[j-1:k].sum()
    df1[len(df1)] = last_count
    return df1
#%%
df_a_day(velo)
# %%
