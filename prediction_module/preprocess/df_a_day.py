import pandas as pd
import numpy as np

def df_a_day(self):
    last_count = self.iloc[-1,3]
    velo = self.drop_duplicates('Date')
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