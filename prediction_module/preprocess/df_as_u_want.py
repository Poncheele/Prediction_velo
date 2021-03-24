import pandas as pd
import numpy as np
import datetime

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
    return df1.to_frame()

def formatedweek(df):
    df.index = pd.date_range(datetime.date(2020,3,12),periods = len(df))
    df['weekday'] = df.index.weekday
    df['week'] = df.index.week
    # set weeks starts from first data don't reset at new year
    x = [0 for i in range(4) ]
    for i in range(1,(len(df)-4)//7+1):
        for j in range(7):
            x.append(i)
    for i in range((len(df)-4)%7):
        x.append(372//7+1)
    df.index = x
    df['week'] = x
    return df