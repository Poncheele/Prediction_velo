#%%
import pandas as pd
import matplotlib.pyplot as plt

import datetime
from copy import deepcopy
from statsmodels.tsa.arima_model import ARIMA

#%%
import prediction_module as pm 
df = pm.Load_db_predict().save_as_df()
#df_day = pm.df_a_day(df)
# %%
time_improved = pd.to_datetime(df['Date'] +
                               ' ' + df['Heure / Time'],
                               format='%d/%m/%Y %H:%M:%S')

# Where d = day, m=month, Y=year, H=hour, M=minutes
time_improved
# %%
df1 = pd.DataFrame()
df1['Date'] = time_improved
df1['velo H'] = df["""Vélos ce jour / Today's total"""]

#%%
df_day = pm.df_a_day(df)

# %%
time_improved = pd.to_datetime(df_day.index.astype(str)+ ' 23:59:59',
                               format='%Y-%m-%d %H:%M:%S')
# %%
df_day['Date'] = time_improved
# %%
df_day['velo H'] = df_day[0]
del df_day[0]
# %%
df_toto = pd.concat((df1,df_day))
# %%
df_toto = df_toto.sort_values(['Date'])
# %%
df_toto.index = df_toto['Date']
# %%
del df_toto['Date']
# %%
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(df["velo H"], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=7,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()
# %%
n_periods = 20
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)

# %%
import prediction_module as pm 
df = pm.Load_db_predict().save_as_df()
df_day = pm.df_a_day(df).iloc[:-1,]
# %%
# %%

df = deepcopy(df_day)
# %%
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(df[0], start_p=1, start_q=1,
                       test='adf',
                       max_p=3, max_q=3, m=7,
                       start_P=0, seasonal=True,
                       d=None, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

smodel.summary()
# %%
n_periods = 3
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(df.index[-1]+datetime.timedelta(days=1),
                            periods=n_periods)
# %%
fitted
# %%
# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot

plt.plot(df[0],figsize=(10,5))
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Final Forecast of a10 - Drug Sales")
plt.show()
# %%


df_toto['week_day'] = df_toto.index.weekday
# %%
df_week = df_toto[df_toto['week_day']!=5]
df_fri = df_week[df_week['week_day']!=6]
# %%
df_fri9 = df_fri[df_fri.index.hour == 9]
df_fri8 = df_fri[df_fri.index.hour == 8]
df_fri23 = df_fri[df_fri.index.hour == 23]
# %%
df_prop = pd.concat((df_fri9,df_fri23,df_fri8))
# %%
df_prop.index = np.arange(len(df_prop))
# %%
df_prop.sort_values(['Date']).iloc[-20:,]
# %%
prop_9 = [455.0/1906.0,	318.0/1970.0, 344.0/1580.0,
          358.0/1696.0, 371.0/1891.0,364.0/1945.0]
# %%
np.median(np.array(prop_9))*1703
# %%
