#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_rows = 50
from download import download
import datetime
from copy import deepcopy
from statsmodels.tsa.arima_model import ARIMA
#%%
import prediction_module as pm 
df = pm.Load_db_predict().save_as_df()
df_day = pm.df_a_day(df)
df1 =pm.formatedweek(df_day)
#df1 = df1.drop(54)
df1 = df1.drop(0)

#%%
#df1= pm.df_new_year(df1)
df = deepcopy(df1)
df

#%%
del df['week']
df['week'] = df.index.week
df
#%%
toto = df[df['weekday']!=6]
df_week = toto[toto['weekday']!=5]
#%%
df_fri = df[df['weekday']==4]

# %%
days = ['Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday']

sns.set_palette("Paired", n_colors=7)
fig, ax = plt.subplots(figsize=(10,5))
df.groupby(['weekday'])[0].plot()
ax.legend(labels=days)
ax.set_xticks = df1.index
plt.show()
# %%
data = df.groupby(['weekday'])[0]
data = data.apply(list)
#%%
for i in range(7-len(df)%7):
        data[6-i].append(0)
# %%
data_cor = pd.DataFrame()
# %%
for i in range(len(days)):
       data_cor[days[i]] = data[i]

# %%
data_cor.corr()
# %%
sns.heatmap(data_cor.iloc[:,:5].corr(), annot=True)
# %%
import numpy as np
from sklearn.linear_model import LinearRegression
X = data_cor.iloc[:,1:4]
y = np.array(data_cor.iloc[:,4])
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
yo = np.array([[1727.0,	1826.0,	1713.0]])
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
# ARIMA SHIT
df = deepcopy(df_day)
df.index = np.arange(len(df))
# %%
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.iloc[295:,:][0].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# %%
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(2, 2, sharex=True)
axes[0, 0].plot(df[0]); axes[0, 0].set_title('Original Series')
plot_acf(df[0], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df[0].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df[0].diff().dropna(), ax=axes[1, 1], )
plt.show()

# %%
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

plot_acf(df[0].diff().dropna())

plt.show()
# %%
# 1,1,2 ARIMA Model
model = ARIMA(df[0], order=(2,1,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# %%
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
# %%
model_fit.plot_predict(dynamic=False)
plt.show()
# %%
# %%
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

model = pm.auto_arima(df[0], start_p=5, start_q=3,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=7, max_q=7, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

# %%
model.plot_diagnostics(figsize=(7,5))
plt.show()

# %%
fc, confint = model.predict(n_periods=5, return_conf_int=True)
# %%
model.plot_diagnostics(figsize=(7,5))
plt.show()
# %%
n_periods = 3
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(datetime.date(2021,3,31),
                                periods = n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df[0])
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()
# %%
model = ARIMA(df[0], order=(6,1,4))
model_fit = model.fit(disp=0)
model_fit.plot_predict(dynamic=False)
plt.show()
# %%
model_fit.predict(start = 300)
# %%
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(df[0], label='Original Series')
axes[0].plot(df[0].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dei
axes[1].plot(df[0], label='Original Series')
axes[1].plot(df[0].diff(7), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('a10 - Drug Sales', fontsize=16)
plt.show()
# %%
# !pip3 install pyramid-arima
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(df[0], start_p=1, start_q=2,
                         test='adf',
                         max_p=1, max_q=2, m=7,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()
# %%
# Forecast
n_periods = 2
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(df.index[-1], periods = n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# # Plot
# plt.plot(data)
# plt.plot(fitted_series, color='darkgreen')
# plt.fill_between(lower_series.index, 
#                  lower_series, 
#                  upper_series, 
#                  color='k', alpha=.15)

# plt.title("SARIMA - Final Forecast of a10 - Drug Sales")
# plt.show()
# %%
n_periods = 15
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
# %%
fitted
# %%
df= df.iloc[:-7,:]
# %%
