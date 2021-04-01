#%%
import prediction_module as pmd
import pmdarima as pm
import pandas as pd
import datetime
import numpy as np

#%%
df = pmd.Load_db_predict().save_as_df()
# %%
df_day = pmd.df_a_day(df).iloc[:-1, ]
# %%
df_day.plot(figsize = (15, 6))
# %%
df_day['bike_passing'] = df_day[0]
# %%
df_day.plot(figsize = (15, 6))
# %%
del df_day[0]
# %%
