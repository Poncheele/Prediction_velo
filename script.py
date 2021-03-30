#%%
import prediction_module as pm 
#%%
import pandas as pd
#%%
import prediction_module as pm 
df = pm.Load_db_predict().save_as_df()
#%%
df
# %%
df_day = pm.df_a_day(df)
# %%
df_day
# %%
time_improved = pd.to_datetime(df['Date'] +
                               ' ' + df['Heure / Time'],
                               format='%d/%m/%Y %H:%M:%S')

# Where d = day, m=month, Y=year, H=hour, M=minutes
time_improved
# %%
pm.Load_db_vis()
# %%
df = pm.save_as_df2()
# %%
