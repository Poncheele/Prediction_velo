#%%
import prediction_module as pm
import datetime
import pandas as pd
import re
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import plotly.express as px
import geopandas as gpd

# %%
df2 = pm.Load_db_vis.save_as_df2('Albert 1er')
df3 = pm.Load_db_vis.save_as_df2('Beracasa')
# %%
df2
# %%

# %%
coords = ((3.788909912109375,43.6291174376414),(3.7827301025390625,43.62464419335915) , 
(3.78204345703125,43.61321105671261), (3.7827301025390625,43.593819465174214),
(3.7957763671875,43.577406313314974),(3.812255859375,43.55750558409851), 
(3.8307952880859375,43.55103643145803), (3.8994598388671875,43.534113825940736),
(3.93310546875,43.560491112629286), (3.954391479492187,43.584867391661994),
(3.947525024414062,43.60277020720797), (3.9420318603515625,43.62663234302636),
(3.937225341796875,43.65048501002724), (3.93035888671875,43.66588482492509), 
(3.8891601562499996,43.66538811835199),(3.788909912109375,43.6291174376414))

#%%
import plotly.express as px
#px.set_mapbox_access_token(open(".mapbox_token").read())
df_car = px.data.carshare()
fig = px.scatter_mapbox(df_car, lat="lat", lon='lon', size="intensity",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
fig.update_layout(mapbox_style="carto-positron")
fig.show()
# %%
df = pd.concat((df2,df3))
# %%
df = pm.Load_db_vis.save_as_df2('Beracasa')

for i in range(1,len(pm.Load_db_vis.name)):
    df_temp = pm.Load_db_vis.save_as_df2(pm.Load_db_vis.name[i])  
    df = pd.concat((df,df_temp))
# %%
import plotly.express as px

# %%
