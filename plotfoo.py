import prediction_module as pm
import datetime
import pandas as pd
import re
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import plotly.express as px
import geopandas as gpd

df2 = pm.Load_db_vis.save_as_df2('Albert 1er')
import plotly.express as px
fig = px.scatter_mapbox(df2, lat="lat", lon="lon",    size="intensity",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10, animation_frame="dateObserved")
fig.update_layout(mapbox_style="carto-positron")
#fig.show()

fig.write_html("stp.html")