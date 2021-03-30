import prediction_module as pm
import pandas as pd
import plotly.express as px


df = pm.Load_db_vis.save_as_df2('Beracasa')

for i in range(1,len(pm.Load_db_vis.name)):
    df_temp = pm.Load_db_vis.save_as_df2(pm.Load_db_vis.name[i])  
    df = pd.concat((df,df_temp))


import plotly.express as px
fig = px.scatter_mapbox(df, lat="lat", lon="lon",    size="intensity", color = 'intensity', range_color= [0,1500],
                  color_continuous_scale=px.colors.diverging.Temps, size_max=20, zoom=12, animation_frame="dateObserved")
fig.update_layout(mapbox_style="carto-positron", showlegend = True)
fig.show()

fig.write_html("vis.html")