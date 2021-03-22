from prediction_module.io import url_db, path_target
from download import download
import pandas as pd
class Load_db:
    def __init__(self,url_db=url_db,path_target=path_target):
        download(url_db, path_target, replace=True)
    @staticmethod
    def save_as_df():
        df_bikes = pd.read_csv(path_target, na_values="", low_memory=False, converters={'data': str, 'heure': str})
        return df_bikes