import pathlib
import yaml 

import tqdm 

from app_v15 import App

### Made to launch several instance of the app in a row , using the yaml config file ### 

def load_yaml_file(file_path:pathlib.Path="config_file.yaml") -> dict:
    """ 
    Load the config file
    """
    with open(file_path , "r" , encoding="utf-8") as file:
        return yaml.safe_load(file)
    

config = load_yaml_file()

for ticker in tqdm.tqdm(config):

    dict_ticker:dict = config[ticker]

    App(ticker=ticker , company_name=dict_ticker["company_name"] ,
        language=dict_ticker.get("language" , "Français") ,
        path_to_save=dict_ticker["path_to_save"]).main()