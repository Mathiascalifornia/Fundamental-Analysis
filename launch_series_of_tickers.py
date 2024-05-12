import argparse
import pathlib
import yaml
import os
import sys

import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "app_fund_analysis"))

from app_fund_analysis.app import App


### Made to launch several instance of the app in a row , using the yaml config file ###
def main(file_path: pathlib.Path):
    def __load_yaml_file(file_path: pathlib.Path) -> dict:
        """
        Load the config file
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    config = __load_yaml_file(file_path)

    for ticker in tqdm.tqdm(config):

        dict_ticker: dict = config[ticker]

        if not os.path.exists(
            os.path.join(dict_ticker["path_to_save"], ticker + ".pptx")
        ):

            App(
                ticker=ticker,
                company_name=dict_ticker["company_name"],
                language=dict_ticker.get("language", "Français"),
                path_to_save=dict_ticker["path_to_save"],
            ).main()

            print(f"{ticker} Done !")


if __name__ == "__main__":

    default_path_config_file = os.path.join(
        os.path.dirname(__file__), "config_files", "config_file.yaml"
    )

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_file_path", default=default_path_config_file)

    args = argparser.parse_args()

    main(file_path=args.config_file_path)
