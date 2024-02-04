from typing import Dict , Union
import pickle
import os 

# Data manipulation
import pandas as pd 
import numpy as np
import re 
import textwrap
from googletrans import Translator
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import talib as ta


# Data visualisation
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns 
from pptx import Presentation
from pptx.util import Inches 
import dataframe_image as dsi



# Api , scraping
import requests
from bs4 import BeautifulSoup
import lxml
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException , UnexpectedAlertPresentException , NoAlertPresentException
import yfinance as yf
import pandas_datareader as web
from pandas_datareader import data
from pandas_datareader import DataReader
yf.pdr_override()


from .api_calls import ApiCaller

# Note ; Les calculs commenceront toujours à year.now - 1
# Il nous faut un petit benchmark , qui sera enregistré , et ne sera recalculé que si l'année change
# Les entreprises de ce secteur seront au nombre de 10 , et seront représentative de ce qu'on attend d'une entreprise à dividende


class DividendScoreCalculator:

    # Old companies , highly representative of what we except from a good dividend company
    benchmark_tickers = ("KO" , "JNJ" , "XOM" , "MMM" , "ITW" , "PM" , "IBM" , "ED" , "O" , "PG" , "EPD" , "BLK" , "VZ" , "NWN")

    PROFITABILITY_SCORE_WEIGHT = 1.5
    STABILITY_SCORE_WEIGHT = 1
    STRIKE_WEIGHT = 0.5

    BENCHMARK_FOLDER = os.path.join(os.path.dirname(__file__) , "benchmark_dividends_scores")

    def __init__(self , df_dividend:pd.DataFrame , df_price:pd.DataFrame):
        self.df_dividend = df_dividend 
        self.df_price = df_price

    def main(self) -> tuple:

        scores_ticker = self.get_all_scores()
        scores_benchmark = self.get_benchmark()

        return scores_ticker , scores_benchmark
        
    def get_all_scores(self) -> Dict[str , Union[int , float]]:

        self.compute_time_series_yearly()

        strike = self.compute_augmentation_strike_score()
        profitability_score = self.compute_profitality_score()
        stability_score = self.compute_stability_score()
        global_score = self.compute_global_score()

        return {
            "strike": strike,
            "profitability_score": profitability_score,
            "stability_score" : stability_score , 
            "global_score" : global_score
               }



    def compute_time_series_yearly(self):
        """ 
        Get the time series of yield over the years 
        """
        df_price = self.df_price.assign(year=self.df_price.index.year)
        df_dividend = self.df_dividend.assign(year=self.df_dividend.index.year)

        grouped = df_dividend.groupby("year" , as_index=False)["Dividends"].sum()

        merged_yearly_div_price = grouped.merge(df_price , how="left" , on="year").dropna(subset=["Adj Close" , "Dividends"])[["year" , "Adj Close" , "Dividends"]]
        merged_yearly_div_price = merged_yearly_div_price.drop_duplicates(subset="year" , keep="last").reset_index(drop=True)

        merged_yearly_div_price["yield"] = (merged_yearly_div_price["Dividends"] / merged_yearly_div_price["Adj Close"]) * 100

        self.merged_yearly_div_price = merged_yearly_div_price


    def compute_augmentation_strike_score(self) -> int:
        """ 
        Compute the number of years where the dividend have been growing , starting from year minus one
        """
        to_iterate = list(self.merged_yearly_div_price["Dividends"][::-1])
        counter_strike = 0

        for i in range(len(to_iterate)-1):
            if to_iterate[i] > to_iterate[i + 1]:
                counter_strike += 1 
            else:
                break 

        return counter_strike


    def compute_stability_score(self) -> float: # Computed with trimestrial data
        """ 
        Compute a stability score based on the standard deviation / trend of the yield and
        the mean , as well as the correlation between the dividend payment and the share price (the higher the correlation , the better)
        """
        def _get_std_pct_change_yield() -> float: # The lower the better
            return self.merged_yearly_div_price["yield"].pct_change().std() * 100
        
        def _get_mean_pct_change_yield() -> float: # The lower the better
            return self.merged_yearly_div_price["yield"].pct_change().mean() * 100
        
        def _get_correlation_price_and_dividends() -> float: # The higher the better
            return self.merged_yearly_div_price["Adj Close"].corr(self.merged_yearly_div_price["Dividends"] , method="pearson")*100
        
        # The higher the better
        return round((_get_correlation_price_and_dividends() - (_get_std_pct_change_yield() + _get_mean_pct_change_yield())) , 3)
        

        
    def compute_profitality_score(self) -> float: # Compute with the yearly data
        """ 
        Compute growth score , based on the median growth in percentage , and the median yield
        To get an idea of the magnitude ; for KO it's 60
        """
        def _compute_median_growth_pct_div() -> float:
            return self.merged_yearly_div_price["Dividends"].pct_change().median() * 100
        
        def _get_median_yield() -> float:
            return self.merged_yearly_div_price["yield"].median()
        
        return round(((_compute_median_growth_pct_div() + _get_median_yield()) / 2) * 10 , 3)
        

    def compute_global_score(self) -> float:
        """ 
        Aggreagate all numbers and return the global dividend quality score
        Here we give more importance 
        """
        self.compute_time_series_yearly()

        strike = self.compute_augmentation_strike_score() 
        stability_score = self.compute_stability_score()
        profitability_score = self.compute_profitality_score()

        strike_weight = DividendScoreCalculator.STRIKE_WEIGHT
        stability_weight = DividendScoreCalculator.STABILITY_SCORE_WEIGHT
        profitability_weight = DividendScoreCalculator.PROFITABILITY_SCORE_WEIGHT 
        sum_weights = sum([strike_weight , stability_weight , profitability_weight])

        return round(((strike*strike_weight) + (stability_score*stability_weight) + (profitability_score*profitability_weight)) / sum_weights , 3)


    @classmethod
    def get_benchmark(cls) -> Dict[str , dict]:
        """ 
        Compute the benchmark , based of the benchmark_tickers class argument
        """

        def _load_pickle_object(file_path):
            """
            Load an object from a pickle file
            """
            with open(file_path, 'rb') as file:
                return pickle.load(file)

        def _save_pickle_object(obj, file_path):
            """
            Save an object to a pickle file
            """
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)

        current_year = str(dt.datetime.now().year)
        path_to_save_benchmark_scores = os.path.join(cls.BENCHMARK_FOLDER , current_year)
        if os.path.exists(path_to_save_benchmark_scores):
            if os.path.basename(path_to_save_benchmark_scores)[:4] == current_year:
                return _load_pickle_object(path_to_save_benchmark_scores)



        # The current year , that we don't want to use to avoid misscalculation for the dividends (only ended fiscal years)
        year_to_remove = dt.datetime.now().year 

        # To aggregate in mean
        global_scores = []
        profitability_scores = []
        stability_scores = []
        strikes = []

        ticker:str
        for ticker in DividendScoreCalculator.benchmark_tickers:

            df_dividend = ApiCaller().get_dividend(ticker=ticker)
            df_price = ApiCaller().get_price(ticker=ticker)

            df_dividend = df_dividend.loc[df_dividend.index.year < year_to_remove]
            df_price = df_price.loc[df_price.index.year < year_to_remove]

            dividend_score_calculator = cls(df_dividend=df_dividend , df_price=df_price)

            scores = dividend_score_calculator.get_all_scores()
            
            global_scores.append(scores["global_score"])
            profitability_scores.append(scores["profitability_score"])
            stability_scores.append(scores["stability_score"])
            strikes.append(scores["strike"])


        benchmark_scores = {
                        "strike": round(np.mean(strikes)),
                        "profitability_score": round(np.mean(profitability_scores),3),
                        "stability_score" : round(np.mean(stability_scores),3) , 
                        "global_score" : round(np.mean(global_scores),3)
                            }

        _save_pickle_object(obj=benchmark_scores , file_path=path_to_save_benchmark_scores)
        return benchmark_scores
