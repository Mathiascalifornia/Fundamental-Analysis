from typing import Dict

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


# Note ; Les calculs commenceront toujours à year.now - 1
# Il nous faut un petit benchmark , qui sera enregistré , et ne sera recalculé que si l'année change
# Les entreprises de ce secteur seront au nombre de 10 , et seront représentative de ce qu'on attend d'une entreprise à dividende


class DividendScoreCalculator:

    # Old companies , highly representative of what we except from a good dividend company
    benchmark_tickers = ("KO" , "JNJ" , "XOM" , "MMM" , "ITW" , "PM" , "IBM" , "ED" , "O" , "PG" , "EPD" , "BLK" , "VZ" , "NWN")

    def __init__(self , df_dividend:pd.DataFrame , df_price:pd.DataFrame):
        self.df_dividend = df_dividend 
        self.df_price = df_price


    def main(self) -> dict:
        """ 
        Compute and return all the scores , in a dict with the score name as key
        """

    ### Not based on the benchmark ###
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
        def _get_std_pct_change_yield() -> float:
            return self.merged_yearly_div_price["yield"].pct_change().std() * 100
        
        def _get_mean_pct_change_yield() -> float:
            return self.merged_yearly_div_price["yield"].pct_change().mean() * 100
        
        def _get_correlation_price_and_dividends() -> float:
            return self.merged_yearly_div_price["Adj Close"].corr(self.merged_yearly_div_price["Dividends"] , method="pearson")
        

        standard_dev_pct_change_yield = _get_std_pct_change_yield()
        mean_pct_pct_change_yield = _get_mean_pct_change_yield()
        correlation_price_and_dividend = _get_correlation_price_and_dividends()

        return -(standard_dev_pct_change_yield + mean_pct_pct_change_yield -  correlation_price_and_dividend) # The higher the better
        

        

    # Based on the benchmark
    def compute_growth_score(self) -> float: # Compute with the yearly data
        """ 
        Compute growth score , based on the placement of the median growth percentage compared to the benchmark
        """
        def _compute_median_growth_pct_div() -> float:
            """ 
            Compute the dividend median growth in percentage 
            """
            return self.merged_yearly_div_price.Dividends.pct_change().median() * 100


    def compute_global_score(self) -> float:
        """ 
        Aggreagate all numbers and return the global dividend quality score
        """

    @classmethod
    def compute_benchmark(cls) -> Dict[str , dict]:
        """ 
        Re compute the benchmark , n of the benchmark_tickers class argument
        """


    