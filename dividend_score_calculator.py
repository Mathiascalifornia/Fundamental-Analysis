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

    # Old companies , highly representative of what we except from a dividend company
    benchmark_tickers = ("KO" , "JNJ" , "XOM" , "MMM" , "ITW" , "PM" , "IBM" , "ED")

    def __init__(self , df_dividend:pd.DataFrame , df_price:pd.DataFrame):
        self.df_dividend = df_dividend 
        self.df_price = df_price


    def compute_time_series_yield(self) -> pd.DataFrame:
        """ 
        Get the time series of yield over the years 
        """

    def compute_augmentation_strike_score(self) -> int:
        """ 
        Compute the number of year where the dividend have been growing , starting from year minus one
        """

    def compute_stability_score(self) -> float:
        """ 
        Compute a stabitlity score based on the standard deviation / trend of the yield ,
        the available payout ratio range , the mean , as well as the correlation between 
        the dividend payment and the share price (the higher the correlation , the better)
        """

    def compute_growth_score(self) -> float:
        """ 
        Compute growth score , based on the placement of the median growth percentage compared to the benchmark
        """

    def compute_global_score(self) -> float:
        """ 
        Aggreagate all numbers and return the global dividend quality score
        """

    @classmethod
    def compute_benchmark(cls):
        """ 
        Re compute the benchmark , made of the 
        """


    