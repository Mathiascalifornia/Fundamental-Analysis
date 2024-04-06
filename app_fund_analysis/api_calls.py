import pandas as pd
import yfinance as yf
from pandas_datareader import data

yf.pdr_override()


class ApiCaller:

    @staticmethod
    def get_price(ticker: str) -> pd.DataFrame:
        """
        Get the stock price and other indicators using yahoo finance
        """
        return data.get_data_yahoo(ticker, start="1975-01-01")

    @staticmethod
    def get_dividend(ticker: str) -> pd.DataFrame:
        """
        Get the dividend history
        """
        to_ret = pd.DataFrame(yf.Ticker(ticker).dividends)
        to_ret.index = pd.to_datetime([val.date() for val in list(to_ret.index)])
        return to_ret

    @staticmethod
    def get_main_institutions(ticker: str) -> list:
        """
        Returns the three biggest institutional holders
        """
        data_ = yf.Ticker(str(ticker).upper())
        main_inst = data_.institutional_holders
        return list(main_inst["Holder"][0:3])
