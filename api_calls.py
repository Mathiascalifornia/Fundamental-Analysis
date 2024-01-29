import pandas as pd 
import yfinance as yf
from pandas_datareader import data
yf.pdr_override()


class ApiCaller:
    def __init__(self , ticker):
        self.ticker = ticker 

    def get_price(self) -> tuple[pd.DataFrame , pd.DataFrame]:
        '''Get the stock price , volume , open , close , high , low ... Using Datareader (data from yahoo finance)'''
        df_price = data.get_data_yahoo(self.ticker , start='1975-01-01')
        sp500_price = data.get_data_yahoo("^GSPC" , start='1975-01-01')
        sp500_price = sp500_price[sp500_price.index >= min(df_price.index)]

        return df_price , sp500_price


    def get_dividend(self) -> pd.DataFrame:
        '''Get the dividend history'''
        to_ret = pd.DataFrame(yf.Ticker(self.ticker).dividends)
        to_ret.index = pd.to_datetime([val.date() for val in list(to_ret.index)])
        return to_ret

    def get_main_institutions(self) -> list:
        '''Returns the three biggest institutional holders'''
        data_ = yf.Ticker(str(self.ticker).upper())
        main_inst = data_.institutional_holders
        return list(main_inst['Holder'][0 : 3])
