import datetime
import os

import pandas as pd
import yfinance as yf
from pandas_datareader import data

yf.pdr_override()


class ApiCaller:
    @staticmethod
    def create_forex_exchange_df() -> pd.DataFrame:

        # List of currencies to convert into dollars
        currencies = {
            "ARS",
            "AUD",
            "BHD",
            "BMD",
            "BOB",
            "BRL",
            "BZD",
            "CAD",
            "CHF",
            "CNY",
            "COP",
            "CRC",
            "CYP",
            "DKK",
            "DOP",
            "EGP",
            "EUR",
            "GBP",
            "HKD",
            "HNL",
            "IDR",
            "ILS",
            "INR",
            "ISK",
            "JMD",
            "JOD",
            "KRW",
            "KWD",
            "LAK",
            "LBP",
            "LKR",
            "LSL",
            "LTL",
            "LVL",
            "MAD",
            "MKD",
            "MUR",
            "MVR",
            "MWK",
            "MYR",
            "MZN",
            "NAD",
            "NGN",
            "NIO",
            "NOK",
            "NPR",
            "NZD",
            "OMR",
            "PEN",
            "PGK",
            "PHP",
            "PKR",
            "PLN",
            "PYG",
            "RON",
            "RUB",
            "SAR",
            "SCR",
            "SEK",
            "SGD",
            "SOS",
            "SRD",
            "SYP",
            "THB",
            "TMT",
            "TND",
            "TRY",
            "TTD",
            "TWD",
            "TZS",
            "UAH",
            "UGX",
            "USD",
            "UYU",
            "UZS",
            "VND",
            "VUV",
            "WST",
            "XPF",
            "ZAR",
        }

        # Create a list of currency pairs in dollars for each currency
        currency_pairs = [currency + "USD=X" for currency in currencies]

        end_date = pd.Timestamp.now()

        start_date_25_years = end_date - pd.DateOffset(years=25)

        historical_data_25_years = data.DataReader(
            currency_pairs, start_date_25_years, end_date
        )["Close"]
        historical_data_25_years.columns = [
            col.replace("USD=X", "").strip() for col in historical_data_25_years.columns
        ]
        historical_data_25_years["USD"] = 1
        historical_data_25_years = historical_data_25_years.fillna(0)

        current_year = datetime.datetime.now().year

        path_ = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            f"historical_forex_{current_year}.csv",
        )

        historical_data_25_years.to_csv(path_)

        return historical_data_25_years

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
