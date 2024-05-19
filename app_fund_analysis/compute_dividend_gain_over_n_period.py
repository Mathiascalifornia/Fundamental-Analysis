import datetime
from typing import Optional
import os

import pandas as pd
import numpy as np
import yfinance as yf

import warnings

warnings.filterwarnings("ignore")

yf.pdr_override()

from api_calls import ApiCaller
from pickle_loader import PickleLoaderAndSaviour
import config

# TODO ; Fixer l'evol dans laquelle ont essai d'adapter les autres currency au dollars


class DividendGainCalculator:
    """
    Compute the gain you would have made by investing in this company n years ago , all dividend reinvested
    """

    TICKER_SUFFIX_CURRENCY = config.ticker_suffix_to_currency

    # Old companies , highly representative of what we except from a good dividend company
    BENCHMARK_TICKERS = config.BENCHMARK_TICKERS
    MINUS_YEARS_TO_COMPUTE = (3, 5, 10, 15, 20)
    BENCHMARK_FOLDER = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "data", "benchmark_backtesting_scores"
        )
    )

    PATH_FOREX_CSV = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            f"data",
            "historical_forex_{}.csv".format(datetime.datetime.now().year),
        )
    )

    pickle_loader = PickleLoaderAndSaviour()
    api_caller = ApiCaller()

    def __init__(self, df_price: pd.DataFrame, df_div: pd.DataFrame, ticker: str):

        self.df_price = df_price
        self.df_div = df_div
        self.ticker = ticker

    def main(self) -> pd.DataFrame:

        if os.path.exists(DividendGainCalculator.PATH_FOREX_CSV):
            self.forex_df: pd.DataFrame = pd.read_csv(
                DividendGainCalculator.PATH_FOREX_CSV,
                index_col="Date",
                parse_dates=["Date"],
            )

        else:
            self.forex_df: pd.DataFrame = (
                DividendGainCalculator.api_caller.create_forex_exchange_df()
            )

        results_ticker: pd.DataFrame = self.get_results()

        results_ticker = results_ticker[
            ["Years of investment", "P&L", "Dividends Gains"]
        ]

        results_benchmark: pd.DataFrame = DividendGainCalculator.get_benchmark()
        results_benchmark = results_benchmark[
            ["Years of investment", "P&L benchmark", "Dividends Gains benchmark"]
        ]

        results_ticker["P&L benchmark"] = results_benchmark["P&L benchmark"]
        results_ticker["Dividends Gains benchmark"] = results_benchmark[
            "Dividends Gains benchmark"
        ]

        results_ticker = results_ticker.dropna()
        results_ticker = results_ticker.astype(int)

        return results_ticker

    @staticmethod
    def detect_currency(ticker: str, dict_equivalence: dict) -> str:

        """
        To adapt the currency to the computations
        """
        if "." in ticker:
            exchange = ticker.split(".")[-1]
            return dict_equivalence.get(exchange, None)

        return "USD"

    @classmethod
    def get_benchmark(cls):

        path_pickle_results_current_year = os.path.join(
            DividendGainCalculator.BENCHMARK_FOLDER, str(datetime.datetime.now().year)
        )

        if os.path.exists(path_pickle_results_current_year):
            return DividendGainCalculator.pickle_loader.load_pickle_object(
                path_pickle_results_current_year
            )

        else:

            results_by_year = {}

            ticker: str
            for ticker in DividendGainCalculator.BENCHMARK_TICKERS:

                df_price: pd.DataFrame = DividendGainCalculator.api_caller.get_price(
                    ticker=ticker
                )
                df_div: pd.DataFrame = DividendGainCalculator.api_caller.get_dividend(
                    ticker=ticker
                )

                result: pd.DataFrame = cls(
                    df_div=df_div, df_price=df_price
                ).get_results()

                year: int
                for year in DividendGainCalculator.MINUS_YEARS_TO_COMPUTE:

                    results_by_year.setdefault(year, {"P&L": [], "Dividends Gains": []})

                    gain_year = result[result["Years of investment"] == year][
                        "P&L"
                    ].values[0]

                    dividend_gains = result[result["Years of investment"] == year][
                        "Dividends Gains"
                    ].iloc[-1]

                    results_by_year[year]["P&L"].append(gain_year)
                    results_by_year[year]["Dividends Gains"].append(dividend_gains)

            results_by_year = {
                year: {
                    "P&L benchmark": round(np.median(results_by_year[year]["P&L"])),
                    "Dividends Gains benchmark": round(
                        np.median(results_by_year[year]["Dividends Gains"])
                    ),
                }
                for year in results_by_year.keys()
            }

            # df_results = pd.DataFrame(results_by_year.items() , columns=["Years of investment", "P&L benchmark"])

            final_result_df = pd.DataFrame.from_dict(results_by_year, orient="index")
            final_result_df["Years of investment"] = final_result_df.index
            df_results = final_result_df.reset_index(drop=True)

            DividendGainCalculator.pickle_loader.save_pickle_object(
                obj=df_results, file_path=path_pickle_results_current_year
            )

            return df_results  # Same format as the results of the 'get_results' method , except the 'P&L' column name

    def get_results(self) -> pd.DataFrame:

        merged_df: pd.DataFrame = self.merge_price_and_div_df(
            df_price=self.df_price, df_div=self.df_div
        )

        # Set the dataframe in the correct time span
        merged_df = self.cut_current_year_values(df=merged_df)

        currency = self.detect_currency(
            ticker=self.ticker, dict_equivalence=self.TICKER_SUFFIX_CURRENCY
        )
        if not currency:
            raise ValueError(
                "Some bad values are present for the forex rates of this currency. Skipping the simulation"
            )

        if currency != "USD":

            # QUICK FIX , to remove
            raise ValueError()
            merged_df = pd.merge(
                merged_df, self.forex_df, left_index=True, right_index=True, how="inner"
            )

            merged_df = merged_df[pd.Timestamp("2004-08-18") :]
            merged_df.to_csv(
                r"C:\Users\User\Desktop\Finance softwares\fundamental_analysis\Fundamental-Analysis\test\merged_df.csv"
            )

            unique_values_curr = list(merged_df[currency])

            if any(not val for val in unique_values_curr) or merged_df.empty:
                raise ValueError(
                    "Some bad values are present for the forex rates of this currency. Skipping the simulation"
                )

            merged_df["Close"] = merged_df["Close"] * merged_df[currency]
            merged_df["Dividends"] = merged_df["Dividends"] * merged_df[currency]

        return self.get_yearly_gains(merged_df=merged_df)

    @staticmethod
    def get_yearly_gains(merged_df: pd.DataFrame) -> pd.DataFrame:
        def __check_enought_time_horizon(
            merged_df: pd.DataFrame, minus_n_years: int
        ) -> bool:
            """
            Check that we have enought data to make the computation for each time spans
            """
            return (
                round((merged_df.index[-1] - merged_df.index[0]).days / 252.25)
                >= minus_n_years
            )

        results = {}
        for year in DividendGainCalculator.MINUS_YEARS_TO_COMPUTE:

            if __check_enought_time_horizon(merged_df=merged_df, minus_n_years=year):

                right_range_merged_df = (
                    DividendGainCalculator.subset_over_minus_n_years(
                        minus_n_years=year, df=merged_df
                    )
                )

                computed_df = DividendGainCalculator.compute_compound_interest(
                    merged_df=right_range_merged_df
                )

                total_gains = (
                    computed_df.iloc[-1]["Capital"] - computed_df.iloc[0]["Capital"]
                )
                dividends_gains = computed_df.iloc[-1]["Dividends Gains"]
                results[year] = {"P&L": total_gains, "Dividends Gains": dividends_gains}
            else:
                results[year] = {"P&L": np.nan, "Dividends Gains": np.nan}

        # return pd.DataFrame(results.items(), columns=["Years of investment", "P&L"])

        final_result_df = pd.DataFrame.from_dict(results, orient="index")
        final_result_df["Years of investment"] = final_result_df.index
        return final_result_df.reset_index(drop=True)

    @staticmethod
    def subset_over_minus_n_years(
        minus_n_years: Optional[int], df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return the time serie dataframe minus n years
        """
        if minus_n_years is None:
            return df

        date_to_subset = df.index[-1]
        lower_boundry = date_to_subset - datetime.timedelta(days=minus_n_years * 365)
        return df[str(lower_boundry) :]

    @staticmethod
    def cut_current_year_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Make sure we don't take this year into account , to be able to compare with the benchmark
        """
        current_year = datetime.datetime.now().year
        first_jan_string = f"{current_year}-01-01"
        return df[:first_jan_string]

    @staticmethod
    def merge_price_and_div_df(
        df_price: pd.DataFrame, df_div: pd.DataFrame
    ) -> pd.DataFrame:

        merged_df = df_div.merge(
            df_price, left_index=True, right_index=True, how="left"
        )
        return merged_df.dropna()  # .ffill().

    @staticmethod
    def compute_compound_interest(
        merged_df: pd.DataFrame, initial_capital: int = 100
    ) -> pd.DataFrame:
        """
        Compute compound interest for a dividend-paying stock investment.

        Args:
            merged_df (pd.DataFrame): DataFrame with stock data including "Adj Close" and "Dividends" columns.
            initial_capital (int, optional): Initial investment capital. Defaults to 100.

        Returns:
            pd.DataFrame: DataFrame with additional "Capital" and "N shares" columns representing capital and shares over time.
        """
        # Initialize empty columns to dynamically update them
        merged_df["Capital"] = 0
        merged_df["N shares"] = 0
        merged_df["Dividends Gains"] = 0

        merged_df.iloc[
            0, merged_df.columns.get_loc("Capital")
        ] = initial_capital  # Set the initial capital
        merged_df.iloc[0, merged_df.columns.get_loc("N shares")] = (
            merged_df.iloc[0]["Capital"] / merged_df.iloc[0]["Close"]
        )  # Initial number of shares

        merged_df["pct_change_price"] = merged_df["Close"].pct_change()

        for i in range(1, len(merged_df)):

            previous = merged_df.iloc[i - 1]
            current = merged_df.iloc[i]

            gain = previous["N shares"] * current["Dividends"]  # Gain in dividends
            new_capital = (
                previous["Capital"]
                + gain
                + (current["pct_change_price"] * previous["Close"])
            )
            new_n_shares = previous["N shares"] + (gain / current["Close"])
            new_gain_in_dividends = previous["Dividends Gains"] + gain

            merged_df.iloc[i, merged_df.columns.get_loc("Capital")] = new_capital
            merged_df.iloc[i, merged_df.columns.get_loc("N shares")] = new_n_shares
            merged_df.iloc[
                i, merged_df.columns.get_loc("Dividends Gains")
            ] = new_gain_in_dividends

        return merged_df

    # sp500_price:Optional[pd.DataFrame]=None): # Optionnal because we don't use it in the benchmark computation

    # @staticmethod
    # def get_buy_and_hold_sp500(sp500_price:pd.DataFrame) -> pd.DataFrame:

    #     def __compound_interest_sp500(sp500_price_:pd.DataFrame) -> pd.DataFrame:

    #         sp500_price_["pct_change_"] = sp500_price_["Close"].pct_change()
    #         sp500_price_["Capital"] = 0

    #         sp500_price_.iloc[0 , sp500_price_.columns.get_loc("Capital")] = 100

    #         for i in range(1 , len(sp500_price_)):

    #             current_pct_change = sp500_price_.iloc[i]["pct_change_"]
    #             previous_capital = sp500_price_.iloc[i-1]["Capital"]

    #             sp500_price_.iloc[i , sp500_price_.columns.get_loc("Capital")] = previous_capital + (previous_capital * current_pct_change)

    #         return sp500_price_

    #     # Start with current_year - 1
    #     sp500_price_ = DividendGainCalculator.cut_current_year_values(df=sp500_price)

    #     results_by_year = {}
    #     year:int
    #     for year in DividendGainCalculator.MINUS_YEARS_TO_COMPUTE:
    #         right_time_span_sp500 = DividendGainCalculator.subset_over_minus_n_years(minus_n_years=year ,
    #                                                                                  df=sp500_price_)

    #         sp500_computed = __compound_interest_sp500(right_time_span_sp500)

    #         gains = sp500_computed.iloc[-1]["Capital"] - sp500_computed.iloc[0]["Capital"]

    #         results_by_year[year] = round(gains)

    #     return pd.DataFrame(results_by_year.items() , columns=["Years of investment", "Gains buy and hold SP500"])
