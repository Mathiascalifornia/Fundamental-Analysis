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

class DividendGainCalculator:
    """ 
    Compute the gain you would have made by investing in this company n years ago , all dividend reinvested
    """
    
    # Old companies , highly representative of what we except from a good dividend company
    BENCHMARK_TICKERS = ("KO" , "JNJ" , "XOM" , "MMM" ,  "ITW" , "IBM" , "O" , "PG" , "EPD" , "BLK" , "VZ" , "NWN" , "HD" , "LEG")
    MINUS_YEARS_TO_COMPUTE = (3 , 5 , 10 , 15 , 20)
    BENCHMARK_FOLDER = "..\\data\\benchmark_backtesting_scores"

    pickle_loader = PickleLoaderAndSaviour()
    api_caller = ApiCaller()

    def __init__(self , df_price:pd.DataFrame , 
                        df_div:pd.DataFrame):
        
        self.df_price = df_price
        self.df_div = df_div




    def main(self) -> pd.DataFrame:
        
        results_ticker:pd.DataFrame = self.get_results()
        results_benchmark:pd.DataFrame = DividendGainCalculator.get_benchmark()

        results_ticker["Gains benchmark"] = results_benchmark["Gains benchmark"]

        results_ticker = results_ticker.dropna()
        results_ticker["Gains"] = results_ticker["Gains"].round().astype(int)
        
        return results_ticker


    @classmethod
    def get_benchmark(cls):

        path_pickle_results_current_year = os.path.join(DividendGainCalculator.BENCHMARK_FOLDER , str(datetime.datetime.now().year))

        if os.path.exists(path_pickle_results_current_year):
            return DividendGainCalculator.pickle_loader.load_pickle_object(path_pickle_results_current_year)
        
        else:

            results_by_year = {}

            ticker:str
            for ticker in DividendGainCalculator.BENCHMARK_TICKERS:

                df_price:pd.DataFrame = DividendGainCalculator.api_caller.get_price(ticker=ticker)
                df_div:pd.DataFrame = DividendGainCalculator.api_caller.get_dividend(ticker=ticker)

                result:pd.DataFrame = cls(df_div=df_div , 
                                          df_price=df_price).get_results()

                year:int
                for year in DividendGainCalculator.MINUS_YEARS_TO_COMPUTE:

                    results_by_year.setdefault(year , [])

                    gain_year = result[result["Years of investment"] == year]["Gains"].values[0]
                    results_by_year[year].append(gain_year)

            
            results_by_year = {year : round(np.median(results_by_year[year])) for year in results_by_year.keys()}
            df_results = pd.DataFrame(results_by_year.items() , columns=["Years of investment", "Gains benchmark"])

            DividendGainCalculator.pickle_loader.save_pickle_object(obj=df_results , file_path=path_pickle_results_current_year)

            return df_results # Same format as the results of the 'get_results' method , except the 'Gain' column name
        
            

    def get_results(self) -> pd.DataFrame:

        merged_df:pd.DataFrame
        merged_df = self.merge_price_and_div_df(df_price=self.df_price , 
                                                df_div=self.df_div)
        
        # Set the dataframe in the correct time span
        merged_df = self.cut_current_year_values(df=merged_df)

        return self.get_yearly_gains(merged_df=merged_df)


    @staticmethod
    def get_yearly_gains(merged_df:pd.DataFrame) -> pd.DataFrame:

        def __check_enought_time_horizon(merged_df:pd.DataFrame , minus_n_years:int) -> bool:
            """ 
            Check that we have enought data to make the computation for each time spans
            """
            return round((merged_df.index[-1] - merged_df.index[0]).days / 365.25) >= minus_n_years
        

        results = {}
        for year in DividendGainCalculator.MINUS_YEARS_TO_COMPUTE:

            if __check_enought_time_horizon(merged_df=merged_df , minus_n_years=year):

                right_range_merged_df = DividendGainCalculator.subset_over_minus_n_years(minus_n_years=year , 
                                                                                         df=merged_df)
                
                
                computed_df = DividendGainCalculator.compute_compound_interest(merged_df=right_range_merged_df)

                total_gains = computed_df.iloc[-1]["Capital"] - computed_df.iloc[0]["Capital"]
                results[year] = total_gains
            else:
                results[year] = np.nan

        return pd.DataFrame(results.items(), columns=["Years of investment", "Gains"])

   

    @staticmethod
    def subset_over_minus_n_years(minus_n_years:Optional[int] , 
                                  df:pd.DataFrame) -> pd.DataFrame:
        
        """ 
        Return the time serie dataframe minus n years
        """
        if minus_n_years is None:
            return df
        
        date_to_subset = df.index[-1]
        lower_boundry = date_to_subset - datetime.timedelta(days=minus_n_years*365)
        return df[str(lower_boundry):]


    @staticmethod
    def cut_current_year_values(df:pd.DataFrame) -> pd.DataFrame:
        """ 
        Make sure we don't take this year into account , to be able to compare with the benchmark
        """
        current_year = datetime.datetime.now().year
        first_jan_string = f"{current_year}-01-01"
        return df[:first_jan_string]


    @staticmethod
    def merge_price_and_div_df(df_price:pd.DataFrame , 
                               df_div:pd.DataFrame) -> pd.DataFrame:
        
        merged_df = df_div.merge(df_price , left_index=True , right_index=True , how="left")
        return merged_df.dropna() # .ffill().
    
    @staticmethod
    def compute_compound_interest(merged_df:pd.DataFrame , 
                                initial_capital:int=100) -> pd.DataFrame:
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

        merged_df.iloc[0 , merged_df.columns.get_loc("Capital")] = initial_capital # Set the initial capital
        merged_df.iloc[0 , merged_df.columns.get_loc("N shares")] = merged_df.iloc[0]["Capital"] / merged_df.iloc[0]["Close"] # Initial number of shares

        merged_df["pct_change_price"] = merged_df["Close"].pct_change()

        for i in range(1 , len(merged_df)):

            previous =  merged_df.iloc[i-1]
            current = merged_df.iloc[i]

            gain = previous["N shares"] * current["Dividends"] # Gain in dividends
            new_capital = previous["Capital"] + gain + (current["pct_change_price"] * previous["Close"])
            new_n_shares = previous["N shares"] + (gain / current["Close"]) 


            merged_df.iloc[i , merged_df.columns.get_loc("Capital")] = new_capital
            merged_df.iloc[i , merged_df.columns.get_loc("N shares")] = new_n_shares
        
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