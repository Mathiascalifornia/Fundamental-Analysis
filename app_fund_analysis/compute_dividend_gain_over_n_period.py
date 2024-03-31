import datetime
from typing import Optional , Any
import pickle
import os

import pandas as pd 
import numpy as np
import yfinance as yf
from pandas_datareader import data

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
    BENCHMARK_TICKERS = ("KO" , "JNJ" , "XOM" , "MMM" , "ITW" , "PM" , "IBM" , "ED" , "O" , "PG" , "EPD" , "BLK" , "VZ" , "NWN")
    MINUS_YEARS_TO_COMPUTE = (3 , 5 , 10 , 15 , 20)
    BENCHMARK_FOLDER = "..\\data\\benchmark_backtesting_scores"

    pickle_loader = PickleLoaderAndSaviour()
    api_caller = ApiCaller()

    def __init__(self , df_price:pd.DataFrame , 
                        df_div:pd.DataFrame):
        
        self.df_price = df_price
        self.df_div = df_div


    @classmethod
    def compute_benchmark(cls):

        path_pickle_results_current_year = os.path.join(DividendGainCalculator.BENCHMARK_FOLDER , str(datetime.datetime.now().year))

        if os.path.exists(path_pickle_results_current_year):
            return DividendGainCalculator.pickle_loader.load_pickle_object(path_pickle_results_current_year)
        
        else:

            results_by_year = {}

            ticker:str
            for ticker in DividendGainCalculator.BENCHMARK_TICKERS:

                df_price:pd.DataFrame = DividendGainCalculator.api_caller.get_price(ticker=ticker)
                df_div:pd.DataFrame = DividendGainCalculator.api_caller.get_dividend(ticker=ticker)

                result:pd.DataFrame = cls(df_div=df_div , df_price=df_price).main()
                
                year:int
                for year in DividendGainCalculator.MINUS_YEARS_TO_COMPUTE:

                    results_by_year.setdefault(year , [])

                    gain_year = result[result["Years of investment"] == year]["Gains"].values[0]
                    results_by_year[year].append(gain_year)

            
            results_by_year = {year : round(np.mean(results_by_year[year])) for year in results_by_year.keys()}
            df_results = pd.DataFrame(results_by_year.items() , columns=["Years of investment", "Gains benchmark"])

            DividendGainCalculator.pickle_loader.save_pickle_object(obj=df_results , file_path=path_pickle_results_current_year)

            return df_results
    

        
            

    def main(self) -> pd.DataFrame:

        merged_df:pd.DataFrame
        merged_df = self.merge_price_and_div_df(df_price=self.df_price , 
                                                df_div=self.df_div)
        
        # Set the dataframe in the correct time span
        merged_df = self.cut_current_year_values(merged_df=merged_df)

        computed_df = self.compute_compound_interest(merged_df=merged_df)

        return self.get_yearly_gains(computed_df=computed_df)


    @staticmethod
    def get_yearly_gains(computed_df:pd.DataFrame) -> pd.DataFrame:

        def __check_enought_time_horizon(computed_df:pd.DataFrame , minus_n_years:int) -> bool:
            return round((computed_df.index[-1] - computed_df.index[0]).days / 365.25) >= minus_n_years
        
        results = {}
        for year in DividendGainCalculator.MINUS_YEARS_TO_COMPUTE:

            if __check_enought_time_horizon(computed_df=computed_df , minus_n_years=year):
                minus_n_df = DividendGainCalculator.subset_over_minus_n_years(minus_n_years=year , 
                                                                            computed_df=computed_df)
                
                
                total_gains = minus_n_df.iloc[-1]["Capital"] - minus_n_df.iloc[0]["Capital"]
                results[year] = total_gains
            else:
                results[year] = np.nan

        return pd.DataFrame(results.items(), columns=["Years of investment", "Gains"])


        

    
    @staticmethod
    def subset_over_minus_n_years(minus_n_years:Optional[int] , 
                                  computed_df:pd.DataFrame) -> pd.DataFrame:
        
        """ 
        Return the merged dataframe minus n years
        """
        if minus_n_years is None:
            return computed_df
        
        date_to_subset = computed_df.index[-1]
        lower_boundry = date_to_subset - datetime.timedelta(days=minus_n_years*365)
        return computed_df[str(lower_boundry):]

    @staticmethod
    def cut_current_year_values(merged_df:pd.DataFrame) -> pd.DataFrame:
        """ 
        Make sure we don't take this year into account , to be able to compare with the benchmark
        """
        current_year = datetime.datetime.now().year
        first_jan_string = f"{current_year}-01-01"
        return merged_df[:first_jan_string]


    @staticmethod
    def merge_price_and_div_df(df_price:pd.DataFrame , 
                            df_div:pd.DataFrame) -> pd.DataFrame:
        
        merged_df = df_div.merge(df_price , left_index=True , right_index=True , how="left")
        return merged_df.ffill().dropna()
    
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
        merged_df.iloc[0 , merged_df.columns.get_loc("N shares")] = merged_df.iloc[0]["Capital"] / merged_df.iloc[0]["Adj Close"] # Initial number of shares

        for i in range(1 , len(merged_df)):

            previous =  merged_df.iloc[i-1]
            current = merged_df.iloc[i]

            gain = previous["N shares"] * current["Dividends"]
            new_capital = previous["Capital"] + gain 
            new_n_shares = previous["N shares"] + (gain / current["Adj Close"]) 


            merged_df.iloc[i , merged_df.columns.get_loc("Capital")] = new_capital
            merged_df.iloc[i , merged_df.columns.get_loc("N shares")] = new_n_shares
        
        return merged_df