from typing import Dict , Union , Tuple
import datetime as dt
import pickle
import os 

import pandas as pd 
import numpy as np

from api_calls import ApiCaller

class DividendScoreCalculator:

    # Old companies , highly representative of what we except from a good dividend company
    benchmark_tickers = ("KO" , "JNJ" , "XOM" , "MMM" , "ITW" , "PM" , "IBM" , "ED" , "O" , "PG" , "EPD" , "BLK" , "VZ" , "NWN")

    PROFITABILITY_SCORE_WEIGHT = 1.5
    STABILITY_SCORE_WEIGHT = 1
    STRIKE_WEIGHT = 0.5

    BENCHMARK_FOLDER = os.path.join(os.path.dirname(__file__) , "benchmark_dividends_scores")

    def __init__(self , df_dividend:pd.DataFrame , df_price:pd.DataFrame , five_years_or_not:bool=False):
        self.df_dividend = df_dividend 
        self.df_price = df_price
        self.five_years_or_not = five_years_or_not

        year_to_remove = dt.datetime.now().year

        self.df_dividend = self.df_dividend.loc[self.df_dividend.index.year < year_to_remove]
        self.df_price = self.df_price.loc[self.df_price.index.year < year_to_remove]

    def main(self) -> tuple:

        scores_ticker = self.get_all_scores()
        scores_benchmark:Tuple[dict , dict] = self.get_benchmark()

        if not self.five_years_or_not:
            return scores_ticker , scores_benchmark[0]
        if self.five_years_or_not:
            return scores_ticker , scores_benchmark[1]
        
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
    def get_benchmark(cls) -> Tuple[dict , dict]:
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
        path_to_save_benchmark_scores_five_years = os.path.join(cls.BENCHMARK_FOLDER , current_year + "_five_years")

        if os.path.exists(path_to_save_benchmark_scores) and os.path.exists(path_to_save_benchmark_scores_five_years):
            if os.path.basename(path_to_save_benchmark_scores)[:4] == current_year:
                return _load_pickle_object(path_to_save_benchmark_scores) , _load_pickle_object(path_to_save_benchmark_scores_five_years)

        # The current year , that we don't want to use to avoid misscalculation for the dividends (only ended fiscal years)
        year_to_remove = dt.datetime.now().year 

        # To aggregate in mean
        global_scores = []
        profitability_scores = []
        stability_scores = []
        strikes = []

        global_scores_five_years = []
        profitability_scores_five_years = []
        stability_scores_five_years = []
        strikes_five_years = []

        ticker:str
        for ticker in DividendScoreCalculator.benchmark_tickers:

            df_dividend = ApiCaller().get_dividend(ticker=ticker)
            df_price = ApiCaller().get_price(ticker=ticker)

            minus_5_years = dt.timedelta(days=365*5)

            df_dividend_five_years = ApiCaller().get_dividend(ticker=ticker)
            df_price_five_years = ApiCaller().get_price(ticker=ticker)

            df_dividend_five_years = df_dividend_five_years[df_dividend_five_years.index[-1] - minus_5_years:]
            df_price_five_years = df_price_five_years[df_price_five_years.index[-1] - minus_5_years:]

            df_dividend = df_dividend.loc[df_dividend.index.year < year_to_remove]
            df_price = df_price.loc[df_price.index.year < year_to_remove]

            df_dividend_five_years = df_dividend_five_years.loc[df_dividend_five_years.index.year < year_to_remove]
            df_price_five_years = df_price_five_years.loc[df_price_five_years.index.year < year_to_remove]

            dividend_score_calculator = cls(df_dividend=df_dividend , df_price=df_price)
            dividend_score_calculator_five_years = cls(df_dividend=df_dividend_five_years , df_price=df_price_five_years)

            scores = dividend_score_calculator.get_all_scores()
            scores_five_years = dividend_score_calculator_five_years.get_all_scores()
            
            global_scores.append(scores["global_score"])
            profitability_scores.append(scores["profitability_score"])
            stability_scores.append(scores["stability_score"])
            strikes.append(scores["strike"])

            global_scores_five_years.append(scores_five_years["global_score"])
            profitability_scores_five_years.append(scores_five_years["profitability_score"])
            stability_scores_five_years.append(scores_five_years["stability_score"])
            strikes_five_years.append(scores_five_years["strike"])

        benchmark_scores = {
                        "strike": round(np.mean(strikes)),
                        "profitability_score": round(np.mean(profitability_scores),3),
                        "stability_score" : round(np.mean(stability_scores),3) , 
                        "global_score" : round(np.mean(global_scores),3)
                            }
        
        benchmark_scores_five_years = {
                        "strike": round(np.mean(strikes_five_years)),
                        "profitability_score": round(np.mean(profitability_scores_five_years),3),
                        "stability_score" : round(np.mean(stability_scores_five_years),3) , 
                        "global_score" : round(np.mean(global_scores_five_years),3)
                                      }

        _save_pickle_object(obj=benchmark_scores , file_path=path_to_save_benchmark_scores)
        _save_pickle_object(obj=benchmark_scores_five_years , file_path=path_to_save_benchmark_scores_five_years)

        return benchmark_scores , benchmark_scores_five_years
