import datetime as dt 

import pandas as pd 
import numpy as np



class FinanceComputationner:
    
    def annualized_return(self , df : pd.DataFrame):
        '''Get the mean annualized return'''
        df_ = df.copy()
        df_ = df_['Adj Close']
        years = pd.to_datetime(dt.date.today()).year - df_.index[0].year

        # Total returns
        total_return = (df_.iloc[-1] - df_.iloc[0]) / df_.iloc[0]

        # Annualized return
        return ((total_return + 1)**(1/years)) -1 , years



    def annualized_return_five_years(self , df : pd.DataFrame):
        '''Get the mean annualized return for the last five years'''
        df_ = df.copy()
        df_ = df_['Adj Close']
        new_index = df_.index[-1] - dt.timedelta(252*5)
        df_ = df_[new_index:]
        years = pd.to_datetime(dt.date.today()).year - df_.index[0].year

        # Total returns
        total_return = (df_.iloc[-1] - df_.iloc[0]) / df_.iloc[0]

        # Annualized return
        return ((total_return + 1)**(1/years)) -1 , years



    def sharpe_and_sortino_ratio(self , df : pd.DataFrame , rfr=0 , target=0):
        '''Return the sharpe and sortino ratio'''
        df_ = df.copy()
        df_ = df_['Adj Close']
        # Number of years
        years = pd.to_datetime(dt.date.today()).year - df_.index[0].year

        # Sharpe ratio
        total_return = (df_.iloc[-1] - df_.iloc[0]) / df_.iloc[0]
        annualized_return = ((1 + total_return) ** (1/years)) -1
        returns = df_.pct_change()
        vol = returns.std() * np.sqrt(252)
        sharpe = ((annualized_return - rfr) / vol)

        # Sortino ratio
        downside_return = returns[returns.values < target]
        mean_return= returns.mean()
        vol_down = downside_return.std()
        sortino = (mean_return-rfr) / vol_down

        return sharpe , sortino



    def sharpe_and_sortino_ratio_five_years(self , df : pd.DataFrame , rfr=0 , target=0):
            '''Returns the sharpe and sortino ratio for the last 5 years'''
            df_ = df.copy()
            df_ = df_['Adj Close']
            new_index = df_.index[-1] - dt.timedelta(252*5)
            df_ = df_[new_index:]

            # Number of years
            years = pd.to_datetime(dt.date.today()).year - df_.index[0].year

            # Sharpe ratio
            total_return = (df_.iloc[-1] - df_.iloc[0]) / df_.iloc[0]
            annualized_return = ((1 + total_return) ** (1/years)) -1
            returns = df_.pct_change()
            vol = returns.std() * np.sqrt(252)
            sharpe = ((annualized_return - rfr) / vol)

            # Sortino ratio
            downside_return = returns[returns.values < target]
            mean_return= returns.mean()
            vol_down = downside_return.std()
            sortino = (mean_return-rfr) / vol_down

            return sharpe , sortino
    