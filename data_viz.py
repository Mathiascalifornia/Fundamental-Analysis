# Data manipulation
import pandas as pd 
import numpy as np
import re 
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

import talib as ta


# Data visualisation
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns 


import yfinance as yf
from pandas_datareader import data
from pandas_datareader import DataReader
yf.pdr_override()

from pres import PresPPT



class DataViz(PresPPT):

    def __init__(self , **kwargs): 

        self.__dict__.update(kwargs)
        self.limit_year = int(int(dt.datetime.today().year) - 1)
        super().__init__() # For the class to use the PresPPT objects

    def plot_element(self, df : pd.DataFrame):
        '''Function to plot the elements of the indicators dataframes over time'''
        cleaning = lambda x : str(x).replace('x' , '').replace('%' , '')
        for col in df.columns:
            new_col = []
            for val in df[col].apply(cleaning):
                    new_col.append(float(val))

            df[col] = new_col
        
        with sns.plotting_context('talk'):

            sns.set_style('darkgrid')
            index = list(df.index)
            for i in range(len(index)):
                plt.figure(figsize=(8,5))
                d = df.iloc[i]
                if (d.isna().sum() / len(d)) < 0.5: # if there is less than 50 per cent of nulls
                    d = d.fillna(method='pad')
                    d.index = pd.to_datetime(d.index).year
                    preds = d[d.index > self.limit_year]
                    plt.plot(d , color='red')
                    plt.plot(preds , linestyle='--' , color='white' , label='Predictions')
                    plt.legend()
                    plt.savefig('data\\fig{i}.png')
                    plt.close('all')
                    if self.english:
                        self.add_picture('data\\fig{i}.png' , self.t(index[i]) , left=1.1 , top=1.9)
                    else:
                        self.add_picture('data\\fig{i}.png' , index[i] , left=1.1 , top=1.9)


    def plot_sentiment_score(self , scores):
        with sns.plotting_context('talk'):
            sns.set_style('darkgrid')
            plt.figure(figsize=(8,5))
            sns.countplot(x='score' , data=scores , palette={'Positive headline' : 'green' , 'Negative headline' : 'red'} , edgecolor='black')
            plt.xlabel('')
            plt.savefig('data\\sentiment.png')
            plt.close('all')
            if self.english == False:
                self.add_picture('data\\sentiment.png' , 'Analyse de sentiment du marché, basée sur les titres d\'actualités.')
            else:
                self.add_picture('data\\sentiment.png' , self.t('Analyse de sentiment du marché , basée sur les titres d\'actualités.'))



    def plot_regression(self , df : pd.DataFrame):
        '''Plot the stock price with a linear regression , to see the trend'''
        
        df_ = df.copy()
        df_.drop(['High', 'Low', 'Open', 'Close', 'Volume'] , axis=1 , inplace=True)
        df_['day_from_start'] = (df_.index - df_.index[-1]).days
        X = df_['day_from_start'].values
        y = df_.drop('day_from_start' , axis=1).values
        a , b = np.polyfit(X , y , deg=1)
        x_line = np.array([np.min(X) , np.max(X)]) # Just the lenght of the regression line
        y_line = a*x_line + b

        with sns.plotting_context('notebook'):
            
            sns.set_style('darkgrid')
            plt.figure(figsize=(8,5))
            plt.plot(X , y ,color='blue')
            plt.plot(x_line , y_line , color='red')
            plt.savefig('data\\linear_regression.png')
            plt.close('all')
            if self.english:
                self.add_picture('data\\linear_regression.png' , 'Linear regression')
            else:
                self.add_picture('data\\linear_regression.png' , 'Régression linéaire')


    def plot_maximum_draw_down(self , df : pd.DataFrame):
            ''' Plot the maximum drow down , which is a measure of an asset's largest price drop from a peak to a trough'''

            _df_ = df.copy()
            _df_.drop(['High', 'Low', 'Open', 'Close', 'Volume'] , axis=1 , inplace=True , errors='ignore')
            # Calculate the max value 
            roll_max = _df_.rolling(center=False,min_periods=1,window=252).max()

            # Calculate the daily draw-down relative to the max
            daily_draw_down = _df_/roll_max - 1.0

            # Calculate the minimum (negative) daily draw-down
            max_daily_draw_down = daily_draw_down.rolling(center=False,min_periods=1, window=252).min()

            # Plot the results
            with sns.plotting_context('notebook'):

                sns.set_style('darkgrid')
                plt.figure(figsize=(8,5))
                plt.plot(daily_draw_down.index, daily_draw_down, label='Daily drawdown')
                plt.plot(max_daily_draw_down.index, max_daily_draw_down, label='Maximum daily drawdown in time-window' , color='red')
                plt.savefig('data\\maximum_draw_down.png')
                plt.legend()
                plt.close('all')
                if self.english:
                    self.add_picture('data\\maximum_draw_down.png' , 'Maximum loss')
                else:
                    self.add_picture('data\\maximum_draw_down.png' , 'Perte de valeur maximum')


    def price_with_dividends(self , df_price , df_dividend_):
            '''Plot the stock price with the dividends payed over time'''
            
            __div = df_dividend_.copy()
            df_ = df_price.copy()
            df_ = df_.drop(['High', 'Low', 'Open', 'Close', 'Volume'] , axis=1 , errors='ignore')
            with sns.plotting_context('notebook'):
                sns.set_style('dark')
                fig , ax = plt.subplots(figsize=(8,5))
                ax.plot(df_.index , df_.values , label='Stock price')
                plt.legend(loc='upper left')
                ax.twinx().plot(__div.index , __div.values , color='red' , label='Dividend')
                plt.legend(loc='center left')
                plt.savefig('data\\price_with_dividends.png')
                plt.close('all')
                if self.english:
                    self.add_picture('data\\price_with_dividends.png' , 'Price and dividends')
                else:
                    self.add_picture('data\\price_with_dividends.png' , 'Prix et dividendes')


    def annual_dividend_history(self):
        '''Group the dividend per year to see the annual dividend history'''

        self.df_dividend['year'] = self.df_dividend.index.year
        d__ = self.df_dividend.groupby('year').sum()
        d__['year'] = d__.index

        with sns.plotting_context('notebook'):
            sns.set_style('darkgrid')
            plt.figure(figsize=(8,5))
            sns.barplot(x='year' , y='Dividends' , data=d__ , edgecolor='black')
            plt.ylabel('Montant')
            plt.xlabel('Année')
            plt.xticks(rotation = 90)
            plt.savefig('data\\annual_dividend.png')
            plt.close('all')
            if self.english:
                self.add_picture('data\\annual_dividend.png' , 'Dividend per year and per share')
            else:
                self.add_picture('data\\annual_dividend.png' , 'Dividende versé par année et par action')


    def payout_ratio(self , df2):   
        '''Function to compute and plot the payout ratio. Uses the second dataframe (table_1)'''

        self.df_dividend['year'] = list(self.df_dividend.index.year.astype(str))
        d_ = self.df_dividend.groupby('year').sum()
        d_['year'] = d_.index
        years = [val for val in df2 if val in d_.index]
        df2.index = [str(val).strip() for val in df2.index]
        bna_years = dict(df2[years].loc['BNA'])
        d_ = d_[d_['year'].isin(years)].drop('year' , axis=1 , errors='ignore')
        d_['BNA'] = list(bna_years.values())
        d_['payout'] = d_['Dividends'] / d_['BNA'] # * 100

        with sns.plotting_context('talk'):
            sns.set_style('darkgrid')
            plt.figure(figsize=(8,5))
            plt.plot(d_.index , d_['payout'].apply(lambda x : round(x*100)) , color='red')
            plt.savefig('data\\payout.png')
            plt.close('all')
            if self.english:
                self.add_picture('data\\payout.png' , 'Pay out ratio')
            else:
                self.add_picture('data\\payout.png' , 'Taux de distribution')



    def plot_against_benmark(self):
            '''Plot the normalized data against a benchmark (the sp500)'''

            df_ = self.df_price.copy()
            bench_ = self.sp500_price.copy()
            df_['normalized'] = MinMaxScaler().fit_transform(df_['Adj Close'].values.reshape(-1,1))   
            bench_['normalized'] = MinMaxScaler().fit_transform(bench_['Adj Close'].values.reshape(-1,1))   

            with sns.plotting_context('notebook'):
                sns.set_style('darkgrid')
                plt.figure(figsize=(8,5))
                plt.plot(df_.index , df_['normalized'] , color='blue' , label=self.ticker)
                plt.plot(bench_.index , bench_['normalized'] , label='SP500' , color='red')
                plt.legend()
                plt.savefig('data\\bench.png')
                plt.close('all')
                if self.english:
                    self.add_picture('data\\bench.png' , f"Normalized stocks price :{self.ticker} vs SP500")
                else:
                    self.add_picture('data\\bench.png' , f"Prix de l'action normalisé {self.ticker} vs SP500")


    def cap_vs_debt(self , table_3):
        '''Plot the proper capital versus the debt. Uses the debt df (table_3)'''

        debt_df = table_3
        debt_df.index = [str(val).lstrip().rstrip() for val in debt_df.index]

        dette = debt_df.loc[['Dette Nette']].T 
        dette = dette.rename(columns={dette.columns[0]:"debt"}).assign(year = dette.index).reset_index(drop=True)

        capitaux = debt_df.loc[["Capitaux Propres"]].T
        capitaux = capitaux.rename(columns={capitaux.columns[0]:"cap"}).assign(year = capitaux.index).reset_index(drop=True)

        new_df = dette.merge(capitaux ,on='year', how='inner')
  
        if new_df['debt'].isna().sum() <= 2 and new_df['cap'].isna().sum() <=2:

            new_df['debt'].fillna(method='pad' , inplace=True)
            new_df['cap'].fillna(method='pad' , inplace=True)
            new_df['debt'] = new_df['debt'].apply(lambda x : float(x))
            new_df['cap'] = new_df['cap'].apply(lambda x : float(x))
            new_df['year'] = new_df['year'].apply(lambda x : int(x))
            prediction = new_df[new_df['year'] > self.limit_year]
            
            with sns.plotting_context('talk'):

                sns.set_style('darkgrid')
                plt.figure(figsize=(8,5))
                if self.english == False:
                    plt.plot(new_df['year'] , new_df['debt'] , color='red' , label='Dette')
                    plt.plot(new_df['year'] , new_df['cap'] , color='blue' , label='Capitaux propres')
                    plt.plot(prediction['year'] , prediction['debt'] , linestyle='--' , color='white' , label='prediction')
                    plt.plot(prediction['year'] , prediction['cap'] , linestyle='--' , color='white')
                    plt.legend()
                    plt.savefig('data\\cap_versus_debt.png')
                    plt.close('all')
                    self.add_picture('data\\cap_versus_debt.png' , 'Capitaux propres vs Dette net')
                else:
                    plt.plot(new_df['year'] , new_df['debt'] , color='red' , label='Debt')
                    plt.plot(new_df['year'] , new_df['cap'] , color='blue' , label='Equity')
                    plt.plot(prediction['year'] , prediction['debt'] , linestyle='--' , color='white' , label='prediction')
                    plt.plot(prediction['year'] , prediction['cap'] , linestyle='--' , color='white')
                    plt.legend()
                    plt.savefig('data\\cap_versus_debt.png')
                    plt.close('all')
                    self.add_picture('data\\cap_versus_debt.png' , 'Equity versus debt')


    def shareholders(self):
        '''Plot a pie plot of the three biggest institutions shareholders'''
        data_ = yf.Ticker(str(self.ticker).upper())

        d_f = data_.major_holders.iloc[:-2]
        particuliers = 1 - d_f.sum().values[0]
        
        color = ['red' , 'lightgreen' , 'gold']
        explode = [0.12 , 0.02 , 0.05]

        values_to_plot = d_f["Value"].values

        color = ['red' , 'lightgreen' , 'gold']
        explode = [0.12 , 0.02 , 0.05]

        if not self.english:
            labels = [f'Initiés : {round(values_to_plot[0] , 3)}%' , f'Institutions:{round(values_to_plot[1] , 3)} %' , f'Autre : {round(particuliers , 3)}%']
        else:
            labels = [f'Initiates : {round(values_to_plot[0] , 3)}%' , f'Institutions:{round(values_to_plot[1] , 3)} %' , f'Other : {round(particuliers , 3)}%']


        plt.figure(figsize=(5,5))
        plt.pie([values_to_plot[0] , values_to_plot[1] , particuliers] , colors=color , explode=explode)
        plt.legend(labels)
        plt.savefig('data\\shareholders.png')
        plt.close('all')

        if not self.english:
            self.add_picture('data\\shareholders.png' , 'Repartition des investisseurs' , left=2.4 , top=1.5)
        else:
            self.add_picture('data\\shareholders.png' , 'Investor distribution' , left=2.4 , top=1.5)



    def plot_rsi(self):
            '''Plot the Relative Strenght Index'''
            start = dt.datetime.now() - dt.timedelta(800)
            self.df_price['RSI'] = ta.RSI(self.df_price['Adj Close'])
            plt.figure(figsize=(10,3))
            plt.plot(self.df_price['RSI'][start:])
            plt.axhline(y=70 , color='red' , linestyle='--')
            plt.axhline(y=30 , color='green' , linestyle='--')
            plt.savefig('data\\rsi.png')
            plt.close('all')
            self.add_picture('data\\rsi.png' , 'Relative Strength Index (RSI)' , left=0 , top=2.5)


    def plot_zoom_candles(self):
            '''Plot a zoom of the last six month , with candles and volume over time'''
            start = dt.datetime.now() - dt.timedelta(175)
            self.df_price['CDL'] = ta.CDLENGULFING(self.df_price['Open'] , self.df_price['High'] , self.df_price['Low'] , self.df_price['Close'])
            colors = mpf.make_marketcolors(up='green' , down='red')
            style = mpf.make_mpf_style(base_mpf_style='yahoo' , marketcolors=colors)
            ax = mpf.plot(self.df_price[start:] , type='candle' , style=style , figsize=(9,6) , volume=True , savefig='data\\Zoom.png')
            if self.english == False:
                self.add_picture('data\\Zoom.png' , 'Zoom des six derniers mois' , left=0.05 , top=1.5)
            else:
                self.add_picture('data\\Zoom.png' , 'Last six months zoom' , left=0.05 , top=1.5)



    def plot_multiple_indicators(self  , df1 : pd.DataFrame , df2 : pd.DataFrame , quaterly=False):
            '''Plot the turnover highlighted with the net result , EBIT , and the margins. 
               Uses table_0 , table_1 and table_2'''
            
           
            df1_ = df1.copy()
            df2_ = df2.copy()
            df2_.index = [col.strip() for col in df2.index]
            df1_.index = [col.strip() for col in df1.index]

            index_ = np.array([int(re.sub(string=year , repl='' , pattern='Q\\d').strip()) for year in df1_.columns])

            if not self.english:
                vals_ = {'Chiffre d\'affaires' : df2_.loc["Chiffre d'affaires"].values ,
                'EBIT' : df2_.loc["Résultat d'exploitation EBIT"].values,
                'Résultat net' : df2_.loc["Résultat net"].values }

                try:
                    marge = {
                        'ME': pd.to_numeric(df2_.loc["Marge d'exploitation"].str.replace('%', ''), errors='coerce').values,
                        'MN': pd.to_numeric(df2_.loc["Marge nette"].str.replace('%', ''), errors='coerce').values
                    }
                except:
                    marge = {
                        'ME': pd.to_numeric(df2_.loc["Marge d'exploitation"], errors='coerce').values,
                        'MN': pd.to_numeric(df2_.loc["Marge nette"], errors='coerce').values
    }

            if self.english:
                vals_ = {'Turnover' : df2_.loc["Chiffre d'affaires"].values ,
                'EBIT' : df2_.loc["Résultat d'exploitation EBIT"].values,
                'Net profit' : df2_.loc["Résultat net"].values }     

                try:
                    marge = {
                        'ME': pd.to_numeric(df2_.loc["Marge d'exploitation"].str.replace('%', ''), errors='coerce').values,
                        'MN': pd.to_numeric(df2_.loc["Marge nette"].str.replace('%', ''), errors='coerce').values
                    }
                except:
                    marge = {
                        'ME': pd.to_numeric(df2_.loc["Marge d'exploitation"], errors='coerce').values,
                        'MN': pd.to_numeric(df2_.loc["Marge nette"], errors='coerce').values
                    }
                    


            date_limit = int(dt.datetime.now().year) - 1
            d = dict(zip(index_ , np.arange(len(index_))))

            x = np.arange(len(index_))
            width = 0.15  # the width of the bars
            multiplier = 0

            with sns.plotting_context('notebook'):
                sns.set_style('dark')
                fig, ax = plt.subplots(layout='constrained' , figsize=(9,5))
                for attribute, measurement in vals_.items():
                    offset = width * multiplier
                    rects = ax.bar(x + offset, [float(val) for val in measurement], width, label=attribute , edgecolor='black' , alpha=1)
                    multiplier += 1


                plt.legend(loc='upper right' , edgecolor='black')
                ax2 = ax.twinx()
                ax2.set_ylim(bottom=min([*marge['MN'] , *marge['ME']])-5 , top=max([*marge['MN'] , *marge['ME']])+5)
                if self.english == False:
                    sns.lineplot(ax=ax2 , x=x , y=marge['MN'] , marker='o' , color='green' , label="Marge nette" , alpha=0.7)
                    sns.lineplot(ax=ax2 , x=x , y=marge['ME'] , marker='o' , color='orange' , label="Marge d'exploitation" , alpha=0.7)
                else:
                    sns.lineplot(ax=ax2 , x=x , y=marge['MN'] , marker='o' , color='green' , label="Net margin" , alpha=0.7)
                    sns.lineplot(ax=ax2 , x=x , y=marge['ME'] , marker='o' , color='orange' , label="Operating margin" , alpha=0.7) 

                ax.set_xticks(x + width, index_)
                ax.axvspan(d[date_limit]+0.85, list(d.values())[-1]+0.65, alpha=0.2, color='gold' , label='Prédiction')
                ax.legend(bbox_to_anchor=(1.1, 0.75), loc='upper left', borderaxespad=0 , edgecolor='black')
                plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0, edgecolor='black')

                plt.savefig('data\\plot_multiple_indicator.png')
                plt.close('all')

                if self.english == False:
                    if not quaterly:
                        self.add_picture('data\\plot_multiple_indicator.png' , 'Évolution du compte de résultat annuel' , left=0.8)
                    if quaterly:
                        self.add_picture('data\\plot_multiple_indicator.png' , 'Évolution du compte de résultat trimestrielle' , left=0.8)
                else:
                    if not quaterly:
                        self.add_picture('data\\plot_multiple_indicator.png' , 'Evolution of the annual income statement' , left=0.8)
                    if quaterly:
                        self.add_picture('data\\plot_multiple_indicator.png' , 'Evolution of the quarterly income statement' , left=0.8)



    def plot_correlation(self):
            ''' Plot spearman correlation between indices and various indicators '''

            def __correlation_merge_on_date(df1 : pd.DataFrame , df2 : pd.DataFrame , column_ : str) -> float :
                df1_ = df1.copy()
                df2_ = df2.copy()
                df1_['date'] = df1_.index 
                df2_['date'] = df2_.index
                m = df1_.merge(df2_ , on='date' , how='inner')
                return m['Adj Close'].corr(m[column_] , method='spearman')

            df_ = self.df_price.copy()
            sp500 = self.sp500_price.copy()
            byield = DataReader('DGS10' , 'fred' , '1975-01-01') 
            fed_fund = DataReader('DFF' , 'fred' , '1975-01-01') 
            inflation = DataReader('CORESTICKM159SFRBATL' , 'fred' , '1975-01-01') 
            dji = data.get_data_yahoo('^DJI' , start='1975-01-01')[['Adj Close']]
            vix = data.get_data_yahoo('^VIX' , start='1975-01-01')[['Adj Close']]
            nas = data.get_data_yahoo('^IXIC' , start='1975-01-01')[['Adj Close']]



            
            corr_vix = df_['Adj Close'].corr(vix['Adj Close'] , method='spearman')
            corr_vix_dow = dji['Adj Close'].corr(vix['Adj Close'] , method='spearman')
            corr_vix_nas = nas['Adj Close'].corr(vix['Adj Close'] , method='spearman')
            corr_vix_sp500 = sp500['Adj Close'].corr(vix['Adj Close'] , method='spearman')

            corr_fed_fund_int_rate = __correlation_merge_on_date(df_ , fed_fund , 'DFF')
            corr_fed_fund_int_rate_dow = __correlation_merge_on_date(dji , fed_fund , 'DFF')
            corr_fed_fund_int_rate_nas = __correlation_merge_on_date(nas , fed_fund , 'DFF')
            corr_fed_fund_int_rate_sp500 = __correlation_merge_on_date(sp500 , fed_fund , 'DFF')


            corr_inf = __correlation_merge_on_date(df_ , inflation , 'CORESTICKM159SFRBATL')
            corr_inf_dow = __correlation_merge_on_date(dji , inflation , 'CORESTICKM159SFRBATL')
            corr_inf_nas = __correlation_merge_on_date(nas , inflation , 'CORESTICKM159SFRBATL')
            corr_inf_sp500 = __correlation_merge_on_date(sp500 , inflation , 'CORESTICKM159SFRBATL') 

            corr_ten_y_byield = __correlation_merge_on_date(df_ , byield , 'DGS10')
            corr_ten_y_byield_dow = __correlation_merge_on_date(dji , byield , 'DGS10')
            corr_ten_y_byield_nas = __correlation_merge_on_date(nas , byield , 'DGS10')
            corr_ten_y_byield_sp500 = __correlation_merge_on_date(sp500 , byield , 'DGS10')



            ind = np.arange(4)
            width = 0.1

            with sns.plotting_context('notebook'):
                        plt.figure(figsize=(9,5))
                        sns.set_style('darkgrid')
                        plt.bar(ind, [corr_ten_y_byield , corr_fed_fund_int_rate , corr_vix , corr_inf], width , edgecolor='black' , label=self.ticker , alpha=0.8)
                        plt.bar(ind+width, [corr_ten_y_byield_dow , corr_fed_fund_int_rate_dow , corr_vix_dow , corr_inf_dow], width , edgecolor='black', label='Dow Jones' , alpha=0.8)
                        plt.bar(ind+width*2, [corr_ten_y_byield_nas , corr_fed_fund_int_rate_nas ,  corr_vix_nas, corr_inf_nas], width , edgecolor='black' , label='Nasdaq' , alpha=0.8)
                        plt.bar(ind+width*3, [corr_ten_y_byield_sp500 , corr_fed_fund_int_rate_sp500 ,corr_vix_sp500 , corr_inf_sp500], width , edgecolor='black' , label='SP500' , alpha=0.8)
                        plt.xticks(ind+0.2 , ['10 years treasury yield' , 'Federal Funds Effective Rate' , 'VIX index' , 'Consumer Price Index'])
                        plt.legend(edgecolor='black')
                        plt.savefig('data\\correlations.png')
                        plt.close('all')

                        if self.english == False:
                            self.add_picture('data\\correlations.png' , 'Corrélations entre différents indices et indicateurs' , left=0.5)
                        else:
                            self.add_picture('data\\correlations.png' ,  'Correlations between various indexes and indicators' , left=0.5)


    def plot_seasonality(self , df):


        to_dec_yearly = df[df.index[-1] - dt.timedelta(252*7):] # for yearly
        to_dec_yearly['month'] = to_dec_yearly.index.month_name()
        decomposition_yearly = seasonal_decompose(to_dec_yearly['Adj Close'], model='additive', period=252)
        to_plot = pd.DataFrame({'seasonal' : decomposition_yearly.seasonal.values , 'trend' : decomposition_yearly.trend.values ,
                                    'month' : to_dec_yearly['month'] , 'date' : decomposition_yearly.seasonal.index ,
                                    'observed' : decomposition_yearly.observed})



        fig, axs = plt.subplots(3, 1, sharex=True , figsize=(9,5))

        axs[0].plot(to_plot['date'], to_plot['observed'], color='blue')
        axs[1].plot(to_plot['date'], to_plot['trend'], color='purple')
        axs[2].plot(to_plot['date'], to_plot['seasonal'], color='red')

        axs[0].set_ylabel('Observed')
        axs[1].set_ylabel('Trend')
        axs[2].set_ylabel('Seasonality')


        # Modifier l'axe x pour afficher les mois
        axs[2].xaxis.set_major_locator(MonthLocator())
        axs[2].xaxis.set_major_formatter(DateFormatter('%b'))
        plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=90)

        plt.xlabel('Date')
        plt.tight_layout()


        plt.savefig('data\\seasonality.png')
        plt.close('all')
        
        if self.english:
            self.add_picture('data\\seasonality.png' , 'Breakdown of annual seasonality within the last years' , left=0.5) 
        else:
            self.add_picture('data\\seasonality.png' , 'Décomposition de la saisonnalité annuelle au cours des dernières années' , left=0.5)


    def pct_change_dividends_summary(self):
                '''Compute summary statistics of change in the dividends by year'''

                df_div_ = self.df_dividend.copy()
                df_div_['year'] = df_div_.index.year
                df_div_ = df_div_[df_div_['year'] != dt.datetime.now().year]
                grp = df_div_.groupby('year' , as_index=False)['Dividends'].sum()
                grp['pct_changes'] = round(grp['Dividends'].pct_change() * 100 , 3)
                grp = grp.dropna()

                with sns.plotting_context('talk'):
                    mean_ = round(grp['pct_changes'].mean() , 3)
                    median_ = round(grp['pct_changes'].median() , 3)
                    min_ = round(grp['pct_changes'].min() , 3)
                    max_ = round(grp['pct_changes'].max() , 3)

                    plt.figure(figsize=(8,5))
                    sns.scatterplot(x=['Max' , 'Mean' , 'Median' , 'Min'] , y=[max_ , mean_ , median_ , min_] , s=200 , edgecolor='black')
                    plt.title(f'Max: {round(max_,2)}% | Mean: {round(mean_,2)}% | Median: {round(median_,2)}% | Min: {round(min_,2)}%')

                    plt.savefig('data\\price_with_dividends.png')
                    plt.close('all')
                    if self.english:
                        self.add_picture('data\\price_with_dividends.png' , 'Dividend percentage changes statistics by year') 
                    else:
                        self.add_picture('data\\price_with_dividends.png' , 'Statistiques de pourcentage de changement du dividende par année')

