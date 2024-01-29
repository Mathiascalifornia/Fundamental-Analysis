import os 
import datetime as dt
import warnings

from googletrans import Translator
import dataframe_image as dsi
import textwrap
import numpy as np 

from scraping import ScrapingSelenium
from data_viz import DataViz
from tkinter_windows import MainWindows
from finance_computation import FinanceComputationner
from api_calls import ApiCaller

warnings.filterwarnings("ignore")



class App:

    def __init__(self , company_name:str , path_to_save:str , ticker:str , language:str):

        self.company_name = company_name
        self.path_to_save = path_to_save
        self.ticker = ticker 
        self.language = language
        self.english = False if self.language != 'English' else True
        self.t = lambda string : Translator().translate(string , src='French').text

        # To get the success or failure of every step
        self.worked_stock_price = False
        self.worked_dividends = False
        self.worked_dataframe = False
        self.worked_share_holders = False
        self.sentiment_score = False
        self.main_institutions_bool = False

        self.data_viz = DataViz(**self.get_attributs()) # Give the same attributes to DataViz 
        self.scraping = ScrapingSelenium(company_name=self.company_name , ticker=self.ticker)
        self.finance_comp = FinanceComputationner()
        self.api_caller = ApiCaller(ticker=self.ticker)
        

    def get_attributs(self):
        """ 
        Get all the instance attributes (will be used for the DataViz class)
        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith("__")}    
    
    def preprocess_description(self):
        """ 
        Preprocess step after scraping of the presentation
        Add the cleanned description in the presentation
        """

        def __jump_line(s : str , every:int=80):
            '''Jump line every 80 words (for the description) , to better suits the power-point format'''
            s = '\n'.join(textwrap.wrap(s , every))
            if len(s) >= 960:
                s1 = s[:960]
                s2 = s[960:]
                return s1 , s2
            
            return s
        
        if not self.english:
            self.description += '\nLa capitalisation , valeur entreprise , chiffre d\'affaire , EBITDA , EBIT , EBT , le résultat net , la dette et trésorerie net , le free cash flow , les capitaux propres , le total des actifs et le Capex sont en millions.'
        
        if self.english:
            self.description = self.t(self.description)
            self.description += self.t('\nLa capitalisation , valeur entreprise , chiffre d\'affaire , EBITDA , EBIT , EBT , le résultat net , la dette et trésorerie net , le free cash flow , les capitaux propres , le total des actifs et le Capex sont en millions.')

        if len(self.description) < 960:
            self.description = __jump_line(self.description)
            self.data_viz.pres_description(self.description , title=self.title)
        else:
            string1 , string2 = __jump_line(self.description)
            self.data_viz.pres_description(string1 , self.title)
            self.data_viz.pres_description(string2 , title='Suite :')


    def plot_dataframes(self):

        dsi.export(self.table_0 , 'data\\df1.png' , table_conversion='matplolib' , fontsize=9)
        if self.english == False:
            self.data_viz.add_picture('data\\df1.png' , 'Tableau 1' , left=0.6 , top=2.1)
        else:
            self.data_viz.add_picture('data\\df1.png' , 'Array 1' , left=0.6, top=2.1)
        self.data_viz.plot_element(self.table_0)


        dsi.export(self.table_1, 'data\\df2.png' , table_conversion='matplolib' , fontsize=12)
        if self.english == False:
            self.data_viz.add_picture('data\\df2.png' , 'Tableau 2' ,  left=0.8, top=2.1)
        else:
            self.data_viz.add_picture('data\\df2.png' , 'Array 2' , left=0.8, top=2.1)
        self.data_viz.plot_element(self.table_1)


        dsi.export(self.table_3 , 'data\\df4.png' , table_conversion='matplolib' , fontsize=11)
        if self.english == False:
            self.data_viz.add_picture('data\\df4.png' , 'Tableau 3' , left=0.7, top=2.1)
        else:
            self.data_viz.add_picture('data\\df4.png' , 'Array 3' , left=0.7, top=2.1)
        self.data_viz.plot_element(self.table_3)


    def preprocess_and_plot_df3(self):
        # Handle missing data for df3
        self.table_2.index = [ind.strip() for ind in self.table_2.index]
        if not self.english:
            if self.table_2.isna().sum(axis=1).sum() / self.table_2.size < 0.15:
                to_repl = str(self.table_2.loc["Marge d'exploitation"].dropna().apply(lambda x : float(str(x).replace('%' , '')))[-1])+'%'
                to_repl_net = str(self.table_2.loc["Marge nette"].dropna().apply(lambda x : float(str(x).replace('%' , '')))[-1])+'%'
                to_repl_ca = self.table_2.loc["Chiffre d'affaires"].dropna().apply(lambda x : float(x))[-1]
                to_repl_ebit = self.table_2.loc["Résultat d'exploitation EBIT"].dropna().apply(lambda x : float(x))[-1]
                to_repl_net_results = self.table_2.loc["Résultat net"].dropna().apply(lambda x : float(x))[-1]

                self.table_2.loc["Marge d'exploitation"] = self.table_2.loc["Marge d'exploitation"].fillna(to_repl)
                self.table_2.loc["Marge nette"] = self.table_2.loc["Marge nette"].fillna(to_repl_net)
                self.table_2.loc["Chiffre d'affaires"] = self.table_2.loc["Chiffre d'affaires"].fillna(to_repl_ca)
                self.table_2.loc["Résultat d'exploitation EBIT"] = self.table_2.loc["Résultat d'exploitation EBIT"].fillna(to_repl_ebit)
                self.table_2.loc["Résultat net"] = self.table_2.loc["Résultat net"].fillna(to_repl_net_results)
                try:
                    self.data_viz.plot_multiple_indicators(df1=self.table_2 , df2=self.table_2 , quaterly=True)
                except Exception as e:
                    print("Problem with quaterly multiple indicators : " , e)

        if self.english:
            if self.table_2.isna().sum(axis=1).sum() / self.table_2.size < 0.15:
                to_repl = str(self.table_2.loc["Marge d'exploitation"].dropna().apply(lambda x : float(str(x).replace('%' , '')))[-1])+'%'
                to_repl_net = str(self.table_2.loc["Marge nette"].dropna().apply(lambda x : float(str(x).replace('%' , '')))[-1])+'%'
                to_repl_ca = self.table_2.loc["Chiffre d'affaires"].dropna().apply(lambda x : float(x))[-1]
                to_repl_ebit = self.table_2.loc["Résultat d'exploitation EBIT"].dropna().apply(lambda x : float(x))[-1]
                to_repl_net_results = self.table_2.loc["Résultat net"].dropna().apply(lambda x : float(x))[-1]

                self.table_2.loc["Marge d'exploitation"] = self.table_2.loc["Marge d'exploitation"].fillna(to_repl)
                self.table_2.loc["Marge nette"] = self.table_2.loc["Marge nette"].fillna(to_repl_net)
                self.table_2.loc["Chiffre d'affaires"] = self.table_2.loc["Chiffre d'affaires"].fillna(to_repl_ca)
                self.table_2.loc["Résultat d'exploitation EBIT"] = self.table_2.loc["Résultat d'exploitation EBIT"].fillna(to_repl_ebit)
                self.table_2.loc["Résultat net"] = self.table_2.loc["Résultat net"].fillna(to_repl_net_results)
                try:
                    self.data_viz.plot_multiple_indicators(df1=self.table_2 , df2=self.table_2 , quaterly=True)   
                except Exception as e:
                    print("Error with multiple indicators : " , e)

    def plot_stock_prices_figures(self):
        self.data_viz.plot_maximum_draw_down(df=self.df_price)
        self.data_viz.plot_against_benmark()
        self.data_viz.plot_regression(df=self.df_price)       
    

        if len(self.df_price) > 1200:
            self.data_viz.plot_rsi()

        self.data_viz.plot_zoom_candles()
        self.data_viz.plot_correlation()
        self.data_viz.plot_seasonality(self.df_price)

        if self.worked_dividends:
            self.data_viz.price_with_dividends(self.df_price[['Adj Close']] , self.df_dividend.drop('year' , axis=1 , errors='ignore'))
            self.data_viz.annual_dividend_history()
            self.data_viz.pct_change_dividends_summary()


    def save_presentation(self):

        # Process the inputs
        self.ticker = self.ticker.upper().replace('"' , '')
        self.path_to_save = os.path.join(self.path_to_save , self.ticker).replace('"' , '') + ".pptx"

        # Save the presentation 
        self.data_viz.pres.save(self.path_to_save)         



    def add_recap_numbers_pres(self):

        def __add_with_shareholders():
            if not self.english:
                try:
                    key_num = f'L\'augmentation du prix moyenne sur {years} ans  est de {round(float(ann_) * 100 , 2)}%\net {round(float(ann_five) * 100 , 2)}% sur les 5 dernières années.\nL\'écart type moyen à l\'année est de {round(std_*100,4)}\net {round(std_five*100,4)} sur les 5 dernières années.\nLes ratios Sharpe et Sortino sont respectivement {round(float(sharpe),4)} et {round(float(sortino),4)}\net de {round(float(sharpe_five),4)} et {round(float(sortino_five),4)} sur les 5 dernières années.\n\nLes trois principaux investisseurs institutionnels sont , par ordre décroissant de détention:\n-{self.main_institutions[0]}\n-{self.main_institutions[1]}\n-{self.main_institutions[2]}.\n\n\n\n\nDate de ce rapport : {dt.datetime.today().date()}'
                except Exception as e:
                    print("Problem with the shareholder on last slide : " , e)
                    key_num = f'L\'augmentation du prix moyenne sur {years} ans  est de {round(float(ann_) * 100 , 2)}%\net {round(float(ann_five) * 100 , 2)}% sur les 5 dernières années.\nL\'écart type moyen moyen à l\'année est de {round(std_*100,4)}\net {round(std_five*100,4)} sur les 5 dernières années.\nLes ratios Sharpe et Sortino sont respectivement {round(float(sharpe),4)} et {round(float(sortino),4)}\net de {round(float(sharpe_five),4)} et {round(float(sortino_five),4)} sur les 5 dernières années.\n\n\n\n\nDate de ce rapport : {dt.datetime.today().date()}'
                
                self.data_viz.pres_description(key_num , 'Chiffres et informations à prendre en compte.')

            if self.english:
                try:
                    key_num = self.t(f'L\'augmentation du prix moyenne sur {years} ans  est de {round(float(ann_) * 100 , 2)}%\net {round(float(ann_five) * 100 , 2)}% sur les 5 dernières années.\nL\'écart type moyen à l\'année est de {round(std_*100,4)}\n      et {round(std_five*100,4)} sur les 5 dernières années.\nLes ratios Sharpe et Sortino sont respectivement {round(float(sharpe),4)} et {round(float(sortino),4)}\net de {round(float(sharpe_five),4)} et {round(float(sortino_five),4)} sur les 5 dernières années.\n\nLes trois principaux investisseurs institutionnels sont , par ordre décroissant de détention:\n-{self.main_institutions[0]}\n-{self.main_institutions[1]}\n-{self.main_institutions[2]}.\n\n\n\n\nDate de ce rapport : {dt.datetime.today().date()}')
                except Exception as e:
                    print("Problem with the shareholder on last slide : " , e)
                    key_num = self.t(f'L\'augmentation du prix moyenne sur {years} ans  est de {round(float(ann_) * 100 , 2)}%\net {round(float(ann_five) * 100 , 2)}% sur les 5 dernières années.\nL\'écart type moyen moyen à l\'année est de {round(std_*100,4)}\net {round(std_five*100,4)} sur les 5 dernières années.\nLes ratios Sharpe et Sortino sont respectivement {round(float(sharpe),4)} et {round(float(sortino),4)}\net de {round(float(sharpe_five),4)} et {round(float(sortino_five),4)} sur les 5 dernières années.\n\n\n\n\nDate de ce rapport : {dt.datetime.today().date()}')
                    
                self.data_viz.pres_description(key_num , 'Numbers and interestings informations.')

        def __add_without_shareholders():
            if not self.english:
                key_num = f'L\'augmentation du prix moyenne sur {years} ans  est de {round(float(ann_) * 100 , 2)}%\net {round(float(ann_five) * 100 , 2)}% sur les 5 dernières années.\nL\'écart type moyen moyen à l\'année est de {round(std_*100,4)}\net {round(std_five*100,4)} sur les 5 dernières années.\nLes ratios Sharpe et Sortino sont respectivement {round(float(sharpe),4)} et {round(float(sortino),4)}\net de {round(float(sharpe_five),4)} et {round(float(sortino_five),4)} sur les 5 dernières années.\n\n\n\n\nDate de ce rapport : {dt.datetime.today().date()}'
                self.data_viz.pres_description(key_num , 'Chiffres et informations à prendre en compte.')

            if self.english:
                key_num = self.t(f'L\'augmentation du prix moyenne sur {years} ans  est de {round(float(ann_) * 100 , 2)}%\net {round(float(ann_five) * 100 , 2)}% sur les 5 dernières années.\nL\'écart type moyen moyen à l\'année est de {round(std_*100,4)}\net {round(std_five*100,4)} sur les 5 dernières années.\nLes ratios Sharpe et Sortino sont respectivement {round(float(sharpe),4)} et {round(float(sortino),4)}\net de {round(float(sharpe_five),4)} et {round(float(sortino_five),4)} sur les 5 dernières années.\n\n\n\n\nDate de ce rapport : {dt.datetime.today().date()}')
                self.data_viz.pres_description(key_num , 'Numbers and interestings informations.')



        ###### Somes interesting numbers ######
        stock_price = self.df_price[['Adj Close']]
        ann_ , years = self.finance_comp.annualized_return(stock_price)
        ann_five , _ = self.finance_comp.annualized_return_five_years(stock_price)
        std_ = float(np.round((np.std(stock_price.pct_change())),4))
        new_index = stock_price.index[-1] - dt.timedelta(252 * 5)
        std_five = float(np.round((np.std(stock_price[new_index:].pct_change())),4))
        sharpe , sortino = self.finance_comp.sharpe_and_sortino_ratio(stock_price)
        sharpe_five , sortino_five = self.finance_comp.sharpe_and_sortino_ratio_five_years(stock_price)

        if self.worked_share_holders:
            __add_with_shareholders()
        else:
            __add_without_shareholders()

    def main(self):

        self.title , self.url , self.url_desc = self.scraping.get_url()

        print('URL finances : ' , self.url)
        print('URL company description : ' , self.url_desc)
        print('Found title of the company : ' , self.title)

        self.description = self.scraping.get_description(url_desc=self.url_desc)

        # Add the description to the presentation
        self.preprocess_description()

        # API calls for stock price and dividends
        try:
            # Get the stock prices
            self.df_price , self.sp500_price = self.api_caller.get_price()
            self.data_viz.df_price = self.df_price 
            self.data_viz.sp500_price = self.sp500_price
            self.worked_stock_price = True 
        except Exception as e:
            print("Problem with the stock price request to yahoo API : " , e)


        try:
            # Get the dividends
            self.df_dividend = self.api_caller.get_dividend()
            self.data_viz.df_dividend = self.df_dividend
            if len(self.df_dividend) > 0:
                self.worked_dividends = True
        except Exception as e:
            print("Problem with the dividend request to yahoo API : " , e) 

        
        # Get the tables from zone bourse
        try:
            self.table_0 , self.table_1 , self.table_2 , self.table_3 = self.scraping.get_tables()
            self.plot_dataframes()
            self.worked_dataframe = True

        except Exception as e:
            print("Error with the dataframes : " , e)

        if self.worked_dataframe:
            if self.worked_dividends:
                try:
                    self.data_viz.payout_ratio(df2=self.table_1)
                except Exception as e:
                    print("Payout ratio not worked : " , e)

            try:
                self.data_viz.cap_vs_debt(table_3=self.table_3)
            except Exception as e:
                print("Cap vs debt did not work : " , e)

            try:
                self.data_viz.plot_multiple_indicators(self.table_0 , self.table_1)
            except Exception as e:
                print("Multiple indicators did not work : " , e)


            try:
                self.preprocess_and_plot_df3()
            except Exception as e:
                print("Problem preprocessing df3 : " , e)
        

        if self.worked_stock_price:
            self.plot_stock_prices_figures()


        try:
            # Plot the shareholders pie
            self.data_viz.shareholders()
            self.worked_share_holders = True

        except Exception as e:
            print("Problem with the shareholders pie : " , e)


        try:
            scores = self.scraping.sentiment_scores()
            self.data_viz.plot_sentiment_score(scores=scores)
            self.sentiment_score = True
        except Exception as e:
            print("Problem with the sentiment scores : " , e)


        try:
            self.main_institutions = self.api_caller.get_main_institutions()
            self.main_institutions_bool = True

        except Exception as e:
            print('Main institutions failed : ' , e)


        self.add_recap_numbers_pres()
        self.save_presentation()

        print('Worked sentiment score : ' , self.sentiment_score)
        print('Worked shareholders : ' , self.worked_share_holders)
        print("Worked stock data : " , self.worked_stock_price)
        print("Worked dividends data : " , self.worked_dividends)
        print("Worked main institution : " , self.main_institutions_bool)


if __name__ == "__main__":
    MainWindows().main(App=App)