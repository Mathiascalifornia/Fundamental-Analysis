
import re 
import time

# Data manipulation
import pandas as pd 
import numpy as np

# NLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Api , scraping
from bs4 import BeautifulSoup
import lxml
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException , NoAlertPresentException
import yfinance as yf

import time
yf.pdr_override()


class ScrapingSelenium:

    def __init__(self , company_name , ticker):
        self.company_name = company_name
        self.ticker = ticker
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))


    def get_url(self) -> tuple:
            """ 
            Get the zone bourse urls (fondamentals and society) using selenium
            """

            self.driver.get(f'https://www.google.com/search?q=zone+bourse+{self.company_name}+finance&sxsrf=ALiCzsbIaWNWrnXJ5acLqlPx2kINT72YMA%3A1670610120483&ei=yHyTY9CSHcmPkdUP3veM2AI&oq=zone+bourse+telenor++finance&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgQIIxAnOggIABCiBBCwA0oECEEYAUoECEYYAFCMBViMBWCHEGgBcAB4AIABKYgBKZIBATGYAQCgAQHIAQPAAQE&sclient=gws-wiz-serp')

            try: # Do the capcha if needed
                WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "(//button)[4]"))
            ).click()

            except TimeoutException:
                pass 

            # Query the first link
            link_first_result = self.driver.find_element(By.XPATH , ('(//div[contains(@data-async-context , "query:zone")]/div//a)[1]')).get_attribute("href")
            self.driver.get(link_first_result)

            time.sleep(2)

            try:
                # Desactivate the alert
                alert = self.driver.switch_to.alert
                alert.accept()
            except NoAlertPresentException:
                pass

            current_url = self.driver.current_url

            splitted_url = current_url.split('/')

            if splitted_url[-2] not in ('fondamentaux' , 'societe' , "cotations"):
                url_desc = current_url + 'societe/'
                url = current_url + 'fondamentaux/'

            elif splitted_url[-2] == "cotations":
                url = current_url.replace("cotations/" , 'fondamentaux/')
                url_desc = current_url.replace("cotations/" , 'societe/')

            elif splitted_url[-2] == 'fondamentaux':
                url_desc = current_url.replace(splitted_url[-2] , 'societe')
                url = current_url

            elif splitted_url[-2] == 'societe':
                url = current_url.replace("societe/" , 'fondamentaux/')
                url_desc = current_url

            self.title = str(url.split('/')[-3]) 
            self.url = url
            self.url_desc = url_desc

            return str(url.split('/')[-3]) , url , url_desc 




    def get_tables(self) -> tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame]:
        
        def clean_and_set_index(df : pd.DataFrame) -> pd.DataFrame:
            index_col = df.columns[df.columns.str.startswith('Période')]
            df.index = df[index_col]
            df = df.drop(index_col , axis=1)
            df.index = [re.sub(string=re.sub(string=str(ind).strip() , repl='' , pattern="(\"|\(|\)|,|\\d)") , repl='' , pattern=r"^(?:')|(?:')$").strip() for ind in df.index]
            return df.drop(df.index[-1])
        
        def clean_table_apply(x):
            return str(x).replace('\u202f' , "").replace('%' , '').replace(',' , '.').replace('x' , '').replace(' ' , '')

        # Get the source code 
        self.driver.get(self.url)
        source_code = self.driver.page_source

        soup = BeautifulSoup(source_code , 'html.parser')
        tables = soup.find_all('div' , class_='card card--collapsible mb-15')
        assert len(tables) == 8 , f'Bad number of tables : {len(tables)}'


        table_0 = pd.read_html(str(tables[0]))[1].apply(lambda x : x.replace('-' , None))
        table_1 = pd.read_html(str(tables[1]))[1].apply(lambda x : x.replace('-' , None))
        table_2 = pd.read_html(str(tables[2]))[1].apply(lambda x : x.replace('-' , None))
        table_3 = pd.read_html(str(tables[3]))[1].apply(lambda x : x.replace('-' , None))

        table_0 = clean_and_set_index(table_0).fillna(np.nan).map(lambda x : float(clean_table_apply(x)) if clean_table_apply(x) else np.nan)
        table_1 = clean_and_set_index(table_1).fillna(np.nan).map(lambda x : float(clean_table_apply(x)) if clean_table_apply(x) else np.nan)
        table_2 = clean_and_set_index(table_2).fillna(np.nan).map(lambda x : float(clean_table_apply(x)) if clean_table_apply(x) else np.nan)
        table_3 = clean_and_set_index(table_3).fillna(np.nan).map(lambda x : float(clean_table_apply(x)) if clean_table_apply(x) else np.nan)

        # Special treatment on table_3 to compute the Capitaux propres , that have been removed from zone bourse , so we compute it here
        values_ = (table_3.loc["Total Actifs"].astype(float) - table_3.loc["Dette Nette"].astype(float)).values
        cols = table_3.columns 
        index_ = "Capitaux Propres"

        table_3 = pd.concat([table_3 , pd.DataFrame(data=values_ , columns=[index_] , index=cols).T])

        return table_0 , table_1 , table_2 , table_3

    def get_description(self , url_desc) -> str:
        
        self.driver.get(url_desc)
        tree = lxml.html.fromstring(self.driver.page_source)
        xpath_expression = '//div[@class="company-logo"]/following-sibling::text()'
        raw_text = tree.xpath(xpath_expression)
        pattern = re.compile(r'\\n|(<.*?>)|\[|\]|\"')
        return re.sub(pattern=pattern , string=str(raw_text) , repl='').strip()



    def sentiment_scores(self) -> pd.DataFrame:
        '''Sentiment analysis using vader SentimentAnalyser (code source modified to fit my needs) , using finviz news title concerning the companie.'''
        if '.' in self.ticker:
            ticker_ = self.ticker.split('.')[0]
            url = f'https://finviz.com/quote.ashx?t={ticker_}&p=d'
        else:
            url = f'https://finviz.com/quote.ashx?t={self.ticker}&p=d'

        
        self.driver.get(url)
        time.sleep(5)
        self.driver.find_element(By.XPATH , ('//*[@id="qc-cmp2-ui"]/div[2]/div/button[3]')).click()
        time.sleep(3)
        source = self.driver.page_source
        self.driver.quit()


        source = BeautifulSoup(source)
        news = source.find_all(class_="news-link-container")

        pattern = re.compile('.*?\"_blank">(.*)</a>.*')

        data = [str(re.findall(pattern , str(new))).replace('[' , '').replace(']' , '').replace("'" , '').replace('"' , "") for new in news]

        new_words = {
        'crushes': 10,
        'fear': -10,
        'greed': 10,
        'beats': 10,
        'misses': -10,
        'trouble': -10,
        'falls': -10,
        'down' : -15,
        'up' : 15,
        'go down' : -15,
        'go up' : 15,
        'increased' : 10,
        'decreased' : -10,
        'increase' : 10,
        'decrease' : -10,
        'safe' : 15,
        'unsafe' : -15,
        'attractive' : 10,
        'dividends cut': -20,
        'dividends increase':20,
        'hits its target' : 10,
        'increase in value' : 10,
        'decrease in value' : -10,
        'top' : 15,
        'buy' : 5,
        'raise' : 5,
        'low' : -5,
        'dangerous' : -10}

        vader = SentimentIntensityAnalyzer()
        vader.lexicon.update(new_words)

        to_plot = []
        for news in data:
            score = vader.polarity_scores(news)
            score.pop('compound' , [])
            if score['pos'] > score['neg']:
                to_plot.append('Positive headline')
                
            elif score['pos'] < score['neg']:
                to_plot.append('Negative headline')

        return pd.DataFrame(data=to_plot , columns=['score'])
