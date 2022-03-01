import pandas as pd
import datetime
import os
import os.path
import time
from functools import reduce

import bs4 as bs
import numpy as np
import pandas as pd
import requests
import yfinance as yf




from utils import EnvConfig

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import warnings
warnings.filterwarnings('ignore')

class MyDataset():

    def __init__(self):
        self.url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.stocks_fname = "sp500_closefull.csv"
        self.outdated_data_files = False
        self.start = datetime.datetime(2010, 1, 1)
        self.stop = datetime.datetime.now()
        #self.Ntrain = datetime.datetime(2020, 9, 30)
        self.Ntest = 300# (self.stop - datetime.datetime(2021, 10, 1)).days
        self.now = time.time()
        self.config = EnvConfig()



    def get_train_test(self) -> pd.DataFrame:

        start = self.start  # datetime.datetime(2019, 1, 1)
        end = self.stop  # datetime.datetime(2019, 7, 17)


        if os.path.isfile(self.stocks_fname):
            filestamp = os.stat(self.stocks_fname).st_mtime
            filecompare = self.now - 1 * 86400
            if filestamp < filecompare:
                self.outdated_data_files = True
                print("Removing old stocks dataset:", self.stocks_fname)
                os.remove(self.stocks_fname)

        if not os.path.isfile(self.stocks_fname):
            resp = requests.get(self.url)
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})
            tickers = []

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            tickers = [s.replace('\n', '') for s in tickers]
            df_sp500 = yf.download(tickers, start=start, end=end)

            df_sp500 = df_sp500['Adj Close']
            #data['Adj Close'].to_csv('sp500_closefull.csv')




            ftes = self.config.stocks
            #df_ftes = web.DataReader("SPY", "yahoo", start=start, end=end)
            print("Downloading FTEs...")
            for fte in ftes:
                df_fte = yf.download(fte, start=start, end=end)
                df_fte = df_fte.loc[:, ['Adj Close']]
                df_fte.columns = [fte]

                df_sp500 = pd.concat([df_sp500, df_fte], axis=1)


            #df0.dropna(axis=1, how='all', inplace=True)
            df_sp500.dropna(axis=0, how='all', inplace=True)
            print("Dropping columns due to nans > 50%:", df_sp500.loc[:, list((100 * (df_sp500.isnull().sum() / len(df_sp500.index)) > 50))].columns)
            df_sp500 = df_sp500.drop(df_sp500.loc[:, list((100 * (df_sp500.isnull().sum() / len(df_sp500.index)) > 50))].columns, 1)
            df_sp500 = df_sp500.ffill().bfill()

            print("Any columns still contain nans:", df_sp500.isnull().values.any())
            print("Assets count:", df_sp500.shape[1])

            df_sp500.to_csv('sp500_closefull.csv')

        df_sp500 = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
        df_returns = pd.DataFrame()
        for name in df_sp500.columns:
            df_returns[name] = np.log(df_sp500[name]).diff()

        #df_returns['SPY_PRICE'] = df0['SPY']
        df_returns[self.config.stocks_adj_close_names] = df_sp500[self.config.stocks]

        # split into train and test
        df_returns.dropna(axis=0, how='any', inplace=True)

        train_data = df_returns.iloc[:-self.Ntest]
         #train_data = df_returns.iloc[:self.Ntrain]
        test_data = df_returns.iloc[-self.Ntest:]

        return train_data, test_data
def transform_to_log_scale(x):

    y = (np.sign(x)) * (np.log(abs(x)))
    return y


if __name__  == "__main__":
   dataset = MyDataset()
   train, test = dataset.get_train_test()
   print(test)

