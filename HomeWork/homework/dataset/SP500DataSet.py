import datetime
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import bs4 as bs
import os

class SP500DataSet:
    def __init__(self, stocks, start=datetime.datetime(2010, 1, 1), stop=datetime.datetime.now()):
        self.df_returns = None
        self.url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.stocks_file_name = "sp500_closefull.csv"
        self.start = start
        self.stop = stop
        self.stocks = stocks
        self.stocks_adj_close_names = [stock + '_Adj_Close' for stock in self.stocks]

    def load(self):
        start = self.start
        end = self.stop
        if os.path.isfile(self.stocks_file_name):
            file_timestamp = os.stat(self.stocks_file_name).st_mtime
            filecompare = time.time() - 1 * 86400
            if file_timestamp < filecompare:
                print("Removing old stocks dataset:", self.stocks_file_name)
                os.remove(self.stocks_file_name)

        if not os.path.isfile(self.stocks_file_name):
            resp = requests.get(self.url)
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})
            tickers = []

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            tickers = [s.replace('\n', '') for s in tickers]
            data = yf.download(tickers, start=start, end=end)
            df_sp500 = data['Adj Close']

            # df0.dropna(axis=1, how='all', inplace=True)
            df_sp500.dropna(axis=0, how='all', inplace=True)
            print("Dropping columns due to nans > 50%:",
                  df_sp500.loc[:, list((100 * (df_sp500.isnull().sum() / len(df_sp500.index)) > 50))].columns)
            df_sp500 = df_sp500.drop(
                df_sp500.loc[:, list((100 * (df_sp500.isnull().sum() / len(df_sp500.index)) > 50))].columns, 1)
            df_sp500 = df_sp500.ffill().bfill()

            print("Any columns still contain nans:", df_sp500.isnull().values.any())
            print("Assets count:", df_sp500.shape[1])

            df_sp500.to_csv(self.stocks_file_name)

        df_sp500 = pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True)
        df_returns = pd.DataFrame()
        for name in df_sp500.columns:
            df_returns[name] = np.log(df_sp500[name]).diff()

        df_returns[self.stocks_adj_close_names] = df_sp500[self.stocks]
        df_returns.dropna(axis=0, how='any', inplace=True)
        return df_returns




