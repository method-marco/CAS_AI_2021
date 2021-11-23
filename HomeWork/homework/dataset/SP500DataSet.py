import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import bs4 as bs
import os


class SP500DataSet:
    def __init__(self, start=datetime.datetime(2010, 1, 1), stop=datetime.datetime.now()):
        self.url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.stocks_file_name = "sp500_closefull.csv"
        self.start = start
        self.stop = stop

    def load(self):

        start = self.start
        end = self.stop

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
            data['Adj Close'].to_csv(self.stocks_file_name)

        df0 = pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True)

        df_spy = yf.download("SPY", start=start, end=end)

        # df_spy = web.DataReader("SPY", "yahoo", start=start, end=end)
        df_spy = df_spy.loc[:, ['Adj Close']]

        df_spy.columns = ['SPY']

        df0 = pd.concat([df0, df_spy], axis=1)

        # df0.dropna(axis=1, how='all', inplace=True)
        df0.dropna(axis=0, how='all', inplace=True)
        print("Dropping columns due to nans > 50%:",
              df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)
        df0 = df0.drop(df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns, 1)
        df0 = df0.ffill().bfill()

        print("Any columns still contain nans:", df0.isnull().values.any())

        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()

        # df_returns['SPY_PRICE'] = df0['SPY']
        # df_returns[self.config.stocks_adj_close_names] = df0[self.config.stocks]
        # split into train and test
        df_returns.dropna(axis=0, how='any', inplace=True)
        df_returns.SPY = [1 if spy > 0 else 0 for spy in df_returns.SPY]
        return df_returns
