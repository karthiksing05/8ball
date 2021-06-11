import plotly.graph_objects as go
import yfinance as yf

import pandas as pd
import numpy as np
import cv2 as cv

from patterns import *
from candle import Candlestick

import datetime
from pprint import pprint

pd.set_option('display.max_rows', 10000000000)

ticker = "DOGE-USD"
START = "2020-01-01" # YYYY-MM-DD
END = datetime.datetime.now().strftime("%Y-%m-%d") # YYYY-MM-DD

def computeRSI(data, time_window=14):
    diff = data.diff(1).dropna()

    up_chg = 0 * diff
    down_chg = 0 * diff
    
    up_chg[diff > 0] = diff[ diff>0 ]
    
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def create_dataset(ticker, start, end):
    ticker = yf.Ticker(ticker.upper())
    df = ticker.history(start=start, end=end, interval="1d")
    df['RSI'] = computeRSI(df['Adj Close'], 14)
    df = df.dropna()
    candles = []
    for i in range(2,df.shape[0]):
        current = df.iloc[i,:]
        realbody = abs(current['Open'] - current['Close'])
        candle_range = current['High'] - current['Low']
        bullish_bearish = "BEARISH" if current["Close"] > current["Open"] else "BEARISH"
        idx = df.index[i]
        candle = Candlestick(
            index=idx.strftime("%Y-%m-%d"),
            body_size=realbody,
            candle_range=candle_range,
            btype=bullish_bearish,
            price_open=current["Open"],
            price_close=current["Close"],
            price_high=current["High"],
            price_low=current["Low"],
            volume=current["Volume"],
            rsi=current["RSI"]
        )
        candles.append(candle)
    pprint(candles)

def show_candlestick_graph(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.show()

create_dataset(ticker, START, END)
show_candlestick_graph(ticker, START, END)
