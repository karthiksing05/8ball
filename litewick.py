# Importing
import plotly.graph_objects as go
# Downloading data
import yfinance as yf

# ML IMPORTS
from sklearn.preprocessing import StandardScaler

# My libs
# All patterns
from patterns import (
    identify_engulfing,
    identify_morning_star,
    identify_shooting_star,
    identify_3_white_soldiers,
    identify_3_black_crows,
    identify_piercing_pattern
)
from candle import Candlestick

import inspect

def _computeRSI(data, time_window=14):
    diff = data.diff(1).dropna()

    up_chg = 0 * diff
    down_chg = 0 * diff

    up_chg[diff > 0] = diff[diff > 0]

    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(
        com=time_window-1, min_periods=time_window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi


def create_candlestick_dataset(ticker, start, end):
    ticker = yf.Ticker(ticker.upper())
    df = ticker.history(start=start, end=end, interval="1d")
    if df.empty:
        return "error"
    df['RSI'] = _computeRSI(df[['Close']])
    df = df.dropna()
    columns = ["Open", "High", "Low", "Close"]
    price_scaler = StandardScaler()
    full_prices = df[columns].values
    df[columns] = price_scaler.fit_transform(full_prices)
    candles = []
    for i in range(2, df.shape[0]):
        current = df.iloc[i, :]
        realbody = abs(current['Open'] - current['Close'])
        candle_range = current['High'] - current['Low']
        bullish_bearish = "BULLISH" if current["Close"] > current["Open"] else "BEARISH"
        idx = df.index[i]
        candle = Candlestick(
            index=idx.strftime("%Y-%m-%d"),
            body_range=realbody,
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
    return candles


def show_candlestick_graph(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.show()

def identify_trend_reversal(candles):
    """
    This function finds recent trend reversals (past 3 days) and returns a confidence that the trend will reverse in the future.
    Note that if you get anything other than a 0, then a trend reversal may be imminent.
    """

    if candles == "error":
        return 0, [False, False, False, False, False, False]

    past_50_dpts = [[(abs(abs(x - len(candles)) - 50) + 1), close_price]
                    for x, close_price in enumerate([candle.close for candle in candles])]
    past_50_dpts = past_50_dpts[-50:]
    past_days = [candle for candle in candles[-3:]]
    list_of_patterns = [
        identify_engulfing,
        identify_piercing_pattern,
        identify_morning_star,
        identify_shooting_star,
        identify_3_white_soldiers,
        identify_3_black_crows
    ]

    condition_lst = []
    good_patterns = []
    for pattern in list_of_patterns:
        num_args = int(len(inspect.getfullargspec(pattern)[0])) - 1
        num_days = len(past_days)
        cond = False
        if num_args == 1:
            # print(pattern)
            while (cond == False):
                if (num_days > 0):
                    cond = pattern(
                        past_days[int(num_days - 1)],
                        past_50_dpts
                    )
                    # print(cond)
                    # print(num_days - 1)
                    num_days -= 1
                else:
                    # print("done")
                    break
        elif num_args == 2:
            # print(pattern)
            while (cond == False):
                if ((num_days - 1) > 0):
                    cond = pattern(
                        past_days[int(num_days - 2)],
                        past_days[int(num_days - 1)],
                        past_50_dpts
                    )
                    # print(cond)
                    # print(num_days - 1, num_days - 2)
                    num_days -= 1
                else:
                    # print("done")
                    break
        elif num_args == 3:
            # print(pattern)
            while (cond == False):
                if ((num_days - 2) > 0):
                    cond = pattern(
                        past_days[int(num_days - 3)],
                        past_days[int(num_days - 2)],
                        past_days[int(num_days - 1)],
                        past_50_dpts
                    )
                    # print(cond)
                    # print(num_days - 1, num_days - 2, num_days - 3)
                    num_days -= 1
                else:
                    # print("done")
                    break
        if cond:
            good_patterns.append(str(pattern.__name__))
        condition_lst.append(cond)
    # print(condition_lst)
    num_true = condition_lst.count(True)
    proba = (num_true / len(condition_lst))
    return proba, good_patterns
