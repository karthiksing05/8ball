# My Custom Library Imports:
from litewick import create_candlestick_dataset, find_recent_trends
from predictor import MarketPredictor
from patterns import define_trend
# Other imports
import pandas as pd

# Python regs
import datetime

# Some constants
START = "2020-01-01"  # YYYY-MM-DD
END = datetime.datetime.now().strftime("%Y-%m-%d")  # YYYY-MM-DD

# Finding volatile stocks that are projected to change
payload = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
df = first_table
symbols = df['Symbol'].values.tolist()
good_tickers = []
for ticker in symbols:
    candles = create_candlestick_dataset(ticker, START, END)
    output, num_true = find_recent_trends(candles)
    if output > 0.2:
        good_tickers.append(ticker)
        print("I am {} confident that {} will have a trend reversal within the next week!".format(
            round(output, 4), ticker))


