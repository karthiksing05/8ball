# My Custom Library Imports:
from archive.litewick import create_candlestick_dataset, identify_trend_reversal
from patterns import define_trend
from eightball import get_final_predictions

# Other imports
import pandas as pd
import yfinance as yf

# Python regs
import datetime
import time


def append_new_line(text_to_append, file_name="screener_log.txt"):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def get_most_recent_weekday(date):
    """
    Accepts date as a datetime.date obj. returns most recent business day if weekend, else returns today.
    """
    new_date = date
    if date.isoweekday() in set((6, 7)):
        new_date += datetime.timedelta(days=-date.isoweekday() + 8)
    if new_date != date:
        new_date -= datetime.timedelta(days=3)
    return new_date


def get_next_weekday(date):
    if date.isoweekday() in set((6, 7)):
        date += datetime.timedelta(days=-date.isoweekday() + 8)
    if date == datetime.datetime.now():
        date += datetime.timedelta(days=1)
    return date


# Some constants
START = "2020-01-01"  # YYYY-MM-DD
END = get_most_recent_weekday(datetime.datetime.today()).strftime(
    "%Y-%m-%d")  # datetime.datetime.now().strftime("%Y-%m-%d")  # YYYY-MM-DD

# Uncomment the lines below if you want to clear the log before screening
file = open("screener_log.txt", "r+")
file.truncate(0)
file.close()

logging_func = append_new_line

logging_func(("Datetime: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Finding volatile stocks that are projected to change
sp500_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = sp500_data[0]
sp500_symbols = df['Symbol'].values.tolist()

allsymbols = sp500_symbols

good_tickers = []
for ticker in allsymbols:
    candles = create_candlestick_dataset(ticker, START, END)
    output, patterns = identify_trend_reversal(candles)
    if output > 0:
        good_tickers.append(ticker)
        df = yf.download(ticker)
        past_dpts = [row[-2] for idx, row in df.iterrows()]
        past_dpts = [(idx, elem) for idx, elem in enumerate(past_dpts)]
        past_dpts = past_dpts[-5:]
        trend = define_trend(past_dpts)
        if trend == "BULLISH":
            logging_func("I am {} confident that {} will have a trend reversal within the next week!".format(
                round(output, 4), ticker))
            logging_func("Before the trend reversal, {} was {}!".format(ticker, trend))
            logging_func("This is because of the following patterns:")
            for pattern in patterns:
                logging_func(pattern)
            logging_func("")
    time.sleep(3)  # delay to prevent the API from overclocking

logging_func("all good tickers" + str(good_tickers))
exit()

for ticker in good_tickers:
    next_business_day = get_next_weekday(
        datetime.datetime.now()).strftime("%Y-%m-%d")
    weighter_res, tr_proba = get_final_predictions(ticker, next_business_day)
    logging_func("Weighted Predictions: " + str(weighter_res))
    logging_func("Probability of Trend Reversal: " + str(tr_proba))
