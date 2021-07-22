# My Imports
from weighter import create_weight_dataset, find_correlation_by_sentiment
from litewick import create_candlestick_dataset, identify_trend_reversal

# Standard Python imports
import datetime
import pickle
import os

def get_final_predictions(stock: str, date_to_predict: str) -> tuple:
    """

    Takes two params:
    - stock -> the ticker of your chosen company
    - date_to_predict -> the date whose prices you would like to predict, taken in the form of
                          YYYY-MM-DD

    A function that first, creates a dataset of predictions, sentiment that day, and the actual prices.
    Then, it uses that dataset to modify the predictions such that they are calibratated with the
    sentiment.

    This function is a quick wrapper for the two methods in the weighter.py file, as well as the 
    predictor.py file's predictor.

    This will get the final, fully weighted and calibrated predictions, which will then be used 
    with Litewick to create a trading bot.
    """

    today = datetime.datetime.now()
    pred_datetime = datetime.datetime.strptime(date_to_predict, "%Y-%m-%d")
    delta = (pred_datetime - today)
    daysfuture = delta.days
    daysfuture += 1
    if daysfuture <= 0:
        raise ValueError("You have predicted a date in the past/today. To use this function, please predict a date in the future.")

    start = (pred_datetime - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    end = (pred_datetime - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if not os.path.exists("benchmark.pickle"):
        data, test_sent_today = create_weight_dataset(stock, daysfuture)
        with open('benchmark.pickle', 'wb') as f:
            pickle.dump([today.strftime("%Y-%m-%d"), [data, test_sent_today]], f)
    else:
        with open('benchmark.pickle', 'rb') as f:
            pickled_info = pickle.load(f)
        dumped_date = pickled_info[0]
        data = pickled_info[1][0]
        test_sent_today = pickled_info[1][1]

        if dumped_date != today.strftime("%Y-%m-%d"):
            data, test_sent_today = create_weight_dataset(stock, daysfuture)
    
    final_preds = find_correlation_by_sentiment(
        data, 
        stock, 
        test_sent_today, 
        date_to_predict, 
        daysfuture
    )

    candles = create_candlestick_dataset(stock, start, end)
    proba, patterns = identify_trend_reversal(candles)

    if proba <= 0:
        return (final_preds, 0)
    else:
        return (final_preds, 1)
