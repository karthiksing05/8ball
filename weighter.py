# Regular Machine Learning imports:
import pandas as pd
import numpy as np

# Some imports for preprocessing:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# My Machine Learning Comparision imports:
import yfinance as yf

# My imports
import predictor
import scraper

# Machine learning imports for final classification
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Reg python imports
from pprint import pprint
import pickle
import os
import datetime
import time

# For various reasons, I am going to wrap the functionality of this module into a SINGLE
# function. Doing so will allow me to easily add this to my stock predictor class.


def create_weight_dataset(stock: str):
    """
    This function will be used to calculate weights for a Regressor using
    trending articles about the subject from no later than the past week.
    """

    date_today = datetime.datetime.now().strftime("%Y-%m-%d")

    linkdates = scraper.get_articles_google(stock)

    all_summaries = scraper.get_summaries(linkdates)

    res = all_summaries

    all_summaries = [x[0] for x in res]
    corresponding_dates = [x[1] for x in res]

    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    for summary in all_summaries:
        sentiment = analyzer.polarity_scores(summary)['compound']
        # Note that Compound scores are ranged from -1 to 1,
        # with -1 being total negativity, 0 being neutrality, and 1 being total positivity
        sentiments.append(sentiment)
    data = pd.DataFrame()
    data['date'] = corresponding_dates
    data['summary'] = all_summaries
    data['polarity'] = sentiments

    sentiment_dict = {}
    for idx, row in data.iterrows():
        date, summary, polarity = list(row)
        if date not in sentiment_dict:
            sentiment_dict[date] = [polarity]
        else:
            sentiment_dict[date].append(polarity)

    sentiment_avgs = {}
    for key, avg_lst in sentiment_dict.items():
        sentiment_avgs[key] = sum(avg_lst)/len(avg_lst)

    # print(sentiment_avgs)

    sentiment_today = sentiment_avgs[date_today]

    # print(sentiment_today)

    if date_today in list(sentiment_avgs):
        del sentiment_avgs[date_today]

    yf_data = yf.download(
        stock, 
        start=(list(sorted(sentiment_avgs))[0]), 
        end=(list(sorted(sentiment_avgs))[-1])
    )
    time.sleep(1)
    yf_data.to_csv("yfdata.csv")
    main_data = pd.read_csv("yfdata.csv")
    main_data = main_data.drop("Volume", axis=1)
    os.remove("yfdata.csv")
    new_headers = [
        "Date", 
        "RealOpen", 
        "RealHigh",
        "RealLow", 
        "RealClose", 
        "RealAdjClose"
    ]
    main_data.columns = new_headers
    main_data["Sentiment"] = [0 for i in range(main_data.shape[0])]

    for sentdate, sentiment_avg in sentiment_avgs.items():
        condition = main_data["Date"] == sentdate
        index = main_data.index[condition]
        main_data.at[index, 'Sentiment'] = sentiment_avg

    dates_to_predict = [date for date in main_data["Date"]]
    all_predictions = {}
    for preddate in dates_to_predict:
        special_end_date = str(datetime.datetime.strptime(
            preddate, r"%Y-%m-%d") - datetime.timedelta(days=1))[:-9]
        mp = predictor.MarketPredictor(stock, end_time=special_end_date)
        mp.load_data()
        mp.fit_inital()
        preds, ranges = mp.predict(preddate)
        preds = list(preds['Output Values'])
        all_predictions[preddate] = preds

    pred_headers = ["PredictedOpen", "PredictedHigh",
                    "PredictedLow", "PredictedClose", "PredictedAdjClose"]

    main_data[pred_headers[0]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[1]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[2]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[3]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[4]] = [None for i in range(main_data.shape[0])]

    for ndate, preds in all_predictions.items():
        condition = main_data["Date"] == ndate
        xindex = main_data.index[condition]
        for idx, pred in enumerate(preds):
            main_data.at[xindex, pred_headers[idx]] = pred

    # Scale the data here!
    scaled_data = pd.DataFrame()
    scaler = MinMaxScaler()
    for idx, col in enumerate(main_data.columns):
        if (col != "Date") and (col != "Sentiment"):
            scaled_data[col] = scaler.fit_transform(
                main_data[col].values.reshape(-1, 1))
        else:
            scaled_data[col] = main_data[col]

    return scaled_data


if __name__ == '__main__':
    print(create_weight_dataset("TSLA"))
