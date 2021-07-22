# Regular Machine Learning imports:
from numpy.testing._private.utils import break_cycles
import pandas as pd
import numpy as np

# Some imports for preprocessing:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# My Machine Learning Comparision imports:
import yfinance as yf

# My imports
import predictor
import scraper

# sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# BiasRegressor!!!
from biaswrappers.regressor import BiasRegressor

# Reg python imports
from pprint import pprint
import simplejson
import os
import datetime
import time

# For various reasons, I am going to wrap the functionality of this module into a SINGLE
# function. Doing so will allow me to easily add this to my stock predictor class.
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def create_weight_dataset(stock: str, daysfuture:int):
    """
    This function will be used to calculate weights for a BiasRegressor using
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
    datetimes = [datetime.datetime.strptime(date[0], "%Y-%m-%d") for date in sentiment_avgs.items()]
    youngest_date = max(datetimes).strftime("%Y-%m-%d")
    sent_ftr = sentiment_avgs[youngest_date]
    daysfuture += int(np.busday_count(youngest_date, date_today))
    if date_today in list(sentiment_avgs):
        del sentiment_avgs[date_today]

    yf_data = None

    yf_data = yf.download(
        stock,
        threads=False,
        start=(list(sorted(sentiment_avgs))[0]),
        end=(list(sorted(sentiment_avgs))[-1])
    )

    main_data = yf_data
    main_data = main_data.drop("Volume", axis=1)
    main_data.reset_index(level=0, inplace=True)
    main_data["Date"] = main_data["Date"].dt.strftime('%Y-%m-%d')
    new_headers = [
        "Date", 
        "RealOpen", 
        "RealHigh",
        "RealLow", 
        "RealClose", 
        "RealAdjClose"
    ]
    main_data.columns = new_headers
    main_data["{}DaySentiment".format(daysfuture)] = [0 for i in range(main_data.shape[0])]

    for sentdate, sentiment_avg in sentiment_avgs.items():
        try:
            condition = main_data["Date"] == sentdate
            index = main_data.index[condition] - daysfuture
            main_data.at[index, "{}DaySentiment".format(daysfuture)] = sentiment_avg
        except:
            pass

    dates_to_predict = [date for date in main_data["Date"]]
    all_predictions = {}
    for preddate in dates_to_predict:
        special_end_date = str(datetime.datetime.strptime(
            preddate, r"%Y-%m-%d") - datetime.timedelta(days=1))[:-9]
        mp = predictor.MarketPredictor(stock, end_time=special_end_date)
        mp.load_data()
        mp.fit_inital()
        preds = mp.predict(preddate)
        preds = list(preds['Output Values'])
        all_predictions[preddate] = preds
        time.sleep(3)

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
    
    return main_data, sent_ftr

def find_correlation_by_sentiment(weight_dataset: pd.DataFrame, ticker: str, sentiment_ftr: float, date: str, daysfuture: int):
    """
    Using the dataset created from _create_weight_dataset, this function will weight
    predictions created by the MarketPredictor from the predictor file.
    """

    main_data = weight_dataset
    fields = ["Open", "High", "Low", "Close", "AdjClose"]

    df = main_data

    model_dict = {}

    for field in fields:
        attributes = ["Predicted"+field, "{}DaySentiment".format(daysfuture)]
        predict = ["Real"+field]
        all_cols = attributes
        all_cols.extend(predict)
        inside_model = Ridge()
        model = BiasRegressor(model=inside_model)
        data = df[all_cols]

        X = data.drop(predict, 1).values
        y = data[predict].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)[0]
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        model_dict[field] = [model, rmse]

    results = []
    mp = predictor.MarketPredictor(ticker)
    mp.load_data()
    mp.fit_inital()
    pred_df = mp.predict(date)
    results = pred_df

    return results