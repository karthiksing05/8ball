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

# sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# plot Imports
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Reg python imports
from pprint import pprint
import pickle
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
    sentiment_today = sentiment_avgs[date_today]
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
    main_data["{}DaySentiment".format(daysfuture)] = [0 for i in range(main_data.shape[0])]


    for sentdate, sentiment_avg in sentiment_avgs.items():
        try:
            condition = main_data["Date"] == sentdate
            index = main_data.index[condition] + daysfuture
            main_data.at[index, "{}DaySentiment".format(daysfuture)] = sentiment_avg
        except:
            pass

    main_data = main_data.dropna()
    for idx, row in main_data.iterrows():
        if float(list(row)[-1]) == 0.0:
            main_data = main_data.drop(idx, axis=0)

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

    main_data.to_csv("IMPORTANT_DO_NOT_DEL.csv")
    
    return main_data, sentiment_today

def find_correlation_by_sentiment(weight_dataset: pd.DataFrame, ticker: str, sentiment_ftr: float, date: str, daysfuture: int):

    main_data = weight_dataset
    fields = ["Open", "High", "Low", "Close", "AdjClose"]

    scaled_data = pd.DataFrame()
    scaler = MinMaxScaler()
    for col in main_data.columns:
        if (col != "Date") and (col != "{}DaySentiment".format(daysfuture)) and (col != "Unnamed: 0"):
            scaled_data[col] = scaler.fit_transform(
                main_data[col].values.reshape(-1, 1))
        else:
            scaled_data[col] = main_data[col]
    
    df = scaled_data

    model_dict = {}

    for field in fields:
        attributes = ["Predicted"+field, "{}DaySentiment".format(daysfuture)]
        predict = ["Real"+field]
        all_cols = attributes
        all_cols.extend(predict)
        model = LinearRegression()
        data = df[all_cols]

        X = data.drop(predict, 1).values
        y = data[predict].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        model = model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        model_dict[field] = [model, rmse]

    results = []
    mp = predictor.MarketPredictor(ticker)
    mp.load_data()
    mp.fit_inital()
    pred_df, ranges = mp.predict(date)
    for field, lst in model_dict.items():
        model = lst[0]        
        ftr = np.array([pred_df['Output Values'][fields.index(field)], sentiment_ftr]).reshape(1, -1)
        scaled_ftr = np.array([scaler.transform(xftr.reshape(-1, 1)) for xftr in ftr]).reshape(1, -1)
        result = model.predict(scaled_ftr)
        result = scaler.inverse_transform(np.array(result).reshape(-1, 1))
        results.append(result[0][0])

    print(results)

if __name__ == '__main__':
    stock = "AMZN"
    daysfuture = 3
    data, test_sent_today = create_weight_dataset(stock, daysfuture)
    data = pd.read_csv("IMPORTANT_DO_NOT_DEL.csv")
    date_to_predict = datetime.datetime(datetime.datetime.now() + datetime.timedelta(days=daysfuture)).strftime("%Y-%m-%d")
    # test_sent_today = 0.5012249999999999
    # print(data)
    x = "Unnamed: 0"
    y = "RealHigh"
    z = "PredictedHigh"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[x], data[y], data[z])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    # plt.show()
    find_correlation_by_sentiment(data, stock, test_sent_today, date_to_predict, daysfuture)
