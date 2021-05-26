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

def get_weighted_preds(stock:str):

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

    yf_data = yf.download(stock, start=(list(sorted(sentiment_avgs))[0]), end=(list(sorted(sentiment_avgs))[-1]))
    time.sleep(1)
    yf_data.to_csv("yfdata.csv")
    main_data = pd.read_csv("yfdata.csv")
    main_data = main_data.drop("Volume", axis=1)
    os.remove("yfdata.csv")
    new_headers = ["Date", "RealOpen", "RealHigh", "RealLow", "RealClose", "RealAdjClose"]
    main_data.columns = new_headers
    main_data["Sentiment"] = [0 for i in range(main_data.shape[0])]

    for sentdate, sentiment_avg in sentiment_avgs.items():
        condition = main_data["Date"] == sentdate
        index = main_data.index[condition]
        main_data.at[index, 'Sentiment'] = sentiment_avg

    dates_to_predict = [date for date in main_data["Date"]]
    all_predictions = {}
    for preddate in dates_to_predict:
        special_end_date = str(datetime.datetime.strptime(preddate, r"%Y-%m-%d") - datetime.timedelta(days=1))[:-9]
        mp = predictor.MarketPredictor(stock, end_time=special_end_date)
        mp.load_data()
        mp.fit_inital()
        preds = mp.predict(preddate)
        preds = list(preds['Output Values'])
        all_predictions[preddate] = preds

    pred_headers = ["PredictedOpen", "PredictedHigh", "PredictedLow", "PredictedClose", "PredictedAdjClose"]

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

    # print(main_data)

    # Scale the data here!
    scaled_data = pd.DataFrame()
    scaler = MinMaxScaler()
    for idx, col in enumerate(main_data.columns):
        if col != "Date":
            scaled_data[col] = scaler.fit_transform(main_data[col].values.reshape(-1, 1))
        else:
            scaled_data[col] = main_data[col]

    # print(scaled_data)

    # Time to do a quick fit with some models

    data = scaled_data
    attributes = list(data.columns)
    try:
        attributes.remove("Date")
    except ValueError:
        pass
    try:
        attributes.remove("Unnamed: 0")
    except ValueError:
        pass
    labels = ["RealOpen", "RealHigh", "RealLow", "RealClose", "RealAdjClose"]
    data = data[attributes]

    num_test_attrs = len([x for x in attributes if x not in labels])

    predict = labels

    # print(num_test_attrs, test_attrs)

    X = np.array(data.drop(predict, 1))
    y = np.array(data[predict])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    increment = 0.01
    parameters = {
                'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                'l1_ratio':np.arange(0.0, 1.0, 0.01).tolist()
                }
        
    rsc = RandomizedSearchCV(
                estimator=ElasticNet(),
                param_distributions=parameters,
                cv=3, 
                scoring='neg_mean_squared_error', 
                verbose=1,
                n_jobs=1
            )

    rxgb_grid = rsc
    rresults = rxgb_grid.fit(X_train, y_train)

    rbest_params = rresults.best_params_
    with open("best_params.txt", "w") as f:
        f.write(str(rbest_params))

    linear = rresults.best_estimator_ # best estimator

    # now to predict today's prices with weight values!

    final_mp = predictor.MarketPredictor(stock)
    final_mp.load_data()
    final_mp.fit_inital()
    preds = final_mp.predict(date_today)
    preds = [preds["Output Values"][x] for x in range(0, 4+1)]

    values_to_predict = preds
    values_to_predict.append(sentiment_today)

    my_values = np.array(values_to_predict)
    my_values = scaler.transform(my_values.reshape(-1, 1))
    my_values = my_values.reshape(-1, num_test_attrs)

    predictions = linear.predict(my_values)
    output_values = scaler.inverse_transform(predictions)
    output_values = list(output_values)
    print(output_values)
    os.remove("weight_models\\model.pickle")


if __name__ == '__main__':
    print(get_weighted_preds("TSLA"))
