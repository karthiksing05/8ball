# Regular Machine Learning imports:
from numpy.core.numeric import count_nonzero
import pandas as pd
import numpy as np

# Some imports for preprocessing:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# My Machine Learning Comparision imports:
import yfinance as yf

# My imports
import predictor
import scraper

# sklearn ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# BiasRegressors!!!
from biaswrappers.regressor import BiasRegressorC1, BiasRegressorC2

# Multiprocessing import; to make the program run faster
import multiprocessing as mp

# Reg python imports
import datetime
import pickle
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# helper function for getting the model names for sklearn
def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]

# helper function for pool
def predict_date(params):
    """
    Wrapper function for multiprocessing; fits and predicts data for a given date
    """
    pool_filename = "pool_data\\data{}.pickle"
    preddate = params[0]
    clone_id = params[1]
    stock = params[2]
    special_end_date = str(datetime.datetime.strptime(
        preddate, r"%Y-%m-%d") - datetime.timedelta(days=1))[:-9]
    mp = predictor.MarketPredictor(stock, clone_id=clone_id, end_date=special_end_date)
    mp.load_data()
    mp.fit_inital()
    preds = mp.predict(preddate)
    preds = list(preds['Output Values'])
    with open(pool_filename.format(clone_id), "wb") as f:
        pickle.dump([preddate, preds], f)
    mp.delete_datasets()

def xBRs(model, numRegs:int):
    for _ in range(numRegs):
        model = BiasRegressorC1(model)
    return model

def create_weight_dataset(stock: str, daysfuture:int):
    """
    This function will be used to calculate weights to be used as features using
    trending articles about the subject from up to the past 3 months
    """

    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    # date_today = "2023-12-28"

    linkdates = scraper.get_articles_google(stock)
    
    all_summaries = scraper.get_news_summaries(linkdates)
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
        majmin = []
        for item in avg_lst:
            if item > 0.5:
                majmin.append(1)
            else:
                majmin.append(0)
        sentiment_avgs[key] = count_nonzero(majmin) / len(majmin)
        del majmin

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

    args = list(zip(
            dates_to_predict, 
            [x for x in range(len(dates_to_predict))],
            [stock for x in range(len(dates_to_predict))]
        ))

    # multiprocessing functions
    pool = mp.Pool(round((mp.cpu_count() - 2)))
    pool.map(
        predict_date, 
        args
    )

    # final dataframe
    pred_headers = ["PredictedOpen", "PredictedHigh",
                    "PredictedLow", "PredictedClose", "PredictedAdjClose"]

    main_data[pred_headers[0]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[1]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[2]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[3]] = [None for i in range(main_data.shape[0])]
    main_data[pred_headers[4]] = [None for i in range(main_data.shape[0])]

    all_predictions = {}
    for directory in os.listdir("pool_data"):
        with open("pool_data\\{}".format(directory), "rb") as f:
            date_and_preds = pickle.load(f)
            all_predictions[date_and_preds[0]] = date_and_preds[1]

    for ndate, preds in all_predictions.items():
        condition = main_data["Date"] == ndate
        xindex = main_data.index[condition]
        for idx, pred in enumerate(preds):
            main_data.at[xindex, pred_headers[idx]] = pred
    
    return main_data, sent_ftr

def find_correlation_by_sentiment(
            weight_dataset: pd.DataFrame, 
            ticker: str, 
            sentiment_ftr: float, 
            date: str, 
            daysfuture: int
        ):
    """
    Using the dataset created from _create_weight_dataset, this function will weight
    predictions created by the MarketPredictor from the predictor file.
    """

    # for directory in os.listdir("pool_data"):
    #     os.remove("pool_data\\" + directory)

    main_data = weight_dataset
    fields = ["Open", "High", "Low", "Close", "AdjClose"]

    df = main_data
    newCols = []
    for col in list(df.columns):
        newCols.append(col.strip())

    df.columns = newCols

    mp = predictor.MarketPredictor(ticker)
    mp.load_data()
    mp.fit_inital()
    pred_df = mp.predict(date)
    raw_results = [pred_df['Output Values'][x] for x in range(5)]

    print("Raw Results:")
    print(raw_results)
    print()

    attributes = []
    predict = []

    for field in fields:
        attributes += [str("Predicted"+field)]
        predict += [str("Real"+field)]
    
    attributes.append("{}DaySentiment".format(daysfuture))

    all_cols = attributes
    all_cols.extend(predict)
    
    data = pd.DataFrame()
    for col in all_cols:
        data[col.strip()] = df[col.strip()]

    X = data.drop(predict, 1).values
    y = data[predict].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    
    reg_models = [BiasRegressorC1(), BiasRegressorC2()]

    def rms_error(actual, predicted):
        mse = mean_squared_error(actual, predicted)
        return np.sqrt(mse)

    scores = {}
    for model in reg_models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        while len(list(preds.shape)) > 2:
            preds = preds[0]

        key = get_model_name(model)
        try:
            key += str(model.model.get_params().get('max_depth', ""))
            key += str(model.model.get_params().get('n_neighbors', ""))
        except:
            pass
        if get_model_name(model)[:-2] == "BiasRegressor":
            key += " with inner model " + get_model_name(model.model)
            key += str(model.model.get_params().get('max_depth', ""))
            key += str(model.model.get_params().get('n_neighbors', ""))
        
        scores[key] = [rms_error(y_test, preds), model]

    modeldf = pd.DataFrame.from_dict(scores, orient='index').sort_values(0)
    modeldf.columns = ['RMSE', 'MODEL_CLASS']

    dfd = modeldf.to_dict()
    best_model = str(min(dfd["RMSE"], key=dfd["RMSE"].get))
    model = dfd['MODEL_CLASS'][best_model]
    print("Best Model for weighted fit: {}".format(best_model))

    br = xBRs(model, 10)

    final_X = np.array([list((raw_results + [sentiment_ftr]))])

    preds = br.predict(final_X)
    while len(list(preds.shape)) > 1:
        preds = preds[0]

    return preds
