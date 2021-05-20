# Web Scraping imports:
import requests
from urllib.request import urlopen, Request
import urllib
from bs4 import BeautifulSoup

# Regular Machine Learning imports:
import pandas as pd
import numpy as np

# Some imports for preprocessing:
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from newspaper import Article

# My Machine Learning Comparision imports:
import yfinance as yf

# THE ALL POWERFUL MarketPredictor
import predictor

# Machine learning imports for final classification
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Reg python imports
import heapq
import re
from pprint import pprint
import pickle
import os
import datetime
import time

# For various reasons, I am going to wrap the functionality of this module into a SINGLE
# function. Doing so will allow me to easily add this to my stock predictor class.

def remove_punc(test_str):
    punc = '''!()-[]{};:'"\|,<>./?@#$%^&*_~'''
    for ele in test_str:
        if ele in punc:
            test_str = test_str.replace(ele, "")
    return test_str

def split_into_sentences(text):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    if "e.g." in text: text = text.replace("e.g.","e<prd>g<prd>")
    if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def get_articles(stock_ticker):
    """
    Scraping BEGINS HERE!
    """

    articles = []
    # Getting articles from FinViz, an online resource that automatically chooses
    # relevant articles to a source.
    url_template = "https://finviz.com/quote.ashx?t={}"

    url = url_template.format(stock_ticker)

    req = Request(url, headers={'user-agent':'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, 'html.parser')

    news_table = html.find(id="news-table")
    rows = news_table.find_all('tr')

    for row in rows:
        link = row.a.get('href')
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        xdata = [link, date]
        articles.append(xdata)

    """
    PREPROCESSING BEGINS HERE!
    """
    # Scraping article text for each list.
    article_text_list = []
    for link in articles:
        xlink = link[0]
        if "finance.yahoo" in xlink:
            response = requests.get(xlink).text
            scraper = BeautifulSoup(response, 'html.parser')
            paragraphs = scraper.find('div', class_='caas-body').find_all('p')
            text = ""
            for p in paragraphs:
                text += p.text + " "
        else:
            try:
                a = Article(xlink.strip())
                a.download()
                a.parse()
                text = a.text
            except:
                continue
        article_text_list.append([text, link[1]])

    summaries = []
    # Creating the summaries for every text in the list.
    for xarticle_text in article_text_list:
        article_text = xarticle_text[0]
        date = xarticle_text[1]

        # Doing some basic preprocessing.
        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        # Tokenizing the article text, and storing it in a new variable
        sentence_list = split_into_sentences(article_text)

        # Calculating word frequencies for each word.
        stopwords = nltk.corpus.stopwords.words('english')

        word_frequencies = {}
        for word in nltk.word_tokenize(article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        try:
            # Normalizing the word frequencies
            maximum_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
        except ValueError:
            continue

        # Create a score for each sentence based on the frequency of each word.
        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        # print(sentence_scores)

        # finally creating a summary based on the highest scored words.
        num_sentences = 7
        summary_sentences = heapq.nlargest(
                        num_sentences, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)
        summaries.append([summary, date])
    return summaries

def get_weighted_preds(stock:str):

    """
    This function will be used to calculate weights for a Regressor using
    trending articles about the subject from no later than the past week.
    """

    all_summaries = get_articles(stock)

    res = []
    for i in all_summaries:
        if i not in res:
            res.append(i)

    all_summaries = [x[0] for x in res]
    corresponding_dates = [x[1] for x in res]
    corresponding_dates = [str(datetime.datetime.strptime(str(x), r"%b-%d-%y").strftime("%Y-%m-%d")) for x in corresponding_dates]

    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    for summary in all_summaries:
        summary = summary
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

    keys_to_delete = []
    for key, value in sentiment_dict.items():
        if len(value) == 1:
            keys_to_delete.append(key)
        
    for key in keys_to_delete:
        del sentiment_dict[key]
    del keys_to_delete

    sentiment_avgs = {}
    for key, avg_lst in sentiment_dict.items():
        sentiment_avgs[key] = sum(avg_lst)/len(avg_lst)

    yf_data = yf.download(stock, start=(list(sentiment_avgs)[-1]), end=(list(sentiment_avgs)[0]))
    time.sleep(1)
    yf_data.to_csv("yfdata.csv")

    main_data = pd.read_csv("yfdata.csv")
    main_data = main_data.drop("Volume", axis=1)
    os.remove("yfdata.csv")
    new_headers = ["Date", "RealOpen", "RealHigh", "RealLow", "RealClose", "RealAdjClose"]
    main_data.columns = new_headers
    main_data["Sentiment"] = [None for i in range(main_data.shape[0])]

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

    print(main_data)

    # Time to do a quick "fit-predict" with a model that has a built-in loss function.

    attributes = list(main_data.columns)
    attributes.remove("Date")
    labels = ["RealOpen", "RealHigh", "RealLow", "RealClose", "RealAdjClose"]

    model = LinearRegression()
    model_path = "weight_model.pickle"

    data = main_data

    data = data[attributes]

    predict = labels

    # print(num_test_attrs, test_attrs)

    X = np.array(data.drop(predict, 1))
    y = np.array(data[predict])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    try:
        best = pickle.load(open(model_path, 'rb'))[1]
    except FileNotFoundError:
        best = 0
    
    num_times = 100000000
    for _ in range(num_times):

        linear = model

        linear.fit(X_train, y_train)

        acc = linear.score(X_test, y_test)
        print("Accuracy:", acc)

        if acc > best:
            with open(model_path, 'wb') as f:
                pickle.dump([linear, acc], f)

            best = acc

    print("Best Acc:", best)

    # print(best_acc_str)
    with open(model_path, 'rb') as pickle_in:
        linear = pickle.load(pickle_in)[0]
    model = linear

if __name__ == '__main__':
    print(get_weighted_preds("TSLA"))
