
# My imports
import yfinance as yf

# Web Scraping imports:
import requests
import urllib.request
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

# Regular Machine Learning imports/abbreviations:
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# The model: 
from sklearn.cluster import KMeans

# Some imports for preprocessing:
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import textblob
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.stem import WordNetLemmatizer

from newspaper import Article


# Reg python imports
from pprint import pprint
import heapq
import datetime
import random
import pickle
import re
import time
import csv
import sys
import os

# For various reasons, I am going to wrap the functionality of this module into a SINGLE
# function. Doing so will allow me to easily add this to my stock predictor class.

np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)

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
    ###########################
    # parsing data from yahoo finance
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
        title = row.a.text
        link = row.a.get('href')
        articles.append(link)

    """
    PREPROCESSING BEGINS HERE!
    """
    # Scraping article text for each list.
    article_text_list = []
    for link in articles:
        xlink = link
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
        article_text_list.append(text)

    summaries = []
    # Creating the summaries for every text in the list.
    for article_text in article_text_list:

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
        summaries.append(summary)
    return summaries

def get_weights(stock):

    """
    This function will be used to calculate weights for a KNeighborsRegressor using
    trending articles about the subject from no later than the past week.
    """

    obj = yf.Ticker(stock)
    lemma = WordNetLemmatizer()

    description = obj.info['longBusinessSummary']

    """with open("sample_summaries.pickle", "rb") as f:
        all_summaries = pickle.load(f)"""

    all_summaries = get_articles(stock)

    pos_summaries = []
    neg_summaries = []

    with open("textindex\\positive-words.txt", "r") as f:
        xpos_words = f.readlines()

    with open("textindex\\negative-words.txt", "r") as f:
        xneg_words = f.readlines()
    
    penalty = 1/len(all_summaries)
    pos_words = [word.strip() for word in xpos_words]
    neg_words = [word.strip() for word in xneg_words]

    for summary in all_summaries:
        score = 0
        summary_words = summary.split()
        for word in summary_words:
            if word in pos_words:
                score = score + penalty
            elif word in neg_words:
                score = score - penalty
            else:
                continue
        if score >= 0.1:
            pos_summaries.append(summary)
        elif score < 0.1:
            neg_summaries.append(summary)

    for summary in all_summaries:
        pass


if __name__ == '__main__':
    start = time.time()
    stock = "TSLA"
    get_weights(stock)
    end = time.time()
    print("Finished in {}.".format(datetime.timedelta(seconds=(end - start))))
