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

# Reg python imports
import heapq
import re
import os
import csv
from time import sleep

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
    try:
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
            articles.append(link)
    except urllib.error.HTTPError:
        ###################################
        # parsing data from yahoo finance #
        ###################################
        headers = {
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'referer': 'https://www.google.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
        }

        def get_article(card):
            """Extract article information from the raw html"""
            headline = card.find('h4', 's-title').text
            source = card.find("span", 's-source').text
            posted = card.find('span', 's-time').text.replace('·', '').strip()
            description = card.find('p', 's-desc').text.strip()
            raw_link = card.find('a').get('href')
            unquoted_link = requests.utils.unquote(raw_link)
            pattern = re.compile(r'RU=(.+)\/RK')
            clean_link = re.search(pattern, unquoted_link).group(1)
            
            article = (headline, source, posted, description, clean_link)
            return article

        """Run the main program"""
        template = 'https://news.search.yahoo.com/search?p={}'
        url = template.format(stock_ticker)
        articles = []
        links = set()
        
        while True:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            cards = soup.find_all('div', 'NewsArticle')
            
            # extract articles from page
            for card in cards:
                article = get_article(card)
                link = article[-1]
                if not link in links:
                    links.add(link)
                    articles.append(article)        
                    
            # find the next page
            try:
                url = soup.find('a', 'next').get('href')
                sleep(1)
            except AttributeError:
                break
                
        # save article data
        with open('results.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Headline', 'Source', 'Posted', 'Description', 'Link'])
            writer.writerows(articles)
            
        articles = pd.read_csv('results.csv')
        os.remove('results.csv')
        articles = articles['Link']

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

def get_weight(stock:str):

    """
    This function will be used to calculate weights for a Regressor using
    trending articles about the subject from no later than the past week.
    """

    CONST1 = -1
    EX1 = 0.5

    all_summaries = get_articles(stock)

    res = []
    for i in all_summaries:
        if i not in res:
            res.append(i)
    
    all_summaries = res

    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    for summary in all_summaries:
        sentiment = analyzer.polarity_scores(summary)['compound']
        # Note that Compound scores are ranged from -1 to 1, 
        # with -1 being total negativity, 0 being neutrality, and 1 being total positivity
        sentiments.append(sentiment)
    data = pd.DataFrame()
    data['summary'] = all_summaries
    data['polarity'] = sentiments

    polarity_lst = []
    for row in data.iterrows():
        row = list(row[1])
        polarity = row[1] + CONST1
        polarity_lst.append(polarity)
    
    avg_sentiment = (sum(polarity_lst) / len(polarity_lst))

    weight = avg_sentiment

    return weight

if __name__ == '__main__':
    print(get_weight("TSLA"))
