# Web Scraping imports:
import requests
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

# PyGoogleNews API
from pygooglenews import GoogleNews

# Tweepy API
import tweepy

# Some imports for preprocessing:
import nltk
from newspaper import Article

# Reg python imports
import heapq
import re
import pickle
from pprint import pprint
import datetime
import time

def remove_punc(test_str):
    punc = '''!()-[]{};:'"\|,<>./?@#$%^&*_~'''
    for ele in test_str:
        if ele in punc:
            test_str = test_str.replace(ele, "")
    return test_str

def split_into_sentences(text:str):
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

def get_articles_finviz(stock_ticker:str):
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

    articles = [[article[0], str(datetime.datetime.strptime(str(article[1]), r"%b-%d-%y").strftime("%Y-%m-%d"))] for article in articles]

    return articles

def get_articles_yahoo(stock_ticker:str):
    current_datetime = datetime.datetime.now()
    headers = {
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'referer': 'https://www.google.com',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
    }

    def get_article(card):
        """Extract article information from the raw html"""
        posted = card.find('span', 's-time').text.replace('·', '').strip()
        raw_link = card.find('a').get('href')
        unquoted_link = requests.utils.unquote(raw_link)
        pattern = re.compile(r'RU=(.+)\/RK')
        clean_link = re.search(pattern, unquoted_link).group(1)
        
        article = (clean_link, posted)
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
            time.sleep(1)
        except AttributeError:
            break

    formatted_articles = []
    for link, time_ago in articles:
        if "hours" in time_ago:
            print("Hours value") # debugging
            time_ago = int(time_ago.split(" hours")[0])
            time_ago = datetime.timedelta(hours=time_ago)
        elif "days" in time_ago:
            print("Days value") # debugging
            time_ago = int(time_ago.split(" days")[0])
            time_ago = datetime.timedelta(days=time_ago)
        else:
            continue
    
        actual_time_ago = (current_datetime - time_ago).strftime("%Y-%m-%d")
        formatted_articles.append([link, actual_time_ago])
    return formatted_articles

def get_articles_google(stock_ticker:str):
    gn_obj = GoogleNews()

    response = gn_obj.search(stock_ticker)

    linkdates = []

    newsitems = response['entries']
    for item in newsitems:
        link = item['link']
        full_date = item['published']
        date = str(full_date[5:16])
        date = datetime.datetime.strptime(date, "%d %b %Y").strftime("%Y-%m-%d")
        linkdates.append([link, date])

    return linkdates

def get_news_summaries(linkdates:list):
    """
    PREPROCESSING BEGINS HERE!
    """
    # Scraping article text for each list.
    article_text_list = []
    for link in linkdates:
        xlink = link[0]
        if "finance.yahoo" in xlink:
            try:
                response = requests.get(xlink).text
                scraper = BeautifulSoup(response, 'html.parser')
                paragraphs = scraper.find('div', class_='caas-body')
                paragraphs = paragraphs.find_all('p')
                text = ""
                for p in paragraphs:
                    text += p.text + " "
            except AttributeError:
                continue
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

def get_tweets(stock_ticker:str):
    """
    This function returns a list of lists. Each list contains a tweet and the date of the tweet.
    This function uses the Tweepy API to get twitter Results.
    """

    with open('tweepy.pickle', 'rb') as f:
        (CONSUMER_API_KEY, CONSUMER_API_SECRET, ACCESS_TOKEN, ACCESS_SECRET, BEARER_TOKEN) = pickle.load(f)

    auth = tweepy.OAuthHandler(CONSUMER_API_KEY, CONSUMER_API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)

    keyword = stock_ticker
    list_of_raw_tweets = api.search(q=keyword, lang='en', count=1000)
    list_of_tweets = []
    tweet_texts = []
    for tweet in list_of_raw_tweets:
        timestamp = str(tweet._json['created_at'])
        year = timestamp[-4:]
        month = timestamp[4:7]
        day = timestamp[8:10]
        dt_str = "{}-{}-{}".format(month, day, year)
        dt = datetime.datetime.strptime(dt_str, r"%b-%d-%Y").strftime("%Y-%m-%d")
        text = tweet._json['text'].rstrip()
        if text not in tweet_texts:
            tweet_texts.append(text)
        else:
            continue
        list_of_tweets.append([dt, text])

    return list_of_tweets

if __name__ == '__main__':
    print(get_tweets("TSLA"))
