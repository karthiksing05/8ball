# Wit.AI Import, to categorize tweets
from wit_clone import Wit

# Dataset preprocessing imports
import pandas as pd

# Scraper for Tweepy API
from scraper import get_tweets

# Reg python Imports
import os
from pprint import pprint
import json
import datetime

ACCESS_TOKEN = "WNGWY6Z443CL6NBBIMJMZ74CVKRMRO4T"

def post_http_req(text, sentiment):

    fp = "data\\json_wit_req.json"

    current_date = datetime.datetime.now()
    current_date = "{}{}{}".format(current_date.year, current_date.month, current_date.day)

    with open(fp, "rb") as f:
        json_dict = json.load(f)

    json_dict[0]["text"] = text[0:280]
    json_dict[0]["intent"] = sentiment

    json_file = json.dumps(json_dict)
    with open(fp, "w") as f:
        f.write(json_file)

    req = """curl -XPOST \"https://api.wit.ai/utterances?v={}\" -H "Authorization: Bearer {}" -H "Content-Type: application/json" -d @{}""".format(current_date, ACCESS_TOKEN, fp)
    
    result = os.popen(req).read()
    return result

def analyze_tweets(stock_ticker: str, debug_attr=False):
    """
    Analyzes tweets and returns a sentiment value given a stock's ticker symbol.
    """

    Client = Wit(ACCESS_TOKEN)

    def get_sentiment(tweet):
        try:
            resp = Client.message(tweet)
            sentiment = str(resp['intents'][0]['name'])
        except:
            sentiment = "oos"
        return sentiment

    recent_tweets = get_tweets(stock_ticker)

    """train_data = pd.read_csv("data\\sorted_tweets.csv")
    train_data = train_data.drop("id", axis=1)

    new_dates = []
    for date in train_data["date"]:
        date = date.split(" ")[0]
        new_dates.append(date)
    train_data["date"] = new_dates
    for idx, row in train_data.iterrows():
        text = list(row)[1]
        intent = list(row)[2]
        debug = post_http_req(text, intent)
        time.sleep(1)
        if debug_attr:
            print(idx)
            print(debug)
            print("\n")"""

    for tweet in recent_tweets:
        sentiment = get_sentiment(tweet[1])
        if sentiment != "oos":
            tweet.append(sentiment)
        else:
            recent_tweets.remove(tweet)

    for tweet in recent_tweets:
        if len(tweet) == 3:
            pass
        else:
            recent_tweets.remove(tweet)

    main_df = pd.DataFrame()
    
    main_df["Date"] = [date for date in [tweet[0] for tweet in recent_tweets]]
    main_df["Text"] = [text for text in [tweet[1] for tweet in recent_tweets]]
    sentiments = []
    for tweet in recent_tweets:
        sentiments.append(tweet[2])

    main_df["Sentiment"] = sentiments

    tweet_spread = [0, 0, 0] # for pos, neu, neg

    for idx, row in main_df.iterrows():
        if row[2] == "negative":
            tweet_spread[2] += 1
        elif row[2] == "positive":
            tweet_spread[0] += 1
        else:
            tweet_spread[1] += 1

    

if __name__ == '__main__':
    analyze_tweets("TSLA")
