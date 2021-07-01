from predictor import MarketPredictor
from threading import Thread
from yahoo_fin import stock_info as si
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import datetime, pickle, time, os, sys

np.set_printoptions(threshold=(sys.maxsize))
pd.set_option('display.max_rows', None, 'display.max_columns', None)

def round_down(n):
    return int(str(n).split('.')[0])

def get_price(ticker):
    return si.get_live_price(ticker)

def write_to_log(text_to_append):
    """Append given text as a new line at the end of file"""
    file_name = 'balancelog.txt'
    with open(file_name, 'a+') as (file_object):
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write('\n')
        file_object.write(text_to_append)


def predictor_trade(ticker, num_stocks_to_buy):
    now = datetime.datetime.now()
    START = datetime.datetime(year=(now.year), month=(now.month), day=(now.day), hour=9, minute=30, second=0)
    END = datetime.datetime(year=(now.year), month=(now.month), day=(now.day), hour=16, minute=0, second=0)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    sp = MarketPredictor(ticker)
    sp.load_data()
    # sp.pick_model(show=False, test_size=0.15)
    sp.fit_inital()
    preds, ranges = sp.predict(datetime.datetime.now().strftime('%Y-%m-%d'))
    print('Predictions: ')
    print(preds)
    PRED_HIGH = preds['Output Values'][1]
    PRED_LOW = preds['Output Values'][2]
    BALANCE_FILENAME = 'test_balance.pickle'
    have_stocks = False
    try:
        with open(BALANCE_FILENAME, 'rb') as (f):
            balance = pickle.load(f)
    except FileNotFoundError:
        balance = 100000
    current_time = datetime.datetime.now()
    write_to_log("Bot Started trading for day {0} on {1}'s stocks.".format(today, ticker))
    write_to_log('Balance: {0}'.format(balance))
    while True:
        current_time = datetime.datetime.now()
        between_start_and_end = current_time >= START and current_time <= END
        if between_start_and_end:
            try:
                current_time = datetime.datetime.now()
                between_start_and_end = current_time >= START and current_time <= END
                current_price = get_price(ticker)
                if current_time.second == 0:
                    print(current_price)
                if not have_stocks:
                    if round_down(current_price) == round_down(PRED_LOW):
                        balance -= num_stocks_to_buy * current_price
                        print('Smart Trader Bought {0} stocks for {1} each'.format(num_stocks_to_buy, current_price))
                        write_to_log('Smart Trader Bought {0} stocks for {1} each'.format(num_stocks_to_buy, current_price))
                        have_stocks = True
                if have_stocks:
                    if round_down(current_price) == round_down(PRED_HIGH):
                        balance += num_stocks_to_buy * current_price
                        print('Smart Trader Sold {0} stocks for {1} each'.format(num_stocks_to_buy, current_price))
                        write_to_log('Smart Trader Sold {0} stocks for {1} each'.format(num_stocks_to_buy, current_price))
                        have_stocks = False
            except ValueError:
                pass

        elif current_time < START:
            print('smart waiting')
            time.sleep(1)
        elif current_time >= END and have_stocks:
            print('Smart Trader Sold {0} stocks for {1} each at the closing price.'.format(num_stocks_to_buy, current_price))
            write_to_log('Smart Trader Sold {0} stocks for {1} each at the closing price.'.format(num_stocks_to_buy, current_price))
            have_stocks = False
            break

    print('Finished Trading!')
    print('Profit: {}'.format(balance))
    write_to_log('Profit for {0}: {1}'.format(today, balance))
    with open(BALANCE_FILENAME, 'wb') as (f):
        pickle.dump(balance, f)


if __name__ == '__main__':
    ticker = 'TSLA'
    num_stocks = 100
    predictor_trade(ticker, num_stocks)
