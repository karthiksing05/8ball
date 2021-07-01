from predictor import MarketPredictor
import datetime

if __name__ == '__main__':
    stock = "CMG"
    preddate = datetime.datetime.now().strftime("%Y-%m-%d")
    mp = MarketPredictor(stock)
    mp.load_data()
    mp.fit_inital()
    preds, ranges = mp.predict(preddate)
    preds = list(preds['Output Values'])
    print(preds)
