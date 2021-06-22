from predictor import MarketPredictor
import datetime

if __name__ == '__main__':
    mp = MarketPredictor("TSLA")
    mp.load_data()
    mp.fit_inital()
    df, ranges = mp.predict("2021-06-21")
    print(df)
