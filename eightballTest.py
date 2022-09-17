from eightball import get_final_predictions
from predictor import MarketPredictor
import datetime

if __name__ == '__main__':
    days_into_future = 1
    date_to_predict = (datetime.datetime.now() + datetime.timedelta(days=days_into_future)).strftime("%Y-%m-%d")
    stock = "AAPL"
    print("Date Predicted: " + date_to_predict)
    print("Stock Predicted: " + stock)
    mp = MarketPredictor(stock)
    mp.load_data()
    mp.fit_inital()
    raw_preds = mp.predict(date_to_predict)
    unweighted_results = [raw_preds['Output Values'][x] for x in range(5)]
    print("\n")
    print("Raw Results:")
    print(unweighted_results)
    weighted_results = get_final_predictions(stock, date_to_predict)
    print("\n")
    print("Weighted Results: ")
    print(weighted_results)
