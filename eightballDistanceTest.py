from eightball import get_final_predictions
from predictor import MarketPredictor
import datetime

if __name__ == '__main__':
    for i in range(-30, -2):
        days_into_future = i
        date_to_predict = (datetime.datetime.now() + datetime.timedelta(days=days_into_future)).strftime("%Y-%m-%d")
        mp = MarketPredictor("INTC", end_date=((datetime.datetime.now() + datetime.timedelta(days=days_into_future - 1)).strftime("%Y-%m-%d")))
        mp.load_data()
        mp.fit_inital()
        print("Date Predicted: " + date_to_predict + " for stock" + "INTC")
        raw_preds = mp.predict(date_to_predict) # will be saved to log
        # weighted_results = get_final_predictions(stock, date_to_predict)
        # print("\n")
        # print("Weighted Results: ")
        # print(weighted_results)