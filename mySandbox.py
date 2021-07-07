from eightball import get_final_predictions
from predictor import MarketPredictor
import datetime

daysfuture = 3
date_to_predict = (datetime.datetime.now() + datetime.timedelta(days=daysfuture)).strftime("%Y-%m-%d")
print("Date Predicted: " + date_to_predict)
stock = "TSLA"
print("Stock Predicted: " + stock)
weighted_results = get_final_predictions(stock, date_to_predict)
mp = MarketPredictor(stock)
mp.load_data()
mp.fit_inital()
raw_preds = mp.predict(date_to_predict)
unweighted_results = [raw_preds['Output Values'][x] for x in range(5)]

# Comparision
print("\n")
print("Raw Results:")
print(unweighted_results)
print("\n")
print("Weighted Results: ")
print(weighted_results)
