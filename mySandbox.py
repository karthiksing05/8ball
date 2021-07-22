from eightball import get_final_predictions
from predictor import MarketPredictor
import datetime

daysfuture = 1
date_to_predict = (datetime.datetime.now() + datetime.timedelta(days=daysfuture)).strftime("%Y-%m-%d")
stock = "TSLA"
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
weighted_results, chance_of_tr = get_final_predictions(stock, date_to_predict)
print("\n")
print("Weighted Results: ")
print(weighted_results)
print("Chance Of Tr: ")
print(chance_of_tr)
