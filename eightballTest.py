from eightball import get_final_predictions
import os

if __name__ == '__main__':
    stocks = ["GOOG", "TSLA", "AMZN", "NVDA", "DIS", "NFLX", "KO", "WMT", "AAPL", "COST"]
    for stock in stocks:
        try:
            os.remove("dataset_benchmark.pickle")
        except FileNotFoundError:
            pass
        for directory in os.listdir("pool_data"):
            os.remove("pool_data\\" + directory)
        date_to_predict = "2023-12-30"
        print("Date Predicted: " + date_to_predict + " for stock " + stock)
        weighted_results = get_final_predictions(stock, date_to_predict)
        print("\n")
        print("Weighted Results: ")
        print(weighted_results)
