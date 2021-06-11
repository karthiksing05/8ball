# Predictor import
from predictor import MarketPredictor

# Standard python libs
from pprint import pprint

watchlist = [
    "BTC-USD",
    "LTC-USD",
    "ETH-USD",
    "DOGE-USD"
]

date = "2021-05-31"

predictors = [MarketPredictor(ticker) for ticker in watchlist]
all_preds = {}
for predictor in predictors:
    predictor.load_data()
    best_acc = predictor.fit_inital()
    preds, ranges = predictor.predict(date)
    all_preds[predictor.ticker] = [preds["Output Values"][x] for x in range(0, 4+1)]

pprint(all_preds)

def projected_results():
    pass
