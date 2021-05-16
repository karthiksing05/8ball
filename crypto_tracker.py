# Predictor import
from predictor_v3 import MarketPredictor

# Standard python libs
from pprint import pprint

watchlist = [
    "BTC-USD",
    "LTC-USD",
    "ETH-USD",
]

date = "2021-05-05"

predictors = [MarketPredictor(ticker) for ticker in watchlist]
all_preds = {}
for predictor in predictors:
    predictor.load_data()
    results = predictor.pick_model(show=True)
    best_acc = predictor.fit_inital()
    preds, attrs, date = predictor.predict(date)
    all_preds[predictor.ticker] = [preds["Output Values"][x] for x in range(0, 4+1)]

def projected_results():
    pass