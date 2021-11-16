from predictor import MarketPredictor
import datetime

mp = MarketPredictor("TSLA")
mp.load_data() # needs to do this to initialize self.benchmark dataset
mp.up_or_down(datetime.datetime.now().strftime("%Y-%m-%d"))