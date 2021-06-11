"""
A quick Candlestick datatype
"""

class Candlestick(object):

    def __init__(self, index, body_size, candle_range, btype, price_open, price_close, price_high, price_low, volume, rsi):
        self.index = index
        self.body_size = body_size
        self.candle_range = candle_range
        self.type = btype
        self.open = price_open
        self.close = price_close
        self.high = price_high
        self.low = price_low
        self.volume = volume
        self.rsi

