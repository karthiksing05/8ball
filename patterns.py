"""
This file is used to identify whether specially made candle objects 
indicate trend reversal patterns.
"""

import numpy as np

def _get_weights(X, y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return W

def define_trend(data_points):
    """
    A list of tuples -> (idx:int, closing_price:float) is given, and output is a string,
    either BEARISH or BULLISH.
    """
    X = np.array([data_point[0] for data_point in data_points])
    y = np.array([data_point[1] for data_point in data_points])
    trend = float(_get_weights(X, y)[1])
    if trend >= 0:
        return "BULLISH"
    elif trend < 0:
        return "BEARISH"

def identify_engulfing(candle_a, candle_b, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if candle_a.type != candle_b.type:
        if (candle_a.open < candle_b.open) and (candle_a.close < candle_b.close):
            if define_trend(past_50_dpts) == candle_b.type:
                return True
    return False

def identify_morning_star(candle_a, candle_b, candle_c, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if candle_a.type != candle_c.type:
        if candle_b.body_size <= 3:
            if candle_c.volume > candle_a.volume:
                if define_trend(past_50_dpts) == candle_a.type:
                    return True
    return False

def identify_3_white_soldiers(candle_a, candle_b, candle_c, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if 40 < candle_a.rsi < 70:
        if 40 < candle_b.rsi < 70:
            if 40 < candle_c.rsi < 70:
                if candle_a.type == candle_b.type == candle_c.type == "BULLISH":
                    if candle_a.close < candle_b.close < candle_c.close:
                        if define_trend(past_50_dpts) == "BEARISH":
                            return True
    return False

def identify_3_black_crows(candle_a, candle_b, candle_c, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if 40 < candle_a.rsi < 70:
        if 40 < candle_b.rsi < 70:
            if 40 < candle_c.rsi < 70:
                if candle_a.type == candle_b.type == candle_c.type == "BULLISH":
                    if candle_a.close < candle_b.close < candle_c.close:
                        if define_trend(past_50_dpts) == "BEARISH":
                            return True
    return False

if __name__ == "__main__":
    m, n = 500, 1
    X = np.random.rand(m, n)
    y = -5 * X + np.random.randn(m, n) * 0.1
    sample_dpts = [[X[idx], y[idx]] for idx in range(len(X))]
    import matplotlib.pyplot as plt
    plt.scatter(X, y)
    plt.show()
    print(define_trend(sample_dpts))
