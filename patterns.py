"""
This file is used to identify whether specially made candle objects 
indicate trend reversal patterns.
"""

import numpy as np

def _about(num1, num2, margin, ndigits):
    if round(num1 + margin, ndigits) == round(num2, ndigits):
        return True
    if round(num1 - margin, ndigits) == round(num2, ndigits):
        return True
    return False

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
    X = np.array([[data_point[0]] for data_point in data_points])
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
    if define_trend(past_50_dpts) == candle_b.type:
        if candle_a.type != candle_b.type:
            if (candle_a.open < candle_b.open) and (candle_a.close < candle_b.close):
                return True
    return False

def identify_morning_star(candle_a, candle_b, candle_c, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    # if _define_trend(past_50_dpts) == candle_a.type:
    if candle_a.type != candle_c.type:
        if round(candle_b.body_range, 3) <= 0.01:
            if candle_c.volume > candle_a.volume:
                return True
    return False

def identify_3_black_crows(candle_a, candle_b, candle_c, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if define_trend(past_50_dpts) == "BULLISH":
        if 25 < candle_a.rsi < 50:
            if 25 < candle_b.rsi < 50:
                if 25 < candle_c.rsi < 50:
                    if (candle_a.type == "BEARISH") and (candle_b.type == "BEARISH") and (candle_c.type == "BEARISH"):
                        if candle_a.close > candle_b.close > candle_c.close:
                            return True
    return False

def identify_3_white_soldiers(candle_a, candle_b, candle_c, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if define_trend(past_50_dpts) == "BEARISH":
        if 35 < candle_a.rsi < 70:
            if 35 < candle_b.rsi < 70:
                if 35 < candle_c.rsi < 70:
                    if (candle_a.type == "BULLISH") and (candle_b.type == "BULLISH") and (candle_c.type == "BULLISH"):
                        if candle_a.close < candle_b.close < candle_c.close:
                            return True
    return False

def identify_piercing_pattern(candle_a, candle_b, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if define_trend(past_50_dpts) == "BEARISH":
        if candle_a.type == "BEARISH":
            if candle_b.type == "BULLISH":
                if candle_a.close > candle_b.open:
                    if candle_b.close > (candle_a.open - (0.5 * candle_a.body_range)):
                        return True
    return False


def identify_shooting_star(candle_a, past_50_dpts):
    """
    Returns boolean indicating whether the candles signify a trend reversal.
    """
    if define_trend(past_50_dpts) == "BULLISH":
        if candle_a.type == "BEARISH":
            if round((candle_a.high - candle_a.open), 3) >= 0.04:
                if 0.002 < round(candle_a.body_range, 4) < 0.004:
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
