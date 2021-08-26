from monkeylearn import MonkeyLearn

API_KEY = "a4c851e916c04215ef782156efa3797caf12a6aa"
MODEL_ID = "cl_YDqS8DxM"

ml = MonkeyLearn(API_KEY)
data = ["I dislike $TSLA's volatility, $TSLA sucks"]

result = ml.classifiers.classify(MODEL_ID, data)

print(result.body)