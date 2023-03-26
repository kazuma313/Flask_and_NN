import numpy as np

def mse(actual, pred):
    return np.square(np.subtract(actual, pred)).mean()

def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100