import numpy as np

def znorm(series, znorm_threshold = 0.01):
    sd = np.std(series)
    if (sd < znorm_threshold):
      return res
    mean = np.mean(series)
    res = np.zeros(len(series))
    for i in range(0, len(series)):
      res[i] = (series[i] - mean) / sd
    return res
