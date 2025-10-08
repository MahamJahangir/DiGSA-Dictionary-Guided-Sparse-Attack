import numpy as np
import statistics

def build_digit_dict(X_test, Y_test):
    digitDict = {}
    for i in range(10):
        mask = (Y_test == i)
        digitDict[i] = X_test[mask]
    for i in range(10):
        print("Digit {0} matrix shape: {1}".format(i, digitDict[i].shape))
    return digitDict

def sample_digits(digitDict):
    norm_arr = []
    sampled = []
    for i in range(10):
        norms = []
        for j in range(len(digitDict[i]) - 1):
            l2 = np.linalg.norm(digitDict[i][j] / 255.0 - digitDict[i][j+1] / 255.0)
            norms = np.append(norms, l2) if len(norms) else np.array([l2])
            norm_arr = np.append(norm_arr, l2) if len(norm_arr) else np.array([l2])
        for k in range(len(digitDict[i]) - 1):
            l2 = np.linalg.norm(digitDict[i][k] / 255.0 - digitDict[i][k+1] / 255.0)
            if (l2 > statistics.mean(norm_arr)):
                sampled = np.append(sampled, digitDict[i][k]) if len(sampled) else np.array([digitDict[i][k]])
        print("digit", i)
        print("current sampled shape", sampled.shape)
    sampled = sampled.reshape(-1, 784)
    print(sampled.shape)
    return sampled
