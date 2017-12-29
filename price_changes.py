import numpy as np

def compute(data):
    up_down = []
    for i in data.index:
        if i == 0:
            continue
        if float(data.iloc[i]) > float(data.iloc[i - 1]):
            up_down.append(1)
        else:
            up_down.append(0)

    return up_down

def compute_with_difference(d, data):
    up_down = []
    for i in data.index:
        if i == 0:
            continue
        if np.abs(float(data.iloc[i]) - float(data.iloc[i - 1])) / float(data.iloc[i - 1]) >= d:
            up_down.append(1)
        else:
            up_down.append(0)

    return up_down

