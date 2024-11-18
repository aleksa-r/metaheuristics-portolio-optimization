import numpy as np


def IsDominated(rsk, ret):
    n = len(rsk)
    f = np.zeros(n, dtype='bool')
    for i in range(n):
        for j in range(i + 1, n):
            if rsk[i] < rsk[j] and ret[i] > ret[j]:
                f[j] = True
            elif rsk[i] > rsk[j] and ret[i] < ret[j]:
                f[i] = True
    return f