import numpy as np
def sampleWidx(idx0, wealth_grid, a0, means, stds, probability_array):
    W = wealth_grid
    mu = means[a0]
    vlty = stds[a0]
    p1 = probability_array[idx0, :, a0]
    idx = np.where(np.random.rand() > p1.cumsum())[0]
    return len(idx)

