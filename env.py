import numpy as np
from scipy.stats import norm

# The grid
def wealth_grid(number_values, lowest_value, highest_value, starting_value):
    lnw_array = np.linspace(np.log(lowest_value), np.log(highest_value), number_values)
    w_array = np.exp(lnw_array)
    idx0 = np.where(np.abs(w_array - starting_value) == np.abs(w_array - starting_value).min())[0]
    w_array = w_array / w_array[idx0]

    return w_array

# Transition probabilities
def transition_probs(means, standard_devs, wealth_grid, time_increments):
    h = time_increments
    grid_length = len(wealth_grid)
    nactions = len(means)
    probs_array = np.zeros((grid_length, grid_length, nactions))

    for i in range(grid_length):
        a = wealth_grid[i]
        for kk in range(nactions):
            mu = means[kk]
            vlty = standard_devs[kk]
            for j in range(grid_length):
                b = wealth_grid[j]
                probs_array[i, j, kk] = norm.pdf((np.log(b / a) - (mu - 0.5*vlty**2)*h) / (vlty * np.sqrt(h)))
            sum_row = probs_array[i, :, kk].sum()
            probs_array[i, :, kk] = probs_array[i, :, kk] / sum_row

    return probs_array
