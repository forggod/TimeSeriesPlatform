import numpy as np


def mutual_information_fuction(x, max_tau):
    n = len(x)
    l = int(np.log2(n)) + 1

    I = np.zeros(max_tau)

    for tau in range(max_tau):
        x_t = x[:-tau] if tau > 0 else x
        x_t_tau = x[tau:]

        hist, _, _ = np.histogram2d(x_t, x_t_tau, bins=l)
        prob = hist / np.sum(hist)

        p_x = np.sum(prob, axis=1)
        p_y = np.sum(prob, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            log = np.log2(prob / np.outer(p_x, p_y))
            log[np.isinf(log)] = 0
            I[tau] = np.sum(prob * log)

    return I


def auto_delay_lm_mif(acf):
    min = 0

    for i, v in enumerate(acf):
        if (v < acf[min]):
            min = i

    return min
