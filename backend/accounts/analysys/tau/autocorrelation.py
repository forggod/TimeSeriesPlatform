import numpy as np


def autocorrelation_function(x, max_tau):
    def y_t(t):
        return x[t] - e_x

    e_x = np.mean(x)
    r = np.zeros(max_tau)

    for tau in range(max_tau):
        n = len(x) - tau
        sum_op = 0.0

        for i in range(n):
            sum_op += y_t(i) * y_t(i + tau)

        r[tau] = sum_op / len(x)

    return r


def auto_delay_zc_acf(acf):
    for i, v in enumerate(acf):
        if v < 0:
            return i

    return None


def auto_delay_lm_acf(acf):
    min = 0

    for i, v in enumerate(acf):
        if (v < acf[min]):
            min = i

    return min
