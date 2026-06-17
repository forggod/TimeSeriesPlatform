import numpy as np


def reconstruct_attractor(series, dim, tau):
    """
    Реконструкция аттрактора методом временных задержек

    Параметры:
    series - временной ряд (1D массив)
    dim - размерность вложения (m)
    tau - временная задержка (в отсчетах)

    Возвращает:
    attractor - реконструированный аттрактор (N x dim массив)
    """
    N = len(series)
    if (dim-1)*tau >= N:
        raise ValueError("Недостаточно данных для реконструкции")

    attractor = np.zeros((N - (dim-1)*tau, dim))

    for i in range(dim):
        attractor[:, i] = series[i*tau: i*tau + N - (dim-1)*tau]

    return attractor
