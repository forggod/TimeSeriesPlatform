import numpy as np

from ..analysys.tau.autocorrelation import (
    autocorrelation_function,
    auto_delay_zc_acf,
    auto_delay_lm_acf
)
from ..analysys.tau.mutual_information import (
    mutual_information_fuction,
    auto_delay_lm_mif
)
from ..analysys.dimension.autocorelation import estimate_embedding_dimension
from ..analysys.dimension.gamma_test import estimate_embedding_dimension_gamma
from ..analysys.dimension.false_nearest_neighbors import (
    false_nearest_neighbors,
    find_optimal_dim
)
from ..analysys.lyapunov.lyapunov import estimate_lyapunov
from ..analysys.takens_rekonstruction import reconstruct_attractor


def compute_acf(series, max_tau=200):
    acf = autocorrelation_function(series, max_tau=max_tau)

    auto_delay = auto_delay_zc_acf(acf)

    if auto_delay is None:
        auto_delay = auto_delay_lm_acf(acf)

    acf = acf/200

    return acf, auto_delay


def compute_mif(series, max_tau=200):
    mif = mutual_information_fuction(series, max_tau=max_tau)

    auto_delay = auto_delay_lm_mif(mif)

    return mif, auto_delay


def compute_dim_auto_correlation(series, tau, max_dim=10):
    auto, _, corr = estimate_embedding_dimension(series, tau, max_m=max_dim)

    return auto, corr


def compute_dim_gamma_test(series, tau, max_dim=10):
    auto, gamma = estimate_embedding_dimension_gamma(
        series, tau, max_m=max_dim)

    return auto, gamma


def compute_dim_false_nearest_neighbors(series, tau, max_dim=10):
    fnn = false_nearest_neighbors(series, tau=tau, max_dim=max_dim)

    auto = find_optimal_dim(fnn)

    return auto, fnn


def compute_lyapunov(series, tau, dim, max_iter=10000):
    value, values = estimate_lyapunov(
        series, tau=tau, m=dim, max_iter=max_iter
    )

    return value, values


def reconstruct_attraktor(series, tau, dim):
    return reconstruct_attractor(series, tau=tau, dim=dim)
