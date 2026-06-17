import numpy as np


def estimate_lyapunov(x, m, tau, max_iter=5000, eps=0.01, eps_max=0.1):
    X = np.array([x[i: i + m * tau: tau] for i in range(len(x) - m * tau)])
    n = len(X)

    sum_l = 0.0
    sum_t = 0
    conv = []

    x0_idx = np.random.randint(0, n - n // 10)
    x0 = X[x0_idx]

    min_dist = np.inf
    x0_nbr = None
    min_idx = 0

    for i in range(n):
        if abs(i - x0_idx) < n // 10:
            continue

        dist = np.linalg.norm(X[i] - x0)

        if dist < min_dist and dist > 0:
            min_dist = dist
            min_idx = i

    x0_nbr = X[min_idx]
    delta = x0_nbr - x0
    delta = delta / np.linalg.norm(delta) * eps

    for _ in range(max_iter):
        x0_idx = min(x0_idx + 1, n - 1)
        nbr_idx = min(min_idx + 1, n - 1)

        new_dist = np.linalg.norm(X[nbr_idx] - X[x0_idx])

        if new_dist > 0 and min_dist > 0:
            sum_l += np.log(new_dist / min_dist)
            sum_t += 1
            conv.append(sum_l / sum_t)

        if new_dist > eps_max:
            delta_dir = (X[nbr_idx] - X[x0_idx])
            delta_dir = delta_dir / np.linalg.norm(delta_dir) * eps

            min_dist = np.inf

            for i in range(n):
                if abs(i - x0_idx) < n // 10:
                    continue
                dist = np.linalg.norm(X[i] + delta_dir - X[x0_idx])

                if dist < min_dist and dist > 0:
                    min_dist = dist
                    min_idx = i

            x0_nbr = X[min_idx] + delta_dir
        else:
            min_dist = new_dist
    estimate = sum_l / sum_t if sum_t > 0 else 0

    return estimate, conv
