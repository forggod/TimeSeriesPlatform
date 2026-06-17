import numpy as np
from scipy.spatial.distance import pdist


def estimate_embedding_dimension(series, tau, max_m=10, epsilons=None):
    """
    Оценка размерности пространства вложения

    Параметры:
    series - временной ряд
    tau - временная задержка
    max_m - максимальная размерность вложения для проверки
    epsilons - массив значений ε (если None, будет сгенерирован автоматически)

    Возвращает:
    m_est - оценка размерности вложения
    d_est - оценка корреляционной размерности
    D_values - значения корреляционной размерности для разных m
    """

    def heaviside(x):
        """Функция Хевисайда"""
        return np.where(x >= 0, 1, 0)

    def correlation_integral(X, epsilon):
        """
        Вычисление корреляционного интеграла C(ε)

        Параметры:
        X - массив точек (M точек в m-мерном пространстве)
        epsilon - радиус окрестности

        Возвращает:
        C(ε) - значение корреляционного интеграла
        """
        M = X.shape[0]
        if M < 2:
            return 0.0

        # Вычисляем попарные расстояния между точками
        distances = pdist(X)

        # Применяем функцию Хевисайда к (ε - расстояния)
        heaviside_values = heaviside(epsilon - distances)

        # Суммируем и нормируем
        C = 2 * np.sum(heaviside_values) / (M * (M - 1))

        return C

    def grassberger_procaccia_algorithm(X, epsilons):
        """
        Алгоритм Грассберга-Прокичиа для оценки корреляционной размерности

        Параметры:
        X - массив точек
        epsilons - массив значений ε

        Возвращает:
        log_eps - логарифмы значений ε
        log_C - логарифмы значений C(ε)
        """
        log_C = []

        for eps in epsilons:
            C = correlation_integral(X, eps)
            if C > 0:
                log_C.append(np.log(C))
            else:
                log_C.append(np.nan)

        log_eps = np.log(epsilons)

        return log_eps, np.array(log_C)

    def estimate_correlation_dimension(log_eps, log_C):
        """
        Оценка корреляционной размерности по методу Грассберга-Прокичиа

        Параметры:
        log_eps - логарифмы значений ε
        log_C - логарифмы значений C(ε)
        plot - флаг для отображения графика

        Возвращает:
        D2 - оценка корреляционной размерности
        """
        # Удаляем NaN значения
        mask = ~np.isnan(log_C)
        log_eps = log_eps[mask]
        log_C = log_C[mask]

        if len(log_eps) < 2:
            return np.nan

        # Находим линейный участок (исключаем крайние значения)
        n = len(log_eps)
        start = n // 4
        end = 3 * n // 4

        # Линейная регрессия на линейном участке
        A = np.vstack([log_eps[start:end], np.ones(end-start)]).T
        slope, _ = np.linalg.lstsq(A, log_C[start:end], rcond=None)[0]

        D2 = slope

        return D2

    def time_delay_embedding(series, m, tau):
        N = len(series)
        if (m-1)*tau + 1 > N:
            raise ValueError("Недостаточно данных для реконструкции")

        X = np.zeros((N - (m-1)*tau, m))
        for i in range(m):
            X[:, i] = series[i*tau: N - (m-1)*tau + i*tau]

        return X

    if epsilons is None:
        # Автоматическая генерация значений ε
        data_range = np.max(series) - np.min(series)
        epsilons = np.logspace(np.log10(data_range/1000),
                               np.log10(data_range/2), 20)

    D_values = []

    for m in range(1, max_m+1):
        # Реконструкция фазового пространства
        X = time_delay_embedding(series, m, tau)

        # Вычисление корреляционной размерности
        log_eps, log_C = grassberger_procaccia_algorithm(X, epsilons)
        D = estimate_correlation_dimension(log_eps, log_C)

        D_values.append(D)

    # Поиск насыщения
    D_values = np.array(D_values)
    diffs = np.diff(D_values)

    # Находим точку, где изменение становится меньше порога (10% от среднего изменения)
    threshold = 0.1 * np.mean(np.abs(diffs))
    saturation_point = np.where(np.abs(diffs) < threshold)[0]

    if len(saturation_point) > 0:
        m_est = saturation_point[0] + 1
        d_est = D_values[m_est-1]
    else:
        m_est = max_m
        d_est = D_values[-1]

    return m_est, d_est, D_values
