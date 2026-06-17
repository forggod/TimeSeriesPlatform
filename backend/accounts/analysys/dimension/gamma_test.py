import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree


def estimate_embedding_dimension_gamma(series, tau, max_m=10, p_max=30):
    """
    Оценка размерности вложения с помощью гамма-теста

    Параметры:
    series - временной ряд
    tau - временная задержка
    max_m - максимальная размерность вложения для проверки
    p_max - максимальное количество ближайших соседей для гамма-теста

    Возвращает:
    m_est - оценка оптимальной размерности вложения
    Gamma_values - значения Gamma_bar для разных размерностей
    """

    def gamma_test(X, y, p_max=30):
        """
        Выполняет гамма-тест для оценки дисперсии неопределенной составляющей

        Параметры:
        X - массив входных векторов (M точек в m-мерном пространстве)
        y - соответствующие значения выходной переменной
        p_max - максимальное количество ближайших соседей для анализа

        Возвращает:
        Delta - массив значений Δ(p)
        Gamma - массив значений Γ(p)
        A - коэффициент сложности поверхности
        Gamma_bar - оценка дисперсии шума (результат гамма-теста)
        """
        M = X.shape[0]
        if M != len(y):
            raise ValueError(
                "Количество точек X и значений y должно совпадать")

        # Создаем KD-дерево для эффективного поиска ближайших соседей
        tree = KDTree(X)

        Delta = np.zeros(p_max)
        Gamma = np.zeros(p_max)

        for p in range(1, p_max + 1):
            # Находим p ближайших соседей для каждой точки (исключая саму точку)
            # k=p+1 потому что включаем саму точку
            distances, indices = tree.query(X, k=p+1)

            # Исключаем саму точку (первый ближайший "сосед" - это сама точка)
            distances = distances[:, 1:p+1]
            indices = indices[:, 1:p+1]

            # Вычисляем Δ(p)
            delta_sum = 0.0
            for h in range(p):
                delta_sum += np.mean(np.linalg.norm(X -
                                     X[indices[:, h]], axis=1)**2)
            Delta[p-1] = delta_sum / p

            # Вычисляем Γ(p)
            gamma_sum = 0.0
            for h in range(p):
                gamma_sum += np.mean((y - y[indices[:, h]])**2) / 2
            Gamma[p-1] = gamma_sum / p

        # Линейная регрессия для определения A и Gamma_bar
        model = LinearRegression().fit(Delta.reshape(-1, 1), Gamma)
        A = model.coef_[0]
        Gamma_bar = model.intercept_

        return Delta, Gamma, A, Gamma_bar

    N = len(series)

    Gamma_values = []

    for m in range(1, max_m + 1):
        # Реконструкция фазового пространства
        X = np.zeros((N - (m - 1) * tau - 1, m))
        y = np.zeros(N - (m - 1) * tau - 1)

        for i in range(m):
            X[:, i] = series[i * tau: N - (m - 1) * tau - 1 + i * tau]

        y[:] = series[(m - 1) * tau + 1: N]

        # Выполнение гамма-теста
        Delta, Gamma, A, Gamma_bar = gamma_test(X, y, p_max)
        Gamma_values.append(Gamma_bar)

    Gamma_values = np.array(Gamma_values)

    # Находим размерность с минимальным Gamma_bar
    m_est = np.argmin(Gamma_values) + 1

    return m_est, Gamma_values
