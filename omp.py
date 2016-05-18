import numpy as np

def omp(y, A, k):
    def index_of_max_correlation(A, active_set, r):
        def correlation(v0, v1):
            d = np.linalg.norm(v0) * np.linalg.norm(v1)
            if d == 0.0:
                raise "zero vector exception"
            else:
                return np.dot(v0, v1) / d

        max_v = float("-inf")
        max_i = -1
        for i in xrange(A.shape[1]):
            if i in active_set:
                continue
            v = abs(correlation(A[:,i], r))
            if max_v < v:
                max_i, max_v = i, v
        return max_i

    def subset_of_atoms(A, active_set):
        m, n = A.shape[0], len(active_set)
        A0 = np.zeros((m, n))
        for i in xrange(len(active_set)):
            A0[:, i] = A[:, active_set[i]]
        return A0

    def solve_equation(A, y):
        return np.linalg.lstsq(A, y)[0]

    def adjust_x(n, x, active_set):
        x0 = np.zeros(n)
        for i in xrange(len(active_set)):
            j = active_set[i]
            x0[j] = x[i]
        return x0

    m, n = A.shape
    r = y
    active_set = []
    max_active_set_len = min(k, n)
    eps = 1e-8
    y_norm = max(np.linalg.norm(y), eps)
    min_ratio = 1e-2

    while (len(active_set) < max_active_set_len and
        min_ratio < (np.linalg.norm(r)/y_norm)):
        i = index_of_max_correlation(A, active_set, r)
        active_set.append(i)
        A0 = subset_of_atoms(A, active_set)
        x = solve_equation(A0, y)
        r = y - np.dot(A0, x)

    if len(active_set) > 0:
        A0 = subset_of_atoms(A, active_set)
        x = solve_equation(A0, y)
        return adjust_x(n, x, active_set)
    else:
        raise "invalid parameter"
