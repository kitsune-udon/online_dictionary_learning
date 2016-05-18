import numpy as np

def lasso_admm(y, A, lambda1, rho, max_iter):
    def soft_thresholding(lambda1, v):
        n = v.shape[0]
        r = np.zeros(n)
        for i in xrange(n):
            if v[i] < -lambda1:
                r[i] = v[i] + lambda1
            elif v[i] <= lambda1:
                r[i] = 0.
            else:
                r[i] = v[i] - lambda1
        return r

    k = A.shape[1]
    b_cur, t_cur, m_cur = np.zeros(k), np.zeros(k), np.zeros(k)
    i = 0
    C0 = np.linalg.inv(np.dot(A.T, A) + rho * np.identity(k))
    C1 = np.dot(A.T, y)

    while i < max_iter:
        C2 = C1 + rho * t_cur - m_cur
        b_succ = np.dot(C0, C2)
        t_succ = soft_thresholding(lambda1/rho, b_succ+m_cur/rho)
        m_succ = m_cur + rho * (b_succ - t_succ)

        if np.allclose(b_cur, b_succ):
            b_cur, t_cur, m_cur = b_succ, t_succ, m_succ
            break

        b_cur, t_cur, m_cur = b_succ, t_succ, m_succ
        i += 1

    print "end at {} th iter".format(i)
    return b_cur

if __name__ == "__main__":
    y = np.array([1,-1,1])
    A = np.array([[1,0,0],[0,1,0],[0,0,1]]).T
    lambda1 = 1e-8
    max_iter = 100
    rho = 1e-9
    x = lasso_admm(y, A, lambda1, rho, max_iter)
    print x
