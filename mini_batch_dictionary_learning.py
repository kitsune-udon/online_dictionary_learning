import numpy as np
import omp

def dictionary_update(D, A, B):
    eps = 1e-9
    max_iter = 100
    n, k = D.shape
    for j in xrange(max_iter):
        flags = np.zeros(k, dtype=bool)
        for i in xrange(k):
            d0 = D[:, i]
            if A[i, i] == 0.:
                s = 1. / eps
            else:
                s = 1. / A[i, i]
            u = s * (B[:, i] - np.dot(D, A[:, i])) + d0
            d1 = (1. / max(1., np.linalg.norm(u))) * u
            D[:, i] = d1
            if np.allclose(d0, d1):
                flags[i] = True
        if np.all(flags):
            break

def learn(y, params):
    D, A, B, sparse_encoder = params
    x = sparse_encoder(y, D)
    A += np.outer(x, x)
    B += np.outer(y, x)
    dictionary_update(D, A, B)


def learn_mini_batch(Y, params):
    assert 2 == Y.ndim
    n_samples = Y.shape[1]
    for i in xrange(n_samples):
        y = Y[:, i]
        learn(y, params)

def initialize_params(n, k, m):
    def normalize(D):
        n = D.shape[1]
        for i in xrange(n):
            d = D[:, i]
            s = np.linalg.norm(d)
            D[:, i] = d / s
        return D

    D = normalize(np.random.random((n, k)))
    A = np.zeros((k, k))
    B = np.zeros((n, k))

    sparse_encoder = lambda y, A: omp.omp(y, A, m)

    params = (D, A, B, sparse_encoder)
    return params

def get_dictionary(params):
    return params[0]

def test_dl():
    params = initialize_params(3, 4, 1)
    input = [[1,0,0]]*100 + [[0,1,0]]*100 + [[0,0,1]]*100
    Y = np.array(input)
    np.random.shuffle(Y)
    learn_mini_batch(Y.T, params)
    print get_dictionary(params)

if __name__ == '__main__':
    test_dl()
