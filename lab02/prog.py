import numpy as np
import pandas as pd


def cholesky(A, r=0):
    n = len(A)
    r = n - r
    for k in range(r):
        if k >= r:
            break
        A[k, k] = np.sqrt(A[k, k])
        A[k+1:n, k] /= A[k, k]
        vk = A[k+1:n, k].flatten()
        for j in range(k + 1, n):
            A[j:n, j] -= A[j:n, k] * vk[j - (k + 1)]
    return A


def generate_A(n):
    A = np.random.rand(n, n)
    return A @ A.T


def main():
    np.random.seed(2124)
    A = generate_A(6)
    print(np.round(A, 2))

    print(np.round(np.linalg.cholesky(A), 2))

    B = np.copy(A)

    L = cholesky(B, 2)

    print(np.round(L, 2))
    # print(np.round(A, 2))


if __name__ == '__main__':
    main()
