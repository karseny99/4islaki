matrix = [
    [8, -3, 9],
    [-3, 8, -2],
    [9, -2, -8],
]


import numpy as np
import copy
import math


def jacobi_rotation(matrix, epsilon=1e-6, max_iterations=1000):
    n = len(matrix)

    A = copy.deepcopy(matrix)
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)] 

    for iteration in range(max_iterations):
        max_val = 0.0
        p, q = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > max_val:
                    max_val = abs(A[i][j])
                    p, q = i, j

        if max_val < epsilon:
            print(f"The method has converged by {iteration} iterations")
            break

        if A[p][p] == A[q][q]:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * A[p][q], A[p][p] - A[q][q])

        c = math.cos(theta)
        s = math.sin(theta)
        G = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        G[p][p] = c
        G[q][q] = c
        G[p][q] = -s
        G[q][p] = s

        # A = G^T * A * G
        A_new = copy.deepcopy(A)
        for i in range(n):
            for j in range(n):
                A_new[i][j] = sum(G[k][i] * A[k][l] * G[l][j] for k in range(n) for l in range(n))
        A = A_new

        # Q = Q * G
        Q_new = [row.copy() for row in Q]
        for i in range(n):
            for j in range(n):
                Q_new[i][j] = sum(Q[i][k] * G[k][j] for k in range(n))
        Q = Q_new

    eigenvalues = [A[i][i] for i in range(n)][::-1]
    eigenvectors = [[Q[i][j] for j in range(n)][::-1] for i in range(n)]

    return eigenvalues, eigenvectors




if __name__ == "__main__":
    eigenvalues, eigenvectors = jacobi_rotation(matrix)

    eigenvalues_np, eigenvectors_np = np.linalg.eigh(matrix)

    print(f"my impl: {eigenvalues}")
    print(f"numpy: {eigenvalues_np}")
    print('\n')
    print(f"my impl:")
    for vec in eigenvectors:
        print(vec)
    print(f"numpy:\n {eigenvectors_np}")