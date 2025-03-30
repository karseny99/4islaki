from math import sqrt
import numpy as np


matrix = [
    [1, 2, 5],
    [-8, 0, -6],
    [9, -7, 3],
]

def sign(x): 
    if x < 0:
        return -1
    elif x == 0:
        return 0
    return 1


def norm2(x): 
    return sqrt(np.sum(np.array(x) ** 2))
    
def solveQuadEql(a, b, c):
    d = b**2 - 4 * a * c
    if d < 0:
        return (-b / 2*a, sqrt(-d) / 2*a), (-b / 2*a, -sqrt(-d) / 2*a)
    elif d == 0:
        return (-b / 2*a, 0), (-b / 2*a, 0)
    else:
        return ((-b + sqrt(d)) / 2*a, 0), ((-b - sqrt(d)) / 2*a, 0)


def BlockError(x1, x2, r1, r2):
    return max(norm2(r1 - x1), norm2(r2 - x2))

     
def QR(A):
    n = len(A)
    A = np.matrix(A)
    Q = np.matrix(np.eye(n))
    for i in range(n - 1):

        v = [0] * n

        v[i] = A[i, i] + sign(A[i, i]) * norm2(A[i:, i])

        v[i+1:] = np.asarray(A[i+1:, i]).T[0]

        v = np.matrix(v).T

        # H = I - 2 * (v*v^t) / (v^t*v)
        H = np.matrix(np.eye(n) - 2 * v * v.T / norm2(v) ** 2)

        # A(i) = H(i)*A(i-1)
        A = H * A

        # Q = H1*H2*...*H(n-1)
        Q *= H

    # A becoming R
    return Q, A
     


def QR_algorithm(A, eps = 0.001): 
    n = len(A)
    RealEv = np.array([0] * n, dtype=float)
    ComplEv = np.array([[[0, 0]] * 2] * n, dtype=float)
    RealConverged = [False] * n
    ComplConverged = [False] * n

    RealConverged[-1] = True

    while True:
        Q, R = QR(A)
        A = R * Q
        for j in range(n - 1):
            
            if not RealConverged[j] or not ComplConverged[j]:

                # Check if A[j][j] converged
                if norm2(A[j+1:, j]) < eps:
                    RealEv[j] = A[j, j]
                    RealConverged[j] = True

                r1, r2 = solveQuadEql(
                    a = 1,
                    b = -A[j+1, j+1]-A[j, j],
                    c = A[j, j] * A[j+1, j+1] - A[j, j+1] * A[j+1, j]
                )
                
                # Check if Complex pair converged
                if BlockError(ComplEv[j, 0], ComplEv[j, 1], r1, r2) < eps:
                    ComplConverged[j], ComplConverged[j+1] = True, True
                
                ComplEv[j] = r1, r2
        
        # Stop if everything converged
        if all([(RealConverged[j] or ComplConverged[j]) for j in range(n)]):
            break

    # Extracting eigenvals
    x = np.array([[0, 0]] * n, dtype=float)
    i = 0
    while i < n:
        if ComplConverged[i]:
            x[i], x[i+1] = ComplEv[i]
            i += 1
        else:
            x[i] = RealEv[i], 0
        i += 1
    return x


if __name__ == "__main__":
    print("My impl:")
    print(QR_algorithm(matrix))    
    eigenvalues_np, eigenvectors_np = np.linalg.eig(matrix)
    
    print("\nNumpy impl:")
    print(eigenvalues_np)
