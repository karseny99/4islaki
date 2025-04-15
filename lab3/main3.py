'''

    x = [0.1 , 0.5 , 0.9 , 1.3 , 1.7 , 2.1]
    y = [-2.2026 , -0.19315 , 0.79464 , 1.5624 , 2.2306 , 2.8419]

'''

class LU:
    EPS = 1e-6

    def __init__(self, U):
        L = np.eye(len(U), dtype=float)
        isDetNeg = False
        permut = np.array(range(len(U)))
        for i in range(len(U)):
            max_idx = i
            for j in range(i + 1, len(U)):
                if abs(U[max_idx][i]) < abs(U[j][i]):
                    max_idx = j
            if max_idx != i:
                U[[i, max_idx]] = U[[max_idx, i]]
                L[[i, max_idx]] = L[[max_idx, i]]
                L[:, [i, max_idx]] = L[:, [max_idx, i]]
                isDetNeg = not isDetNeg
                permut[[i, max_idx]] = permut[[max_idx, i]]
            if abs(U[i][i]) < self.EPS: continue
            for j in range(i + 1, len(U)):
                mu = U[j][i] / U[i][i]
                L[j][i] = mu
                for k in range(len(U)):
                    U[j][k] -= mu * U[i][k]
        det = U.diagonal().prod()
        if isDetNeg: det = -det
        self._permut = permut
        self.L = L
        self.U = U
        self.det = det

    def solve(self, b):
        b = np.array([ b[pi] for pi in self._permut ], dtype=float)
        z = np.array([0] * len(b), dtype=float)
        for i in range(len(b)):
            z[i] = b[i]
            for j in range(i):
                z[i] -= self.L[i, j] * z[j]
        x = np.array([0] * len(b), dtype=float)
        for i in range(len(b)-1, -1, -1):
            if abs(self.U[i, i]) < self.EPS: continue
            x[i] = z[i]
            for j in range(len(b)-1, i, -1):
                x[i] -= x[j] * self.U[i, j]
            x[i] /= self.U[i, i]
        return x


import numpy as np

x = np.array([0.1, 0.5, 0.9, 1.3, 1.7, 2.1])
y = np.array([-2.2026, -0.19315, 0.79464, 1.5624, 2.2306, 2.8419])
# x = np.array([0.0, 1.7, 3.4, 5.1, 6.8, 8.5])
# y = np.array([0.0, 1.3038, 1.8439, 2.2583, 2.6077, 2.9155])

def least_squares_poly(x, y, n):
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    
    for i in range(n+1):
        for j in range(n+1):
            A[i,j] = np.sum(x**(i+j))
        b[i] = np.sum(y * x**i)

    coeffs = LU(A).solve(b)

    return coeffs

def poly_func(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))

coeffs_1 = least_squares_poly(x, y, 1)
coeffs_2 = least_squares_poly(x, y, 2)

y_pred_1 = np.array([poly_func(coeffs_1, xi) for xi in x])
y_pred_2 = np.array([poly_func(coeffs_2, xi) for xi in x])

sse_1 = np.sum((y - y_pred_1)**2)
sse_2 = np.sum((y - y_pred_2)**2)

print("1st degree polynomial coefficients:", coeffs_1)
print("Sum of squared errors (1st degree):", sse_1)
print("\n2nd degree polynomial coefficients:", coeffs_2)
print("Sum of squared errors (2nd degree):", sse_2)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Original data')
x_plot = np.linspace(min(x), max(x), 100)
y_plot_1 = np.array([poly_func(coeffs_1, xi) for xi in x_plot])
y_plot_2 = np.array([poly_func(coeffs_2, xi) for xi in x_plot])
plt.plot(x_plot, y_plot_1, label=f'1st degree (SSE={sse_1:.4f})')
plt.plot(x_plot, y_plot_2, label=f'2nd degree (SSE={sse_2:.4f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Polynomial Approximation')
plt.legend()
plt.grid(True)
plt.show()
