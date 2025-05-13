import numpy as np
import math
import matplotlib.pyplot as plt

def solve_bvp(N):
    a = 0
    b = math.pi / 6
    h = (b - a) / N

    y0 = 2
    yN = 2.5 - 0.5 * math.log(3)

    x = np.linspace(a, b, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    y[N] = yN

    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    C = np.zeros(N + 1)

    for k in range(1, N):
        xk = x[k]
        tan_xk = math.tan(xk)
        A[k] = 1 + (h / 2) * tan_xk
        B[k] = -2 + 2 * h**2
        C[k] = 1 - (h / 2) * tan_xk

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    alpha[1] = -C[1] / B[1]
    beta[1] = (-A[1] * y[0]) / B[1]

    for k in range(2, N):
        denom = B[k] + A[k] * alpha[k - 1]
        alpha[k] = -C[k] / denom
        beta[k] = (-A[k] * beta[k - 1]) / denom

    for k in range(N - 1, 0, -1):
        y[k] = alpha[k] * y[k + 1] + beta[k]

    return x, y

# Решение на грубой сетке (N=50)
N1 = 10
x1, y1 = solve_bvp(N1)

# Решение на точной сетке (N=100)
N2 = 20
x2, y2 = solve_bvp(N2)

# Интерполяция y1 на сетку y2 для сравнения
y1_interp = np.interp(x2, x1, y1)

# Оценка погрешности по Рунге-Ромбергу
p = 2  # порядок точности метода
error = np.abs(y2 - y1_interp) / (2**p - 1)

# Уточненное решение
y_refined = y2 + (y2 - y1_interp) / (2**p - 1)

# Точное решение
def exact_solution(x):
    sin_x = np.sin(x)
    mask = np.abs(sin_x - 1) < 1e-10  # избегаем деления на 0
    result = sin_x + 2 - sin_x * np.log((1 + sin_x) / (1 - sin_x))
    result[mask] = 2
    return result

y_exact = exact_solution(x2)

# Построение графиков
# plt.figure(figsize=(12, 8))

# plt.plot(x2, y2, 'b-', linewidth=2, label=f'Численное решение (N={N2})')
# plt.plot(x2, y_exact, 'r--', linewidth=2, label='Точное решение')
# plt.xlabel('x')
# plt.ylabel('y(x)')
# plt.title('Сравнение численного и точного решений')
# plt.legend()
# plt.grid(True)

# plt.show()

# Погрешность относительно точного решения
error_exact = np.abs(y2 - y_exact)


# print(error)
# Вывод максимальной погрешности
max_error = np.max(error)
_error = np.max(error_exact)
print(f"Максимальная оценка погрешности (Рунге-Ромберг): {max_error}")
print(f"Максимальная оценка погрешности (abs): {_error}")