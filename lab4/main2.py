'''

    y'' - tg(x) * y' + 2y = 0
    y(0) = 2
    y(pi / 6) = 2.5 - 0.5 * ln3

    y(x) = sinx + 2 - sinx * ln((1+sinx) / (1 - sinx))

'''
import numpy as np
import math
import matplotlib.pyplot as plt

# Параметры задачи
a = 0
b = math.pi / 6
y_a = 2  # y(0) = 2
y_b = 2.5 - 0.5 * math.log(3)  # y(pi/6) = 2.5 - 0.5*ln(3)

# Точное решение
def exact_solution(x):
    sin_x = math.sin(x)
    if abs(sin_x - 1) < 1e-10:
        return 2
    return sin_x + 2 - sin_x * math.log((1 + sin_x) / (1 - sin_x))

# Система ОДУ: y' = z, z' = tan(x)*z - 2*y
def ode_system(x, u):
    y, z = u
    return np.array([z, math.tan(x) * z - 2 * y])

# Метод Рунге-Кутты 4-го порядка
def runge_kutta_4(f, x0, u0, h, steps):
    x = np.linspace(x0, x0 + steps * h, steps + 1)
    u = np.zeros((steps + 1, 2))
    u[0] = u0
    for i in range(steps):
        k1 = h * f(x[i], u[i])
        k2 = h * f(x[i] + h/2, u[i] + k1/2)
        k3 = h * f(x[i] + h/2, u[i] + k2/2)
        k4 = h * f(x[i] + h, u[i] + k3)
        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x, u

# Функция для метода стрельбы: возвращает y(b) - y_b
def shoot(eta, h, steps):
    _, u = runge_kutta_4(ode_system, a, np.array([y_a, eta]), h, steps)
    return u[-1, 0] - y_b

# Метод секущих для поиска eta = y'(0)
def secant_method(f, eta0, eta1, h, steps, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        f0 = f(eta0, h, steps)
        f1 = f(eta1, h, steps)
        if abs(f1) < tol:
            return eta1
        eta_new = eta1 - f1 * (eta1 - eta0) / (f1 - f0)
        eta0, eta1 = eta1, eta_new
    return eta1

# Параметры сетки
steps_fine = 20  # Мелкая сетка (h)
steps_coarse = 10  # Грубая сетка (2h)
h_fine = (b - a) / steps_fine
h_coarse = (b - a) / steps_coarse

# Находим eta (y'(0)) для мелкой сетки
eta0_guess = -1.0
eta1_guess = -2.0
eta = secant_method(shoot, eta0_guess, eta1_guess, h_fine, steps_fine)
print(f"Найденное eta (y'(0)): {eta:.6f}")

# Решение на мелкой сетке (h)
x_fine, u_fine = runge_kutta_4(ode_system, a, np.array([y_a, eta]), h_fine, steps_fine)
y_fine = u_fine[:, 0]

# Решение на грубой сетке (2h) — решаем отдельно!
x_coarse, u_coarse = runge_kutta_4(ode_system, a, np.array([y_a, eta]), h_coarse, steps_coarse)
y_coarse = u_coarse[:, 0]

# Узлы грубой сетки должны совпадать с каждым вторым узлом мелкой
assert np.allclose(x_coarse, x_fine[::2]), "Сетки не согласованы!"

# Оценка погрешности по Рунге-Ромбергу (p=4 для RK4)
error_rr = np.abs(y_fine[::2] - y_coarse) / (2**4 - 1)

# Точное решение в узлах грубой сетки
y_exact_coarse = np.array([exact_solution(xi) for xi in x_coarse])

# Погрешность относительно точного решения
error_exact = np.abs(y_coarse - y_exact_coarse)

# Графики
# plt.figure(figsize=(12, 6))
# plt.plot(x_fine, y_fine, 'b-', label=f'Численное решение (h={h_fine:.4f})')
# plt.plot(x_coarse, y_exact_coarse, 'r--', label='Точное решение')
# plt.xlabel('x')
# plt.ylabel('y(x)')
# plt.title('Сравнение численного и точного решений')
# plt.legend()
# plt.grid()

# plt.figure(figsize=(12, 6))
# plt.plot(x_coarse, error_rr, 'g-', label='Погрешность (Рунге-Ромберг)')
# plt.plot(x_coarse, error_exact, 'm--', label='Погрешность (от точного решения)')
# plt.xlabel('x')
# plt.ylabel('Погрешность')
# plt.title('Оценка погрешности')
# plt.legend()
# plt.grid()

# plt.show()

# Вывод максимальных погрешностей
print(f"Максимальная погрешность (Рунге-Ромберг): {np.max(error_rr)}")
print(f"Максимальная погрешность (от точного решения): {np.max(error_exact)}")