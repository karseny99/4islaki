'''

    (x^2 - 1)y'' -2xy' + 2y = 0
    y(2) = 7
    y'(2) = 5
    x in [2, 3]
    h = 0.1

    y = x^2 + x + 1

'''

import numpy as np
import matplotlib.pyplot as plt

# Аналитическое решение
def exact_solution(x):
    return x**2 + x + 1

# Правая часть ОДУ y'' = f(x, y, y')
def equation(x, y, z):
    return (2*x*z - 2*y) / (x**2 - 1)

# Метод Эйлера
def euler_method(f, y0, z0, a, b, h):
    x = np.arange(a, b + h, h)
    n = len(x)
    y = np.zeros(n)
    z = np.zeros(n)
    y[0], z[0] = y0, z0
    
    for i in range(n - 1):
        y[i+1] = y[i] + h * z[i]
        z[i+1] = z[i] + h * f(x[i], y[i], z[i])
    return x, y

# Метод Рунге-Кутты 4-го порядка
def runge_kutta_method(f, y0, z0, a, b, h):
    x = np.arange(a, b + h, h)
    n = len(x)
    y = np.zeros(n)
    z = np.zeros(n)
    y[0], z[0] = y0, z0
    
    for i in range(n - 1):
        k1 = h * z[i]
        l1 = h * f(x[i], y[i], z[i])
        
        k2 = h * (z[i] + l1/2)
        l2 = h * f(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        
        k3 = h * (z[i] + l2/2)
        l3 = h * f(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        
        k4 = h * (z[i] + l3)
        l4 = h * f(x[i] + h, y[i] + k3, z[i] + l3)
        
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        z[i+1] = z[i] + (l1 + 2*l2 + 2*l3 + l4)/6
    return x, y

def runge_kutta_method_with_z(f, y0, z0, a, b, h):
    x = np.arange(a, b + h, h)
    n = len(x)
    y = np.zeros(n)
    z = np.zeros(n)
    y[0], z[0] = y0, z0
    
    for i in range(n - 1):
        k1 = h * z[i]
        l1 = h * f(x[i], y[i], z[i])
        
        k2 = h * (z[i] + l1/2)
        l2 = h * f(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        
        k3 = h * (z[i] + l2/2)
        l3 = h * f(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        
        k4 = h * (z[i] + l3)
        l4 = h * f(x[i] + h, y[i] + k3, z[i] + l3)
        
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        z[i+1] = z[i] + (l1 + 2*l2 + 2*l3 + l4)/6
    
    return x, y, z

# Метод Адамса 4-го порядка
def adams_method(f, y0, z0, a, b, h):
    x = np.arange(a, b + h, h)
    n = len(x)
    y = np.zeros(n)
    z = np.zeros(n)
    
    # Получаем первые 4 точки методом Рунге-Кутты
    rk_x, rk_y, rk_z = runge_kutta_method_with_z(f, y0, z0, a, a + 3*h, h)
    y[:4] = rk_y[:4]
    z[:4] = rk_z[:4]
    
    # Коэффициенты метода Адамса-Бэшфорта 4-го порядка
    coef = np.array([55, -59, 37, -9]) / 24
    
    for i in range(3, n - 1):
        # Прогноз для y и z
        f_vals = [f(x[i-j], y[i-j], z[i-j]) for j in range(1, 5)]
        z_pred = z[i] + h * np.dot(coef, f_vals)
        y_pred = y[i] + h * np.dot(coef, [z[i-j] for j in range(1, 5)])
        
        # Коррекция
        z[i+1] = z[i] + h/24 * (9*f(x[i+1], y_pred, z_pred) + 
                               19*f(x[i], y[i], z[i]) - 
                               5*f(x[i-1], y[i-1], z[i-1]) + 
                               f(x[i-2], y[i-2], z[i-2]))
        
        y[i+1] = y[i] + h/24 * (9*z_pred + 
                               19*z[i] - 
                               5*z[i-1] + 
                               z[i-2])
    
    return x, y


def plot_solutions(x, y, method_name):
    """Построение графиков численного и точного решений"""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label=f'{method_name} (численное)')
    plt.plot(x, exact_solution(x), 'r--', label='Точное решение')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(f'Сравнение решений: {method_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

def print_results(x, y, rr_error, method_name):
    """Вывод результатов с ошибками"""
    print(f"\n{method_name}:")
    print("x\t\ty_num\t\ty_exact\terror\t\tRR_error")
    
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        y_ex = exact_solution(xi)
        err = abs(yi - y_ex)
        
        # Для RR_error выводим только для четных индексов (где есть сравнение)
        if i % 2 == 0 and i//2 < len(rr_error):
            rr_err = rr_error[i//2]
            print(f"{xi:.1f}\t{yi:.6f}\t{y_ex:.6f}\t{err:.2e}\t{rr_err:.2e}")
        else:
            print(f"{xi:.1f}\t{yi:.6f}\t{y_ex:.6f}\t{err:.2e}\t-")

def solve_and_plot(method, f, y0, z0, a, b, h, method_name, p_order):
    """Полный цикл решения, анализа и визуализации"""
    # Решение с шагом h
    x_h, y_h = method(f, y0, z0, a, b, h)
    
    # Решение с шагом h/2 для оценки погрешности
    x_2h, y_2h = method(f, y0, z0, a, b, h/2)
    
    # Оценка погрешности Рунге-Ромберга
    rr_error = []
    for i in range(len(y_2h)):
        if 2*i < len(y_h):
            rr_error.append(abs(y_h[2*i] - y_2h[i]) / (2**p_order - 1))
    
    # Вывод результатов
    print_results(x_h, y_h, rr_error, method_name)
    
    # Построение графиков
    plot_solutions(x_h, y_h, method_name)
    
    return x_h, y_h, rr_error

# Основные параметры задачи
a, b = 2.0, 3.0  # Интервал
h = 0.1           # Шаг
y0, z0 = 7.0, 5.0 # Начальные условия
# Решение и анализ для каждого метода

print("="*50 + "\nАнализ методов решения ОДУ\n" + "="*50)

# Метод Эйлера (порядок точности p=1)
x_euler, y_euler, rr_euler = solve_and_plot(
    euler_method, equation, y0, z0, a, b, h, "Метод Эйлера", 1)

# Метод Рунге-Кутты (порядок точности p=4)
x_rk, y_rk, rr_rk = solve_and_plot(
    runge_kutta_method, equation, y0, z0, a, b, h, "Метод Рунге-Кутты", 4)

# Метод Адамса (порядок точности p=4)
x_adams, y_adams, rr_adams = solve_and_plot(
    adams_method, equation, y0, z0, a, b, h, "Метод Адамса", 4)
