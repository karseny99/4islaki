from scipy.interpolate import CubicSpline

def tridiagonal_method(a, b, c, d):
    '''
    Решение трехдиагональной системы методом прогонки
    a(i) * x(i-1) + b(i) * x(i) + c(i) * x(i+1) = d(i)
    '''
    n = len(d)
    p = [0] * n
    q = [0] * n

    # Прямой ход
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = a[i] * p[i - 1] + b[i]
        p[i] = -c[i] / denominator
        q[i] = (d[i] - a[i] * q[i - 1]) / denominator
    
    # Обратный ход
    x = [0] * n
    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    
    return x

def calculate_c(x, y):
    """ Вычисление коэффициентов c с помощью метода прогонки """
    n = len(x)
    a = []  # main diag
    b = []  # low diag
    c = []  # up diag
    d = []  # free values
    
    for i in range(1, n - 1):
        h_i = x[i + 1] - x[i]
        h_i_ = x[i] - x[i - 1]
        a.append(2.0 * (h_i_ + h_i))
        b.append(h_i)
        c.append(h_i_)
        d.append(3.0 * ((y[i + 1] - y[i]) / h_i - (y[i] - y[i - 1]) / h_i_))

    c_i = tridiagonal_method(b, a, c, d)
    return c_i

def print_spline_polynomial(a, b, c, d, x_i, x_i1):
    """ Вывод многочлена сплайна для отрезка в читаемом виде """
    terms = []
    terms.append(f"{a:.6f}")
    
    if b != 0:
        sign = '+' if b > 0 else '-'
        terms.append(f" {sign} {abs(b):.6f}*(x - {x_i:.1f})")
    
    if c != 0:
        sign = '+' if c > 0 else '-'
        terms.append(f" {sign} {abs(c):.6f}*(x - {x_i:.1f})^2")
    
    if d != 0:
        sign = '+' if d > 0 else '-'
        terms.append(f" {sign} {abs(d):.6f}*(x - {x_i:.1f})^3")
    
    polynomial = ''.join(terms)
    print(f"S(x) = {polynomial}  для x ∈ [{x_i:.1f}, {x_i1:.1f}]")

def cubic_spline():
    import numpy as np

    x = np.array([0.1, 0.5, 0.9, 1.3, 1.7])
    y = np.array([-2.2026, -0.19315, 0.79464, 1.5624, 2.2306])
    x_ = 0.8

    n = len(x)
    c = calculate_c(x, y)
    c.insert(0, 0.0)  # Естественный сплайн: c0 = 0
    c.append(0.0)     # Естественный сплайн: cn = 0

    a = []
    b = []
    d = []
    border = -1
    
    print("\nКубические сплайны для каждого отрезка:")
    for i in range(n - 1):
        if x_ <= x[i + 1] and border == -1:
            border = i
            
        h_i = x[i + 1] - x[i]
        a_i = y[i]
        b_i = (y[i + 1] - y[i]) / h_i - h_i * (2 * c[i] + c[i + 1]) / 3
        d_i = (c[i + 1] - c[i]) / (3 * h_i)
        
        a.append(a_i)
        b.append(b_i)
        d.append(d_i)
        
        # Вывод многочлена для текущего отрезка
        print_spline_polynomial(a_i, b_i, c[i], d_i, x[i], x[i + 1])
        print(f"Коэффициенты: a={a_i:.6f}, b={b_i:.6f}, c={c[i]:.6f}, d={d_i:.6f}\n")

    if border != -1:
        res = a[border] + b[border]*(x_ - x[border]) + c[border]*(x_ - x[border])**2 + d[border]*(x_ - x[border])**3
        print(f"\nЗначение кубического сплайна в точке {x_}: {res:.6f}")

cubic_spline()