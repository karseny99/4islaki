'''
    y = ln(x) + x
    
    x_i = 0.1, 0.5, 0.9, 1.3
    x_i = 0.1, 0.5, 1.1, 1.3

    x* = 0.8
'''

import math

source_eq = lambda x: math.log(x) + x

def lagrange_polynomial_str(X, Y):
    n = len(X)
    polynomial_terms = []
    
    for i in range(n):
        numerator_parts = []
        denominator = 1.0
        
        for j in range(n):
            if j != i:
                numerator_parts.append(f"(x - {X[j]:.1f})")
                denominator *= (X[i] - X[j])
        
        numerator = " * ".join(numerator_parts)
        basis_poly = f"({numerator})"
        
        term = f"{Y[i] / denominator:.6f} * {basis_poly}"
        polynomial_terms.append(term)
    
    polynomial = " + ".join(polynomial_terms)
    return polynomial

def newton_polynomial_str(X, Y):
    from sympy import symbols, expand
    n = len(X)
    x = symbols('x')
    
    # Таблица разделённых разностей
    divided_diff = [[0] * n for _ in range(n)]
    for i in range(n):
        divided_diff[i][0] = Y[i]
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (X[i + j] - X[i])
    
    # Формируем многочлен
    polynomial = divided_diff[0][0]
    poly_str = f"{divided_diff[0][0]:.6f}"
    
    for i in range(1, n):
        term = divided_diff[0][i]
        term_str = f"{divided_diff[0][i]:.6f}"
        
        for j in range(i):
            term *= (x - X[j])
            term_str += f" * (x - {X[j]:.1f})"
        
        polynomial += term
        poly_str += " + " + term_str
    
    return poly_str

def lagrange_interpolation(X, Y, _x): 
    '''
        L(x) = sum(i=1..n)(
           y_i * prod(j=1..n)( [_x - x(j)] / [x(i) - x(j)] ) 
        )
    '''

    n = len(X)

    res = 0.0
    # mn = [0.0] * n
    for i in range(n):
        tmp = Y[i]
        
        for j in range(n):
            if j != i:
                tmp *= (_x - X[j]) / (X[i] - X[j])

        res += tmp
    return res

def newton_interpolation(X, Y, _x):
    '''
        P(x) = sum(i=1..n)(
           f[x[0..n]] * (x - x0)...(x - x(n))
        )

        f[x[i]] = y_i, f[x[i], x[j]] = ( f[x[j]] - f[x[i]] ) / (x(j) - x(i))

    '''

    import numpy as np
    n = len(X)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = Y

    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = \
            (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / \
            (X[i + j] - X[i])

    res = divided_diff[0][0]
    for i in range(1, n):
        tmp = divided_diff[0][i]

        for j in range(i):
            tmp *= (_x - X[j])

        res += tmp

    return res



X1 = [0.1 , 0.5 , 0.9 , 1.3]
Y1 = [source_eq(x) for x in X1]

X2 = [0.1 , 0.5 , 1.1 , 1.3]
Y2 = [source_eq(x) for x in X2]

_x = 0.8
_y = source_eq(_x)

err1 = abs(lagrange_interpolation(X1, Y1, _x) - _y)
err2 = abs(lagrange_interpolation(X2, Y2, _x) - _y)

print("Lagrange: \n")
print(lagrange_polynomial_str(X1, Y1))
print('\n')
print("Newton: \n")
print(newton_polynomial_str(X1, Y1))
print('\n')
# print(_y, lagrange_interpolation(X2, Y2, _x))
print(f"Lagrange error for (X1, Y1): {err1},\n for (X2, Y2): {err2}\n")

err1 = abs(newton_interpolation(X1, Y1, _x) - _y)
err2 = abs(newton_interpolation(X2, Y2, _x) - _y)
print(f"Newton error for (X1, Y1): {err1},\n for (X2, Y2): {err2}\n")



# def draw_graphic():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     dots = np.linspace(0.01, 3, 40)
#     lagr = []
#     new = []

#     for tol in dots:
#         lagr.append(lagrange_interpolation(X1, Y1, tol))
#         new.append(newton_interpolation(X1, Y1, tol))
       
#     plt.figure(figsize=(10, 6))
#     plt.plot(dots, lagr, 'bo-', label="lagrange interpolation")
#     plt.plot(dots, new, 'yo-', label="newton interpolation")
#     plt.plot(dots, [source_eq(x) for x in dots],'g--', label="ln(x) + x")
#     plt.grid(True)
#     plt.scatter(X1, Y1, color='red', s=[90 for x in X1])
#     plt.legend()
#     plt.show()

# draw_graphic()  