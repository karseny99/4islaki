'''

    X* = 0.8

    x = [0.1, 0.5 , 0.9, 1.3, 1.7]
    f = [-1.8647, -0.63212, 1.0, 3.7183, 9.3891]
    
'''


import numpy as np

def tridiagonal_method(a, b, c, d):
    n = len(d)
    p = np.zeros(n)
    q = np.zeros(n)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = a[i] * p[i-1] + b[i]
        p[i] = -c[i] / denominator
        q[i] = (d[i] - a[i] * q[i-1]) / denominator

    x = np.zeros(n)
    x[-1] = q[-1]

    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]

    return x

def natural_cubic_spline(x, f, _x):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(f) / h

    a = np.zeros(n-2)
    b = np.zeros(n-2)
    c = np.zeros(n-2)
    d = np.zeros(n-2)

    for i in range(1, n-1):
        a[i-1] = h[i-1] / 6
        b[i-1] = (h[i-1] + h[i]) / 3
        c[i-1] = h[i] / 6
        d[i-1] = delta[i] - delta[i-1]

    M_inner = tridiagonal_method(a, b, c, d)
    M = np.zeros(n)
    M[1:-1] = M_inner

    for i in range(n-1):
        if x[i] <= _x <= x[i+1]:
            break
    
    dx = _x - x[i]
    a_i = f[i]
    b_i = (f[i+1] - f[i]) / h[i] - h[i] * (2 * M[i] + M[i+1]) / 6
    c_i = M[i] / 2
    d_i = (M[i+1] - M[i]) / (6 * h[i])
    
    S_x = a_i + b_i * dx + c_i * dx**2 + d_i * dx**3
    return S_x

x = np.array([0.1, 0.5, 0.9, 1.3, 1.7])
f = np.array([-1.8647, -0.63212, 1.0, 3.7183, 9.3891])
_x = 0.8

res = natural_cubic_spline(x, f, _x)
print("f(x*) = ", res)