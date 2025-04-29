'''

    y = x^2 / (x^4 + 256)

    X(0) = 0, X(k) = 2

    h(1) = 0.5, h(2) = 0.25


'''

import numpy as np
import scipy.integrate as integrate

def rectangle_method(x, f):
    h = x[1] - x[0]
    f = np.vectorize(f)
    return h * sum(f((x[:-1] + x[1:]) / 2))
     

def trapezoidal_method(x, f):
    h = x[1] - x[0]
    f = np.vectorize(f)
    return h * ((f(x[0])+f(x[-1]))/2 + sum(f(x[1:-1])))

def simpson_method(x, f):
    h = x[1] - x[0]
    f = np.vectorize(f)
    return h / 3 * (f(x[0]) + f(x[-1]) + 4*sum(f(x[1:-1:2])) + 2*sum(f(x[2:-1:2])))


def runge_romberg_richardson_method(F_h, F_kh, k, p):
    return F_h + (F_h - F_kh) / (k**p - 1)


def main():
    y = lambda x: x ** 2 / (x ** 4 + 256)
    x0 = 0
    xk = 2

    h1 = 0.5
    h2 = 0.25

    x1 = np.linspace(x0, xk, int((xk - x0) / h1)+1)
    x2 = np.linspace(x0, xk, int((xk - x0) / h2)+1)

    ans = integrate.quad(y, x0, xk)[0]

    # y = lambda x: x / (3*x+4)**2
    # x0, xk = -1, 1
    # h1, h2 = 0.5, 0.25
    # x1 = np.linspace(x0, xk, int((xk - x0) / h1)+1)
    # x2 = np.linspace(x0, xk, int((xk - x0) / h2)+1)

    # ans = integrate.quad(y, x0, xk)

    res1 = [
        rectangle_method(x1, y),
        trapezoidal_method(x1, y),
        simpson_method(x1, y),
    ]

    res2 = [
        rectangle_method(x2, y),
        trapezoidal_method(x2, y),
        simpson_method(x2, y),
    ]

    rrr = [
        runge_romberg_richardson_method(res2[0], res1[0], h1 / h2, 2),
        runge_romberg_richardson_method(res2[1], res1[1], h1 / h2, 2),
        runge_romberg_richardson_method(res2[2], res1[2], h1 / h2, 4),
    ]

    err = [
        abs(rrr[0] - ans),
        abs(rrr[1] - ans),
        abs(rrr[2] - ans),
    ]

    print("\t\t\tRect\t\tTrapez\t\t\tSimpson")
    print(f"h1 = {h1}:\t{res1}")
    print(f"h2 = {h2}:\t{res2}")
    print(f"error rrr...:\t{rrr}")
    print(f"abs error:\t{err}")
    print(f"actual is:\t{integrate.quad(y, x0, xk)[0]}")

if __name__ == "__main__":
    main()