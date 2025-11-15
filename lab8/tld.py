from copy import deepcopy


def tridiagonal_method(a, b, c, d):

    '''
        a(i) * x(i-1) + b(i) * x(i) + c(i) * x(i+1) = d(i)
        x(i-1) = p(i-1) * x(i) + q(i-1)
    '''

    # a, b, c, d = deepcopy(matrix)

    n = len(d)

    p = [0] * n
    q = [0] * n

    # forward

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = a[i] * p[i - 1] + b[i]
        p[i] = -c[i] / denominator
        q[i] = (d[i] - a[i] * q[i - 1]) / denominator
    
    # backward
        
    x = [0] * n
    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    
    return x