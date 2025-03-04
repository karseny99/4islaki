
matrix = [
    [-5, -1, -3, -1, 18],
    [-2, 0, 8, -4, -12],
    [-7, -2, 2, -2, 6],
    [2, -4, -4, 4, -12],
]

import numpy as np
import copy

def gauss_forward(matrix):

    matrix_copy = copy.deepcopy(matrix)

    n = len(matrix_copy)
    m = len(matrix_copy[0])

    for i in range(n):
        
        # row replacement with max val in col
        max_row = max(range(i, n), key=lambda k: abs(matrix_copy[k][i]))
        matrix_copy[i], matrix_copy[max_row] = matrix_copy[max_row], matrix_copy[i]

        # row normalizing (diag element becomes 1)
        div = matrix_copy[i][i]
        for j in range(i, m):
            matrix_copy[i][j] /= div

        # row substraction 
        for row in range(i + 1, n):
            mult = matrix_copy[row][i]
            for j in range(m):
                matrix_copy[row][j] -= mult * matrix_copy[i][j]

    return matrix_copy


def gauss_backward(matrix):

    matrix_copy = copy.deepcopy(matrix)

    n = len(matrix_copy)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = matrix_copy[i][n]
        for j in range(i + 1, n):
            x[i] -= matrix_copy[i][j] * x[j]

    return x


def gauss_determinant(matrix):
    
    '''
        input matrix -> upper triangular -> det is multuplication of diagonals
    '''

    matrix_copy = copy.deepcopy(matrix)

    determinant = 1

    n = len(matrix_copy)

    for i in range(n):
        
        # row replacement with max val in col
        max_row = max(range(i, n), key=lambda k: abs(matrix_copy[k][i]))

        # switch sign if rows are swapped
        if max_row != i:
            determinant *= -1

        matrix_copy[i], matrix_copy[max_row] = matrix_copy[max_row], matrix_copy[i]

        # row normalizing (diag element becomes 1)
        div = matrix_copy[i][i]
        # multiply determinant by leading elem
        determinant *= div
        for j in range(i, n):
            matrix_copy[i][j] /= div

        # row substraction 
        for row in range(i + 1, n):
            mult = matrix_copy[row][i]

            for j in range(n):
                matrix_copy[row][j] -= mult * matrix_copy[i][j]

    return determinant




def first_part(matrix): 

    def slay(matrix):
        return gauss_backward(gauss_forward(matrix))

    def check_with_np(A, b):
        x = np.linalg.solve(A, b)
        return x

    A = np.array([
        [-5, -1, -3, -1],
        [-2, 0, 8, -4],
        [-7, -2, 2, -2],
        [2, -4, -4, 4],
    ])

    b = np.array([18, -12, 6, -12])

    print(f"numpy: {check_with_np(A, b)}")
    print(f"my impl: {slay(matrix)}")


def second_part(matrix):
    def inv(matrix):

        '''
            A * A(-1) = I  
            [A | I] --> [I | A(-1)]
        '''

        n = len(matrix)
        augmented_matrix = []

        for i in range(n):
            new_row = matrix[i].copy()
            for j in range(n):
                new_row.append(1 if i == j else 0)
            augmented_matrix.append(new_row)

        augmented_matrix = gauss_forward(augmented_matrix)

        # backward gauss
        for i in range(n - 1, -1, -1):
            for k in range(i - 1, -1, -1):
                mult = augmented_matrix[k][i]
                for j in range(2 * n):
                    augmented_matrix[k][j] -= mult * augmented_matrix[i][j]

        inv = [row[n:] for row in augmented_matrix]
        return inv
    
    A = np.array([
        [-5, -1, -3, -1],
        [-2, 0, 8, -4],
        [-7, -2, 2, -2],
        [2, -4, -4, 4],
    ])

    matrix_for_inverse = [row[:-1] for row in matrix]

    print(f"numpy: {np.linalg.inv(A)}")
    print(f"my impl: {np.matrix(inv(matrix_for_inverse))}")


def third_part(matrix):

    det = gauss_determinant(matrix)

    A = np.array([
        [-5, -1, -3, -1],
        [-2, 0, 8, -4],
        [-7, -2, 2, -2],
        [2, -4, -4, 4],
    ])

    print(f"NumPy determinant: {np.linalg.det(A)}")
    print(f"determinant: {det}")


def tridiagonal_method(matrix):

    '''
        a(i) * x(i-1) + b(i) * x(i) + c(i) * x(i+1) = d(i)
        x(i-1) = p(i-1) * x(i) + q(i-1)
    '''

    a, b, c, d = copy.deepcopy(matrix)

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



def fourth_part():

    matrix = [[0, 2, -9, -4, 7],        # a - under diag
            [18, -9, 21, -10, 12],      # b - diag
            [-9, -4, -8, 5, 0],         # c - above  
            [-81, 71, -39, 64, 3]]      # d - constants

    
    def is_matrix_good(matrix):
        first_cond = True
        second_cond = True

        for i in range(len(matrix[0])):
            first_cond = first_cond and (abs(matrix[1][i]) >= abs(matrix[0][i]) + abs(matrix[2][i]))
            second_cond = second_cond or (abs(matrix[1][i]) > abs(matrix[0][i]) + abs(matrix[2][i]))

        return first_cond and second_cond

    def check_with_np(A, b):
        x = np.linalg.solve(A, b)
        return x

    A = np.array([
        [18, -9, 0, 0, 0],
        [2, -9, -4, 0, 0],
        [0, -9, 21, -8, 0],
        [0, 0, -4, -10, 5],
        [0, 0, 0, 7, 12]
    ])

    b = np.array([-81, 71, -39, 64, 3])

    print(f"numpy: {check_with_np(A, b)}")
    print(f"my impl: {tridiagonal_method(matrix)}\nIs matrix good? {is_matrix_good(matrix)}")



def fixed_point_iterations(matrix, epsilon=1e-6, max_iter=1000):

    '''
        x(k+1) = ( b - A_i * x(k) ) * A(-1))
    '''

    matrix_copy = copy.deepcopy(matrix)
    n = len(matrix_copy)
    A = [row[:-1] for row in matrix_copy]
    b = [row[-1] for row in matrix_copy]

    x = [0.0] * n

    for iter in range(max_iter):
        x_new = [0.0] * n

        for i in range(n):
            sm = 0.0
            for j in range(n):
                if i != j:
                    sm += A[i][j] * x[j]
            x_new[i] = (b[i] - sm) / A[i][i]
        
        diff = 0.0
        for i in range(n):
            diff += abs(x_new[i] - x[i])
        
        if diff < epsilon:
            print(f"Converged by {iter} iterations")
            return x_new
        
        x = x_new
    
    print(f"Max iter exceeded")
    return x


def seidel_method(matrix, epsilon=1e-6, max_iter=1000):
    '''
        x(k+1) = ( b - L * x(k+1) - U * x(k) ) / A(-1)
    '''

    matrix_copy = copy.deepcopy(matrix)
    n = len(matrix_copy)
    A = [row[:-1] for row in matrix_copy]
    b = [row[-1] for row in matrix_copy]

    x = [0.0] * n

    for iter in range(max_iter):
        x_prev = x.copy()

        for i in range(n):
            sm = 0.0
            for j in range(n):
                if i != j:
                    sm += A[i][j] * x[j]
            x[i] = (b[i] - sm) / A[i][i]
        
        diff = 0.0
        for i in range(n):
            diff += abs(x_prev[i] - x[i])
        
        if diff < epsilon:
            print(f"Converged by {iter} iterations")
            return x
            
    print(f"Max iter exceeded")
    return x


def fifth_part():

    matrix = [[21, -6, -9, -4, 127],
              [-6, 20, -4, 2, -144],
              [-2, -7, -20, 3, 236],
              [4, 9, 6, 24, 0, -5]]
    
    A = np.array([
        [21, -6, -9, -4],
        [-6, 20, -4, 2],
        [-2, -7, -20, 3],
        [4, 9, 6, 24]
    ])

    b = np.array([127, -144, 236, -5])

    def check_with_np(A, b):
        x = np.linalg.solve(A, b)
        return x

    print(f"numpy: {check_with_np(A, b)}")
    print(f"fixed-point iteration: {fixed_point_iterations(matrix)}")
    print(f"Seidel method: {seidel_method(matrix)}")


if __name__ == "__main__":
    first_part(matrix) # gauss 
    print('\n')
    second_part(matrix) # inverse
    print('\n')
    third_part(matrix) # determinant
    print('\n')
    fourth_part() # progonka
    print('\n')
    fifth_part() # fixed-point iteration