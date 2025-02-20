
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


if __name__ == "__main__":
    first_part(matrix)
    second_part(matrix)
    third_part(matrix)