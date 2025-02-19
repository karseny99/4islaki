
matrix = [
    [-5, -1, -3, -1, 18],
    [-2, 0, 8, -4, -12],
    [-7, -2, 2, -2, 6],
    [2, -4, -4, 4, -12],
]

import numpy as np

def gauss_forward(matrix):

    n = len(matrix)
    m = len(matrix[0])

    for i in range(n):
        
        # row replacement with max val in col
        max_row = max(range(i, n), key=lambda k: abs(matrix[k][i]))
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

        # row normalizing (diag element becomes 1)
        div = matrix[i][i]
        for j in range(i, m):
            matrix[i][j] /= div

        # row substraction 
        for row in range(i + 1, n):
            mult = matrix[row][i]
            for j in range(m):
                matrix[row][j] -= mult * matrix[i][j]

    return matrix


def gauss_backward(matrix):
    n = len(matrix)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = matrix[i][n]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]

    return x


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

    # Убираем последний столбец (правую часть СЛАУ) для нахождения обратной матрицы
    matrix_for_inverse = [row[:-1] for row in matrix]

    print(f"numpy: {np.linalg.inv(A)}")
    print(f"my impl: {np.matrix(inv(matrix_for_inverse))}")



if __name__ == "__main__":
    # first_part(matrix)
    second_part(matrix)
    # third_part()