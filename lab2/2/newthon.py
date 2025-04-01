'''

    2x - cosy = 0
    2y - exp(x) = 0

    - J*delta = fx
    
    - x(n+1) = x(n) - delta
    
'''

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


def solve_linear_system(matrix):
    return gauss_backward(gauss_forward(matrix))


def newton_system(f, x0, tol=1e-6, max_iter=100, dx=1e-6):
    x = np.array(x0, dtype=float)
    n = len(x)
    
    for iteration in range(max_iter):

        fx = f(x)
        
        J = np.zeros((n, n))
        for j in range(n):
            x_perturbed = x.copy()
            x_perturbed[j] += dx
            
            J[:, j] = (f(x_perturbed) - fx) / dx
        
        try:
            augmented_matrix = np.hstack((J, -fx.reshape(-1, 1)))
            delta_x = solve_linear_system(augmented_matrix)
        except np.linalg.LinAlgError:
            raise RuntimeError("Singular Jacobian encountered")
        
        x += delta_x
        
        if np.linalg.norm(delta_x) < tol:
            return x, iteration + 1
    
    raise RuntimeError(f"Failed to converge in {max_iter} iterations")


def source_system(x):
    return np.array([
        2*x[0] - np.cos(x[1]),  # 2x - cos(y) = 0
        2*x[1] - np.exp(x[0])    # 2y - exp(x) = 0
    ])


def draw_graphic():
    import matplotlib.pyplot as plt
    tolerances = np.logspace(-1, -10, 40)  # From 1e-1 to 1e-10
    newton_iterations = []

    for tol in tolerances:
        newton_iterations.append(newton_system(source_system, [0.5, 0.5], tol)[1])
       
    plt.figure(figsize=(10, 6))
    plt.semilogx(tolerances, newton_iterations, 'bo-', label='Our Newton')
    plt.gca().invert_xaxis()  
    plt.xlabel('Tolerance (ε)')
    plt.ylabel('Iterations')
    plt.title('Convergence: Iterations vs Tolerance')
    plt.legend()
    plt.grid(True)
    plt.show()


x0 = np.array([0.5, 0.5])
solution, iterations = newton_system(source_system, x0)

print(f"Solution: x = {solution[0]:.6f}, y = {solution[1]:.6f}")
print(f"Found in {iterations} iterations")

from scipy.optimize import fsolve
numpy_solution = fsolve(source_system, [0.5, 0.5])
print("NumPy fsolve solution:", numpy_solution)

draw_graphic()

