

import numpy as np

def source_system(x):
    return np.array([
        2*x[0] - np.cos(x[1]),  # 2x - cos(y) = 0
        2*x[1] - np.exp(x[0])    # 2y - exp(x) = 0
    ])

def phi_system(x):
    return np.array([
        np.cos(x[1]) / 2,
        np.exp(x[0]) / 2
    ])

def simple_iteration_method(phi, x0, tol=1e-6, max_iter=100, dx=1e-6):
    x = x0.copy()
    iterations = 0
    n = len(x0)

    
    for iterations in range(max_iter):
        x_new = phi(x)

        J = np.zeros((n, n))
        for j in range(n):
            x_perturbed = x.copy()
            x_perturbed[j] += dx
            J[:, j] = (phi(x_perturbed) - phi(x)) / dx
        
        # Check matrix norm condition (||J|| < 1)
        matrix_norm = np.linalg.norm(J, 2)
        
        if matrix_norm >= 1:
            return x_new, iterations + 1, "DIVERGES (||J|| ≥ 1)"
        
        if np.linalg.norm(x_new - x, 2) < tol:
            return x_new, iterations + 1, "CONVERGED"
            
        x = x_new
    
    return x, max_iter, "MAX ITERATIONS REACHED"

def draw_graphic():
    import matplotlib.pyplot as plt
    tolerances = np.logspace(-1, -10, 40)  # From 1e-1 to 1e-10
    newton_iterations = []

    for tol in tolerances:
        newton_iterations.append(simple_iteration_method(phi_system, [0.5, 0.5], tol)[1])
       
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
solution, iterations, status = simple_iteration_method(phi_system, x0)

print(f"Solution: x = {solution[0]:.6f}, y = {solution[1]:.6f}")
print(f"Found in {iterations} iterations")
print(f"Status is {status}")

draw_graphic()