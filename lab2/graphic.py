import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('iter.txt')  
simple_iter, newton, eps = data[:, 0], data[:, 1], data[:, 2]

plt.figure(figsize=(12, 6))

plt.plot(eps, simple_iter, 'bo-', label='Simple Iteration', markersize=8, linewidth=2)
plt.plot(eps, newton, 'rs-', label="Newton's Method", markersize=8, linewidth=2)

plt.gca().invert_xaxis()
plt.xscale('log')

plt.xlabel('Tolerance (eps)')
plt.ylabel('Iterations')
plt.title('Iterations vs Tolerance (Log-Log Scale)')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()