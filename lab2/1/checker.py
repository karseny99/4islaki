import numpy as np
from scipy.optimize import fsolve

def equation(x):
    return x * np.exp(x) + x**2 - 1

root = fsolve(equation, 0.5)[0] 
print(f"Root: {root:.16f}")
