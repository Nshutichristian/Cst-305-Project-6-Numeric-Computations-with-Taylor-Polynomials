import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def diff_eq(x, y):
    return [y[1], (x - y[0]) / (x**2 + 4)]

# Initial conditions (assuming y(0) = a0, y'(0) = a1)
a0 = 1  # Example value for a0
a1 = 0  # Example value for a1
y0 = [a0, a1]

# Solve over the interval [0, 5]
sol = solve_ivp(diff_eq, [0, 5], y0, t_eval=np.linspace(0, 0.5, 100))

# Plot the solution
plt.plot(sol.t, sol.y[0], label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution of the Differential Equation')
plt.legend()
plt.grid()
plt.show()