# Taylor Polynomial Approximation for the Differential Equation:
# y'' - 2xy' + x^2y = 0
# with initial conditions: y(0) = 1, y'(0) = -1

import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
x0 = 0
y0 = 1
y1_0 = -1  # y'(0)

# Define the function for y''
def y2(x, y1, y):
    return 2 * x * y1 - x**2 * y

# Define the function for y'''
def y3(x, y, y1, y2_val):
    return 2 * y1 + 2 * x * y2_val - 2 * x * y - x**2 * y1

# Define the function for y⁽⁴⁾
def y4(x, y, y1, y2_val, y3_val):
    return 4 * y2_val + 2 * x * y3_val - 2 * y - 4 * x * y1 - x**2 * y2_val

# Compute all derivatives at x0
x = x0
y = y0
y1 = y1_0
y2_val = y2(x, y1, y)
y3_val = y3(x, y, y1, y2_val)
y4_val = y4(x, y, y1, y2_val, y3_val)

# Define the Taylor polynomial function
def taylor_y(x_val):
    return y + y1 * (x_val - x0) + (y2_val / 2) * (x_val - x0)**2 + (y3_val / 6) * (x_val - x0)**3 + (y4_val / 24) * (x_val - x0)**4

# Estimate y at x = 3.5
x_target = 3.5
y_approx = taylor_y(x_target)
print(f"Estimated y(3.5) ≈ {y_approx}")

# Plot the Taylor approximation
x_vals = np.linspace(0, 4, 100)
y_vals = [taylor_y(val) for val in x_vals]

plt.plot(x_vals, y_vals, label="Function Y(x)")
plt.title("Function Y(x) part A and Its Taylor Polynomial")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.show()