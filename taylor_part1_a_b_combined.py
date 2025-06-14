# Programmer:Christian Nshuti Manzi
# CST-305 Benchmark Project - Part 1(a) and 1(b)
# Taylor Polynomial Approximation for Two ODEs
# Part 1(a): y'' - 2xy' + x^2y = 0 with y(0) = 1, y'(0) = -1
# Part 1(b): y'' = (x - 2)y' - 2y with y(3) = 6, y'(3) = 1


import numpy as np
import matplotlib.pyplot as plt

### --- PART 1(A) --- ###
x0_a = 0
y0_a = 1
y1_0_a = -1

def y2_a(x, y1, y):
    return 2 * x * y1 - x**2 * y

def y3_a(x, y, y1, y2_val):
    return 2 * y1 + 2 * x * y2_val - 2 * x * y - x**2 * y1

def y4_a(x, y, y1, y2_val, y3_val):
    return 4 * y2_val + 2 * x * y3_val - 2 * y - 4 * x * y1 - x**2 * y2_val

x = x0_a
y = y0_a
y1 = y1_0_a
y2_val = y2_a(x, y1, y)
y3_val = y3_a(x, y, y1, y2_val)
y4_val = y4_a(x, y, y1, y2_val, y3_val)

def taylor_y_a(x_val):
    return y + y1 * (x_val - x0_a) + (y2_val / 2) * (x_val - x0_a)**2 + (y3_val / 6) * (x_val - x0_a)**3 + (y4_val / 24) * (x_val - x0_a)**4

x_vals_a = np.linspace(0, 4, 100)
y_vals_a = [taylor_y_a(val) for val in x_vals_a]

plt.figure(1)
plt.plot(x_vals_a, y_vals_a, label="Taylor Polynomial Approximation", color='darkblue', linestyle='--', linewidth=2)
plt.title("Part 1(a): Taylor Polynomial for y'' - 2xy' + x²y = 0")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.legend()

### --- PART 1(B) --- ###
x0_b = 3
y0_b = 6
y1_0_b = 1

def y2_b(x, y1, y):
    return (x - 2) * y1 - 2 * y

def taylor_y_b(x_val):
    y2_val_b = y2_b(x0_b, y1_0_b, y0_b)
    return y0_b + y1_0_b * (x_val - x0_b) + (y2_val_b / 2) * (x_val - x0_b)**2

x_vals_b = np.linspace(2, 4, 100)
y_vals_b = [taylor_y_b(val) for val in x_vals_b]

plt.figure(2)
plt.plot(x_vals_b, y_vals_b, label="Taylor Polynomial (2nd Order)", color='green', linestyle='-', linewidth=2)
plt.title("Part 1(b): Taylor Polynomial for y'' = (x-2)y' - 2y")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.legend()

plt.show()
