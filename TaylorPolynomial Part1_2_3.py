"""
CST-305 Benchmark Project - Combined Parts 1, 2, and 3
Programmer: Chritian Nshuti Manzi
professor: Ricardo Citro
institution: Grand canyon University
date: 6/14/2025
Packages: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# -------------------------------
# PART 1(A) - Taylor Polynomial
# -------------------------------
def part1a():
    x0 = 0
    y0 = 1
    y1_0 = -1

    def y2(x, y1, y): return 2 * x * y1 - x**2 * y
    def y3(x, y, y1, y2_val): return 2 * y1 + 2 * x * y2_val - 2 * x * y - x**2 * y1
    def y4(x, y, y1, y2_val, y3_val): return 4 * y2_val + 2 * x * y3_val - 2 * y - 4 * x * y1 - x**2 * y2_val

    x = x0
    y = y0
    y1 = y1_0
    y2_val = y2(x, y1, y)
    y3_val = y3(x, y, y1, y2_val)
    y4_val = y4(x, y, y1, y2_val, y3_val)

    def taylor_y(x_val):
        return y + y1 * (x_val - x0) + (y2_val / 2) * (x_val - x0)**2 + \
               (y3_val / 6) * (x_val - x0)**3 + (y4_val / 24) * (x_val - x0)**4

    x_vals = np.linspace(0, 4, 100)
    y_vals = [taylor_y(val) for val in x_vals]

    plt.figure()
    plt.plot(x_vals, y_vals, label="Part 1(a): y(x)", color='darkblue')
    plt.title("Part 1(a): Taylor Polynomial Approximation")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()

# -------------------------------
# PART 1(B) - Taylor Polynomial
# -------------------------------
def part1b():
    x0 = 3
    y0 = 6
    y1_0 = 1

    def y2(x, y1, y): return (x - 2) * y1 - 2 * y

    def taylor_y(x_val):
        y2_val = y2(x0, y1_0, y0)
        return y0 + y1_0 * (x_val - x0) + (y2_val / 2) * (x_val - x0)**2

    x_vals = np.linspace(2, 4, 100)
    y_vals = [taylor_y(val) for val in x_vals]

    plt.figure()
    plt.plot(x_vals, y_vals, label="Part 1(b): y(x)", color='green')
    plt.title("Part 1(b): Taylor Approximation for 2nd ODE")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()

# -------------------------------
# PART 2 - solve_ivp Numerical
# -------------------------------
def part2():
    def diff_eq(x, y):
        return [y[1], (x - y[0]) / (x**2 + 4)]

    a0 = 1
    a1 = 0
    y0 = [a0, a1]
    sol = solve_ivp(diff_eq, [0, 0.5], y0, t_eval=np.linspace(0, 0.5, 100))

    plt.figure()
    plt.plot(sol.t, sol.y[0], label='Part 2: y(x)', color='orange')
    plt.title("Part 2: Numerical Solution Using solve_ivp")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()

# -------------------------------
# PART 3 - ODEINT Performance Model
# -------------------------------
def part3():
    def performance_ode(P, t): return -0.1 * P - 1.8

    P0 = 100
    t = np.linspace(0, 50, 100)
    P = odeint(performance_ode, P0, t)

    plt.figure(figsize=(10, 5))
    plt.plot(t, P, label='Part 3: Performance P(t)', color='blue')
    plt.title("Part 3: Computer System Performance Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Performance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():
    part1a()
    part1b()
    part2()
    part3()
    plt.show()

if __name__ == "__main__":
    main()
