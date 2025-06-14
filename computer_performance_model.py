"""
Programmer:Christian Nshuti Manzi
Packages Used: numpy, scipy, matplotlib
Approach: Solves a simplified ODE model of computer performance using ODEINT and visualizes the performance curve.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE model for performance decay
def performance_ode(P, t):
    return -0.1 * P - 1.8

def main():
    # Initial performance level
    P0 = 100

    # Time range from 0 to 50 seconds
    t = np.linspace(0, 50, 100)

    # Solve the ODE using odeint
    P = odeint(performance_ode, P0, t)

    # Plot the performance curve
    plt.figure(figsize=(10, 5))
    plt.plot(t, P, label='Performance P(t)', color='blue')
    plt.title('Computer System Performance Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_plot.png")  # Save plot as image
    print("Plot saved as performance_plot.png")

if __name__ == "__main__":
    main()
