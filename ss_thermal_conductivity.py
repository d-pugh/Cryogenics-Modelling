"""
Thermal conductivity module for 304 Stainless Steel.
Uses NIST cryogenic curve fit coefficients.
"""

import numpy as np
import matplotlib.pyplot as plt

# NIST coefficients for thermal conductivity (304 stainless)
a = -1.4087
b = 1.3982
c = 0.2543
d = -0.6260
e = 0.2334
f = 0.4256
g = -0.4658
h = 0.1650
i = -0.0199


def thermal_conductivity(T):
    """
    Calculate thermal conductivity of 304 Stainless Steel using NIST coefficients.
    
    Parameters:
    T : float or array
        Temperature in Kelvin (valid range: 1-300 K)
    
    Returns:
    k : float or array
        Thermal conductivity in W/(m·K)
    """
    logT = np.log10(T)
    logy = (a + b*logT + c*logT**2 + d*logT**3 + e*logT**4 + 
            f*logT**5 + g*logT**6 + h*logT**7 + i*logT**8)
    return 10**logy


def plot_thermal_conductivity(T_min=0.8, T_max=5, num_points=1000):
    """
    Plot thermal conductivity vs temperature.
    
    Parameters:
    T_min : float
        Minimum temperature in K
    T_max : float
        Maximum temperature in K
    num_points : int
        Number of points for the plot
    """
    T = np.linspace(T_min, T_max, num_points)
    k = thermal_conductivity(T)
    
    plt.figure()
    plt.plot(T, k, label="NIST Fit", color='blue')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Thermal Conductivity (W/m·K)")
    plt.title("Thermal Conductivity vs Temperature\n304 Stainless Steel (NIST Fit)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Demo: plot thermal conductivity
    plot_thermal_conductivity()