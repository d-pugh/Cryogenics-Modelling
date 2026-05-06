import numpy as np

R = 8.314 # molar gas constant J/(mol K)

# Helium evaporation rate calculation for cryogenic system.
def calculate_n_l(Q_dot, operating_hours, latent_heat):
    """
    Calculate total helium moles evaporated.
    Parameters:
    Q_dot : float
        Heat leak in watts (J/s)
    operating_hours : float
        Desired hold time in hours
    latent_heat : float
        Latent heat of vaporization for helium in J/mol
    Returns:    
    n : float
    """
    # Evaporation rate (mol/s)
    n_dot = Q_dot / latent_heat
    
    # Operating time (seconds)
    t = operating_hours * 3600
    
    # Total moles evaporated
    n = n_dot * t
    
    return n

def calculate_n_e(P, T, V_evap, n_liq, V_mol):

    V_gas = V_evap - n_liq * V_mol

    return (P * V_gas) / (R * T)

def T_z(z, T_hot, T_cold, L):
    """
    Calculate the temperature at a given height assuming a linear gradient.

    """
    grad_T = (T_hot - T_cold) / L
    T = T_cold + grad_T * z

    return T

def calculate_n_t(P, radius, length, T_func, int_points=1000):
    """
    Calculate the total moles of gas in a tube with a temperature gradient.

    """
    A = np.pi * radius**2
    heights = np.linspace(0, length, int_points)
    integrand = 1/T_func(heights)
    n_t = (A*P/R) * np.trapezoid(integrand, heights) 
    return n_t

def n_adsorbed(P, T, n_max, E, P0_val):
    """
    Calculate amount of gas adsorbed according to the D-R equation.
    """    
    exponent = (R * T / E) * np.log(P0_val / P)
    return n_max * np.exp(-exponent ** 2)

def calculate_n_p(P, T, V_pump, n_max, E, P0_val):
    """
    Calculate the total moles of gas in the system including adsorbed and free gas.
    """
    n_ads = n_adsorbed(P, T, n_max, E, P0_val)
    n_gas = (P * V_pump) / (R * T)

    return n_gas+n_ads