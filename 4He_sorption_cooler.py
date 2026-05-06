import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import sorption_cooler_functions as scf
from ss_thermal_conductivity import thermal_conductivity
plt.rcParams['font.family'] = 'Times New Roman'
f = 12
R = 8.314               # J/(mol K)
n_max = 0.2               # maximum adsorbate capacity (mol)
E = 1000.0              # characteristic adsorption energy (J/mol)
T_STP = 273.15         # K
P_STP = 101325         # Pa
L4 = 84.0               # latent heat of vaporization (J/mol)
V_mol_liq4 = 27.57e-6 # m^3/mol, molar volume of liquid He-4 at 1K and SVP
r = 0.5e-2              # m, radius of pumping tube
l_pump = 10e-2         # m, length of pumping tube
t = 0.25e-3             # m, wall thickness of pumping tube

# Critical point from literature
T_c = 5.2                     # K
P_c = 0.227e6                 # Pa

file_path = os.path.join(os.path.dirname(__file__), 'vapour_pressure_He4.txt')
T_data, P_data = np.loadtxt(file_path, delimiter=',', skiprows=1, unpack=True)
P_data = P_data * 100          # mbar → Pa

lnP_data = np.log(P_data)
lnP_interp = PchipInterpolator(T_data, lnP_data, extrapolate=False)
Tmin, Tmax = T_data[0], T_data[-1]
slope_low = (lnP_data[1] - lnP_data[0]) / (T_data[1] - T_data[0])

def P0_helium(T):
    """Saturation pressure of He-4 [Pa]. Works for scalars and arrays."""
    T = np.asarray(T, dtype=float)
    scalar = T.ndim == 0
    if scalar:
        T = np.array([T])
    lnP = np.empty_like(T)
    mask_low = T < Tmin
    mask_mid = (T >= Tmin) & (T <= T_c)
    mask_high = T > T_c
    lnP[mask_low] = lnP_data[0] + slope_low * (T[mask_low] - Tmin)
    lnP[mask_mid] = lnP_interp(T[mask_mid])
    lnP[mask_high] = np.log(P_c) + 2.0 * np.log(T[mask_high] / T_c)
    result = np.exp(lnP)
    if scalar:
        return result[0]
    return result

def conductive_heat_leak(k_T, radius, length, thickness, T_hot, T_cold):
    r = radius
    w = thickness
    area = 2*np.pi*r*w

    integral_k, error = quad(k_T, T_cold, T_hot, epsabs=1e-10)
    Q_cond = (area / length) * integral_k
    return Q_cond, error

def total_charge(Qdot, hold_time, P, T_cond, T_evap, V_evap, V_pump):
    n_l = scf.calculate_n_l(Qdot, hold_time, L4)    # assume all liquid evaporates
    n_e = scf.calculate_n_e(P, T_evap, V_evap, n_l, V_mol_liq4)
    n_t = scf.calculate_n_t(P, r, l_pump, lambda z: scf.T_z(z, T_cond, T_evap, l_pump))
    n_p = scf.calculate_n_p(P, T_cond, V_pump, n_max, E, P0_helium(T_cond))

    return n_l + n_e + n_t + n_p

def V_stp(n):
    R = 8.31 
    T = 273.15 # K
    P = 101325 # Pa
    return (n*R*T)/P

if __name__ == "__main__":
    T_cond = 4.0 # K
    T_evap = 1.0 # K
    V_evap = 20e-6 # m^3, volume of evaporator
    V_pump = 10e-6 # m^3, volume of pump not occupied by charcoal
    Qdot = conductive_heat_leak(lambda T: thermal_conductivity(T), r, l_pump, t, T_cond, T_evap)[0]
    #Qdot = 16.8e-6 # W
    hold_times = np.linspace(1, 12, 100) # hours
    charges = [total_charge(Qdot, t, P0_helium(T_evap), T_cond, T_evap, V_evap, V_pump) for t in hold_times]
    charges = np.array(charges)
    V_stp_values = np.array([V_stp(charge) for charge in charges])*1e3


    # Example: Calculate charge for 6 hours
    t_hold = 6
    P = P0_helium(T_evap)
    n_l = scf.calculate_n_l(Qdot, t_hold, L4)
    V_l = n_l * V_mol_liq4 
    n_e = scf.calculate_n_e(P, T_evap, V_evap, n_l, V_mol_liq4)
    n_t = scf.calculate_n_t(P, r, l_pump, lambda z: scf.T_z(z, T_cond, T_evap, l_pump))
    n_p = scf.calculate_n_p(P, T_cond, V_pump, n_max, E, P0_helium(T_cond))
    n_ads = scf.n_adsorbed(P, T_cond, n_max, E, P0_helium(T_cond))
    total_n = n_l + n_e + n_t + n_p
    total_V = V_stp(total_n)
    print("=== He-4 Sorption Cooler Charge Calculation ===")
    print(f"Heat leak Qdot = {Qdot:.3e} W")
    print(f"n_l = {n_l:.3e} mol")
    print(f"volume of liquid = {V_l:.3e} m^3")
    print(f"n_gas = {n_e+n_t+n_p-n_ads:.3e} mol")
    print(f"n_t = {n_t:.3e} mol")
    print(f"n_p = {n_p:.3e} mol")
    print(f"n_ads = {n_ads:.3e} mol")
    print(f"theta = {n_ads/n_max:.3f}")
    print(f"Total charge = {total_n:.3e} mol")
    print(f"Standard volume at STP = {total_V*1e3:.3f} liters")
    print(f"Pressure at evaporator = {P:.3e} Pa = {P*1e-5:.3e} bar")

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('turbo')
    E_vals = [500, 1000, 1500, 2000, 3000]  # J/mol
    for E in E_vals:
        charges = [total_charge(Qdot, t, P0_helium(T_evap), T_cond, T_evap, V_evap, V_pump) for t in hold_times]
        charges = np.array(charges)
        V_stp_values = np.array([V_stp(charge) for charge in charges])*1e3
        plt.plot(hold_times, V_stp_values, label=f'E={E}', color=cmap(E / max(E_vals)))
    plt.xlabel('Hold Time (hours)', fontsize=f)
    plt.ylabel('Total STP Volume of Gas(l)', fontsize=f)
    plt.title(f'Gas Charge vs Hold Time (varying E, $n_{0}$=0.1,)', fontsize=f+2)
    plt.grid(True)
    plt.legend()
    plt.savefig('he4_gas_charge_vs_hold_time1.png', dpi=600)
    plt.show()
    plt.close()

    E = 1000.0  # J/mol
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('cool')
    n_max_vals = [0.05, 0.1, 0.15, 0.2]  # J/mol
    for n_max in n_max_vals:
        charges = [total_charge(Qdot, t, P0_helium(T_evap), T_cond, T_evap, V_evap, V_pump) for t in hold_times]
        charges = np.array(charges)
        V_stp_values = np.array([V_stp(charge) for charge in charges])*1e3
        plt.plot(hold_times, V_stp_values, label=f'$n_{{max}}$={n_max}', color=cmap(n_max / max(n_max_vals)))
    plt.xlabel('Hold Time (hours)', fontsize=f)
    plt.ylabel('Total STP Volume of Gas(l)', fontsize=f)
    plt.title('Gas Charge vs Hold Time (varying n_max, E=1000)', fontsize=f+2)
    plt.grid(True)
    plt.legend()
    plt.savefig('he4_gas_charge_vs_hold_time2.png', dpi=600)
    plt.show()
    plt.close()
