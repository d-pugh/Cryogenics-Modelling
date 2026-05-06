import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import os
plt.rcParams['font.family'] = 'Times New Roman'
# ===================
# Physical constants & pump parameters
# ===================
R = 8.314               # J/(mol K)
V_g = 1e-3              # free gas volume (m³)
n_max = 1             # maximum adsorbate capacity (mol) (likely much lower than 1 in reality)
E = 1000.0              # characteristic adsorption energy (J/mol)
T_STP = 273.15         # K
P_STP = 101325         # Pa
# ===================
# Measured saturation data for He-4
# ===================
file_path = os.path.join(os.path.dirname(__file__), 'vapour_pressure_He4.txt')
T_data, P_data = np.loadtxt(file_path, delimiter=',', skiprows=1, unpack=True)
P_data = P_data * 100          # mbar → Pa

# Critical point from literature
T_c = 5.2                     # K
P_c = 0.227e6                 # Pa

# ===================
# Interpolation of vapour pressure
# ===================
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

# ===================
# Dubinin‑Radushkevich
# ===================
def n_adsorbed(P, T, n_max, E, P0_val):
    exponent = (R * T / E) * np.log(P0_val / P)
    return n_max * np.exp(-exponent ** 2)

def adsorption_efficiency(P, T, E, P0_val):
    exponent = (R * T / E) * np.log(P0_val / P)
    return np.exp(-exponent ** 2)

# ===================
# Smooth residual (continuous, with quadratic penalty for P >= P0)
# ===================
def residual(P, T, n_tot, V_g, n_max, E, P0_func):
    P0 = P0_func(T)
    if P <= 0:
        return -n_tot
    n_gas = P * V_g / (R * T)
    if P < P0:
        n_ads = n_adsorbed(P, T, n_max, E, P0)
        return n_gas + n_ads - n_tot
    else:
        # Quadratic penalty to keep function smooth; at P = P0, n_ads = n_max
        penalty = 1e6 * (P - P0)**2
        return n_gas + n_max - n_tot + penalty

# ===================
# Robust pressure solver using a logarithmic scan + Brent's method
# ===================
def solve_pressure(T, n_tot, V_g, n_max, E, P0_func):
    P0 = P0_func(T)
    # Log scan from an extremely low pressure up to just below P0
    P_min = 1e-30          # virtually zero, safe for ln
    P_max = P0 * 0.9999
    n_scan = 300
    P_scan = np.logspace(np.log10(P_min), np.log10(P_max), n_scan)

    # Evaluate residual at scanned points
    f_scan = np.array([residual(p, T, n_tot, V_g, n_max, E, P0_func) for p in P_scan])

    # Find where sign changes
    signs = np.sign(f_scan)
    sign_change = np.where(np.diff(signs))[0]

    if len(sign_change) == 0:
        # No sign change – no root (should not happen for physical n_tot)
        return np.nan

    # Use the first crossing to set a narrow bracket
    idx = sign_change[0]
    a, b = P_scan[idx], P_scan[idx + 1]
    try:
        return brentq(lambda p: residual(p, T, n_tot, V_g, n_max, E, P0_func),
                      a, b, xtol=1e-14)
    except ValueError:
        return np.nan

# ===================
# Get pressure and efficiency for a given T and n_tot
# ===================
def equilibrium_state(T, n_tot):
    P = solve_pressure(T, n_tot, V_g, n_max, E, P0_helium)
    if np.isnan(P):
        return np.nan, np.nan
    P0 = P0_helium(T)
    eff = adsorption_efficiency(P, T, E, P0)
    return P, eff


# ===================
# Trim isotherm data to remove the meaningless near-zero tail
# ===================
def trim_isotherm_tail(P_range, eff_range, threshold_percent=1.0, min_points=10):
    """
    Truncate an isotherm curve where the efficiency drops below a threshold.
    
    Parameters:
    -----------
    P_range : array of pressures
    eff_range : array of efficiencies (0-100%)
    threshold_percent : efficiency below which we consider the data meaningless
    min_points : minimum number of points to keep (avoid over-trimming)
    
    Returns:
    --------
    P_trimmed, eff_trimmed : truncated arrays
    """
    # Convert threshold to fraction
    threshold = threshold_percent / 100.0
    
    # Find all points above threshold
    valid_mask = eff_range >= threshold_percent
    
    if np.sum(valid_mask) < min_points:
        # Not enough points above threshold; keep original
        return P_range, eff_range
    
    # Find the last index where efficiency is above threshold
    # We want to keep everything from the start to this point
    valid_indices = np.where(valid_mask)[0]
    last_valid_idx = valid_indices[-1]
    
    # Add a small buffer (keep a few more points after the threshold crossing
    # to show the natural decay, but not the flat tail)
    buffer = min(5, len(P_range) - last_valid_idx - 1)
    end_idx = min(last_valid_idx + buffer, len(P_range) - 1)
    
    return P_range[:end_idx + 1], eff_range[:end_idx + 1]


# ===================
# Main simulation
# ===================
if __name__ == "__main__":
    # Temperature ranges
    T_full = np.linspace(1, 60, 200)      # full warm‑up
    T_crit_range = np.linspace(0.6, 5.2, 100)

    n_tot_list = [0.05, 0.1, 0.2, 0.5, 1.0]            # test fill amounts

    # Compute curves
    results = {}
    for n_tot in n_tot_list:
        P_arr = []
        eff_arr = []
        for T in T_full:
            P, eff = equilibrium_state(T, n_tot)
            P_arr.append(P)
            eff_arr.append(eff)
        results[n_tot] = {
            'P': np.array(P_arr),
            'eff': np.array(eff_arr)
        }
        valid = np.sum(~np.isnan(eff_arr))
        print(f"n_tot={n_tot:.2f} mol: {valid}/{len(T_full)} valid points")

    # ===================
    # Plotting
    # ===================
    # ---- Fig 1: Saturation curve ----
    fig1, (ax_sat) = plt.subplots(1, 1, figsize=(10, 6))

    P_sat = P0_helium(T_crit_range)
    ax_sat.plot(T_crit_range, P_sat, 'b-', lw=2, label='Interpolation')
    ax_sat.plot(T_data, P_data, 'rx', ms=5, label='Data')
    ax_sat.axvline(T_c, color='gray', ls='--', alpha=0.7, label=f'$T_c$ = {T_c} K')
    ax_sat.set_yscale('log')
    ax_sat.set_xlabel('Temperature (K)')
    ax_sat.set_ylabel('Pressure (Pa)')
    ax_sat.set_title('He-4 Saturated Vapour Pressure')
    ax_sat.legend()
    ax_sat.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('He4_saturation_curve.png', dpi=300)
    plt.show()
    plt.close(fig1)

    # ---- Fig 2, Panel 1: Adsorption efficiency vs Temperature ----
    fig2, (ax_eff1, ax_eff2) = plt.subplots(1, 2, figsize=(14, 6))
    label_size = 16
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(n_tot_list)))
    for n_tot, color in zip(n_tot_list, colors):
        eff = results[n_tot]['eff'] * 100
        valid = ~np.isnan(eff)
        if np.any(valid):
            ax_eff1.plot(T_full[valid], eff[valid], color=color, lw=2,
                        label=f'$n_{{\\rm tot}}$ = {n_tot:.2f} $n_{{\\rm 0}}$')
    ax_eff1.axvline(T_c, color='gray', ls='--', alpha=0.7, label=f'$T_c$ = {T_c} K')
    ax_eff1.set_xlabel('Temperature (K)', fontsize=label_size)
    ax_eff1.set_ylabel('Adsorption Efficiency θ (%)', fontsize=label_size)
    ax_eff1.legend(fontsize=14, loc='upper right')
    ax_eff1.grid(True, which='both', ls='--', alpha=0.4)

    # ---- Fig 2, Panel 2: Gas adsorbed vs Temperature ----
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(n_tot_list)))
    for n_tot, color in zip(n_tot_list, colors):
        eff = results[n_tot]['eff']
        valid = ~np.isnan(eff)
        if np.any(valid):
            ax_eff2.plot(T_full[valid], eff[valid]*n_max/n_tot, color=color, lw=2,
                        label=f'$n_{{\\rm tot}}$ = {n_tot:.2f} $n_{{\\rm 0}}$')
    ax_eff2.axvline(T_c, color='gray', ls='--', alpha=0.7, label=f'$T_c$ = {T_c} K')
    ax_eff2.set_xlabel('Temperature (K)', fontsize=label_size)
    ax_eff2.set_ylabel(f'Fraction of gas adsorbed ($n/n_{{\\rm max}}$)', fontsize=label_size)
    ax_eff2.legend(fontsize=14, loc='upper right')
    ax_eff2.grid(True, which='both', ls='--', alpha=0.4)
    fig2.suptitle('Charcoal Adsorption Efficiency vs Temperature'.format(n_max), fontsize=16)
    plt.tight_layout()
    #plt.savefig('He4_adsorption_efficiencyvstemp.png', dpi=300)
    plt.show()
    plt.close(fig2)


       # ---- Fig 3: D‑R Efficiency Isotherms (ignore the trimming stuff) ----
    
    fig3, (ax_iso1) = plt.subplots(1, 1, figsize=(14, 7))

    pump_temps = [1.0, 2.5, 4.2, 10.0, 20.0, 50.0, 100.0]
    iso_colors = plt.cm.viridis(np.linspace(0, 1, len(pump_temps)))
    
    
    TRIM_THRESHOLD = 1.0  # percent efficiency below which we cut off
    
    for Tp, col in zip(pump_temps, iso_colors):
        P0 = P0_helium(Tp)      
        P_range = np.logspace(-12, np.log10(P0 * 0.9999), 300)
        eff_range = np.array([adsorption_efficiency(p, Tp, E, P0)*100 for p in P_range])
        
        # ---- APPLY THE TRIMMING PROCEDURE (doesn't work) ----
        P_trimmed, eff_trimmed = trim_isotherm_tail(P_range, eff_range, 
                                                      threshold_percent=TRIM_THRESHOLD)
        
        ax_iso1.semilogx(P_trimmed, eff_trimmed, color=col, lw=2,
                        label=f'$T_{{\\rm pump}}$ = {Tp} K')
    
    ax_iso1.axvline(16.2, color='red', ls='--', alpha=0.5, 
                   label='Typical 1K operating P (⁴He)')
    ax_iso1.set_xlabel('Pressure (Pa)', fontsize=16)
    ax_iso1.set_ylabel('θ (%)', fontsize=16)
    #ax_iso1.set_xscale('log')
    #ax_iso1.set_yscale('log')
    ax_iso1.set_title('Dubinin‑Radushkevich Isotherms'.format(n_max), fontsize=18)
    ax_iso1.legend(fontsize=9, loc='lower right')
    ax_iso1.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
  # plt.savefig('He4_DR_isotherms.png', dpi=300)
    plt.show()
    plt.close(fig3)


    # ===================
    # Bonus: Print trimming statistics
    # ===================
    print("\n" + "="*60)
    print("ISOTHERM TRIMMING STATISTICS")
    print("="*60)
    print(f"Trimming threshold: θ = {TRIM_THRESHOLD}%")
    for Tp in pump_temps:
        P0 = P0_helium(Tp)
        P_range = np.logspace(-12, np.log10(P0 * 0.9999), 300)
        eff_range = np.array([adsorption_efficiency(p, Tp, E, P0)*100 for p in P_range])
        P_trimmed, eff_trimmed = trim_isotherm_tail(P_range, eff_range, 
                                                      threshold_percent=TRIM_THRESHOLD)
        original_points = len(P_range)
        trimmed_points = len(P_trimmed)
        removed = original_points - trimmed_points
        print(f"T_pump = {Tp:5.1f} K: {original_points} points → {trimmed_points} points "
              f"(removed {removed} points, {removed/original_points*100:.1f}%)")
        
