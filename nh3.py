import numpy as np
from eos import nh3_gao as gao
from eos import ch4_nh3_mandy as mandy
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import gaussian_filter1d
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
erg_to_kJ = (u.erg/u.gram).to('kJ/g')


"""
This module contains the equation of state for ammonia (NH3) using
the analytic fits of Gao et al. (2023) and the DFT data from
Bethkenhagen et al. (2017). It interpolates smoothly between the two
methods using a logistic function. The pressure is calculated
using the analytic fit for low densities and the DFT data for high
densities.
"""
# ---------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------
def logistic(x, k=10.0):
    return 1.0 / (1.0 + np.exp(-k * x))

# ---------------------------------------------------------------------
# blended pressure
# ---------------------------------------------------------------------

def pressure_blend(rho, T,
                   rho_switch   = 0.60,  delta_rho = 0.001,
                   delta_T      = 5.0,  k        = 10.0,
                   Tmax_analytic= 1000.0,
                   p_switch     = 1e10,  # 1 GPa  (dyn cm⁻²)
                   delta_p      = None,
                   sigma=3.0):
    """
    • Analytic EOS for ρ < rho_switch   (hard cut)
    • ρ–T logistic blend for rho>rho_switch, P<=p_switch
    • Force DFT EOS (smoothly) whenever BOTH ρ>rho_switch & P>p_switch
    """
    rho = np.atleast_1d(rho).astype(float)
    T   = np.atleast_1d(T).astype(float)

    p_a = pressure_analytic(rho, T)
    p_d = pressure_dft     (rho, T)

    # ---------- region 1 : low-density, always analytic ----------------
    out       = p_a.copy()
    mask_hi   = rho > rho_switch        # points where we MAY blend/override
    if not np.any(mask_hi):
        return out.squeeze()            # nothing to blend

    # work only on the high-density subset
    rho_hi, T_hi  = rho[mask_hi], T[mask_hi]
    p_a_hi, p_d_hi= p_a[mask_hi], p_d[mask_hi]

    # ---------- analytic EOS validity in T -----------------------------
    # find the last temperature where analytic EOS is positive
    mask_pos = (p_a_hi >= 0.0) & (T_hi <= Tmax_analytic)
    if np.any(mask_pos):
        T0     = T_hi[mask_pos].max()
        p_peak = p_a_hi[mask_pos].max()
    else:
        T0, p_peak = Tmax_analytic, 0.0

    # clamp analytic P above T0 to p_peak so the logistic has a plateau
    p_a_clamped = np.where(T_hi > T0, p_peak, p_a_hi)

    # ---------- logistic weights --------------------------------------
    w_r = logistic((rho_hi - rho_switch) / delta_rho, k=k)     # density switch
    w_t = logistic((T_hi   - T0        ) / delta_T,   k=k)     # temperature
    w_dt = w_r * w_t                                          # AND (ρ & T)

    if delta_p is None:
        delta_p = 0.1 * p_switch                              # 10 % window
    # pressure override applies ONLY when rho>rho_switch
    w_p  = w_r * logistic((p_a_clamped - p_switch) / delta_p, k=k)

    # final weight: smooth OR  -> max(w_dt, w_p)
    w = 1.0 - (1.0 - w_dt)*(1.0 - w_p)

    # ---------- blend & store back -------------------------------------
    out[mask_hi] = (1.0 - w)*p_a_clamped + w*p_d_hi
    return out.squeeze()

def _blend_scalar(quantity_analytic, quantity_dft,
                  rho, T,
                  rho_switch=0.60, delta_rho=0.001,
                  delta_T=5.0,  k=10.0,
                  Tmax_analytic=1000.0,
                  p_switch=1e10, delta_p=None,
                  sigma=3.0):
    """
    Generic helper: blends `quantity_analytic` with `quantity_dft`
    using the same switches as pressure_blend.
    """
    rho = np.atleast_1d(rho).astype(float)
    T   = np.atleast_1d(T).astype(float)

    q_a = quantity_analytic(rho, T)
    q_d = quantity_dft     (rho, T)

    # region 1 – always analytic
    out     = q_a.copy()
    mask_hi = rho > rho_switch
    if not np.any(mask_hi):
        return out.squeeze()

    # subset with rho > rho_switch
    rho_hi, T_hi = rho[mask_hi], T[mask_hi]
    q_a_hi, q_d_hi = q_a[mask_hi], q_d[mask_hi]

    # ---- prepare analytic pressure for the same points ---------------
    p_a_hi = pressure_analytic(rho_hi, T_hi)  # plain analytic P
    # clamp analytic P above its valid T range (for logistic stability)
    mask_pos = (p_a_hi >= 0.0) & (T_hi <= Tmax_analytic)
    if np.any(mask_pos):
        T0     = T_hi[mask_pos].max()
        p_peak = p_a_hi[mask_pos].max()
    else:
        T0, p_peak = Tmax_analytic, 0.0
    p_a_clamped = np.where(T_hi > T0, p_peak, p_a_hi)

    # logistic weights
    w_r  = logistic((rho_hi - rho_switch) / delta_rho, k=k)
    w_t  = logistic((T_hi   - T0       ) / delta_T,   k=k)
    w_dt = w_r * w_t                            # ρ AND T

    if delta_p is None:
        delta_p = 0.1 * p_switch
    w_p  = w_r * logistic((p_a_clamped - p_switch) / delta_p, k=k)

    # OR-style combination
    w = 1.0 - (1.0 - w_dt)*(1.0 - w_p)

    # blend and write back
    out[mask_hi] = (1.0 - w) * q_a_hi + w * q_d_hi
    return out.squeeze()


# ---------------------------------------------------------------------
# user-facing wrappers
# ---------------------------------------------------------------------
def energy_blend(rho, T, **kwargs):
    """
    Blended specific internal energy (erg g⁻¹) following the same
    rules as pressure_blend.
    """
    return _blend_scalar(energy_analytic, energy_dft, rho, T, **kwargs)


def entropy_blend(rho, T, **kwargs):
    """
    Blended specific entropy (erg g⁻¹ K⁻¹) following the same
    rules as pressure_blend.
    """
    return _blend_scalar(entropy_analytic, entropy_dft, rho, T, **kwargs)


def pressure_analytic(rho, T):
    return gao.get_p_rhot(rho, T)

def pressure_dft(rho, T):
    return mandy.get_p_rhot(rho, T, molecule='ammonia')

def energy_analytic(rho, T):
    return gao.get_u_rhot(rho, T)+159023421379.96707

def energy_dft(rho, T):
    return mandy.get_u_rhot(rho, T, molecule='ammonia')

def entropy_analytic(rho, T):
    return gao.get_s_rhot(rho, T)+1.03e8

def entropy_dft(rho, T):
    return mandy.get_s_rhot(rho, T, molecule='ammonia')


# ---------------------------------------------------------------------
# P-T Combined NH3 EOS
# ---------------------------------------------------------------------

data_ammonia = np.load('eos/methane_ammonia/ammonia_eos_pt_extended.npz')

rgi_args = {'method': 'linear', 'bounds_error': False, 'fill_value': None}

rho_pt_rgi_ammonia = RGI((data_ammonia['logT'], data_ammonia['logP']),
                            data_ammonia['logrho'], **rgi_args)
s_pt_rgi_ammonia = RGI((data_ammonia['logT'], data_ammonia['logP']),
                            data_ammonia['s'], **rgi_args)
u_pt_rgi_ammonia = RGI((data_ammonia['logT'], data_ammonia['logP']),
                            data_ammonia['u'], **rgi_args)

def get_logrho_pt(_lgp, _lgt):

    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_pt_rgi_ammonia(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt(_lgp, _lgt):

    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_pt_rgi_ammonia(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_u_pt(_lgp, _lgt):

    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_pt_rgi_ammonia(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result