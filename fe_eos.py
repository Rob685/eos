import numpy as np
from astropy import units as u
#from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root
import os

from eos import ideal_eos

"""
    This file provides access to an Iron EOS table from Jisheng Zhang. 

    As with the H-He tables, this file reads pre-computed inverted tables and 
    includes the inversion functions used to produce the tables.

    The pre-computed table function names end with _tab.
    
    Authors: Jisheng Zhang, Roberto Tejada Arevalo
    
"""

# for guesses
ideal_z = ideal_eos.IdealEOS(m=56)

mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
J_to_erg = (u.J / (u.kg * u.K)).to('erg/(K*g)')
J_to_kbbar = (u.J / (u.kg * u.K)).to(k_B/mp)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

### S, P ###

s_grid = np.loadtxt('eos/zhang_eos/iron/Fel_s1.txt')*J_to_kbbar
logpgrid = np.log10(np.loadtxt('eos/zhang_eos/iron/Fel_P1.txt')*10)

logtvals = np.log10(np.loadtxt('eos/zhang_eos/iron/Fel_T_map1.txt'))
logrhovals = np.log10(np.loadtxt('eos/zhang_eos/iron/Fel_rho_map1.txt')*1e-3)
loguvals = np.log10(np.loadtxt('eos/zhang_eos/iron/Fel_E_map1.txt')*J_to_erg)

rho_rgi = RGI((s_grid, logpgrid), logrhovals, method='linear', bounds_error=False, fill_value=None)
t_rgi = RGI((s_grid, logpgrid), logtvals, method='linear', bounds_error=False, fill_value=None)
u_rgi = RGI((s_grid, logpgrid), loguvals, method='linear', bounds_error=False, fill_value=None)

def get_logrho_sp_tab(s, p):
    if np.isscalar(s):
        return float(rho_rgi(np.array([s, p]).T))
    return rho_rgi(np.array([s, p]).T)
def get_logt_sp_tab(s, p):
    if np.isscalar(s):
        return float(t_rgi(np.array([s, p]).T))
    return t_rgi(np.array([s, p]).T)
    
def get_logu_sp_tab(s, p):
    return u_rgi(np.array([s, p]).T)

# def get_rhot_sp_tab(s, p):
#     return get_rho_sp_tab(s, p), get_t_sp_tab(s, p)

### P, T ###

logtgrid = np.arange(2.455, 5.05, 0.05)

s_res_pt, logrho_res_pt = np.load('%s/zhang_eos/iron/pt_base.npy' % CURR_DIR)

get_s_rgi_pt = RGI((logpgrid, logtgrid), s_res_pt, method='linear', \
            bounds_error=False, fill_value=None)
get_rho_rgi_pt = RGI((logpgrid, logtgrid), logrho_res_pt, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_pt_tab(p, t):
    if np.isscalar(p):
        return float(get_s_rgi_pt(np.array([p, t]).T))
    else:
        return get_s_rgi_pt(np.array([p, t]).T)

def get_rho_pt_tab(p, t):
    if np.isscalar(p):
        return float(get_rho_rgi_pt(np.array([p, t]).T))
    else:
        return get_rho_rgi_pt(np.array([p, t]).T)

def get_u_pt_tab(p, t):
    s = get_s_pt_tab(p, t)*erg_to_kbbar
    return get_logu_sp_tab(s, p)

### rho, T ###

logrhogrid = np.arange(0.85, 2.01, 0.01)

logp_res_rhot, s_res_rhot = np.load('%s/zhang_eos/iron/rhot_base.npy' % CURR_DIR)

get_p_rgi_rhot = RGI((logrhogrid, logtgrid), logp_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot = RGI((logrhogrid, logtgrid), s_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t):
    if np.isscalar(rho):
        return float(get_p_rgi_rhot(np.array([rho, t]).T))
    else:
        return get_p_rgi_rhot(np.array([rho, t]).T)

def get_s_rhot_tab(rho, t):
    if np.isscalar(rho):
        return float(get_s_rgi_rhot(np.array([rho, t]).T))
    else:
        return get_s_rgi_rhot(np.array([rho, t]).T)

### S, \rho ###

svals_srho = np.arange(0.01, 0.5, 0.01)

logp_res_srho, logt_res_srho = np.load('%s/zhang_eos/iron/srho_base.npy' % CURR_DIR)

get_p_rgi_srho = RGI((svals_srho, logrhogrid), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhogrid), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_p_rgi_srho(np.array([s, r]).T))
    else:
        return get_p_rgi_srho(np.array([s, r]).T)

def get_t_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_t_rgi_srho(np.array([s, r]).T))
    else:
        return get_t_rgi_srho(np.array([s, r]).T)

##### Inversion Functions #####

def err_s_pt(s, pval, tval):
    logt = get_t_sp_tab(s, pval)
    return (logt/tval) - 1

def err_p_rhot(lgp, rhoval, tval):
    s = get_s_pt(lgp, tval)*erg_to_kbbar
    logrho = get_rho_sp_tab(s, lgp)
    return (logrho/rhoval) - 1

def err_t_srho(lgt, sval, rhoval):
    logp = get_p_rhot(rhoval, lgt)
    s = get_s_pt(logp, lgt)*erg_to_kbbar
    return (s/sval) - 1

def get_s_pt(p, t):
    # returns in kb/bar
    if np.isscalar(p):
        p, t = np.array([p]), np.array([t])
    guess = ideal_z.get_s_pt(p, t, 0)
    sol = root(err_s_pt, guess, tol=1e-8, method='hybr', args=(p, t))
    return sol.x/erg_to_kbbar

def get_rho_pt(p, t):
    s = get_s_pt(p, t)
    return get_rho_sp_tab(s, p)

def get_p_rhot(rho, t):
    if np.isscalar(rho):
        rho, t = np.array([rho]), np.array([t])
    guess = ideal_z.get_p_rhot(rho, t, 0)
    sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t))
    return sol.x

def get_t_srho(s, rho):
    if np.isscalar(s):
        s, rho = np.array([s]), np.array([rho])
    guess = ideal_z.get_t_srho(s, rho, 0)
    sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho))
    return sol.x


def get_s_pt_inv(_lgp, _lgt, tab_guess=True, arr_guess=None, method='newton_brentq'):

    """
    Compute the pressure given density, temperature, helium abundance, and metallicity.

    Parameters:
        _lgrho (array_like): Log10 density values.
        _lgt (array_like): Log10 temperature values.
        tab_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
        logt_guess (array_like, optional): User-provided initial guess for log temperature when `tab_guess` is False.

    Returns:
        ndarray: Computed temperature values.
    """

    _lgp = np.atleast_1d(_lgp)
    _lgt = np.atleast_1d(_lgt)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _lgp, _lgt = np.broadcast_arrays(_lgp, _lgt)

    if tab_guess:
        guess = iron.get_s_pt_tab(_lgp, _lgt)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when tab_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(lgp_i, lgt_i, guess_i):
        def err(_s):
            # Error function for logt(S, logp)
            logt_test = get_logt_sp_tab(_s*erg_to_kbbar, lgp_i)
            return (logt_test/lgt_i) - 1

        if method == 'root':
            sol = root(err, guess_i, tol=1e-8)
            if sol.success:
                return sol.x[0], True
            else:
                return np.nan, False  # Assign np.nan to non-converged elements

        elif method == 'newton':
            try:
                sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                return sol_root, True
            except RuntimeError:
                #Convergence failed
                return np.nan, False
            except Exception as e:
                #Handle other exceptions
                return np.nan, False

        elif method == 'brentq':
            # Define an initial interval around the guess
            delta = 0.1  # Initial interval half-width
            a = guess_i - delta
            b = guess_i + delta

            # Try to find a valid interval where the function changes sign
            max_attempts = 5
            factor = 2.0  # Factor to expand the interval if needed

            for attempt in range(max_attempts):
                try:
                    fa = err(a)
                    fb = err(b)
                    if np.isnan(fa) or np.isnan(fb):
                        raise ValueError("Function returned NaN.")

                    if fa * fb < 0:
                        # Valid interval found
                        sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                        return sol_root, True
                    else:
                        # Expand the interval and try again
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor  # Increase delta for next iteration
                except ValueError:
                    # If err() cannot be evaluated, expand the interval
                    a -= delta * factor
                    b += delta * factor
                    delta *= factor

        elif method == 'newton_brentq':
            # Try the Newton method first
            try:
                sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                return sol_root, True
            except RuntimeError:
                # Fall back to the Brentq method if Newton fails
                delta = 0.1
                a = guess_i - delta
                b = guess_i + delta
                max_attempts = 5
                factor = 2.0

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")
                        if fa * fb < 0:
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    except ValueError:
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor
                return np.nan, False
            # If no valid interval is found after max_attempts
            return np.nan, False
        else:
            raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
    # Vectorize the root_func
    vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

    # Apply the vectorized function
    density, converged = vectorized_root_func(_lgp, _lgt, guess)

    return density, converged

############## derivatives ##############

def get_dtdrho_srho(s, rho, drho=0.01):
    R0 = 10**rho
    R1 = R0*(1+drho)
    T0 = 10**get_t_srho_tab(s, np.log10(R0))
    T1 = 10**get_t_srho_tab(s, np.log10(R1))

    return (T1 - T0)/(R1 - R0)

def get_dtds_srho(s, rho, ds=0.01):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho)

    return (T1 - T0)/(S1 - S0)

def get_c_v(s, rho, ds=0.1):
    # ds/dlogT_{rho, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_srho_tab(S0*erg_to_kbbar, rho)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, rho)
 
    return (S1 - S0)/(T1 - T0)

def get_c_p(s, p, ds=0.1):
    # ds/dlogT_{P, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_sp_tab(S0*erg_to_kbbar, p)
    T1 = get_t_sp_tab(S1*erg_to_kbbar, p)

    return (S1 - S0)/(T1 - T0)