import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

from eos import ideal_eos

ideal_z = ideal_eos.IdealEOS(m=18) # ideal EOS for water

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

rhot_data = np.load('eos/mazevet/mazevet_water_2021_rhot.npz')
pt_data = np.load('eos/mazevet/mazevet_water_2021_pt.npz')

# rho, T basis
logrhovals_rhot = rhot_data['logrhovals'] # log g/cm^3
logtvals_rhot = rhot_data['logtvals'] # log K
pvals_grid_rhot = rhot_data['pvals'] # in dyn/cm2
s_grid_rhot = rhot_data['svals'] # in erg/g/K
u_grid_rhot = rhot_data['uvals'] # in erg/g

# P, T basis
logpvals_pt = pt_data['logpvals'] # log g/cm^3
logtvals_pt = pt_data['logtvals'] # log K
logrho_grid_pt = pt_data['logrhovals'] # in dyn/cm2
s_grid_pt = pt_data['svals'] # in erg/g/K
u_grid_pt = pt_data['uvals'] # in erg/g

# INTERPOLATION FUNCTIONS

s_rgi_rhot = RGI((logtvals_rhot, logrhovals_rhot),  s_grid_rhot, method='linear', \
            bounds_error=False, fill_value=None)

p_rgi_rhot = RGI((logtvals_rhot, logrhovals_rhot), pvals_grid_rhot, method='linear', \
            bounds_error=False, fill_value=None)

u_rgi_rhot = RGI((logtvals_rhot, logrhovals_rhot), u_grid_rhot, method='linear', \
            bounds_error=False, fill_value=None)

rho_rgi_pt = RGI((logpvals_pt, logtvals_pt), logrho_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
s_rgi_pt = RGI((logpvals_pt, logtvals_pt), s_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
u_rgi_pt = RGI((logpvals_pt, logtvals_pt), u_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)

# TABLE FUNCTIONS

def get_p_rhot_tab(_lgrho, _lgt): # returns in dyn/cm^2
    args = (_lgt, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = p_rgi_rhot(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_rhot_tab(_lgrho, _lgt): # returns in erg/g/K
    args = (_lgt, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_rhot(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result
def get_u_rhot_tab(_lgrho, _lgt): # returns in erg/g
    args = (_lgt, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_rgi_rhot(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_pt_tab(_lgp, _lgt): 
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt_tab(_lgp, _lgt): # returns in erg/g/K
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_u_pt_tab(_lgp, _lgt): # returns in erg/g
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

# INVERSION FUNCTION

def get_logrho_pt_inv(_lgp, _lgt, ideal_guess=True, arr_guess=None, method='newton_brentq'):

    """
    Compute the pressure given density, temperature, helium abundance, and metallicity.

    Parameters:
        _lgrho (array_like): Log10 density values.
        _lgt (array_like): Log10 temperature values.
        ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
        logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

    Returns:
        ndarray: Computed temperature values.
    """

    _lgp = np.atleast_1d(_lgp)
    _lgt = np.atleast_1d(_lgt)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _lgp, _lgt = np.broadcast_arrays(_lgp, _lgt)

    if ideal_guess:
        guess = ideal_z.get_rho_pt(_lgp, _lgt, 0)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(lgp_i, lgt_i, guess_i):
        def err(_lgrho):
            # Error function for logt(S, logp)
            logp_test = get_p_rhot(_lgrho, lgt_i)
            return (logp_test/10**lgp_i) - 1

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