import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

# from eos import ideal_eos
from eos import ice_aneos_eos as ice

# ideal_z = ideal_eos.IdealEOS(m=18) # ideal EOS for water

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

# rhot_data = np.load('eos/mazevet/mazevet_water_2021_rhot.npz')
pt_data = np.load('eos/mazevet/mazevet_aneos_pt.npz')

# P, T basis
logpvals_pt = pt_data['logpvals'] # log g/cm^3
logtvals_pt = pt_data['logtvals'] # log K
logrho_grid_pt = pt_data['logrho_pt'] # in dyn/cm2
s_grid_pt = pt_data['s_pt'] # in erg/g/K
logu_grid_pt = pt_data['logu_pt'] # in erg/g

# INTERPOLATION FUNCTIONS

rho_rgi_pt = RGI((logtvals_pt, logpvals_pt), logrho_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
s_rgi_pt = RGI((logtvals_pt, logpvals_pt), s_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_pt = RGI((logtvals_pt, logpvals_pt), logu_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)

# TABLE FUNCTIONS

def get_logrho_pt_tab(_lgp, _lgt): 
    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt_tab(_lgp, _lgt): # returns in erg/g/K
    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_pt_tab(_lgp, _lgt): # returns in erg/g
    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

# INVERSION FUNCTION

def get_logp_rhot_inv(_lgrho, _lgt, ideal_guess=True, arr_guess=None, method='newton_brentq'):

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

    _lgrho = np.atleast_1d(_lgrho)
    _lgt = np.atleast_1d(_lgt)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _lgrho, _lgt = np.broadcast_arrays(_lgrho, _lgt)

    if ideal_guess:
        guess = ideal_water.get_p_rhot(_lgrho, _lgt, 0)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(lgrho_i, lgt_i, guess_i):
        def err(_lgp):
            # Error function for logt(S, logp)
            logrho_test = ice.get_logrho_pt_tab(_lgp, lgt_i)
            return (logrho_test/lgrho_i) - 1

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
    density, converged = vectorized_root_func(_lgrho, _lgt, guess)

    return density, converged