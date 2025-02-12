import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
from eos import fe_eos as iron
from eos import iron2_eos as iron2
from eos import ideal_eos

ideal_z = ideal_eos.IdealEOS(m=56) # ideal EOS for water

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

srho_data = srho_data = np.load('eos/aneos/aneos_iron_srho.npz')
sp_data = np.load('eos/aneos/aneos_iron_sp.npz')
pt_data = np.load('eos/aneos/aneos_fe_pt.npz')

svals_srho = srho_data['s_vals']
logrhovals_srho = srho_data['logrhovals']

logp_grid_srho = srho_data['logpvals_srho']
logt_grid_srho = srho_data['logtvals_srho']
logu_grid_srho = srho_data['loguvals_srho']

logp_rgi_srho = RGI((svals_srho, logrhovals_srho), logp_grid_srho, method='linear', \
            bounds_error=False, fill_value=None)
logt_rgi_srho = RGI((svals_srho, logrhovals_srho), logt_grid_srho, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_srho = RGI((svals_srho, logrhovals_srho), logu_grid_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_logp_srho_tab(_s, _lgrho, _frock=0.0, _firon=0.0): 
    args = (_s, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logp_rgi_srho(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logt_srho_tab(_s, _lgrho, _frock=0.0, _firon=0.0): # returns in erg/g/K
    args = (_s, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logt_rgi_srho(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_srho_tab(_s, _lgrho, _frock=0.0, _firon=0.0): # returns in erg/g
    args = (_s, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_srho(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_sp_inv(_s, _lgp, ideal_guess=True, arr_guess=None, method='newton_brentq'):

    _s = np.atleast_1d(_s)
    _lgp = np.atleast_1d(_lgp)
    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _s, _lgp = np.broadcast_arrays(_s, _lgp)

    if ideal_guess:
        guess = iron2.get_logrho_sp_tab(_s, _lgp)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(s_i, lgp_i, guess_i):
        def err(_lgrho):
            # Error function for logt(S, logp)
            
            logp_test = get_logp_srho_tab(s_i, _lgrho)
            return (logp_test/lgp_i) - 1

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
    density, converged = vectorized_root_func(_s, _lgp, guess)

    return density


def get_logrho_logt_logu_sp_inv(_s, _lgp):
    logrho = get_logrho_sp_inv(_s, _lgp)
    return logrho, get_logt_srho_tab(_s, logrho), get_logu_srho_tab(_s, logrho)

# S, P basis
svals_sp = sp_data['s_vals'] # erg/g/K
logpvals_sp = sp_data['logpvals'] # log K

logrho_grid_sp = sp_data['logrho_sp'] # in g/cm^3
logt_grid_sp = sp_data['logt_sp'] # in K
logu_grid_sp = sp_data['logu_sp'] # in erg/g

logt_rgi_sp = RGI((svals_sp, logpvals_sp), logt_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)
logrho_rgi_sp = RGI((svals_sp, logpvals_sp), logrho_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_sp = RGI((svals_sp, logpvals_sp), logu_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)

def get_logt_sp_tab(_s, _lgp, _frock=0.0, _firon=0.0): 
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logt_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_sp_tab(_s, _lgp, _frock=0.0, _firon=0.0): # returns in erg/g/K
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logrho_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_sp_tab(_s, _lgp, _frock=0.0, _firon=0.0): # returns in erg/g
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt_inv(_lgp, _lgt, ideal_guess=True, arr_guess=None, method='newton_brentq'):

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
        guess = iron.get_s_pt_tab(_lgp, _lgt)#ideal_z.get_s_pt(_lgp, _lgt, 0) / erg_to_kbbar
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
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
    entropy, converged = vectorized_root_func(_lgp, _lgt, guess)

    return entropy

def get_s_logrho_logu_pt_inv(_lgp, _lgt):
    s = get_s_pt_inv(_lgp, _lgt)
    logrho = get_logrho_sp_tab(s, _lgp)
    logu = get_logu_srho_tab(s, logrho)
    return s, logrho, logu

logpvals_pt = pt_data['logpvals'] # log g/cm^3
logtvals_pt = pt_data['logtvals'] # log K
logrho_grid_pt = pt_data['logrhovals'] # in dyn/cm2
s_grid_pt = pt_data['svals'] # in erg/g/K
logu_grid_pt = pt_data['loguvals'] # in erg/g


rho_rgi_pt = RGI((logpvals_pt, logtvals_pt), logrho_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
s_rgi_pt = RGI((logpvals_pt, logtvals_pt), s_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
u_rgi_pt = RGI((logpvals_pt, logtvals_pt), logu_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)

def get_logrho_pt_val(_lgp, _lgt, _frock=0.0, _firon=0.0): 
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt_val(_lgp, _lgt, _frock=0.0, _firon=0.0): # returns in erg/g/K
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_pt_val(_lgp, _lgt, _frock=0.0, _firon=0.0): # returns in erg/g
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result


#### RELEVANT DERIVATIVES #####

def get_c_v(_s, _lgrho, _frock=0.0, _firon=0.0, ds=1e-3):
    # ds/dlogT_{rho, Y}

    lgt2 = get_logt_srho_tab(_s + ds, _lgrho, _frock, _firon)
    lgt1 = get_logt_srho_tab(_s - ds, _lgrho, _frock, _firon)
 
    return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))

def get_c_p(_s, _lgp, _frock=0.0, _firon=0.0, ds=1e-3):
    # ds/dlogT_{P, Y}

    lgt2 = get_logt_sp_tab(_s + ds, _lgp, _frock, _firon)
    lgt1 = get_logt_sp_tab(_s - ds, _lgp, _frock, _firon)
 
    return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))