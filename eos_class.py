import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
import const
import pdb
from tqdm import tqdm
from numba import njit
from eos import ideal_eos
from scipy.optimize import root, newton, brentq, brenth
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

ideal_xy = ideal_eos.IdealHHeMix()

mh = 1
mhe = 4.0026

##### useful unit conversions #####

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

class mixtures:
    def __init__(self, hhe_eos, z_eos, hg=True):
    
        if hhe_eos == 'cms':
            if hg:
                self.pt_data = np.load('eos/cms/{}_hg_{}_pt.npz'.format(hhe_eos, z_eos))
                self.rhot_data = np.load('eos/cms/{}_hg_{}_rhot.npz'.format(hhe_eos, z_eos))
                self.sp_data = np.load('eos/cms/{}_hg_{}_sp.npz'.format(hhe_eos, z_eos))
                self.rhop_data = np.load('eos/cms/{}_hg_{}_rhop.npz'.format(hhe_eos, z_eos))
            else:
                self.pt_data = np.load('eos/cms/{}_{}_pt.npz'.format(hhe_eos, z_eos))
        else:
            self.pt_data = np.load('eos/{}/{}_{}_pt.npz'.format(hhe_eos, hhe_eos, z_eos))
    
        # 1-D independent grids
        self.logpvals = self.pt_data['logpvals'] # these are shared. Units: log10 dyn/cm^2
        self.logtvals = self.pt_data['logtvals'] # log10 K

        self.logrhovals = self.rhot_data['logrhovals'] # log10 g/cc
        self.logrhovals_rhop = self.rhop_data['logrhovals'] # log10 g/cc -- rho, P table range
        self.svals = self.sp_data['s_vals'] # kb/baryon

        self.yvals_pt = self.pt_data['yvals'] # mass fraction -- yprime
        self.zvals_pt = self.pt_data['zvals'] # mass fraction

        self.yvals_rhot = self.rhot_data['yvals']
        self.zvals_rhot = self.rhot_data['zvals']

        self.yvals_sp = self.sp_data['yvals']
        self.zvals_sp = self.sp_data['zvals']

        self.yvals_rhop = self.rhop_data['yvals']
        self.zvals_rhop = self.rhop_data['zvals']

        # 4-D dependent grids
        self.s_pt_tab = self.pt_data['s_pt'] # erg/g/K
        self.logrho_pt_tab = self.pt_data['logrho_pt'] # log10 g/cc
        self.logu_pt_tab = self.pt_data['logu_pt'] # log10 erg/g

        self.s_rhot_tab = self.rhot_data['s_rhot']
        self.logp_rhot_tab = self.rhot_data['logp_rhot']

        self.logt_sp_tab = self.sp_data['logt_sp']
        self.logrho_sp_tab = self.sp_data['logrho_sp']

        self.s_rhop_tab = self.rhop_data['s_rhop']
        self.logt_rhop_tab = self.rhop_data['logt_rhop']

        # RGI interpolation functions
        rgi_args = {'method': 'linear', 'bounds_error': False, 'fill_value': None}

        self.s_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt), self.s_pt_tab, **rgi_args)
        self.logrho_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt), self.logrho_pt_tab, **rgi_args)
        self.logu_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt), self.logu_pt_tab, **rgi_args)

        self.s_rhot_rgi = RGI((self.logrhovals, self.logtvals, self.yvals_rhot, self.zvals_rhot), self.s_rhot_tab[0], **rgi_args)
        self.logp_rhot_rgi = RGI((self.logrhovals, self.logtvals, self.yvals_rhot, self.zvals_rhot), self.logp_rhot_tab[0], **rgi_args)

        self.logt_sp_rgi = RGI((self.svals, self.logpvals, self.yvals_sp, self.zvals_sp), self.logt_sp_tab[0], **rgi_args)
        self.logrho_sp_rgi = RGI((self.svals, self.logpvals, self.yvals_sp, self.zvals_sp), self.logrho_sp_tab[0], **rgi_args)

        self.s_rhop_rgi = RGI((self.logrhovals_rhop, self.logpvals, self.yvals_rhop, self.zvals_rhop), self.s_rhop_tab[0], **rgi_args)
        self.logt_rhop_rgi = RGI((self.logrhovals_rhop, self.logpvals, self.yvals_rhop, self.zvals_rhop), self.logt_rhop_tab[0], **rgi_args)

    ####### EOS table calls #######

    # logp, logt tables
    def get_s_pt(self, _lgp, _lgt, _y, _z):
        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.s_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logrho_pt(self, _lgp, _lgt, _y, _z):
        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logrho_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logu_pt(self, _lgp, _lgt, _y, _z):
        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logu_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # logrho, logt tables
    def get_s_rhot_tab(self, _lgrho, _lgt, _y, _z):
        args = (_lgrho, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.s_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logp_rhot_tab(self, _lgrho, _lgt, _y, _z):
        args = (_lgrho, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logp_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # S, logp tables
    def get_logt_sp_tab(self, _s, _lgp, _y, _z):
        args = (_s, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_sp_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logrho_sp_tab(self, _s, _lgp, _y, _z):
        args = (_s, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logrho_sp_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # logrho, logp tables
    def get_s_rhop_tab(self, _lgrho, _lgp, _y, _z):
        args = (_lgrho, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.s_rhop_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logt_rhop_tab(self, _lgrho, _lgp, _y, _z):
        args = (_lgrho, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_rhop_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result


    ### Inversion Functions ###

    def get_logt_sp(self, _s, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton'):

        """
        Compute the temperature given entropy, pressure, helium abundance, and metallicity.
    
        Parameters:
            _s (array_like): Entropy values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
    
        _s = np.atleast_1d(_s)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgp, _y, _z = np.broadcast_arrays(_s, _lgp, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_t_sp(_s, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            guess = arr_guess

    # Define a function to compute root and capture convergence
        def root_func(s_i, lgp_i, y_i, z_i, guess_i):
            def err(_lgt):
                # Error function for logt(S, logp)
                s_test = self.get_s_pt(lgp_i, _lgt, y_i, z_i) * const.erg_to_kbbar
                return (s_test/s_i) - 1

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

                # If no valid interval is found after max_attempts
                return np.nan, False

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
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgp, _y, _z, guess)

        return temperatures, converged

    def get_logrho_sp(self, _s, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        logt, conv = self.get_logt_sp( _s, _lgp, _y, _z, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        #logt_interp = self.interpolate_non_converged_temperatures_1d(_z, logt, conv, interp_kind='quadratic')
        return self.get_logrho_pt(_lgp, logt, _y, _z)

    def get_logp_rhot(self, _lgrho, _lgt, _y, _z, ideal_guess=True, arr_guess=None, method='newton'):

        """
        Compute the temperature given entropy, pressure, helium abundance, and metallicity.
    
        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgt (array_like): Log10 temperature values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
    
        _lgrho = np.atleast_1d(_lgrho)
        _lgt = np.atleast_1d(_lgt)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgt, _y, _z = np.broadcast_arrays(_lgrho, _lgt, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
       # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgt_i, y_i, z_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                logrho_test = self.get_logrho_pt(_lgp, lgt_i, y_i, z_i)
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
        pressure, converged = vectorized_root_func(_lgrho, _lgt, _y, _z, guess)

        return pressure, converged

    def get_s_rhot(self, _lgrho, _lgt, _y, _z, ideal_guess=True, arr_guess=None):
        logp = self.get_logp_rhot(_lgrho, _lgt, _y, _z, ideal_guess=ideal_guess, arr_guess=arr_guess)
        return self.get_s_pt(logp, _lgt, _y, _z)

    def get_logp_srho(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='newton'):

        """
        Compute the temperature given entropy, pressure, helium abundance, and metallicity.
    
        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
        
        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_p_srho(_s, _lgrho, _y)[0]
        else:
            if logt_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
    # Define a function to compute root and capture convergence
        def root_func(s_i, lgrho_i, y_i, z_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                logrho_test = self.get_logrho_sp(s_i, _lgp, y_i, z_i)
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

                # If no valid interval is found after max_attempts
                return np.nan, False

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
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgrho, _y, _z, guess)

        return temperatures, converged

    def get_logp_logt_srho(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None):
        """
        Compute temperature and pressure given entropy, density, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            arr_guess (tuple of array_like, optional): User-provided initial guesses for log temperature and log pressure when `ideal_guess` is False.

        Returns:
            logt_values (ndarray): Computed log10 temperature values.
            logp_values (ndarray): Computed log10 pressure values.
            converged (ndarray): Boolean array indicating convergence for each point.
        """

        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        # Prepare output arrays
        shape = _s.shape
        logt_values = np.empty(shape)
        logp_values = np.empty(shape)
        converged = np.zeros(shape, dtype=bool)

        # Initial guesses for log temperature and log pressure
        if ideal_guess:
            # Use the ideal EOS for the initial guesses
            #pdb.set_trace()
            guess_lgp, guess_lgt = ideal_xy.get_pt_srho(_s, _lgrho, _y).T
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            else:
                guess_lgp, guess_lgt = arr_guess

        # Flatten arrays for iteration
        lgrho_flat = _lgrho.flatten()
        s_flat = _s.flatten()
        y_flat = _y.flatten()
        z_flat = _z.flatten()
        guess_lgp_flat = guess_lgp.flatten()
        guess_lgt_flat = guess_lgt.flatten()

        # Iterate over each element
        for idx in range(len(s_flat)):
            lgrho_i = lgrho_flat[idx]
            s_i = s_flat[idx]
            y_i = y_flat[idx]
            z_i = z_flat[idx]
            guess_lgp_i = guess_lgp_flat[idx]
            guess_lgt_i = guess_lgt_flat[idx]

            def equations(vars):
                lgp, lgt = vars
                s_calc = self.get_s_pt(lgp, lgt, y_i, z_i) * const.erg_to_kbbar
                lgrho_calc = self.get_logrho_pt(lgp, lgt, y_i, z_i)
                
                # Convert s_calc and lgrho_calc to scalars if they are arrays
                if isinstance(s_calc, np.ndarray):
                    s_calc = s_calc.item()
                if isinstance(lgrho_calc, np.ndarray):
                    lgrho_calc = lgrho_calc.item()
                
                err1 = (s_calc/s_i) - 1
                err2 = (lgrho_calc/lgrho_i) - 1
                return np.array([err1, err2]) 

            try:
                sol = root(
                    equations, [guess_lgp_i, guess_lgt_i], method='hybr', tol=1e-6
                )
                if sol.success:
                    logp_values.flat[idx], logt_values.flat[idx] = sol.x
                    converged.flat[idx] = True
                else:
                    logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                    converged.flat[idx] = False
            except Exception as e:
                logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                converged.flat[idx] = False

        # Reshape output arrays to original shape
        logp_values = logp_values.reshape(shape)
        logt_values = logt_values.reshape(shape)
        converged = converged.reshape(shape)

        return logp_values, logt_values, converged


    def get_logt_rhop(self, _lgrho, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='root'):

        """
        Compute the temperature given entropy, pressure, helium abundance, and metallicity.
    
        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
        
        _lgrho = np.atleast_1d(_lgrho)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgp, _y, _z = np.broadcast_arrays(_lgrho, _lgp, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            else:
                guess = arr_guess
        # sol = root(err, guess, tol=1e-6)
        # return sol.x, sol.success

    # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgp_i, y_i, z_i, guess_i):
            def err(_lgt):
                logrho_test = self.get_logrho_pt(lgp_i, _lgt, y_i, z_i)
                return (logrho_test / lgrho_i) - 1


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

                # If no valid interval is found after max_attempts
                return np.nan, False

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
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_lgrho, _lgp, _y, _z, guess)

        return temperatures, converged

    # def get_s_rhop(self, _lgrho, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='root'):

    #     """
    #     Compute the temperature given entropy, pressure, helium abundance, and metallicity.
    
    #     Parameters:
    #         _lgrho (array_like): Log10 density values.
    #         _lgp (array_like): Log10 pressure values.
    #         _y (array_like): Helium mass fraction values.
    #         _z (array_like): Heavy metal mass fraction values.
    #         ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
    #         logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
    #     Returns:
    #         ndarray: Computed temperature values.
    #     """
        
    #     _lgrho = np.atleast_1d(_lgrho)
    #     _lgp = np.atleast_1d(_lgp)
    #     _y = np.atleast_1d(_y)
    #     _z = np.atleast_1d(_z)

    #     # Ensure inputs are numpy arrays and broadcasted to the same shape
    #     _lgrho, _lgp, _y, _z = np.broadcast_arrays(_lgrho, _lgp, _y, _z)

    #     if ideal_guess:
    #         guess = ideal_xy.get_s_rhop(_lgrho, _lgp, _y)
    #     else:
    #         if arr_guess is None:
    #             raise ValueError("logt_guess must be provided when ideal_guess is False.")
    #         else:
    #             guess = arr_guess
    #     # sol = root(err, guess, tol=1e-6)
    #     # return sol.x, sol.success

    # # Define a function to compute root and capture convergence
    #     def root_func(lgrho_i, lgp_i, y_i, z_i, guess_i):
    #         def err(_s):
    #             logrho_test = self.get_logrho_sp(_s, lgp_i, y_i, z_i)
    #             return (logrho_test / lgrho_i) - 1


    #         if method == 'root':
    #             sol = root(err, guess_i, tol=1e-8)
    #             if sol.success:
    #                 return sol.x[0], True
    #             else:
    #                 return np.nan, False  # Assign np.nan to non-converged elements

    #         elif method == 'newton':
    #             try:
    #                 sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
    #                 return sol_root, True
    #             except RuntimeError:
    #                 #Convergence failed
    #                 return np.nan, False
    #             except Exception as e:
    #                 #Handle other exceptions
    #                 return np.nan, False

    #         elif method == 'brentq':
    #             # Define an initial interval around the guess
    #             delta = 0.1  # Initial interval half-width
    #             a = guess_i - delta
    #             b = guess_i + delta

    #             # Try to find a valid interval where the function changes sign
    #             max_attempts = 5
    #             factor = 2.0  # Factor to expand the interval if needed

    #             for attempt in range(max_attempts):
    #                 try:
    #                     fa = err(a)
    #                     fb = err(b)
    #                     if np.isnan(fa) or np.isnan(fb):
    #                         raise ValueError("Function returned NaN.")

    #                     if fa * fb < 0:
    #                         # Valid interval found
    #                         sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
    #                         return sol_root, True
    #                     else:
    #                         # Expand the interval and try again
    #                         a -= delta * factor
    #                         b += delta * factor
    #                         delta *= factor  # Increase delta for next iteration
    #                 except ValueError:
    #                     # If err() cannot be evaluated, expand the interval
    #                     a -= delta * factor
    #                     b += delta * factor
    #                     delta *= factor

    #             # If no valid interval is found after max_attempts
    #             return np.nan, False

    #         elif method == 'newton_brentq':
    #             # Try the Newton method first
    #             try:
    #                 sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
    #                 return sol_root, True
    #             except RuntimeError:
    #                 # Fall back to the Brentq method if Newton fails
    #                 delta = 0.1
    #                 a = guess_i - delta
    #                 b = guess_i + delta
    #                 max_attempts = 5
    #                 factor = 2.0

    #                 for attempt in range(max_attempts):
    #                     try:
    #                         fa = err(a)
    #                         fb = err(b)
    #                         if np.isnan(fa) or np.isnan(fb):
    #                             raise ValueError("Function returned NaN.")
    #                         if fa * fb < 0:
    #                             sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
    #                             return sol_root, True
    #                         else:
    #                             a -= delta * factor
    #                             b += delta * factor
    #                             delta *= factor
    #                     except ValueError:
    #                         a -= delta * factor
    #                         b += delta * factor
    #                         delta *= factor
    #                 return np.nan, False
    #         else:
    #             raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
    #     # Vectorize the root_func
    #     vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

    #     # Apply the vectorized function
    #     temperatures, converged = vectorized_root_func(_lgrho, _lgp, _y, _z, guess)

    #     return temperatures, converged


    def interpolate_non_converged_temperatures_1d(self, _lgrho, temperatures, converged, interp_kind='linear'):

        # Get converged and non-converged indices
        converged_indices = np.where(converged)
        non_converged_indices = np.where(~converged)

        # Extract converged data
        lgrho_converged = _lgrho[converged_indices]
        temperatures_converged = temperatures[converged_indices]

        # Sort data for interpolation
        sorted_indices = np.argsort(lgrho_converged)
        lgrho_converged_sorted = lgrho_converged[sorted_indices]
        temperatures_converged_sorted = temperatures_converged[sorted_indices]

        # Create interpolation function
        interp_func = interp1d(
            lgrho_converged_sorted, temperatures_converged_sorted, kind=interp_kind, fill_value="extrapolate"
        )

        # Interpolate temperatures for non-converged points
        temperatures_interpolated = temperatures.copy()
        temperatures_interpolated[non_converged_indices] = interp_func(_lgrho[non_converged_indices])

        return temperatures_interpolated

    def inversion(self, a_arr, b_arr, y_arr, z_arr, basis, inversion_method='newton_brentq'):

        """
        Invert the EOS table to compute new thermodynamic quantities.

        Parameters:
            a_arr (array_like): Array of 'a' values (e.g., entropy).
            b_arr (array_like): Array of 'b' values (e.g., pressure).
            y_arr (array_like): Array of helium mass fraction values.
            z_arr (array_like): Array of metallicity values.
            basis (str): The basis for inversion ('sp', 'rhot', 'srho', 'rhop').

        Returns:
            Two arrays of the inverted quantities.
        """

        res1_list = []
        res2_list = []
        for a_ in tqdm(a_arr):
            res1_b = []
            res2_b = []
            for b_ in b_arr:
                res1_y = []
                res2_y = []
                prev_res1_temp = None  # Initialize previous res1_temp to None
                prev_res2_temp = None # For double inversion
                for y_ in y_arr:
                    a_const = np.full_like(z_arr, a_)
                    b_const = np.full_like(z_arr, b_)
                    y_const = np.full_like(z_arr, y_)
                    if basis == 'sp':
                        #pdb.set_trace()
                        if prev_res1_temp is None:

                            res1_temp, conv = self.get_logt_sp(
                                a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                )
                        else:
                            res1_temp, conv = self.get_logt_sp(
                                a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                )

                        res1 = self.interpolate_non_converged_temperatures_1d(
                            z_arr, res1_temp, conv, interp_kind='quadratic'
                        )

                        res2 = self.get_logrho_pt(b_const, res1, y_const, z_arr)
                        prev_res1_temp = res1

                    elif basis == 'rhot':
                        if prev_res1_temp is None:

                            res1_temp, conv = self.get_logp_rhot(
                                a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                )
                        else:
                            res1_temp, conv = self.get_logp_rhot(
                                a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                )

                        res1 = self.interpolate_non_converged_temperatures_1d(
                            z_arr, res1_temp, conv, interp_kind='quadratic'
                        )

                        res2 = self.get_s_pt(res1, b_const, y_const, z_arr)
                        prev_res1_temp = res1

                    elif basis == 'srho':
                        if prev_res1_temp is None:

                            res1_temp, res2_temp, conv = self.get_logp_logt_srho(
                                a_const, b_const, y_const, z_arr, ideal_guess=True
                                )
                        else:
                            res1_temp, res2_temp, conv = self.get_logp_logt_srho(
                                a_const, b_const, y_const, z_arr, ideal_guess=False, arr_guess=[prev_res1_temp, prev_res2_temp]
                                )

                        try:

                            res1 = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='quadratic'
                            )

                            res2 = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res2_temp, conv, interp_kind='quadratic'
                            )

                        except:
                            print(a_const[0], b_const[0], y_const[0], z_arr)
                            raise

                        prev_res1_temp = res1
                        prev_res2_temp = res2

                    elif basis == 'rhop':
                        if prev_res1_temp is None:

                            res1_temp, conv = self.get_logt_rhop(
                                a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                            )
                        else:
                            res1_temp, conv = self.get_logt_rhop(
                                a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                            )

                        try:
                            res1 = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='quadratic'
                            )
                        except:
                            if prev_res1_temp is not None:
                                res1_temp, conv = self.get_logt_rhop(
                                            a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                        )
                                try:
                                    res1 = self.interpolate_non_converged_temperatures_1d(
                                        z_arr, res1_temp, conv, interp_kind='quadratic'
                                    )
                                except:
                                    print(a_const[0], b_const[0], y_const[0], z_arr)
                                    raise
                            else:
                                print(a_const[0], b_const[0], y_const[0], z_arr)
                                raise


                        res2 = self.get_s_pt(b_const, res1_temp, y_const, z_arr)
                        prev_res1_temp = res1  # Update prev_res1_temp for the next iteration
                    else:
                        raise ValueError('Unknown inversion basis. Please choose sp, rhot, srho, or rhop')

                    res1_y.append(res1)
                    res2_y.append(res2)
                res1_b.append(res1_y)
                res2_b.append(res2_y)

            res1_list.append(res1_b)
            res2_list.append(res2_b)

        return np.array([res1_list]), np.array([res2_list])


    ########### useful mixture functions from mixtures_eos.py ###########

    def Y_to_n(self, _y):
        ''' Change between mass and number fraction OF HELIUM'''
        return ((_y/mhe)/(((1 - _y)/mh) + (_y/mhe)))

    def n_to_Y(self, x):
        ''' Change between number and mass fraction OF HELIUM'''
        return (mhe * x)/(1 + 3.0026*x)

    def x_H(self, _y, _z, mz):
        yeff = _y#/(1 - _z)
        Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
        return (1-yeff)*(1-_z)/mh/Ntot

    def x_Z(self, _y, _z, mz):
        yeff = _y#/(1 - _z)
        Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
        return (_z/mz)/Ntot

    def guarded_log(self, x):
        ''' Used to calculate ideal enetropy of mixing: xlogx'''
        if np.isscalar(x):
            if x == 0:
                return 0
            elif x  < 0:
                raise ValueError('Number fraction went negative.')
            return x * np.log(x)
        return np.array([self.guarded_log(x_) for x_ in x])

    def get_smix_id_y(self, Y):
        #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
        xhe = self.Y_to_n(Y)
        xh = 1 - xhe
        q = mh*xh + mhe*xhe
        return -1*(self.guarded_log(xh) + self.guarded_log(xhe)) / q

    def get_smix_id_yz(self, Y, Z, mz):
        #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
        xh = self.x_H(Y, Z, mz)
        xz = self.x_Z(Y, Z, mz)
        xhe = 1 - xh - xz
        q = mh*xh + mhe*xhe + mz*xz
        return -1*(self.guarded_log(xh) + self.guarded_log(xhe) + self.guarded_log(xz)) / q

class derivatives(mixtures):
    def __init__(self, hhe_eos, z_eos, hg=True):
        super().__init__(hhe_eos, z_eos, hg)

