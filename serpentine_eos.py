import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from eos import ideal_eos, aneos
import os

from astropy import units as u
#from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu

mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ideal_fe = ideal_eos.IdealEOS(m=56)

aneos_serp = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='serpentine')

def get_s_pt_tab(p, t):
    if np.isscalar(p):
        return float((10**aneos_serp.get_logs(p, t)/11605.333))
    return 10**aneos_serp.get_logs(p, t)/11605.333

def get_rho_pt_tab(p, t):
    if np.isscalar(p):
        return float(aneos_serp.get_logrho(p, t))
    return aneos_serp.get_logrho(p, t)

#### S, P ####

sgrid = np.arange(0.0006, 1.1, 0.003)

logrho_res_sp, logt_res_sp = np.load('%s/aneos/aneos_serpentine_sp_base.npy' % CURR_DIR)

get_t_sp_rgi = RGI((sgrid, aneos_serp.logpvals), logt_res_sp, \
                    method='linear', bounds_error=False, fill_value=None)
get_rho_sp_rgi = RGI((sgrid, aneos_serp.logpvals), logrho_res_sp, \
                    method='linear', bounds_error=False, fill_value=None)

def get_t_sp_tab(s, p):
    if np.isscalar(p):
        return float(get_t_sp_rgi(np.array([s, p]).T)) # in log GPa for now
    return get_t_sp_rgi(np.array([s, p]).T)

def get_rho_sp_tab(s, p):
    if np.isscalar(p):
        return float(get_rho_sp_rgi(np.array([s, p]).T)) # in log GPa for now
    return get_rho_sp_rgi(np.array([s, p]).T)

#### error functions ####

def err_t_sp(logt, s_val, logp, alg):
    s_ = get_s_pt_tab(logp, logt)*erg_to_kbbar
   # s_val /= erg_to_kbbar # in cgs
    return (s_/s_val) - 1

#### inversions ####

def get_t_sp(s, p, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, p = np.array([s]), np.array([p])
            guess = ideal_fe.get_t_sp(s, p, 0)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, alg))
            return float(sol.x)

        guess = ideal_fe.get_t_sp(s, p, 0)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, alg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            #try:
            sol = root_scalar(err_t_sp, bracket=[0, 8], xtol=1e-8, method='brenth', args=(s, p, alg)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
            return sol.root
            #except:
            #    raise
        sol = np.array([get_t_sp(s_, p_) for s_, p_ in zip(s, p)])
        return sol