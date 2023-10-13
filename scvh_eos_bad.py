import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
#from scipy.interpolate import RectBivariateSpline as RBS
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy.constants import m_p
#from numba import jit
import os
from eos import ideal_eos, aqua_eos, scvh_man

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

eos_scvh = scvh_man.eos(path_to_data='%s/scvh_mesa' % CURR_DIR)

get_s_pt = eos_scvh.get_logs
get_rho_pt = eos_scvh.get_logrho
get_u_pt = eos_scvh.get_logu

pd.options.mode.chained_assignment = None
ideal_z = ideal_eos.IdealEOS(m=18)
ideal_xy = ideal_eos.IdealHHeMix()
ideal_x = ideal_eos.IdealEOS(m=2)
mz = 18
mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

mh = 1 #* amu.value
mhe = 4.0026

#### inverted tables ####


### S(rho, T, Y), P(rho, T, Y) ###
logrhovals_rhot = np.arange(-4, 1.0, 0.1)
logtvals_rhot = np.arange(2.1, 5.1, 0.1)
yvals_rhot = np.arange(0.05, 0.40, 0.05)

s_res_rhot, logp_res_rhot = np.load('eos/scvh/rhot_basis.npy')

get_p_rhot_rgi = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), logp_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rhot_rgi = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), s_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t, y):
    if np.isscalar(rho):
        return float(get_p_rhot_rgi(np.array([rho, t, y]).T))
    else:
        return get_p_rhot_rgi(np.array([rho, t, y]).T)

def get_s_rhot_tab(rho, t, y):
    if np.isscalar(rho):
        return float(get_s_rhot_rgi(np.array([rho, t, y]).T))
    else:
        return get_s_rhot_rgi(np.array([rho, t, y]).T)


TBOUNDS = [2, 7] # s(rho, P, Y) only works for these bounds... [0, 7] even when the top limit of the CMS table is logT<5
PBOUNDS = [0, 15]

##### error functions #####

XTOL = 1e-4

def err_t_sp(logt, s_val, logp, y):
    s = 10**get_s_pt(logp, logt, y)
    sval = s_val/erg_to_kbbar

    return (s/sval) - 1

def err_p_rhot(lgp, lgt, rhoval, y, alg):
    #lgp, lgt = pt_pair
    logrho = get_rho_pt(lgp, lgt, y)
    #s *= erg_to_kbbar
    if alg == 'root':
        return  logrho/rhoval - 1
    elif alg == 'brenth':
        return float(logrho/rhoval) - 1

##### inverted functions #####

def get_t_sp(s, p, y, alg='root'):
    if alg == 'root':
        if np.isscalar(s):
            s, p, y = np.array([s]), np.array([p]), np.array([y])
        guess = ideal_xy.get_t_sp(s, p, y)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, p, y)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
                return sol.root
            except:
                #print('s={}, p={}, y={}'.format(s, p, y))
                raise
        sol = np.array([get_t_sp(s_, p_, y_) for s_, p_, y_ in zip(s, p, y)])
        return sol

def get_rhot_sp(s, p, y, alg='root'):
    logt = get_t_sp(s, p, y, alg)
    return get_rho_pt(p, logt, y), logt

def get_p_rhot(rho, t, y, alg='root'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t, y = np.array([rho]), np.array([t]), np.array([y])
        guess = ideal_xy.get_p_rhot(rho, t, y)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(t, rho, y, alg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(t, rho, y, alg))
                return sol.root
            except:
                #print('rho={}, t={}, y={}'.format(rho, t, y))
                raise
        sol = np.array([get_p_rhot(rho_, t_, y_) for t_, rho_, y_ in zip(t, rho, y)])
        return sol

def get_s_rhot(rho, t, y, alg='root'):
    logp = get_p_rhot(rho, t, y, alg)
    s = get_s_pt(logp, t, y)
    return s