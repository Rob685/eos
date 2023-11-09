import numpy as np
import matplotlib.pyplot as plt
from eos import mixtures_eos
erg_to_kbbar = mixtures_eos.erg_to_kbbar
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
import os

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

metals_eos = mixtures_eos.metals_eos
ideal_eos = mixtures_eos.ideal_eos

ideal_z = ideal_eos.IdealEOS(m=40)

def guarded_log(x):
    if np.isscalar(x):
        if x == 0:
            return 0
        elif x  < 0:
            raise ValueError('Number fraction went negative.')
        return x * np.log(x)
    return np.array([guarded_log(x_) for x_ in x])

def x_z1(_z2, _z3, mz1, mz2, mz3):
    Ntot = (1-_z2)*(1-_z3)/mz1 + (_z2*(1-_z3)/mz2) + _z3/mz3
    return (1-_z2)*(1-_z3)/mz1/Ntot

def x_z3(_z2, _z3, mz1, mz2, mz3):
    Ntot = (1-_z2)*(1-_z3)/mz1 + (_z2*(1-_z3)/mz2) + _z3/mz3
    return (_z3/mz3)/Ntot

def get_s_pt_tab(_lgp, _lgt, _z2, _z3):
    s_z1 = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='aqua')
    s_z2 = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='ppv')
    s_z3 = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='iron')

    m_z1 = 18
    m_z2 = 100.3
    m_z3 = 56

    xz1 = x_z1(_z2, _z3, m_z1, m_z2, m_z3)
    xz3 = x_z3(_z2, _z3, m_z1, m_z2, m_z3)

    xz2 = 1 - xz1 - xz3

    if np.any(xz1 + xz2 + xz3) != 1.0:
        raise Exception('X + Y + Z != 0')

    #s_id_zmix = (guarded_log(xz1) + guarded_log(xz2) + guarded_log(xz3)) / erg_to_kbbar

    return (1 - _z2) * (1 - _z3) * s_z1 + _z2 * (1 - _z3) * s_z2 + _z3 * s_z3 #+ s_id_zmix

def get_rho_pt_tab(_lgp, _lgt, _z2, _z3):
    rho_aqua = 10**metals_eos.get_rho_pt_tab(_lgp, _lgt, eos='aqua')
    rho_ppv = 10**metals_eos.get_rho_pt_tab(_lgp, _lgt, eos='ppv')
    rho_fe = 10**metals_eos.get_rho_pt_tab(_lgp, _lgt, eos='iron')

    mix = ((1 - _z2) * (1 - _z3) / rho_aqua) + (_z2 * (1 - _z3) / rho_ppv) + _z3 / rho_fe

    return np.log10(1/mix)

def err_t_sp(_lgt, _s, _lgp, _z2, _z3):
    s_test = get_s_pt_tab(_lgp, _lgt, _z2, _z3)*erg_to_kbbar
    return (s_test/_s) - 1

def err_p_rhot(_lgp, _lgrho, _lgt, _z2, _z3):
    logrho_test = get_rho_pt_tab(_lgp, _lgt, _z2, _z3)
    return (logrho_test/_lgrho) - 1

def err_t_srho(_lgt, _s, _lgrho, _z2, _z3):
    # logp = get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, alg=alg, z_eos=z_eos)
    # s_test = get_s_pt(logp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)*erg_to_kbbar
    #logp = get_p_rhot(_lgrho, _lgt, _z2, _z3)
    s_test = get_s_rhot_tab(_lgrho, _lgt)*erg_to_kbbar # fixed at _z2 and _z3 quarter fornow
    #s_test = get_s_pt_tab(logp, _lgt, _z2, _z3)*erg_to_kbbar
    return (s_test/_s) - 1

TBOUNDS = [2, 7]
PBOUNDS = [0, 15]

XTOL = 1e-16

def get_t_sp(_s, _lgp, _z2, _z3, alg='brenth'):
    if alg == 'root':
        if np.isscalar(_s):
            _s, _lgp, _z2, _z3 = np.array([_s]), np.array([_lgp]), np.array([_z2]), np.array([_z3])
            guess = ideal_z.get_t_sp(_s, _lgp, 0)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(_s, _lgp, _z2, _z3))
            return float(sol.x)
        guess = ideal_z.get_t_sp(_s, _lgp, np.zeros(len(_s)))#*(1 - _z) + _z*ideal_z.get_t_sp(_s, _lgp, 0) # just a guess...
        sol = root(err_t_sp, guess, tol=XTOL, method='hybr', args=(_s, _lgp, _z2, _z3))
        return sol.x

    elif alg == 'brenth':
        if np.isscalar(_s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(_s, _lgp, _z2, _z3)) # range should be 2, 5 but doesn'_lgt converge for higher _z unless it'_s lower
                return sol.root
            except:
                raise
        sol = np.array([get_t_sp(s_, p_, y_, z_) for s_, p_, y_, z_ in zip(_s, _lgp, _z2, _z3)])
        return sol

def get_p_rhot(_lgrho, _lgt, _z2, _z3, alg='brenth'):
    if alg == 'root':
        if np.isscalar(_lgrho):
            _lgrho, _lgt, _z2, _z3 = np.array([_lgrho]), np.array([_lgt]), np.array([_z2]), np.array([_z3])
            guess = ideal_z.get_p_rhot(_lgrho, _lgt, 0)
            sol = root(err_p_rhot, guess, tol=XTOL, method='hybr', args=(_lgrho, _lgt, _z2, _z3))
            return float(sol.x)
        guess = ideal_z.get_p_rhot(_lgrho, _lgt, np.zeros(len(_lgrho)))
        sol = root(err_p_rhot, guess, tol=XTOL, method='hybr', args=(_lgrho, _lgt, _z2, _z3))
        return sol.x

    elif alg == 'brenth':
        if np.isscalar(_lgrho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(_lgrho, _lgt, _z2, _z3))
                return sol.root
            except:
                raise
        sol = np.array([get_p_rhot(rho_, t_, y_, z_) for rho_, t_, y_, z_ in zip(_lgrho, _lgt, _z2, _z3)])
        return sol

def get_t_srho(_s, _lgrho, _z2, _z3, alg='brenth'):
    if alg == 'root':
        if np.isscalar(_s):
            _s, _lgrho, _z2, _z3 = np.array([_s]), np.array([_lgrho]), np.array([_z2]), np.array([_z3])
            guess = ideal_z.get_t_srho(_s, _lgrho, 0)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _z2, _z3))
            return float(sol.x)

        guess = ideal_z.get_t_srho(_s, _lgrho, np.zeros(len(_s)))
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _z2, _z3))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(_s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(_s, _lgrho, _z2, _z3))
                return sol.root
            except:
                #print('_s={}, _lgrho={}, _y={}'.format(_s, _lgrho, _y))
                raise
        sol = np.array([get_t_srho(s_, rho_, y_, z_) for s_, rho_, y_, z_ in zip(_s, _lgrho, _z2, _z3)])
        return sol

############################### Tabulated EOS Functions ###############################

###### S, P ######

svals_sp = np.arange(0.08, 3.01, 0.01)
logpvals_sp = np.arange(11, 15.05, 0.05)

logrho_res_sp_mix, logt_res_sp_mix = np.load('%s/metal_mixtures/sp_aqua_ppv_iron_half_qtr.npy' % CURR_DIR)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp), logrho_res_sp_mix, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp), logt_res_sp_mix, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(_s, _lgp):
    if np.isscalar(_s):
        return float(get_rho_rgi_sp(np.array([_s, _lgp]).T))
    else:
        return get_rho_rgi_sp(np.array([_s, _lgp]).T)

def get_t_sp_tab(_s, _lgp):
    #if z_eos == 'aqua':
    if np.isscalar(_s):
        return float(get_t_rgi_sp(np.array([_s, _lgp]).T))
    else:
        return get_t_rgi_sp(np.array([_s, _lgp]).T)

###### Rho, T ######

# logrhovals_rhot = np.linspace(-4.5, 2.0, 100)
# logtvals_rhot = np.arange(2, 5.05, 0.05)
# yvals_rhot = np.arange(0.05, 0.55, 0.05)
# zvals_rhot = np.arange(0, 0.55, 0.05)

logrhovals_rhot = np.arange(0.05, 2.51, 0.01)
logtvals_rhot = np.arange(2.1, 7.05, 0.05)

logp_res_rhot_mix, s_res_rhot_mix = np.load('%s/metal_mixtures/rhot_aqua_ppv_iron_half_qtr.npy' % CURR_DIR)

get_p_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot), logp_res_rhot_mix, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot), s_res_rhot_mix, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(_lgrho, _lgt):
    if np.isscalar(_lgrho):
        return float(get_p_rgi_rhot(np.array([_lgrho, _lgt]).T))
    else:
        return get_p_rgi_rhot(np.array([_lgrho, _lgt]).T)

def get_s_rhot_tab(_lgrho, _lgt):
    if np.isscalar(_lgrho):
        return float(get_s_rgi_rhot(np.array([_lgrho, _lgt]).T))
    else:
        return get_s_rgi_rhot(np.array([_lgrho, _lgt]).T)

###### S, Rho ######

logrhovals_srho = np.arange(0.05, 2.1, 0.01)
svals_srho = np.arange(0.01, 3.11, 0.01)

logp_res_srho_mix, logt_res_srho_mix = np.load('%s/metal_mixtures/srho_aqua_ppv_iron_half_qtr.npy' % CURR_DIR)

get_p_rgi_srho = RGI((svals_srho, logrhovals_srho), logp_res_srho_mix, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhovals_srho), logt_res_srho_mix, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(_s, _lgrho):
    if np.isscalar(_s):
        return float(get_p_rgi_srho(np.array([_s, _lgrho]).T))
    else:
        return get_p_rgi_srho(np.array([_s, _lgrho]).T)

def get_t_srho_tab(_s, _lgrho):
    if np.isscalar(_s):
        return float(get_t_rgi_srho(np.array([_s, _lgrho]).T))
    else:
        return get_t_rgi_srho(np.array([_s, _lgrho]).T)