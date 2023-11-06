import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu
import os
from eos import ideal_eos, aqua_eos, ppv_eos
import pdb

from eos import cms_eos, mls_eos, metals_eos

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

pd.options.mode.chained_assignment = None

ideal_xy = ideal_eos.IdealHHeMix()
ideal_z = ideal_eos.IdealEOS(m=40)

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

mh = 1
mhe = 4.0026

###### H-He mixing ######

def Y_to_n(_y):
    return ((_y/mhe)/(((1 - _y)/mh) + (_y/mhe)))
def n_to_Y(x):
    return (mhe * x)/(1 + 3.0026*x)

def x_H(_y, _z, mz):
    Ntot = (1-_y)*(1-_z)/mh + (_y*(1-_z)/mhe) + _z/mz
    return (1-_y)*(1-_z)/mh/Ntot

def x_Z(_y, _z, mz):
    Ntot = (1-_y)*(1-_z)/mh + (_y*(1-_z)/mhe) + _z/mz
    return (_z/mz)/Ntot

def guarded_log(x):
    if np.isscalar(x):
        if x == 0:
            return 0
        elif x  < 0:
            raise ValueError('Number fraction went negative.')
        return x * np.log(x)
    return np.array([guarded_log(x_) for x_ in x])

#### P, _lgt mixtures ####

def get_s_pt(_lgp, _lgt, _y, _z, hhe_eos='cms', z_eos=None):
    if hhe_eos == 'cms':
        xy_eos = cms_eos
    elif hhe_eos == 'mls':
        xy_eos = mls_eos
    else:
        raise Exception('Only cms and mls (CMS19 and MLS22) allowed for now')

    s_nid_mix = xy_eos.get_smix_nd(_y, _lgp, _lgt) # in cgs
    s_h = 10 ** xy_eos.get_s_h(_lgt, _lgp) # in cgs
    s_he = 10 ** xy_eos.get_s_he(_lgt, _lgp)

    if z_eos == 'aqua':
        mz = 18.015
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='aqua')
    elif z_eos == 'ppv':
        mg = 24.305
        si = 28.085
        o3 = 48.000
        mz = mg+si+o3 
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='ppv')
    elif z_eos is None:
        mz = 2.0 # doesn't matter because xz should be 0
        xz = 0.0
        s_z = 0.0
        xh = x_H(_y, _z, mz)
    else:
        raise Exception('z_eos must be either None, ideal, aqua, or ppv')

    xhe = 1 - xh - xz
    if np.any(xh + xhe + xz) != 1.0:
        raise Exception('X + Y + Z != 0')
    s_id_zmix = (guarded_log(xh) + guarded_log(xz) + guarded_log(xhe)) / erg_to_kbbar
    if np.isscalar(_lgp):
        return float((1 - _y)* (1 - _z) * s_h + _y * (1 - _z) * s_he + s_z * _z + s_nid_mix*(1 - _z) - s_id_zmix)

    return (1 - _y)* (1 - _z) * s_h + _y * (1 - _z) * s_he + s_z * _z + s_nid_mix*(1 - _z) - s_id_zmix

def get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos='cms', z_eos=None):
    if hhe_eos == 'cms':
        xy_eos = cms_eos
    elif hhe_eos == 'mls':
        xy_eos = mls_eos
    else:
        raise Exception('Only cms and mls (CMS19 and MLS22) allowed for now')

    rho_hhe = 10**xy_eos.get_rho_pt(_lgp, _lgt, _y)

    if z_eos is not None:
        rho_z = 10**metals_eos.get_rho_pt_tab(_lgp, _lgt, z_eos)
    if z_eos is None:
        rho_z = 1.0 # doesn't matter because _z should be 0
        _z = 0.0
    return np.log10(1/((1 - _z)/rho_hhe + _z/rho_z))

def get_u_pt(_lgp, _lgt, _y, _z, hhe_eos='cms', z_eos=None):
    if hhe_eos == 'cms':
        xy_eos = cms_eos
    elif hhe_eos == 'mls':
        xy_eos = mls_eos
    else:
        raise Exception('Only cms and mls (CMS19 and MLS22) allowed for now')

    u_h = 10**xy_eos.get_logu_h(_lgt, _lgp) # MJ/kg to erg/g
    u_he = 10**xy_eos.get_logu_he(_lgt, _lgp)

    if z_eos is not None:
        u_z = 10**metals_eos.get_u_pt_tab(_lgp, _lgt, z_eos)
    if z_eos is None:
        u_z = 1.0 # doesn't matter because _z should be 0
        _z = 0.0
    
    return np.log10((1 - _y)*(1 - _z) * u_h + _y * (1 - _z)* u_he + _z * u_z)

### error functions ###

def err_t_sp(_lgt, _s, _lgp, _y, _z, hhe_eos, z_eos):
    s_test = get_s_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)*erg_to_kbbar
    return (s_test/_s) - 1

def err_p_rhot(_lgp, _lgrho, _lgt, _y, _z, hhe_eos, z_eos):
    logrho_test = get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return (logrho_test/_lgrho) - 1

def err_t_srho(_lgt, _s, _lgrho, _y, _z, hhe_eos, alg, z_eos):
    # logp = get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, alg=alg, z_eos=z_eos)
    # s_test = get_s_pt(logp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)*erg_to_kbbar
    s_test = get_s_rhot_tab(_lgrho, _lgt, _y, _z)*erg_to_kbbar
    return (s_test/_s) - 1

def err_p_srho(_lgp, _s, _lgrho, _y, _z, hhe_eos, alg, z_eos): # only CMS and AQUA for now, add these options later!
    #logrho_test = get_rho_sp(_s, _lgp, _y, _z, hhe_eos=hhe_eos, alg=alg, z_eos=z_eos)
    logt = get_t_sp(_s, _lgp, _y, _z, hhe_eos=hhe_eos, alg=alg, z_eos=z_eos)
    logrho_test = get_rho_pt(_lgp, logt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return (logrho_test/_lgrho) - 1

def err_t_rhop(_lgt, _lgrho, _lgp, _y, _z, hhe_eos, z_eos):
    logrho_test = get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return (logrho_test/_lgrho) - 1

############################### inversion functions ###############################

TBOUNDS = [2, 7]
PBOUNDS = [0, 15]

XTOL = 1e-16

###### S, P ######

def get_t_sp(_s, _lgp, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    if np.any(_z) > 0.0 and z_eos is None:
        raise Exception('You gotta chose a z_eos if you want metallicities!')
    if alg == 'root':
        if np.isscalar(_s):
            _s, _lgp, _y, _z = np.array([_s]), np.array([_lgp]), np.array([_y]), np.array([_z])
            guess = ideal_xy.get_t_sp(_s, _lgp, _y)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(_s, _lgp, _y, _z, hhe_eos, z_eos))
            return float(sol.x)
        guess = ideal_xy.get_t_sp(_s, _lgp, _y)#*(1 - _z) + _z*ideal_z.get_t_sp(_s, _lgp, 0) # just a guess...
        sol = root(err_t_sp, guess, tol=XTOL, method='hybr', args=(_s, _lgp, _y, _z, hhe_eos, z_eos))
        return sol.x

    elif alg == 'brenth':
        if np.isscalar(_s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(_s, _lgp, _y, _z, hhe_eos, z_eos)) # range should be 2, 5 but doesn'_lgt converge for higher _z unless it'_s lower
                return sol.root
            except:
                raise
        sol = np.array([get_t_sp(s_, p_, y_, z_, z_eos) for s_, p_, y_, z_ in zip(_s, _lgp, _y, _z)])
        return sol

def get_rho_sp(_s, _lgp, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    logt = get_t_sp(_s, _lgp, _y, _z, hhe_eos=hhe_eos, alg=alg, z_eos=z_eos)
    return get_rho_pt(_lgp, logt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

###### rho, T ######

def get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    if alg == 'root':
        if np.isscalar(_lgrho):
            _lgrho, _lgt, _y = np.array([_lgrho]), np.array([_lgt]), np.array([_y]), np.array([_z])
            guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
            sol = root(err_p_rhot, guess, tol=XTOL, method='hybr', args=(_lgrho, _lgt, _y, _z, hhe_eos, z_eos))
            return float(sol.x)
        guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
        sol = root(err_p_rhot, guess, tol=XTOL, method='hybr', args=(_lgrho, _lgt, _y, _z, hhe_eos, z_eos))
        return sol.x

    elif alg == 'brenth':
        if np.isscalar(_lgrho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(_lgrho, _lgt, _y, _z, hhe_eos, z_eos))
                return sol.root
            except:
                raise
        sol = np.array([get_p_rhot(rho_, t_, y_) for rho_, t_, y_ in zip(_lgrho, _lgt, _y, _z)])
        return sol

###### S, rho ######

def get_t_srho(_s, _lgrho, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    if alg == 'root':
        if np.isscalar(_s):
            _s, _lgrho, _y, _z = np.array([_s]), np.array([_lgrho]), np.array([_y]), np.array([_z])
            guess = ideal_xy.get_t_srho(_s, _lgrho, _y)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _y, _z, hhe_eos, alg, z_eos))
            return float(sol.x)

        guess = ideal_xy.get_t_srho(_s, _lgrho, _y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _y, _z, hhe_eos, alg, z_eos))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(_s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(_s, _lgrho, _y, _z, hhe_eos, alg, z_eos))
                return sol.root
            except:
                #print('_s={}, _lgrho={}, _y={}'.format(_s, _lgrho, _y))
                raise
        sol = np.array([get_t_srho(s_, rho_, y_, z_) for s_, rho_, y_, z_ in zip(_s, _lgrho, _y, _z)])
        return sol

def get_p_srho(_s, _lgrho, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    if alg == 'root':
        if np.isscalar(_s):
            _s, _lgrho, _y, _z = np.array([_s]), np.array([_lgrho]), np.array([_y]), np.array([_z])
            guess = ideal_xy.get_p_srho(_s, _lgrho, _y)
            sol = root(err_p_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _y, _z, hhe_eos, alg, z_eos)) # add the options later
            return float(sol.x)

        guess = ideal_xy.get_p_srho(_s, _lgrho, _y)
        sol = root(err_p_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _y, _z, hhe_eos, alg, z_eos))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(_s):
        #guess = 2.5
            try:
                sol = root_scalar(err_p_srho, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(_s, _lgrho, _y, _z, hhe_eos, alg, z_eos))
                return sol.root
            except:
                #print('_s={}, _lgrho={}, _y={}'.format(_s, _lgrho, _y))
                raise
        sol = np.array([get_p_srho(s_, rho_, y_, z_) for s_, rho_, y_, z_ in zip(_s, _lgrho, _y, _z)])
        return sol

###### rho, P ######

def get_t_rhop(_lgrho, _lgp, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    if alg == 'root':
        if np.isscalar(_lgrho):
            _lgrho, _lgp, _y, _z = np.array([_lgrho]), np.array([_lgp]), np.array([_y]), np.array([_z])
            guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
            sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(_lgrho, _lgp, _y, _z, hhe_eos, z_eos))
            return float(sol.x)

        guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
        sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(_lgrho, _lgp, _y, _z, hhe_eos, z_eos))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(_lgrho):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_rhop, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(_lgrho, _lgp, _y, _z, hhe_eos, z_eos))
                return sol.root
            except:
                raise
        sol = np.array([get_t_rhop(rho_, p_, y_) for rho_, p_, y_ in zip(_lgrho, _lgp, _y)])
        return sol

############################### Tabulated EOS Functions ###############################

###### S, P ######

svals_sp = np.arange(5.5, 9.05, 0.05)
logpvals_sp = np.arange(5.5, 14, 0.05)
yvals_sp = np.arange(0.05, 0.55, 0.05)
zvals_sp = np.arange(0, 0.55, 0.05)

logrho_res_sp_cms_aqua, logt_res_sp_cms_aqua = np.load('%s/cms/sp_base_z_aqua.npy' % CURR_DIR)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp, zvals_sp), logrho_res_sp_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp, zvals_sp), logt_res_sp_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(_s, _lgp, _y, _z):
    if np.isscalar(_s):
        return float(get_rho_rgi_sp(np.array([_s, _lgp, _y, _z]).T))
    else:
        return get_rho_rgi_sp(np.array([_s, _lgp, _y, _z]).T)

def get_t_sp_tab(_s, _lgp, _y, _z):
    if np.isscalar(_s):
        return float(get_t_rgi_sp(np.array([_s, _lgp, _y, _z]).T))
    else:
        return get_t_rgi_sp(np.array([_s, _lgp, _y, _z]).T)

###### rho, T ######

logrhovals_rhot = np.linspace(-4.5, 2.0, 100)
logtvals_rhot = np.arange(2, 5.05, 0.05)
yvals_rhot = np.arange(0.05, 0.55, 0.05)
zvals_rhot = np.arange(0, 0.55, 0.05)

logp_res_rhot_cms_aqua, s_res_rhot_cms_aqua = np.load('%s/cms/rhot_base_z_aqua.npy' % CURR_DIR)

get_p_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), logp_res_rhot_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), s_res_rhot_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(_lgrho, _lgt, _y, _z):
    if np.isscalar(_lgrho):
        return float(get_p_rgi_rhot(np.array([_lgrho, _lgt, _y, _z]).T))
    else:
        return get_p_rgi_rhot(np.array([_lgrho, _lgt, _y, _z]).T)

def get_s_rhot_tab(_lgrho, _lgt, _y, _z):
    if np.isscalar(_lgrho):
        return float(get_s_rgi_rhot(np.array([_lgrho, _lgt, _y, _z]).T))
    else:
        return get_s_rgi_rhot(np.array([_lgrho, _lgt, _y, _z]).T)