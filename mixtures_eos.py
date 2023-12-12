import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu
import os
from eos import ideal_eos, metals_eos, cms_eos, mls_eos, scvh_eos
import pdb

# aqua_eos = metals_eos.aqua_eos
# serpentine_eos = metals_eos.serpentine_eos
# ppv_eos = metals_eos.ppv_eos
# fe_eos = metals_eos.fe_eos

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
    elif hhe_eos == 'scvh':
        xy_eos = scvh_eos
    else:
        raise Exception('Only cms and mls (CMS19, MLS22, and SCvH95) allowed for now')

    if hhe_eos == 'scvh':
        s_nid_mix = 0.0
        s_xy = xy_eos.get_s_pt_tab(_lgp, _lgt, _y)
    else:
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
    elif z_eos == 'serpentine':
        mz = 56
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='serpentine')
    elif z_eos == 'iron':
        mz = 56
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='iron')
    elif z_eos == 'ideal':
        mz = 18
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='ideal')
    elif z_eos == 'mixture':
        mz = 40
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='mixture')
    elif z_eos is None:
        mz = 2.0 # doesn't matter because xz should be 0
        xz = 0.0
        s_z = 0.0
        xh = x_H(_y, _z, mz)
    else:
        raise Exception('Acceptable z_eos: aqua, ppv, serpentine, iron, ideal, mixture, None')

    xhe = 1 - xh - xz
    if np.any(xh + xhe + xz) != 1.0:
        raise Exception('X + Y + Z != 0')
    s_id_zmix = (guarded_log(xh) + guarded_log(xz) + guarded_log(xhe)) / erg_to_kbbar
    if hhe_eos == 'scvh':
        #s_id_zmix = 0.0
        return (1 - _z)*s_xy + s_z * _z #- s_id_zmix
    else:
        if np.isscalar(_lgp):
            return float((1 - _y)* (1 - _z) * s_h + _y * (1 - _z) * s_he + s_z * _z + s_nid_mix*(1 - _z) - s_id_zmix)

        return (1 - _y)* (1 - _z) * s_h + _y * (1 - _z) * s_he + s_z * _z + s_nid_mix*(1 - _z) - s_id_zmix

def get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos='cms', z_eos=None):
    if hhe_eos == 'cms':
        xy_eos = cms_eos
    elif hhe_eos == 'mls':
        xy_eos = mls_eos
    elif hhe_eos == 'scvh':
        xy_eos = scvh_eos
    else:
        raise Exception('Only cms and mls (CMS19, MLS22, and SCvH95) allowed for now')

    if hhe_eos == 'scvh':
        rho_hhe = 10**xy_eos.get_rho_pt_tab(_lgp, _lgt, _y)
    else:
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
    elif hhe_eos == 'scvh':
        xy_eos = scvh_eos 
    else:
        raise Exception('Only cms and mls (CMS19, MLS22, and SCvH95) allowed for now')

    if z_eos is not None:
        u_z = 10**metals_eos.get_u_pt_tab(_lgp, _lgt, z_eos)
    if z_eos is None:
        u_z = 1.0 # doesn't matter because _z should be 0
        _z = 0.0

    if hhe_eos == 'scvh':
        u_xy = 10**xy_eos.get_u_pt(_lgp, _lgt, _y)
        return np.log10((1 - _z)*u_xy + _z*u_z)
    else:
        u_h = 10**xy_eos.get_logu_h(_lgt, _lgp) # MJ/kg to erg/g
        u_he = 10**xy_eos.get_logu_he(_lgt, _lgp)
        
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
    s_test = get_s_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos)*erg_to_kbbar
    return (s_test/_s) - 1

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
        sol = np.array([get_t_sp(s_, p_, y_, z_, z_eos=z_eos) for s_, p_, y_, z_ in zip(_s, _lgp, _y, _z)])
        return sol

def get_rho_sp(_s, _lgp, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    logt = get_t_sp(_s, _lgp, _y, _z, hhe_eos=hhe_eos, alg=alg, z_eos=z_eos)
    return get_rho_pt(_lgp, logt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

###### Rho, T ######

def get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos='cms', alg='brenth', z_eos=None):
    if alg == 'root':
        if np.isscalar(_lgrho):
            _lgrho, _lgt, _y, _z = np.array([_lgrho]), np.array([_lgt]), np.array([_y]), np.array([_z])
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

###### S, Rho ######

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

###### Rho, P ######

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

svals_sp_aqua = np.arange(3.0, 9.1, 0.1)
logpvals_sp_aqua = np.arange(6, 14.1, 0.1)
yvals_sp = np.arange(0.05, 0.95, 0.1)
zvals_sp = np.arange(0, 1.0, 0.1)

logrho_res_sp_cms_aqua, logt_res_sp_cms_aqua = np.load('%s/cms/sp_base_z_aqua_extended.npy' % CURR_DIR)
logrho_res_sp_scvh_aqua, logt_res_sp_scvh_aqua = np.load('%s/scvh/sp_base_z_aqua_extended.npy' % CURR_DIR)

get_rho_rgi_sp_cms = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logrho_res_sp_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_cms = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logt_res_sp_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_rho_rgi_sp_scvh = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logrho_res_sp_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_scvh = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logt_res_sp_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)

# svals_sp_serp = np.arange(0.1, 9.1, 0.05)
# logpvals_sp_serp = np.arange(11, 14.5, 0.05)

# logrho_res_sp_cms_serp, logt_res_sp_cms_serp = np.load('%s/cms/sp_base_z_serpentine_extended.npy' % CURR_DIR)

# get_rho_rgi_sp_serp = RGI((svals_sp_serp, logpvals_sp_serp, yvals_sp, zvals_sp), logrho_res_sp_cms_serp, method='linear', \
#             bounds_error=False, fill_value=None)
# get_t_rgi_sp_serp = RGI((svals_sp_serp, logpvals_sp_serp, yvals_sp, zvals_sp), logt_res_sp_cms_serp, method='linear', \
#             bounds_error=False, fill_value=None)

# logrho_res_sp_cms_mix, logt_res_sp_cms_mix = np.load('%s/cms/sp_base_z_mixture.npy' % CURR_DIR)

# get_rho_rgi_sp_mix = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logrho_res_sp_cms_mix, method='linear', \
#             bounds_error=False, fill_value=None)
# get_t_rgi_sp_mix = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logt_res_sp_cms_mix, method='linear', \
#             bounds_error=False, fill_value=None)

def get_rho_sp_tab(_s, _lgp, _y, _z, hhe_eos='cms', z_eos='aqua'):
    if hhe_eos == 'cms':
        if np.isscalar(_s):
            return float(get_rho_rgi_sp_cms(np.array([_s, _lgp, _y, _z]).T))
        else:
            return get_rho_rgi_sp_cms(np.array([_s, _lgp, _y, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_rho_rgi_sp_scvh(np.array([_s, _lgp, _y, _z]).T))
        else:
            return get_rho_rgi_sp_scvh(np.array([_s, _lgp, _y, _z]).T)

def get_t_sp_tab(_s, _lgp, _y, _z, hhe_eos='cms', z_eos='aqua'):

    if hhe_eos == 'cms':
        if np.isscalar(_s):
            return float(get_t_rgi_sp_cms(np.array([_s, _lgp, _y, _z]).T))
        else:
            return get_t_rgi_sp_cms(np.array([_s, _lgp, _y, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_t_rgi_sp_scvh(np.array([_s, _lgp, _y, _z]).T))
        else:
            return get_t_rgi_sp_scvh(np.array([_s, _lgp, _y, _z]).T)

def get_rhot_sp_tab(_s, _lgp, _y, _z, hhe_eos='cms', z_eos='aqua'):
    rho, t = get_rho_sp_tab(_s, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos), get_t_sp_tab(_s, _lgp , _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return rho, t

###### Rho, T ######

# logrhovals_rhot = np.linspace(-4.5, 2.0, 100)
# logtvals_rhot = np.arange(2, 5.05, 0.05)
# yvals_rhot = np.arange(0.05, 0.55, 0.05)
# zvals_rhot = np.arange(0, 0.55, 0.05)

logrhovals_rhot = np.linspace(-5.0, 2.0, 100)
logtvals_rhot = np.arange(2, 5.05, 0.05)
yvals_rhot = np.arange(0.05, 0.95, 0.1)
zvals_rhot = np.arange(0, 1.0, 0.1)

logp_res_rhot_cms_aqua, s_res_rhot_cms_aqua = np.load('%s/cms/rhot_base_z_aqua_extended.npy' % CURR_DIR)

logp_res_rhot_scvh_aqua, s_res_rhot_scvh_aqua = np.load('%s/scvh/rhot_base_z_aqua_extended.npy' % CURR_DIR)

get_p_rgi_rhot_cms = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), logp_res_rhot_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_cms = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), s_res_rhot_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_scvh = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), logp_res_rhot_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_scvh = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), s_res_rhot_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos='cms', z_eos='aqua'):
    if hhe_eos == 'cms':
        if np.isscalar(_lgrho):
            return float(get_p_rgi_rhot_cms(np.array([_lgrho, _lgt, _y, _z]).T))
        else:
            return get_p_rgi_rhot_cms(np.array([_lgrho, _lgt, _y, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_lgrho):
            return float(get_p_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y, _z]).T))
        else:
            return get_p_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y, _z]).T)

def get_s_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos='cms', z_eos='aqua'):
    if hhe_eos == 'cms':
        if np.isscalar(_lgrho):
            return float(get_s_rgi_rhot_cms(np.array([_lgrho, _lgt, _y, _z]).T))
        else:
            return get_s_rgi_rhot_cms(np.array([_lgrho, _lgt, _y, _z]).T)
    elif hhe_eos == 'scvh':
        if np.isscalar(_lgrho):
            return float(get_s_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y, _z]).T))
        else:
            return get_s_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y, _z]).T)

##### S, Rho #####

svals_srho = np.arange(3.0, 9.1, 0.05)
logrhovals_srho = np.linspace(-5.0, 2.0, 100)
yvals_srho = np.arange(0.05, 0.95, 0.1)
zvals_srho = np.arange(0, 1.0, 0.1)

logp_res_srho_cms_aqua, logt_res_srho_cms_aqua = np.load('%s/cms/srho_base_z_aqua_extended.npy' % CURR_DIR)

logp_res_srho_scvh_aqua, logt_res_srho_scvh_aqua = np.load('%s/scvh/srho_base_z_aqua_extended.npy' % CURR_DIR)

get_p_rgi_srho_cms = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logp_res_srho_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_cms = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logt_res_srho_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_srho_scvh = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logp_res_srho_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_scvh = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logt_res_srho_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua'):
    if hhe_eos == 'cms':
        if np.isscalar(_s):
            return float(get_p_rgi_srho_cms(np.array([_s, _lgrho, _y, _z]).T))
        else:
            return get_p_rgi_srho_cms(np.array([_s, _lgrho, _y, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_p_rgi_srho_scvh(np.array([_s, _lgrho, _y, _z]).T))
        else:
            return get_p_rgi_srho_scvh(np.array([_s, _lgrho, _y, _z]).T)

def get_t_srho_tab(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua'):
    if hhe_eos == 'cms':
        if np.isscalar(_s):
            return float(get_t_rgi_srho_cms(np.array([_s, _lgrho, _y, _z]).T))
        else:
            return get_t_rgi_srho_cms(np.array([_s, _lgrho, _y, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_t_rgi_srho_scvh(np.array([_s, _lgrho, _y, _z]).T))
        else:
            return get_t_rgi_srho_scvh(np.array([_s, _lgrho, _y, _z]).T)



def get_u_srho_tab(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua'):
    _lgp, _lgt = get_p_srho_tab(_s, _lgrho, _y, _z, hhe_eos = hhe_eos, z_eos=z_eos), get_t_srho_tab(_s, _lgrho , _y, _z, hhe_eos = hhe_eos, z_eos=z_eos)
    return get_u_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

############################### Derivatives ###############################

def get_dudy_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', dy=0.01):
    U0 = 10**get_u_srho_tab(_s, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    U1 = 10**get_u_srho_tab(_s, _lgrho, _y*(1+dy), _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #U2 = 10**get_u_srho_tab(_s, _lgrho, _y, _z*(1+dz))

    dudy_srhoz = (U1 - U0)/(_y*dy)
    #dudz_srhoy = (U2 - U0)/(_z*dz)
    return dudy_srhoz# + dudz_srhoy

def get_dudz_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', dz=0.01):
    U0 = 10**get_u_srho_tab(_s, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #U1 = 10**get_u_srho_tab(_s, _lgrho, _y*(1+dy), _z)
    U2 = 10**get_u_srho_tab(_s, _lgrho, _y, _z*(1+dz), hhe_eos=hhe_eos, z_eos=z_eos)

    #dudy_srhoz = (U1 - U0)/(_y*dy)
    dudz_srhoy = (U2 - U0)/(_z*dz)
    return dudz_srhoy

def get_dsdy_rhop_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', ds=0.01, dy=0.01):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)

    P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _y*(1+dy), _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #P3 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z*(1+dz))     
    
    dpds_rhoyz = (P1 - P0)/(S1 - S0)
    dpdy_srhoz = (P2 - P0)/(_y*dy)
    #dpdz_srhoy = (P3 - P0)/(_z*dz)

    dsdy_rhopz = -dpdy_srhoz/dpds_rhoyz
    #dsdy_rhopy = -dpdz_srhoy/dpds_rhoyz

    return dsdy_rhopz #+ dsdy_rhopy # should be able to add arbitrary components, this is temporary

def get_dsdz_rhop_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', ds=0.01, dz=0.01):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)

    P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _y*(1+dy), _z)
    P3 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z*(1+dz), hhe_eos=hhe_eos, z_eos=z_eos)     
    
    dpds_rhoyz = (P1 - P0)/(S1 - S0)
    #dpdy_srhoz = (P2 - P0)/(_y*dy)
    dpdz_srhoy = (P3 - P0)/(_z*dz)

    #dsdy_rhopz = -dpdy_srhoz/dpds_rhoyz
    dsdz_rhopy = -dpdz_srhoy/dpds_rhoyz # triple product rule

    return dsdz_rhopy #+ dsdy_rhopy # should be able to add arbitrary components, this is temporary

def get_dsdy_pt(_lgp, _lgt, _y, _z, hhe_eos='cms', z_eos='aqua', dy=0.01):
    S0 = get_s_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    S1 = get_s_pt(_lgp, _lgt, _y*(1+dy), _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #S2 = get_s_pt(_lgp, _lgt, _y, _z*(1+dz))

    dsdy_ptz = (S1 - S0)/(_y*dy) # constant P, T, Z
    #dsdz_pty = (S2 - S0)/(_z*dz) # constant P, T, Y

    return dsdy_ptz# + dsdz_pty

def get_dsdz_pt(_lgp, _lgt, _y, _z, hhe_eos='cms', z_eos='aqua', dz=0.01):
    S0 = get_s_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #S1 = get_s_pt(_lgp, _lgt, _y*(1+dy), _z)
    S2 = get_s_pt(_lgp, _lgt, _y, _z*(1+dz), hhe_eos=hhe_eos, z_eos=z_eos)

    #dsdy_ptz = (S1 - S0)/(_y*dy) # constant P, T, Z
    dsdz_pty = (S2 - S0)/(_z*dz) # constant P, T, Y

    return dsdz_pty

def get_c_s(_s, _lgp, _y, _z,hhe_eos='cms', z_eos='aqua',  dp=0.1):
    P0 = 10**_lgp
    P1 = P0*(1+dp)
    R0 = get_rho_sp_tab(_s, np.log10(P0), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    R1 = get_rho_sp_tab(_s, np.log10(P1), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

    return np.sqrt((P1 - P0)/(10**R1 - 10**R0))

def get_dtdrho_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', drho=0.01):
    R0 = 10**_lgrho
    R1 = R0*(1+drho)
    T0 = 10**get_t_srho_tab(_s, np.log10(R0), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    T1 = 10**get_t_srho_tab(_s, np.log10(R1), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

    return (T1 - T0)/(R1 - R0)

def get_dtds_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', ds=0.01):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

    return (T1 - T0)/(S1 - S0)

def get_dtdy_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', dy=0.01):
    T0 = 10**get_t_srho_tab(_s, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    T1 = 10**get_t_srho_tab(_s, _lgrho, _y*(1+dy), _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #T2 = 10**get_t_srho(_s, _lgrho, _y, _z*(1+dz))

    dtdy_srhoz = (T1 - T0)/(_y*dy)
    #dtdz_srhoy = (T2 - T0)/(_z*dz)
    return dtdy_srhoz

def get_dtdz_srho(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', dz=0.01):
    T0 = 10**get_t_srho_tab(_s, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    #T1 = 10**get_t_srho(_s, _lgrho, _y*(1+dy), _z)
    T2 = 10**get_t_srho_tab(_s, _lgrho, _y, _z*(1+dz), hhe_eos=hhe_eos, z_eos=z_eos)

    #dtdy_srhoz = (T1 - T0)/(_y*dy)
    dtdz_srhoy = (T2 - T0)/(_z*dz)
    return dtdz_srhoy

def get_drhodt_py(p, t, y, z, hhe_eos='cms', z_eos='aqua', dt=0.1):
    #y = cms.n_to_Y(x)
    rho0 = get_rho_pt(p, t, y, z, hhe_eos=hhe_eos, z_eos=z_eos)
    rho1 = get_rho_pt(p, t*(1+dt), y, z, hhe_eos=hhe_eos, z_eos=z_eos)

    drhodt = (rho1 - rho0)/(t*dt)

    return drhodt

def get_c_v(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', ds=0.1):
    # ds/dlogT_{_lgrho, Y}
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = get_t_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
 
    return (S1 - S0)/(T1 - T0)

def get_c_p(_s, _lgp, _y, _z, hhe_eos='cms', z_eos='aqua', ds=0.1):
    # ds/dlogT_{P, Y}
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = get_t_sp_tab(S0*erg_to_kbbar, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    T1 = get_t_sp_tab(S1*erg_to_kbbar, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

    return (S1 - S0)/(T1 - T0)

def get_gamma1(_s, _lgp, _y, _z, hhe_eos='cms', z_eos='aqua', dp = 0.01):
    # dlogP/dlogrho_S, Y, Z
    R0 = get_rho_sp_tab(_s, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    R1 = get_rho_sp_tab(_s, _lgp*(1+dp), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return (_lgp*dp)/(R1 - R0)

def get_nabla_ad(_s, _lgp, _y, _z, hhe_eos='cms', z_eos='aqua', dp=0.01):
    T0 = get_t_sp_tab(_s, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    T1 = get_t_sp_tab(_s, _lgp*(1+dp), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return (T1 - T0)/(_lgp*dp)

def get_gruneisen(_s, _lgrho, _y, _z, hhe_eos='cms', z_eos='aqua', drho = 0.01):
    T0 = get_t_srho_tab(_s, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    T1 = get_t_srho_tab(_s, _lgrho*(1+drho), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return (T1 - T0)/(_lgrho*drho)