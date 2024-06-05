from eos import cms_eos, aqua_eos, ideal_eos
import numpy as np
from scipy.optimize import root
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator as RGI
import os
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu

"""
    This file provides tables for Hydrogen-Water mixtures 
    from the CMS19 H and AQUA tables (Haldemann et al. 2020). 
    The AQUA table accounts for different phases of water.

    As with other mixture tables, this file reads pre-computed inverted tables and 
    includes the inversion functions used to produce the tables.

    The tables were produced in the H+AQUA.ipynb notebook.

    The pre-computed table function names end with _tab.
    
    Authors: Roberto Tejada Arevalo
    
"""

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

mh = 1
mz = 18.01528

def Y_to_x(_z):
    ''' Change between mass and number fraction OF WATER'''
    return ((_z/mz)/(((1 - _z)/mh) + (_z/mz)))

def guarded_log(x):
    ''' Used to calculate ideal enetropy of mixing: xlogx'''
    if np.isscalar(x):
        if x == 0:
            return 0
        elif x  < 0:
            raise ValueError('Number fraction went negative.')
        return x * np.log(x)
    return np.array([guarded_log(x_) for x_ in x])

###### HYDROGEN COMPONENTS ######

def get_s_h(lgp, lgt): # hydrogen entropy
    if np.isscalar(lgp):
        return float(10**cms_eos.get_s_h_rgi(np.array([lgt, lgp]).T))
    else:
        return 10**cms_eos.get_s_h_rgi(np.array([lgt, lgp]).T)
    
def get_rho_h(lgp, lgt): # hydrogen density
    if np.isscalar(lgp):
        return float(cms_eos.get_rho_h_rgi(np.array([lgt, lgp]).T))
    else:
        return cms_eos.get_rho_h_rgi(np.array([lgt, lgp]).T)
    
def get_logu_h(lgp, lgt): # hydrogen internal energy
    if np.isscalar(lgp):
        return float(cms_eos.get_logu_h_rgi(np.array([lgt, lgp]).T))
    else:
        return cms_eos.get_logu_h_rgi(np.array([lgt, lgp]).T)

###### MIXTURES ######
    
def get_s_id(_z): # indeal entropy of mixing
    xz = Y_to_x(_z)
    xh = 1 - xz
    q = mh*xh + mz*xz
    return (guarded_log(xh) + guarded_log(xz)) / q
    
def get_s_pt(lgp, lgt, z, sid=True): # entropy of mixutre
    s_h = get_s_h(lgp, lgt)
    s_z = aqua_eos.get_s_pt_tab(lgp, lgt)
    if sid: return s_h*(1-z) + s_z*z + (get_s_id(z) / erg_to_kbbar)
    else: return s_h*(1-z) + s_z*z
    
def get_rho_pt(lgp, lgt, z): # density of mixture
    rho_h = 10**get_rho_h(lgp, lgt)
    rho_z = 10**aqua_eos.get_rho_pt_tab(lgp, lgt)
    
    return np.log10(1/((1 - z)/rho_h + z/rho_z))

def get_u_pt(lgp, lgt, z): # internal energy of mixture
    u_h = 10**get_logu_h(lgp, lgt)
    u_z = 10**aqua_eos.get_u_pt_tab(lgp, lgt)

    return np.log10(u_h*(1-z) + u_z*z)

###### error functions ######

"""
These functions are used by the inversion functions below for better convergence.
"""
def err_t_sp(_lgt, _s, _lgp, _z):
    s_test = get_s_pt(_lgp, _lgt, _z)*erg_to_kbbar
    return (s_test/_s) - 1

def err_p_rhot(_lgp, _lgrho, _lgt, _z):
    rho_test = get_rho_pt(_lgp, _lgt, _z,)
    return (rho_test/_lgrho) - 1

def err_t_srho(_lgt, _s, _lgrho, _z):
    s_test = get_s_rhot_tab(_lgrho, _lgt, _z)*erg_to_kbbar
    return (s_test/_s) - 1

def err_grad(s_trial, _lgp, _z):
    grad_a = get_nabla_ad(s_trial, _lgp, _z)
    logt = get_t_sp_tab(s_trial, _lgp, _z)
    grad_prof = np.gradient(logt)/np.gradient(_lgp)
    return (grad_a/grad_prof) - 1

###### inversion functions ######

TBOUNDS = [2, 7]
PBOUNDS = [0, 15]

XTOL = 1e-16

ideal_xy = ideal_eos.IdealHHeMix(m_he=mz)
###### S, P ######

def get_t_sp(_s, _lgp, _z):
    #if alg == 'root':
    if np.isscalar(_s):
        _s, _lgp, _z = np.array([_s]), np.array([_lgp]), np.array([_z])
        guess = ideal_xy.get_t_sp(_s, _lgp, _z)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(_s, _lgp, _z))
        return float(sol.x)
    #guess = ideal_xy.get_t_sp(_s, _lgp, _z)
    #sol = root(err_t_sp, guess, tol=XTOL, method='hybr', args=(_s, _lgp, _z))
    sol = np.array([get_t_sp(s, lgp, z) for s, lgp, z in zip(_s, _lgp, _z)])
    return sol
    
def get_p_rhot(_lgrho, _lgt, _z):
    #if alg == 'root':
    if np.isscalar(_lgrho):
        _lgrho, _lgt, _z = np.array([_lgrho]), np.array([_lgt]),np.array([_z])
        guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _z)
        sol = root(err_p_rhot, guess, tol=XTOL, method='hybr', args=(_lgrho, _lgt, _z))
        return float(sol.x)
    #guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _z)
    sol = np.array([get_p_rhot(rho, t, z) for rho, t, z in zip(_lgrho, _lgt, _z)])
    return sol

def get_s_rhot(_lgrho, _lgt, _z):
    logp = get_p_rhot(_lgrho, _lgt, _z)
    return get_s_pt(logp, _lgt, _z)
    
def get_t_srho(_s, _lgrho, _z):
    #if alg == 'root':
    if np.isscalar(_s):
        _s, _lgrho, _z = np.array([_s]), np.array([_lgrho]), np.array([_z])
        guess = ideal_xy.get_t_srho(_s, _lgrho, _z)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _z))
        return float(sol.x)

    # guess = ideal_xy.get_t_srho(_s, _lgrho, _z)
    # sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _z))
    sol = np.array([get_t_srho(s, rho, z) for s, rho, z in zip(_s, _lgrho, _z)])
    return sol

####### Inverted Tables #######

### rho(s, p), t(s, p) ###

logrho_res_sp, logt_res_sp = np.load('%s/h_aqua/sp_base.npy' % CURR_DIR)

logpvals_sp = np.arange(6, 14.1, 0.1)
svals_sp = np.arange(2.0, 9.05, 0.05)
zvals_sp = np.arange(0.01, 1.0, 0.01)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp, zvals_sp), logrho_res_sp, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp, zvals_sp), logt_res_sp, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(s, p, z):
    if np.isscalar(s):
        return float(get_rho_rgi_sp(np.array([s, p, z]).T))
    else:
        return get_rho_rgi_sp(np.array([s, p, z]).T)

def get_t_sp_tab(s, p, z):
    if np.isscalar(s):
        return float(get_t_rgi_sp(np.array([s, p, z]).T))
    else:
        return get_t_rgi_sp(np.array([s, p, z]).T)

def get_rhot_sp_tab(s, p, z):
    return get_rho_sp_tab(s, p, z), get_t_sp_tab(s, p, z)

def get_u_sp(s, p, z):
    t = get_t_sp_tab(s, p, z)
    return get_u_pt(p, t, z)

### p(rho, t), s(rho, t) ###

logp_res_rhot, s_res_rhot = np.load('%s/h_aqua/rhot_base.npy' % CURR_DIR)

logrhovals_rhot = np.linspace(-5, 2.0, 100)
logtvals_rhot = np.arange(2.1, 5.1, 0.05)
zvals_rhot = np.arange(0.01, 1.0, 0.01)

get_p_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, zvals_rhot), logp_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, zvals_rhot), s_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t, z):
    if np.isscalar(rho):
        return float(get_p_rgi_rhot(np.array([rho, t, z]).T))
    else:
        return get_p_rgi_rhot(np.array([rho, t, z]).T)

def get_s_rhot_tab(rho, t, z):
    if np.isscalar(rho):
        return float(get_s_rgi_rhot(np.array([rho, t, z]).T))
    else:
        return get_s_rgi_rhot(np.array([rho, t, z]).T)

def get_u_rhot(rho, t, z):
    p = get_p_rhot_tab(rho, t, z) 
    return get_u_pt(p, t, z)

### p(s, rho), t(s, rho) ###

logp_res_srho, logt_res_srho = np.load('%s/h_aqua/srho_base.npy' % CURR_DIR)

svals_srho = np.arange(1.0, 9.05, 0.05)
logrhovals_srho = np.linspace(-5, 2.0, 100)
zvals_srho = np.arange(0.01, 1.0, 0.01)

get_p_rgi_srho = RGI((svals_srho, logrhovals_srho, zvals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhovals_srho, zvals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, rho, z):
    if np.isscalar(s):
        return float(get_p_rgi_srho(np.array([s, rho, z]).T))
    else:
        return get_p_rgi_srho(np.array([s, rho, z]).T)

def get_t_srho_tab(s, rho, z):
    if np.isscalar(s):
        return float(get_t_rgi_srho(np.array([s, rho, z]).T))
    else:
        return get_t_rgi_srho(np.array([s, rho, z]).T)

def get_pt_srho(s, rho, z):
    return get_p_srho_tab(s, rho, z), get_t_srho_tab(s, rho, z)

def get_u_srho(s, rho, z):
    p, t = get_p_srho_tab(s, rho, z), get_t_srho_tab(s, rho, z)
    return get_u_pt(p, t, z)

def get_s_ad(_lgp, _lgt, _z):
    """This function returns the entropy value
    required for nabla - nabla_a = 0 at
    pressure and temperature profiles"""
    guess = get_s_pt(_lgp, _lgt, _z) * erg_to_kbbar
    
    sol = root(err_grad, guess, tol=1e-8, method='hybr', args=(_lgp, _z))
    return sol.x

############################### Derivatives ###############################


def get_dudz_srho(_s, _lgrho, _z, dz=0.01):
    U0 = 10**get_u_srho_tab(_s, _lgrho, _z)
    #U1 = 10**get_u_srho_tab(_s, _lgrho, _y*(1+dy), _z)
    U2 = 10**get_u_srho_tab(_s, _lgrho, _z*(1+dz))

    #dudy_srhoz = (U1 - U0)/(_y*dy)
    dudz_srhoy = (U2 - U0)/(_z*dz)
    return dudz_srhoy

def get_dsdz_rhop_srho(_s, _lgrho, _z, ds=0.01, dz=0.01):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)

    P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _z)
    P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, _lgrho, _z)
    #P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _y*(1+dy), _z)
    P3 = 10**get_p_srho_tab(S0*erg_to_kbbar, _lgrho, _z*(1+dz))     
    
    dpds_rhoyz = (P1 - P0)/(S1 - S0)
    dpdz_srhoy = (P3 - P0)/(_z*dz)

    dsdz_rhopy = -dpdz_srhoy/dpds_rhoyz

    return dsdz_rhopy

def get_dsdz_pt(_lgp, _lgt, _z, dz=0.1):
    S0 = get_s_pt(_lgp, _lgt, _z)
    S1 = get_s_pt(_lgp, _lgt, _z*(1+dz))
   # S2 = get_s_pt(_lgp, _lgt, _z*(1-dz))

    return (S1 - S0)/(_z*dz)

def get_c_s(_s, _lgp, _z,  dp=0.1):
    P0 = 10**_lgp
    P1 = P0*(1+dp)
    R0 = get_rho_sp_tab(_s, np.log10(P0), _z)
    R1 = get_rho_sp_tab(_s, np.log10(P1), _z)

    return np.sqrt((P1 - P0)/(10**R1 - 10**R0))

def get_dtdrho_srho(_s, _lgrho, _z, drho=0.01):
    R0 = 10**_lgrho
    R1 = R0*(1+drho)
    T0 = 10**get_t_srho_tab(_s, np.log10(R0), _z)
    T1 = 10**get_t_srho_tab(_s, np.log10(R1), _z)

    return (T1 - T0)/(R1 - R0)

def get_dtds_srho(_s, _lgrho, _z, ds=0.01):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, _lgrho, _z)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, _lgrho, _z)

    return (T1 - T0)/(S1 - S0)

def get_dtds_sp(_s, _lgp, _z, ds=0.01):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_sp_tab(S0*erg_to_kbbar, _lgp, _z)
    T1 = 10**get_t_sp_tab(S1*erg_to_kbbar, _lgp, _z)

    return (T1 - T0)/(S1 - S0)

def get_dtdy_srho(_s, _lgrho, _z, dz=0.01):
    T0 = 10**get_t_srho_tab(_s, _lgrho, _z)
    T1 = 10**get_t_srho_tab(_s, _lgrho, _z*(1+dz))
    #T2 = 10**get_t_srho(_s, _lgrho, _z*(1+dz))

    dtdy_srhoz = (T1 - T0)/(_z*dz)
    #dtdz_srhoy = (T2 - T0)/(_z*dz)
    return dtdy_srhoz
    

def get_dtdz_srho(_s, _lgrho, _z, dz=0.01):
    T0 = 10**get_t_srho_tab(_s, _lgrho, _z)
    #T1 = 10**get_t_srho(_s, _lgrho, _y*(1+dy), _z)
    T2 = 10**get_t_srho_tab(_s, _lgrho, _z*(1+dz))

    #dtdy_srhoz = (T1 - T0)/(_y*dy)
    dtdz_srhoy = (T2 - T0)/(_z*dz)
    return dtdz_srhoy

def get_drhodt_pz(p, t, z, dt=0.1):
    #y = cms.n_to_Y(x)
    rho0 = get_rho_pt(p, t, z)
    rho1 = get_rho_pt(p, t*(1+dt), z)

    drhodt = (rho1 - rho0)/(t*dt)

    return drhodt

def get_drhodz_pt(p, t, z, dz=0.1):
    #y = cms.n_to_Y(x)
    rho0 = get_rho_pt(p, t, z)
    rho1 = get_rho_pt(p, t, z*(1+dz))

    drhodz = (rho1 - rho0)/(z*dz)

    return drhodz

def get_c_v(_s, _lgrho, _z, ds=0.1, tab=True):
    # ds/dlogT_{_lgrho, Y}
    S0 = _s/erg_to_kbbar
    S1 = S0*(1-ds)
    S2 = S0*(1+ds)
    if tab:
        T0 = get_t_srho_tab(S0*erg_to_kbbar, _lgrho, _z)
        T1 = get_t_srho_tab(S1*erg_to_kbbar, _lgrho, _z)
        T2 = get_t_srho_tab(S2*erg_to_kbbar, _lgrho, _z)
    else:
        T0 = get_t_srho(S0*erg_to_kbbar, _lgrho, _z)
        T1 = get_t_srho(S1*erg_to_kbbar, _lgrho, _z)
        T2 = get_t_srho(S2*erg_to_kbbar, _lgrho, _z)
 
    return (S2 - S1)/(T2 - T1)

def get_c_p(_s, _lgp, _z, ds=0.1, tab=True):
    # ds/dlogT_{P, Y}
    S0 = _s/erg_to_kbbar
    S1 = S0*(1-ds)
    S2 = S0*(1+ds)
    if tab:
        T0 = get_t_sp_tab(S0*erg_to_kbbar, _lgp, _z)
        T1 = get_t_sp_tab(S1*erg_to_kbbar, _lgp, _z)
        T2 = get_t_sp_tab(S2*erg_to_kbbar, _lgp, _z)
    else:
        T0 = get_t_sp(S0*erg_to_kbbar, _lgp, _z)
        T1 = get_t_sp(S1*erg_to_kbbar, _lgp, _z)
        T2 = get_t_sp(S2*erg_to_kbbar, _lgp, _z)

    return (S2 - S1)/(T2 - T1)

def get_gamma1(_s, _lgp, _z, dp = 0.01):
    # dlogP/dlogrho_S, Y, Z
    #if tab:
    R0 = get_rho_sp_tab(_s, _lgp, _z)
    R1 = get_rho_sp_tab(_s, _lgp*(1-dp), _z)
    R2 = get_rho_sp_tab(_s, _lgp*(1+dp), _z)

    return (2*_lgp*dp)/(R2 - R1)

def get_nabla_ad(_s, _lgp, _z, dp=0.01, tab=True):
    if tab:
        T0 = get_t_sp_tab(_s, _lgp, _z)
        T1 = get_t_sp_tab(_s, _lgp*(1-dp), _z)
        T2 = get_t_sp_tab(_s, _lgp*(1+dp), _z)
    else:
        T0 = get_t_sp(_s, _lgp, _z)
        T1 = get_t_sp(_s, _lgp*(1-dp), _z)
        T2 = get_t_sp(_s, _lgp*(1+dp), _z)
    return (T2 - T1)/(_lgp*2*dp)

def get_gruneisen(_s, _lgrho, _z, drho = 0.01):
    T0 = get_t_srho_tab(_s, _lgrho, _z)
    T1 = get_t_srho_tab(_s, _lgrho*(1+drho), _z)
    return (T1 - T0)/(_lgrho*drho)

def get_K(_lgp, _lgt, _z, dp = 0.01):
    P0 = 10**_lgp
    P1 = P0*(1+dp)
    R0 = 10**get_rho_pt(_lgp, _lgt, _z, )
    R1 = 10**get_rho_pt(np.log10(P1), _lgt, _z, )

    return -R0*(P1 - P0)/(R1 - R0)

def get_alpha(_lgp, _lgt, _z, dt=0.1):
    '''
    Coefficient of thermal expansion
    '''
    T0 = 10**_lgt
    T1 = T0*(1+dt)
    R0 = 10**get_rho_pt(_lgp, _lgt, _z)
    R1 = 10**get_rho_pt(_lgp, np.log10(T1), _z)
    return R0*((1/R1 - 1/R0)/(T1 - T0))
