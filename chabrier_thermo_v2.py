from eos import cms_hhe as eos_hhe
import numpy as np
from astropy import units as u
from astropy.constants import u as amu
from astropy.constants import k_B
from scipy.interpolate import RectBivariateSpline as RBS
#import cms_newton_raphson as cms
#import pandas as pd
#import main_eos as eos

""" Try to compute the derivatives of different bases on this separate file """
tp_bases = eos_hhe.cms_thermo_tp(2019)
tr_bases = eos_hhe.cms_thermo_tr(2019)

def get_s_mix(lgp, lgt, y):
    s_h = 10 ** tp_bases.get_s_h.ev(lgt, lgp)
    s_he = 10 ** tp_bases.get_s_he.ev(lgt, lgp)
    smix = tp_bases.smix_interp.ev(lgt, lgp) # HG23 correction
    return (1 - y) * s_h + y * s_he + smix*(1 - y)*y

def get_rho_mix(lgp, lgt, y):
    rho_h = 10 ** tp_bases.get_rho_h.ev(lgt, lgp)
    rho_he = 10 ** tp_bases.get_rho_he.ev(lgt, lgp)
    vmix = tp_bases.vmix_interp.ev(lgt, lgp) # HG23 correction
    return 1/((1 - y) / rho_h + y / rho_he + vmix*(1 - y)*y)

def get_u_mix_p(lgp, lgt, y):
    u_h = 10**tp_bases.get_u_h.ev(lgt, lgp)
    u_he = 10**tp_bases.get_u_h.ev(lgt, lgp)
    return (1 - y)*u_h + y*u_he

def get_dudt_mix_p(lgp, lgt, y, log_deriv):
    u_h = 10**tp_bases.get_u_h.ev(lgt, lgp)
    u_he = 10**tp_bases.get_u_he.ev(lgt, lgp)
    u_mix = get_u_mix_p(lgp, lgt, y)
    u_t_h = tp_bases.get_u_h.ev(lgt, lgp, dx=1)
    u_t_he = tp_bases.get_u_he.ev(lgt, lgp, dx=1)

    if log_deriv:
        fac = 1
    else:
        fac = (u_mix/10**lgt)

    return ((1 - y) * (u_mix/u_h) * u_t_h + y * (u_mix/u_he) * u_t_he)*fac

def get_drdp_mix_t(lgp, lgt, lgr, y, log_deriv): # eq. 43 in SCvH95

    # rho_mix = get_rho_mix(lgp, lgt, y)
    rho_mix = 10**lgr
    rho_h = 10**tp_bases.get_rho_h.ev(lgt, lgp)
    rho_he = 10**tp_bases.get_rho_he.ev(lgt, lgp)

    rhop_h = tp_bases.get_rhop_h.ev(lgt, lgp)
    rhop_he = tp_bases.get_rhop_he.ev(lgt, lgp)

    # rhop_h = tp_bases.get_rhop_h(lgt, lgp) # trying RBS spline derivatives instead.
    # rhop_he = tp_bases.get_rhop_he(lgt, lgp)
    drhodp = (1 - y) * (rho_mix/rho_h) * rhop_h + y * (rho_mix/rho_he) * rhop_he

    if log_deriv:
        fac = 1
    else:
        fac = (rho_mix/(10**lgp)) 
    
    return drhodp*fac#drho/dP in linear space

def get_drdt_mix_p(lgp, lgt, lgr, y, log_deriv): # eq. 44 in SCvH95
    
    #rho_mix = get_rho_mix(lgp, lgt, y)
    rho_mix = 10**lgr
    rho_h = 10**tp_bases.get_rho_h.ev(lgt, lgp)
    rho_he = 10**tp_bases.get_rho_he.ev(lgt, lgp)

    rhot_h = tp_bases.get_rhot_h.ev(lgt, lgp)
    rhot_he = tp_bases.get_rhot_he.ev(lgt, lgp)
    # rhot_h = tp_bases.get_rhot_h(lgt, lgp)
    # rhot_he = tp_bases.get_rhop_he(lgt, lgp)

    drhodt = (1 - y) * (rho_mix/rho_h) * rhot_h + y * (rho_mix/rho_he) * rhot_he

    if log_deriv:
        fac = 1
    else:
        fac = (rho_mix/(10**lgt))
    
    return drhodt*fac #drho/dP in linear space

### CMS19 equations ###
''' The functions below until the next section calculate the thermodynamic quantities according to the definitions
given by the CMS19 paper, combined with some definitions from the SCvH95 paper.'''

def get_chi_t(lgp, lgt, lgr, y): # equation 5 in CMS19
    return -get_drdt_mix_p(lgp, lgt, lgr, y, log_deriv=True)/get_drdp_mix_t(lgp, lgt, lgr, y, log_deriv=True)

def get_chi_rho(lgp, lgt, lgr, y):
    return 1/get_drdp_mix_t(lgp, lgt, lgr, y, log_deriv=True)

#def get_dpdt_mix_

# required to calculate grad_ad_mix
def get_dsdt_mix_p(lgp, lgt, y): # eq. 45 in SCvH95
    smix = get_s_mix(lgp, lgt, y)
    s_h = 10**tp_bases.get_s_h.ev(lgt, lgp)
    s_he = 10**tp_bases.get_s_he.ev(lgt, lgp)

    # s_corr = tp_bases.smix_interp.ev(lgt, lgp)
    # scorr_t = tp_bases.smix_interp.ev(lgt, lgp, dx=1) # dS/dlogT

    # s_t_h = tp_bases.get_st_h.ev(lgt, lgp)
    # s_t_he = tp_bases.get_st_he.ev(lgt, lgp)
    s_t_h = tp_bases.get_s_h.ev(lgt, lgp, dx=1)
    s_t_he = tp_bases.get_s_he.ev(lgt, lgp, dx=1)

    # s_t_h = tp_bases.get_st_h(lgt, lgp)
    # s_t_he = tp_bases.get_st_he(lgt, lgp)

    return (1 - y) * (smix/s_h) * s_t_h + y * (smix/s_he) * s_t_he #+ scorr_t#*s_corr**2/smix  # test derivative of the entropy of mixing. Take out if it doesn't work.

def get_dsdp_mix_t(lgp, lgt, y): # eq. 46 in SCvH95
    smix = get_s_mix(lgp, lgt, y)
    s_h = 10**tp_bases.get_s_h.ev(lgt, lgp)
    s_he = 10**tp_bases.get_s_he.ev(lgt, lgp)

    # s_corr = tp_bases.smix_interp.ev(lgt, lgp)
    # scorr_p = tp_bases.smix_interp.ev(lgt, lgp, dy=1)

    # s_p_h = tp_bases.get_sp_h.ev(lgt, lgp)
    # s_p_he = tp_bases.get_sp_he.ev(lgt, lgp)

    s_p_h = tp_bases.get_s_h.ev(lgt, lgp, dy=1)
    s_p_he = tp_bases.get_s_he.ev(lgt, lgp, dy=1)

    # s_p_h = tp_bases.get_sp_h(lgt, lgp)
    # s_p_he = tp_bases.get_sp_he(lgt, lgp)

    return (1 - y) * (smix/s_h) * s_p_h + y * (smix/s_he) * s_p_he #+ scorr_p

def get_grada_mix(lgp, lgt, y): # eq. 47 in SCvH95
    sp = get_dsdp_mix_t(lgp, lgt, y)
    st = get_dsdt_mix_p(lgp, lgt, y)

    return -sp/st

# def get_cp_p(lgp, lgt, y): # eq. gt.7b in notes
#     rho_mix = get_rho_mix(lgp, lgt, y)
#     dudt_p = get_dudt_mix_p(lgp, lgt, y, log_deriv=False)
#     drhodt_p = get_drdt_mix_p(lgp, lgt, y,  log_deriv=False)
#     return dudt_p - ((10**lgp)/rho_mix**2)*drhodt_p

def get_cp(lgp, lgt, y):
    s = get_s_mix(lgp, lgt, y)
    dsdt = get_dsdt_mix_p(lgp, lgt, y) # already in log space

    return s*dsdt

def get_gamma1_alt(lgp, lgt, lgr, y):
    grada = get_grada_mix(lgp, lgt, y)
    chirho = get_chi_rho(lgp, lgt, lgr, y)
    chit = get_chi_t(lgp, lgt, lgr, y)

    return chirho/(1 - chit*grada)

def get_cv_alt(lgp, lgt, lgr, y):
    gamma1 = get_gamma1_alt(lgp, lgt, lgr, y)
    chirho = get_chi_rho(lgp, lgt, lgr, y)
    cp = get_cp(lgp, lgt, y)
    return cp*chirho/gamma1



def get_cv(lgp, lgt, lgr, y): 
    rho = 10**lgr
    p = 10**lgp
    t = 10**lgt
    chi_rho = get_chi_rho(lgp, lgt, lgr, y)
    chi_t = get_chi_t(lgp, lgt, lgr, y)
    cp = get_cp(lgp, lgt, y)

    return cp - (p/(rho * t))*(chi_t**2 / chi_rho)

def get_grad_ad(lgp, lgt, lgr, y): # eq. 12 in cms19
    rho = 10**lgr
    p = 10**lgp
    t = 10**lgt
    chi_rho = get_chi_rho(lgp, lgt, lgr, y)
    chi_t = get_chi_t(lgp, lgt, lgr, y)
    cv = get_cv(lgp, lgt, lgr, y)

    return chi_t/(chi_t**2 + chi_rho*(cv/(p/(rho*t))))

def get_gamma1(lgp, lgt, lgr, y): # from eq. 18 in cms
    rho = 10**lgr
    p = 10**lgp
    t = 10**lgt
    chi_t = get_chi_t(lgp, lgt, lgr, y)
    cv = get_cv(lgp, lgt, lgr, y) 

    grun = (p/(rho * t * cv))*chi_t # Grüneisen parameter
    gradad = get_grad_ad(lgp, lgt, lgr, y)

    return grun/gradad

def get_csound(lgp, lgt, y): # eq. 15 in cms
    rho = get_rho_mix(lgp, lgt, y)
    p = 10**lgp
    chi_rho = get_chi_rho(lgp, lgt, y)
    cv = get_cv(lgp, lgt, y)
    cp = get_cp(lgp, lgt, y)

    return  np.sqrt(p/rho)*np.sqrt(cp/cv)*np.sqrt(chi_rho)

def get_compress(lgp, lgt, y):
    rho = get_rho_mix(lgp, lgt, y)
    cs = get_csound(lgp, lgt, y)
    return 1/(rho * cs**2)

###############################################################

''' The functions below take derivatives at constant densities. Use them only if you have to.'''

## t, rho bases ##

# def get_p_mix(lgr, lgt, y): # this doesn't work and shouldn't be used
#     p_h = 10**tr_bases.get_p_h.ev(lgt, lgr)
#     p_he = 10**tr_bases.get_p_h.ev(lgt, lgr)
#     return (1 - y)*p_h + y*p_he

def get_u_mix_r(lgr, lgt, y):
    u_h = 10**tr_bases.get_u_h.ev(lgt, lgr)
    u_he = 10**tr_bases.get_u_h.ev(lgt, lgr)
    return (1 - y)*u_h + y*u_he

def get_dudt_mix_r(lgr, lgt, y, log_deriv): # taking eqns. 43-46 in SCvH95
    u_h = 10**tr_bases.get_u_h.ev(lgt, lgr)
    u_he = 10**tr_bases.get_u_he.ev(lgt, lgr)
    u_mix = get_u_mix_r(lgr, lgt, y)

    if log_deriv:
        fac = 1
    else:
        fac = (u_mix/10**lgt)

    u_t_h = tr_bases.get_u_h.ev(lgt, lgr, dx=1) # at constant density, derivatives in logspace
    u_t_he = tr_bases.get_u_he.ev(lgt, lgr, dx=1) # at constant density

    # the interpolation must be multiplied by u/T since it's in log space
    return ((1 - y) * (u_mix/u_h) * u_t_h + y * (u_mix/u_he) * u_t_he)*fac

def get_dudrho_mix_t(lgr, lgt, y, log_deriv):
    u_h = 10**tr_bases.get_u_h.ev(lgt, lgr)
    u_he = 10**tr_bases.get_u_he.ev(lgt, lgr)
    u_mix = get_u_mix_r(lgr, lgt, y)
    u_rho_h = tr_bases.get_u_h.ev(lgt, lgr, dy=1) # at constant temperature
    u_rho_he = tr_bases.get_u_he.ev(lgt, lgr, dy=1) # at constant temperature

    if log_deriv:
        fac = 1
    else:
        fac = (u_mix/10**lgr)

     # the interpolation must be multiplied by u/rho since it's in log space
    return ((1 - y) * (u_mix/u_h) * u_rho_h + y * (u_mix/u_he) * u_rho_he)*fac


# Can't take this derivative because we don't have a non-ideal correction to the pressure

def get_dpdt_mix_r(lgp, lgt, lgr, y, log_deriv):# dP/dT_const_rho
    p_h = 10**tr_bases.get_p_h.ev(lgt, lgr)
    p_he = 10**tr_bases.get_p_he.ev(lgt, lgr)
    p_mix = 10**lgp
    p_t_h = tr_bases.get_pt_h(lgt, lgr) # at constant density
    p_t_he = tr_bases.get_pt_he(lgt, lgr) # at constant density

    if log_deriv:
        fac = 1
    else:
        fac = (p_mix/10**lgt)

    return ((1 - y) * (p_mix/p_h) * p_t_h + y * p_mix/p_he * p_t_he)*fac

def get_dpdr_mix_t(lgp, lgt, lgr, y, log_deriv):
    p_h = 10**tr_bases.get_p_h.ev(lgt, lgr)
    p_he = 10**tr_bases.get_p_he.ev(lgt, lgr)
    p_mix = 10**lgp
    p_rho_h = tr_bases.get_prho_h(lgt, lgr) # at constant temp
    p_rho_he = tr_bases.get_prho_he(lgt, lgr) # at constant temp

    if log_deriv:
        fac = 1
    else:
        fac = (p_mix/10**lgr)

    return ((1 - y) * (p_mix/p_h) * p_rho_h + y * (p_mix/p_he) * p_rho_he)*fac


### Thermodynamic quantities ###

def get_cv_r(lgr, lgt, y): # equation gt. 7a and gt.14b in thermo notes
    return get_dudt_mix_r(lgr, lgt, y, log_deriv=False) 

def get_cp_r(lgp, lgt, lgr, y): # equation gt.14c in thermo notes
    cv = get_cv_r(lgr, lgt, y)
    #P = get_p_mix(lgr, lgt, y)
    rho = 10**lgr
    T = 10**lgt
    drdp_t = get_drdp_mix_t(lgp, lgt, lgr, y, log_deriv=False)
    drdt_p = get_drdt_mix_p(lgp, lgt, lgr, y,log_deriv=False)
    dpdt_r = drdp_t/drdt_p
    # dpdt_r = get_dpdt_mix_r(lgr, lgt, y, log_deriv=False)
    # dpdr_t = get_dpdr_mix_t(lgr, lgt, y, log_deriv=False)

    return cv + (T/rho**2) * (dpdt_r**2 * drdp_t)

def get_grada_r(lgp, lgt, lgr, y): # equation gt.14d in thermo notes
    cp = get_cp_r(lgp, lgt, lgr, y)
    cv = get_cv_r(lgr, lgt, y)
    dtdp_r = 1/get_dpdt_mix_r(lgp, lgt, lgr, y, log_deriv=True) # log derivative per eq. gt.14.d

    return ((cp - cv)/cp)*dtdp_r

def get_gamma1_r(lgp, lgt, lgr, y): # equation gt.14e in thermo notes
    cp = get_cp_r(lgp, lgt, lgr, y)
    cv = get_cv_r(lgp, lgt, lgr, y)
    dpdr_t = get_dpdr_mix_t(lgp, lgt, lgr, y, log_deriv=True) # log derivative per eq. gt.14.e

    return (cp/cv)*dpdr_t

# def get_gamma1_p(lgp, lgt, lgr, y):
#     cp = get_cp_p(lgp, lgt, y)
#     cv = get_cv_r(lgr, lgt, y)
#     dpdr_t = get_dpdr_mix_t(lgr, lgt, y, log_deriv=True)

#     return (cp/cv)*dpdr_t
