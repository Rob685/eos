import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from eos import ideal_eos, ppv_eos, aqua_eos, serpentine_eos, aneos_fe_eos
import os

from astropy import units as u
#from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu

mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ideal_water = ideal_eos.IdealEOS(m=18) # default for ideal eos is water for now

#### P, T ####

def get_rho_pt_tab(p, t, eos):
    if eos == 'aqua':
        return aqua_eos.get_rho_pt_tab(p, t)
    elif eos == 'ppv':
        return ppv_eos.get_rho_pt_tab(p, t)
    elif eos == 'serpentine':
        return serpentine_eos.get_rho_pt_tab(p, t)
    elif eos == 'iron':
        return aneos_fe_eos.get_rho_pt_tab(p, t)
    elif eos == 'ideal':
        return ideal_water.get_rho_pt(p, t, 0)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_s_pt_tab(p, t, eos):
    if eos == 'aqua':
        return aqua_eos.get_s_pt_tab(p, t)
    elif eos == 'ppv':
        return ppv_eos.get_s_pt_tab(p, t)
    elif eos == 'serpentine':
        return serpentine_eos.get_s_pt_tab(p, t)
    elif eos == 'iron':
        return aneos_fe_eos.get_s_pt_tab(p, t)
    elif eos == 'ideal':
        return ideal_water.get_s_pt(p, t, 0)/erg_to_kbbar
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

#### rho, T ####

def get_p_rhot_tab(rho, t, eos):
    if eos == 'aqua':
        return aqua_eos.get_p_rhot_tab(rho, t)
    elif eos == 'ppv':
        return ppv_eos.get_p_rhot_tab(rho, t)
    # elif eos == 'serpentine':
    #     #return serpentine_eos.get_p_rhot_tab(rho, t)
    elif eos == 'iron':
        return aneos_fe_eos.get_p_rhot_tab(rho, t)
    elif eos == 'ideal':
        return ideal_water.get_p_rhot(rho, t, 0)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_s_rhot_tab(rho, t, eos):
    if eos == 'aqua':
        return aqua_eos.get_s_rhot_tab(rho, t)
    elif eos == 'ppv':
        return ppv_eos.get_s_rhot_tab(rho, t)
    # elif eos == 'serpentine':
    #     #return serpentine_eos.get_p_rhot_tab(rho, t)
    elif eos == 'iron':
        return aneos_fe_eos.get_s_rhot_tab(rho, t)
    elif eos == 'ideal':
        return ideal_water.get_s_rhot(rho, t, 0)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

#### S, P ####

def get_t_sp_tab(s, p, eos):
    if eos == 'aqua':
        return aqua_eos.get_t_sp_tab(s, p)
    elif eos == 'ppv':
        return ppv_eos.get_t_sp_tab(s, p)
    elif eos == 'serpentine':
        return serpentine_eos.get_t_sp_tab(s, p)
    elif eos == 'iron':
        return aneos_fe_eos.get_t_sp_tab(s, p)
    elif eos == 'ideal':
        return ideal_water.get_t_sp(s, p, 0)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_rho_sp_tab(s, p, eos):
    if eos == 'aqua':
        return aqua_eos.get_rho_sp_tab(s, p)
    elif eos == 'ppv':
        return ppv_eos.get_rho_sp_tab(s, p)
    elif eos == 'serpentine':
        return serpentine_eos.get_rho_sp_tab(s, p)
    elif eos == 'iron':
        return aneos_fe_eos.get_rho_sp_tab(s, p)
    elif eos == 'ideal':
        return ideal_water.get_rho_sp(s, p, 0)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_rhot_sp_tab(s, p, eos):
    return get_rho_sp_tab(s, p, eos), get_t_sp_tab(s, p, eos)

#### S, rho ####

def get_t_srho_tab(s, rho, eos):
    if eos == 'aqua':
        return aqua_eos.get_t_srho_tab(s, rho)
    elif eos == 'ppv':
        return ppv_eos.get_t_srho_tab(s, rho)
    elif eos == 'serpentine':
        return serpentine_eos.get_t_srho_tab(s, rho)
    elif eos == 'iron':
        return aneos_fe_eos.get_t_srho_tab(s, rho)
    elif eos == 'ideal':
        return ideal_water.get_t_srho(s, rho, 0)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_p_srho_tab(s, rho, eos):
    if eos == 'aqua':
        return aqua_eos.get_p_srho_tab(s, rho)
    elif eos == 'ppv':
        return ppv_eos.get_p_srho_tab(s, rho)
    elif eos == 'serpentine':
        return serpentine_eos.get_p_srho_tab(s, rho)
    elif eos == 'iron':
        return aneos_fe_eos.get_p_srho_tab(s, rho)
    elif eos == 'ideal':
        return ideal_water.get_p_srho(s, rho, 0)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

############## derivatives ##############

def get_dtdrho_srho(s, rho, eos, drho=0.01):
    R0 = 10**rho
    R1 = R0*(1+drho)
    T0 = 10**get_t_srho_tab(s, np.log10(R0), eos)
    T1 = 10**get_t_srho_tab(s, np.log10(R1), eos)

    return (T1 - T0)/(R1 - R0)

def get_dtds_srho(s, rho, eos, ds=0.01):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho, eos)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho, eos)

    return (T1 - T0)/(S1 - S0)

def get_c_v(s, rho, eos, ds=0.1):
    # ds/dlogT_{rho, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_srho_tab(S0*erg_to_kbbar, rho, eos)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, rho, eos)
 
    return (S1 - S0)/(T1 - T0)

def get_c_p(s, p, eos, ds=0.1):
    # ds/dlogT_{P, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_sp_tab(S0*erg_to_kbbar, p, eos)
    T1 = get_t_sp_tab(S1*erg_to_kbbar, p, eos)

    return (S1 - S0)/(T1 - T0)