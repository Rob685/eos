import numpy as np
from scipy.optimize import brenth, brentq
from . import cms_newton_raphson as cms
#import cms_tables_rgi as cms_rgi
# import aneos
from scipy.interpolate import RegularGridInterpolator as RGI

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
# eos_aneos = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='ice')

erg_to_kbbar = 1.202723550011625e-08

def err(logt, logp, y, z, s_val, m=15.5, corr=True):

    #s_ = cms.get_s_mix(logp, logt, y, corr)
    s_ = float(cms.get_smix_z(y, z, logp, logt, mz=m))
    s_val /= erg_to_kbbar # in cgs
    #print((s_/s_val) - 1, logt, logp)
    #return (s_/s_val) - 1
    return s_ - s_val

def get_rho_p_ideal(s, logp, m=15.5):
    # done from ideal gas
    # note: 10 is average molecular weight for solar comp
    # for Y = 0.25, m = 3 * (1 * 1 + 1 * 0) + (1 * 4) / 7 = 1
    p = 10**logp
    return np.log10(np.maximum(np.exp((2/5) * (5.096 - s)) * (np.maximum(p, 0) / 1e11)**(3/5) * m**(8/5), np.full_like(p, 1e-10)))

# def get_rho_aneos(logp, logt):
#     return eos_aneos.get_logrho(logp, logt)

def rho_mix(p, t, y, z, ideal, m=15.5):
    rho_hhe = float(cms.get_rho_mix(p, t, y, hc_corr=True))
    try:
        #t = get_t(s, p, y, z)
        rho_z = 10**get_rho_id(p, t, m=m)
        # if ideal:
        #     rho_z = 10**get_rho_id(p, t)
        # elif not ideal:
        #     rho_z = 10**get_rho_aneos(p, t)
    except:
        print(p, y, z)

    return np.log10(1/((1 - z)/rho_hhe + z/rho_z))
    # except:
    #     print(s, p, y, z)
    # raise
    # #if z > 0:

    # elif z == 0:
    #      return np.log10(rho_hhe)

def get_t(s, p, y, z, m=15.5):
    t_root = brenth(err, 0, 7, args=(p, y, z, s, m)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
    return t_root

# def get_rho(s, p, t, y, z, ideal):
#     try:
#         t = get_t(s, p, y, z)
#     except:
#         print(s, p, y, z)
#         raise
#     return rho_mix(s, p, t, y, z, ideal)

def get_rhot(s, p, y, z, ideal, m=15.5):
    try:
        t = get_t(s, p, y, z, m=m)
    except:
        print(s, p, y, z)
        raise
    rho = rho_mix(p, t, y, z, ideal, m=m)
    return rho, t

###### derivatives ######

s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, gamma1_arr, grada_arr = np.load('%s/cms/cms_hg_thermo.npy' % CURR_DIR)

get_rho_ = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)
get_t_ = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)

get_cp = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
get_cv = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

get_chirho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
get_chit = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

get_grada = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
get_gamma1 = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

def get_rho_t(s, p, y):
    return get_rho_(np.array([y, s, p]).T), get_t_(np.array([y, s, p]).T)

def get_c_p(s, p, y):
    cp_res = get_cp(np.array([y, s, p]).T)
    return cp_res

def get_c_v(s, p, y):
    cv_res = get_cv(np.array([y, s, p]).T)
    return cv_res

def get_chi_rho(s, p, y):
    chirho_res = get_chirho(np.array([y, s, p]).T)
    return chirho_res

def get_chi_t(s, p, y):
    chit_res = get_chit(np.array([y, s, p]).T)
    return chit_res

def get_grad_ad(s, p, y):
    grada = get_grada(np.array([y, s, p]).T)
    return grada

def get_gamma_1_hhe(s, p, y):
    return get_gamma1(np.array([y, s, p]).T)

def get_logrho_mix(s, p, y, z, m=15.5):
    if not np.isscalar(s):
        return np.array([get_logrho_mix(si, pi, yi, zi, m=m)
                         for si, pi, yi, zi in zip(s, p, y, z)])
    try:
        t = get_t(s, p, y, z, m=m)
    except:
        print(s, p, y, z)
        raise
    rho_hhe = float(cms.get_rho_mix(p, t, y, hc_corr=True)) # already in cgs
    rho_z = 10**get_rho_p_ideal(s, p, m=m)

    return np.log10(1/((1 - z)/rho_hhe + z/rho_z))

def get_gamma1_calc(s, p, y, z, m=15.5, dp=0.001):
    delta_logrho = (-get_logrho_mix(s, p, y, z, m=m)
                    + get_logrho_mix(s, p*(1+dp), y, z, m=m))
    dlogrho_dlogP = delta_logrho/(p*dp)
    return 1/dlogrho_dlogP

def get_rho_id(logp, logt, m=15.5):
    return np.log10(((10**logp) * m*1.6605390666e-24) / (1.380649e-16 * (10**logt)))
