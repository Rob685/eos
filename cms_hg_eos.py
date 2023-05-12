import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from state import cms_newton_raphson as cms
#import thermo_chabrier_v2 as pt_cms

y_arr = np.array([0.2 , 0.22, 0.24, 0.26, 0.28, 0.3 , 0.32, 0.34, 0.36, 0.38, 0.4 ,0.42, 0.44, 0.46, 0.48])

s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, grada_arr, gamma1_arr = np.load('state/cms/cms_hg_thermo.npy')


get_t = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)
get_rho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)

get_cp = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
get_cv = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

get_chirho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
get_chit = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

get_grada = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
get_gamma1 = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

def get_s(p, t, y):
    s = cms.get_s_mix(p, t, y, hc_corr=True)
    return s

def get_rho_t(s, p, y):
    t_res = get_t(np.array([y, s, p]).T)
    rho_res = get_rho(np.array([y, s, p]).T)
    return rho_res, t_res

def get_c_p(s, p, y):
    cp_res = get_cp(np.array([y, s, p]).T)
    #cv_res = get_cv(np.array([y, s, p]).T)
    return cp_res

def get_c_v(s, p, y):
    #cp_res = get_cp(np.array([y, s, p]).T)
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

def get_gamma_1(s, p, y):
    gamma1 = get_gamma1(np.array([y, s, p]).T)
    return gamma1

#################### comp gradients ####################

# dlogrho_dy, dlogs_dy = np.load('state/cms/comp_derivatives_cms.npy')

# logtvals = np.linspace(2, 5, 100)
# logpvals = np.linspace(5, 14, 300)


# ygrid = np.arange(0.22, 1.0, 0.01)

# get_dlogrho_dy = RGI((logtvals, logpvals, ygrid), dlogrho_dy, method='linear', bounds_error=False, fill_value=None)

# def get_dlogrhody(p, t, y):
#     drhody = get_dlogrho_dy(np.array([t, p, y]).T)
#     return drhody
