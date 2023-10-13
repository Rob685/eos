import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import scvh_nr

# y_arr = np.array([0.22, 0.25, 0.28, 0.30, 0.32])

# s_arr, p_arr, t_arr, r_arr, cp_arr, cv_arr, chirho_arr, chit_arr, grada_arr, gamma1_arr = np.load('inverted_eos_data/eos_data/scvh_main_thermo.npy')

# get_t = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)
# get_rho = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)

# get_cp = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
# get_cv = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

# get_chirho = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
# get_chit = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

# get_grada = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
# get_gamma1 = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

# def get_rho_t(y, s, p):
#     t_res = get_t(np.array([y, s, p]).T)
#     rho_res = get_rho(np.array([y, s, p]).T)
#     return rho_res, t_res

# def get_c_p(y, s, p):
#     cp_res = get_cp(np.array([y, s, p]).T)
#     #cv_res = get_cv(np.array([y, s, p]).T)
#     return cp_res

# def get_c_v(y, s, p):
#     #cp_res = get_cp(np.array([y, s, p]).T)
#     cv_res = get_cv(np.array([y, s, p]).T)
#     return cv_res

# def get_chi_rho(y, s, p):
#     chirho_res = get_chirho(np.array([y, s, p]).T)
#     return chirho_res

# def get_chi_t(y, s, p):
#     chit_res = get_chit(np.array([y, s, p]).T)
#     return chit_res

# def get_grad_ad(y, s, p):
#     grada = get_grada(np.array([y, s, p]).T)
#     return grada

# def get_gamma_1(y, s, p):
#     gamma1 = get_gamma1(np.array([y, s, p]).T)
#     return gamma1

########### calculating entropy #############

bounds_s, stab = scvh_nr.scvh_reader('stabnew_adam.dat')
R1, R2, T1, T2, T11, T12 = bounds_s[2:8] # same bounds

stab_names = ['s22scz.dat', 'stabnew_adam.dat', 's28scz.dat', 's30.dat']
stabs = np.array([scvh_nr.scvh_reader(name)[-1] for name in stab_names])

def x(R):
    return (R - R1)*(300 - 1)/(R2 - R1)

def deltaT(R):
    return (R - R1)*((T12 - T11) - (T2 - T1))/(R2 - R1) + (T2 - T1)

def T1p(R):
    m1 = (T11 - T1)/(R2 - R1)
    return m1*(R - R1) + T1

def y(R, T):
    return (T - T1p(R))*(100 - 1)/deltaT(R)

def get_s(R, T, yhe):
     # all of these have the same temperature ranges so I don't need to change the bounds each time
    
    if 0.22 <= yhe < 0.25:
        y1, y2 = 0.22, 0.25
        tab1 = stabs[0]
        tab2 = stabs[1]

    elif 0.25 <= yhe < 0.28:
        y1, y2 = 0.25, 0.28
        tab1 = stabs[1]
        tab2 = stabs[2]
        
    elif 0.28 <= yhe <= 0.30:
        y1, y2 = 0.28, 0.30
        tab1 = stabs[2]
        tab2 = stabs[3]
    
    x_arr = np.arange(0, 300, 1)
    y_arr = np.arange(0, 100, 1)
    
    eta1 = (y2 - yhe)/(y2 - y1)
    eta2 = 1 - eta1
    
    smix = eta1*tab1 + eta2*tab2
    
    x_arr = np.arange(0, 300, 1)
    y_arr = np.arange(0, 100, 1)
    
    interp_stab = RGI((x_arr, y_arr), smix, method='linear', bounds_error=False, fill_value=None)
    
    return interp_stab((x(R), y(R, T)))

########### composition gradients #############

# dlogrho_dy, dlogs_dy = np.load('data/comp_gradients_log_scvh.npy') #dlogho/dY, dlogS/dY
# logpgrid = np.linspace(5, 14, 100)
# logtgrid = np.linspace(2.1, 5, 100)

# ygrid = np.linspace(np.min(y_arr), np.max(y_arr), 100)
# get_drho_dy = RGI((logpgrid, logtgrid, ygrid), dlogrho_dy, method='linear', bounds_error=False, fill_value=None)
# get_ds_dy = RGI((logpgrid, logtgrid, ygrid), dlogs_dy, method='linear', bounds_error=False, fill_value=None)

# def get_drhody(self, p, t, y):
#     drhody = self.get_drho_dy(np.array([p, t, y]).T)
#     return drhody

# def get_dsdy(self, p, t, y):
#     dsdy = self.get_ds_dy(np.array([p, t, y]).T)
#     return dsdy