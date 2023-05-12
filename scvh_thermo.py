import numpy as np
import scvh_nr
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib.pyplot as plt


yvals = np.array([0.22, 0.25, 0.28, 0.32])

s_arr = []
p_arr = []
t_arr = []
r_arr = []

cp_arr = []
cv_arr = []
chirho_arr = []
chit_arr = []
grada_arr = []
gamma1_arr = []

for y in yvals:
    s, p, t, r, cp, cv, chirho, chit, grada, gamma1 = np.load('inverted_eos_data/eos_data/{}_{}.npy'.format('scvh_main_thermo', int(y*100)))
    s_arr.append(s)
    p_arr.append(p)
    t_arr.append(t)
    r_arr.append(r)

    cp_arr.append(cp)
    cv_arr.append(cv)
    chirho_arr.append(chirho)
    chit_arr.append(chit)

    grada_arr.append(grada)
    gamma1_arr.append(gamma1)

# saving to self to use later for composition gradients (03/23/23)    
y_arr = np.array([y for y in sorted(yvals)])
s_arr = np.array([x for _, x in sorted(zip(y_arr, s_arr))])
p_arr = np.array([x for _, x in sorted(zip(y_arr, p_arr))])
t_arr = np.array([x for _, x in sorted(zip(y_arr, t_arr))])
r_arr = np.array([x for _, x in sorted(zip(y_arr, r_arr))])

cp_arr = np.array([x for _, x in sorted(zip(y_arr, cp_arr))])
cv_arr = np.array([x for _, x in sorted(zip(y_arr, cv_arr))])
chirho_arr = np.array([x for _, x in sorted(zip(y_arr, chirho_arr))])
chit_arr = np.array([x for _, x in sorted(zip(y_arr, chit_arr))])
grada_arr = np.array([x for _, x in sorted(zip(y_arr, grada_arr))])
gamma1_arr = np.array([x for _, x in sorted(zip(y_arr, gamma1_arr))])

get_t = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)
get_rho = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)

get_cp = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
get_cv = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

get_chirho = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
get_chit = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

get_grada = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
get_gamma1 = RGI((y_arr, s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

# def get_s(r, t):
    
#     sres = 10**np.array([scvh_nr.get_sp_sixpt(r[i], t[i])[0] for i in range(len(r))])

#     return sres

def get_rho_t(y, s, p):
    t_res = get_t(np.array([y, s, p]).T)
    rho_res = get_rho(np.array([y, s, p]).T)
    return rho_res, t_res

def get_c_p(y, s, p):
    cp_res = get_cp(np.array([y, s, p]).T)
    #cv_res = get_cv(np.array([y, s, p]).T)
    return cp_res

def get_c_v(y, s, p):
    #cp_res = get_cp(np.array([y, s, p]).T)
    cv_res = get_cv(np.array([y, s, p]).T)
    return cv_res

def get_chi_rho(y, s, p):
    chirho_res = get_chirho(np.array([y, s, p]).T)
    return chirho_res

def get_chi_t(y, s, p):
    chit_res = get_chit(np.array([y, s, p]).T)
    return chit_res

def get_grad_ad(y, s, p):
    grada = get_grada(np.array([y, s, p]).T)
    return grada

def get_gamma_1(y, s, p):
    gamma1 = get_gamma1(np.array([y, s, p]).T)
    return gamma1

#################### pressure ####################

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

# yvals_p = np.array([0.22, 0.25, 0.28, 0.32])

# p_arr_p, t_arr_p, rho_arr_p, s_arr_p = np.load('inverted_eos_data/eos_data/scvh_main_p.npy')

# get_rho_p = RGI((yvals_p, p_arr_p[0][:,0], t_arr_p[0,:][0]), rho_arr_p, method='linear', bounds_error=False, fill_value=None)
# get_s_p = RGI((yvals_p, p_arr_p[0][:,0], t_arr_p[0,:][0]), s_arr_p, method='linear', bounds_error=False, fill_value=None)

# def get_rhop(y, p, t):
#     rho_res = get_rho_p(np.array([y, p, t]).T)
#     return rho_res

# def get_s(y, p, t):
#     s_res = get_s_p(np.array([y, p, t]).T)
#     return s_res

# def scvh_reader(tab_name):
#     tab = []
#     head = []
#     with open('/Users/Helios/burrows_research/scvh_eos_adam/eos/'+tab_name) as file:
#         for j, line in enumerate(file):
#             line = line.rstrip('\n')
#             if line.startswith("#"):
#                 line = line.lstrip('#')
#                 head.append(line)

#             else:# j > head: # skipping the header
#                 tab.append(list(float(line[i:i+8]) for i in range(0, len(line), 8)))

#     header = list(filter(None, ' '.join(head).split(' ')))

#     tab = np.array(tab)
#     tab_ = np.reshape(tab, (300, 100))
#     return [float(val) for val in header], tab_

# bounds_s, stab = scvh_reader('stabnew_adam.dat')
# R1, R2, T1, T2, T11, T12 = bounds_s[2:8] # same bounds
# logrho_array = np.linspace(R1, R2, 100)

# stab_names = ['s22scz.dat', 'stabnew_adam.dat', 's28scz.dat', 's30.dat']
# ptab_names = ['p22scz.dat', 'ptabnew_adam.dat', 'p28scz.dat', 'p30.dat']

# stabs = np.array([scvh_reader(name)[-1] for name in stab_names])
# ptabs = np.array([scvh_reader(name)[-1] for name in ptab_names])

# def x(R):
#     return (R - R1)*(300 - 1)/(R2 - R1)

# def deltaT(R):
#     return (R - R1)*((T12 - T11) - (T2 - T1))/(R2 - R1) + (T2 - T1)

# def T1p(R):
#     m1 = (T11 - T1)/(R2 - R1)
#     return m1*(R - R1) + T1

# def T1p(R):
#     m1 = (T11 - T1)/(R2 - R1)
#     return m1*(R - R1) + T1

# def y(R, T):
#     return (T - T1p(R))*(100 - 1)/deltaT(R)

# def get_tarr(R):
#     m1 = (T12 - T1)/(R2 - R1)
#     b = T1 - m1*R1
#     T1p = R*m1 + b
#     #T1p = (R - R1)*m1 + b
#     T2p = T1p + deltaT(R)

#     return np.linspace(T1p, T2p, 100)

# def deltaT(R):
#     eta = (R - R1)/(R2 - R1)
#     return eta*(T2 - T1) + (1-eta)*(T12 - T11)

# def get_tarr(R):
#     m1 = (T11 - T1)/(R2 - R1)
#     b = T1 - m1*R1
#     T1p = R*m1 + b
#     #T1p = (R - R1)*m1 + b
#     T2p = T1p + deltaT(R)

#     return np.linspace(T1p, T2p, 100)

# Y_he_arr = np.array([0.22, 0.25, 0.28, 0.30])
# x_arr = np.arange(0, 300, 1)
# y_arr = np.arange(0, 100, 1)

# interp_s = RGI((Y_he_arr, x_arr, y_arr), stabs, method='linear', bounds_error=False, fill_value=None)
# interp_p = RGI((Y_he_arr, x_arr, y_arr), ptabs, method='linear', bounds_error=False, fill_value=None)

# def get_s(Y, r, t):
#     # test
#     #y_arr = y(logrho_array, get_tarr(r))
#     interp_s = RGI((Y_he_arr, x_arr, y_arr), stabs, method='linear', bounds_error=False, fill_value=None)
    
#     return interp_s(np.array([Y, x(r), y(r, t)]).T)

# def get_p(Y, r, t):
#     #test
#     #y_arr = y(logrho_array, get_tarr(r))
#     interp_p = RGI((Y_he_arr, x_arr, y_arr), ptabs, method='linear', bounds_error=False, fill_value=None)
#     return interp_p(np.array([Y, x(r), y(r, t)]).T)



#################### comp gradients ####################

#yvals_p = []
# dlogrho_dy = np.load('data/dlogrho_dY_scvh.npy')

# logpgrid = np.linspace(5, 14, 100)
# logtgrid = np.linspace(2.1, 5, 100)

# ygrid = np.linspace(0.22, 0.32, 100)
# get_dlogrho_dy = RGI((logpgrid, logtgrid, ygrid), dlogrho_dy, method='linear', bounds_error=False, fill_value=None)

# def get_dlogrhody(p, t, y):
#     drhody = get_dlogrho_dy(np.array([p, t, y]).T)
#     return drhody

#rhoarr, sarr = np.load()
