import numpy as np
#import scvh_nr
from scipy.optimize import root
from scipy.interpolate import RegularGridInterpolator as RGI

import os
erg_to_kbbar = 1.202723550011625e-08
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def scvh_reader(tab_name):
    tab = []
    head = []
    with open('/Users/Helios/planet_interiors/state/scvh/eos/'+tab_name) as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            if line.startswith("#"):
                line = line.lstrip('#')
                head.append(line)

            else:# j > head: # skipping the header
                tab.append(list(float(line[i:i+8]) for i in range(0, len(line), 8)))

    header = list(filter(None, ' '.join(head).split(' ')))

    tab = np.array(tab)
    tab_ = np.reshape(tab, (300, 100))

    return [float(val) for val in header], tab_

bounds_s, stab = scvh_reader('stabnew_adam.dat')
_, INDEX, R1, R2, T1, T2, T11, T12 = bounds_s # same bounds

stab_names = ['s22scz.dat', 'stabnew_adam.dat', 's28scz.dat', 's30.dat']
ptab_names = ['p22scz.dat', 'ptabnew_adam.dat', 'p28scz.dat', 'p30.dat']

stabs = np.array([scvh_reader(name)[-1] for name in stab_names])
ptabs = np.array([scvh_reader(name)[-1] for name in ptab_names])

bounds_s, stab = scvh_reader('stabnew_adam.dat')
R1, R2, T1, T2, T11, T12 = bounds_s[2:8] # same bounds

logrho_arr = np.linspace(R1, R2, 300)
stab_names = ['s22scz.dat', 'stabnew_adam.dat', 's28scz.dat', 's30.dat']
stabs = np.array([scvh_reader(name)[-1] for name in stab_names])
ptab_names = ['p22scz.dat', 'ptabnew_adam.dat', 'p28scz.dat', 'p30.dat']
ptabs = np.array([scvh_reader(name)[-1] for name in ptab_names])

def x(R):
    return (R - R1)*(300)/(R2 - R1)

def T1p(R):
    m1 = (T11 - T1)/(R2 - R1)
    return m1*(R - R1) + T1

def T2p(R):
    m2 = (T12 - T2)/(R2 - R1)
    return m2*(R - R1) + T2

def get_logtarr(R):
    return np.linspace(T1p(R), T2p(R), 100)

def y(R, T):
    return ((T - T1p(R))/(T2p(R) - T1p(R)))*(100)

yhe_arr = np.array([0.22, 0.25, 0.28, 0.30])

x_arr = np.arange(0, 300, 1)
y_arr = np.arange(0, 100, 1)
interp_s = RGI((yhe_arr, x_arr, y_arr), stabs, method='linear', bounds_error=False, fill_value=None)
interp_p = RGI((yhe_arr, x_arr, y_arr), ptabs, method='linear', bounds_error=False, fill_value=None)

def get_s(r, t, yhe):

    if not hasattr(r, '__len__'):
        return float(interp_s((yhe, r, t)))
    else:
        return interp_s(np.array([yhe, x(r), y(r, t)]).T)

def get_p(r, t, yhe):
    if not hasattr(r, '__len__'):
        return float(interp_p((yhe, x(r), t)))
    else:
        return interp_p(np.array([yhe, x(r), y(r, t)]).T)

def get_sp(r, t, yhe):
    if not hasattr(r, '__len__'):
        return float(interp_s((yhe, x(r), y(r, t)))), float(interp_p((yhe, x(r), y(r, t))))
    else:
        return interp_s(np.array([yhe, x(r), y(r, t)]).T), interp_p(np.array([yhe, x(r), y(r, t)]).T)

def err_scvh(rt_pair, sval, pval, y):
    rho, temp = rt_pair
    s, p = get_sp(rho, temp, y)

    return  s/sval - 1, p/pval -1

def get_rho_p_ideal(s, logp, m=15.5):
    # done from ideal gas
    # note: 15.5 is average molecular weight for solar comp
    # for Y = 0.25, m = 3 * (1 * 1 + 1 * 0) + (1 * 4) / 7 = 1
    p = 10**logp
    return np.log10(np.maximum(np.exp((2/5) * (5.096 - s)) * (np.maximum(p, 0) / 1e11)**(3/5) * m**(8/5), np.full_like(p, 1e-10)))

def get_rhot(s, p, y, z, ideal=None): # in-situ inversion
    s = np.log10(s)
    sol = root(err_scvh, [-2, 2.1], args=(s, p, y))

    # if z > 0:
    #     rho_hhe = 10**sol.x[0]
    #     rho_z = 10**get_rho_p_ideal(s, p)
    #     return np.log10(1/((1 - z)/rho_hhe + z/rho_z)), sol.x[1]
    # else:
    return sol.x

##### pressure-temperature #####

logp_res, logt_res, logrho_res, s_res = np.load('%S/SCVH/scvh_pt.npy' % CURR_DIR)
yvals = np.array([0.22, 0.25, 0.28, 0.30])

get_rho_pt = RGI((yvals, logt_res[0][:,0], logp_res[0][0]), logrho_res)
get_s_pt = RGI((yvals, logt_res[0][:,0], logp_res[0][0]), s_res)

def get_rhos_p_t(p, t, y):
    return get_rho_pt(np.array([y, t, p]).T), get_s_pt(np.array([y, t, p]).T)


###### derivatives ######

s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, gamma1_arr, grada_arr = np.load('eos/scvh/scvh_thermo.npy')

get_rho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)
get_t= RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)

get_cp = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
get_cv = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

get_chirho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
get_chit = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

get_grada = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
get_gamma1 = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

def get_rho_t(s, p, y):
    # cp_res = get_cp(np.array([y, s, p]).T)
    # cv_res = get_cv(np.array([y, s, p]).T)
    return get_rho(np.array([y, s, p]).T), get_t(np.array([y, s, p]).T)

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
    gamma1_hhe = get_gamma1(np.array([y, s, p]).T)
    # rho_tot = 10**np.array([get_rho(s[i], p[i], y[i], z[i], ideal) for i in range(len(p))])
    # rho_hhe = 10**np.array([get_rho(s[i], p[i], y[i], 0, ideal) for i in range(len(p))])
    # rho_z = 10**get_rho_p_ideal(s, p)
    return gamma1_hhe
    #return 1/((1-z)*(rho_hhe/rho_tot) * (1/gamma1_hhe) + z*(rho_z/rho_tot) * (3/5))

####### composition derivatives #######

dlogrho_dy, dlogs_dy = np.load('/Users/Helios/planet_interiors/state/scvh/comp_derivatives_scvh.npy')

logtvals = np.linspace(2.1, 5, 100)
logpvals = np.linspace(5, 14, 300)

ygrid = np.arange(0.22, 1.0, 0.01)

get_dlogrho_dy = RGI((logtvals, logpvals, ygrid), dlogrho_dy, method='linear', bounds_error=False, fill_value=None)

def get_dlogrhody(p, t, y):
    drhody = get_dlogrho_dy(np.array([t, p, y]).T)
    return drhody
