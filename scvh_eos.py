import numpy as np
#import scvh_nr
from scipy.optimize import root, root_scalar
from scipy.interpolate import RegularGridInterpolator as RGI
from eos import aneos#, scvh_man
import os
erg_to_kbbar = 1.202723550011625e-08
mh = 1 
mhe = 4.0026

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

eos_aneos = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='serpentine')
#eos_scvh = scvh_man.eos(path_to_data='%s/scvh_mesa' % CURR_DIR)

def scvh_reader(tab_name):
    tab = []
    head = []
    with open('%s/scvh/eos/%s' % (CURR_DIR, tab_name)) as file:
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


###### derivatives ######

s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, gamma1_arr, grada_arr = np.load('%s/scvh/scvh_thermo.npy' % CURR_DIR)

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

# def get_rho_z(s, p, y, z, z_eos='ideal'): # y should be scaled outside
#     if z_eos == 'ideal':
#         rho_z = 10**get_rho_p_ideal(s, p)
#     elif z_eos == 'aneos':
#         rho_z = 10**eos_aneos.get_logrho(p, get_t(np.array([y/(1-z), s, p]).T))
#     rho = 10**get_rho(np.array([y/(1-z), s, p]).T)
#     rho_mix = np.log10(1/((1-z)/rho + z/rho_z))
#     return rho_mix

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


def get_s_rhot(r, t, yhe):

    if np.isscalar(r):
        return (10**float(interp_s((yhe, x(r), y(r, t)))))/erg_to_kbbar
    else:
        return (10**interp_s(np.array([yhe, x(r), y(r, t)]).T))/erg_to_kbbar

def get_p_rhot(r, t, yhe):
    if np.isscalar(r):
        return float(interp_p((yhe, x(r), y(r, t))))
    else:
        return interp_p(np.array([yhe, x(r), y(r, t)]).T)

def get_sp_rhot(r, t, yhe):
    if np.isscalar(r):
        return (10**float(interp_s((yhe, x(r), y(r, t)))))/erg_to_kbbar, float(interp_p((yhe, x(r), y(r, t))))
    else:
        return (10**interp_s(np.array([yhe, x(r), y(r, t)]).T))/erg_to_kbbar, interp_p(np.array([yhe, x(r), y(r, t)]).T)

### error functions ###
def err_rhot(rt_pair, sval, pval, y):
    rho, t = rt_pair
    s, p = get_s_rhot(rho, t, y), get_p_rhot(rho, t, y)
    sval /= erg_to_kbbar
    return  s/sval - 1, p/pval -1

def err_rho_pt(lgrho, pval, t, y):
    logp_ = get_p_rhot(lgrho, t, y)
    return logp_/pval - 1

def err_t_rhop(lgt, lgp, rhoval, y):
    #lgp, lgt = pt_pair
    logrho_ = get_rho_pt(lgp, lgt, y)
    #s *= erg_to_kbbar
    return  logrho_/rhoval - 1

def err_t_sp(logt, logp, s_val, y):
    s_ = get_s_pt(logp, logt, y)
    s_val /= erg_to_kbbar # in cgs

    return (s_/s_val) - 1

def err_t_srho(lgt, lgr, sval, y):
    s = get_s_rhot(lgr, lgt, y)
    sval /= erg_to_kbbar
    return s/sval - 1

### inversions ###

TBOUNDS = [0, 7] # logrho and logt_sp test passes with [0, 6]
PBOUNDS = [0, 15] # works with 1, 15
RHOBOUNDS = [R1+0, R2-2] # works with +1 and -2

XTOL = 1e-8

def get_rho_pt(p, t, y):
    if np.isscalar(p):
        try:
            sol = root_scalar(err_rho_pt, bracket=RHOBOUNDS, xtol=XTOL, method='brenth', args=(p, t, y))
            return sol.root
        except:
            print('p={}, t={}, y={}'.format(p, t, y))
            # err1 = err_rho_pt(-5, p, TBOUNDS[0], y)
            # err1 = err_rho_pt(-5, p, TBOUNDS[0], y)
            #print('errors at temp bounds=[{},{}]')
            raise

    sol = np.array([get_rho_pt(p_, t_, y_) for p_, t_, y_ in zip(p, t, y)])
    return sol

def get_s_pt(p, t, y):
    rho = get_rho_pt(p, t, y)
    #rho, T = get_rhot_sp()
    return get_s_rhot(rho, t, y)

def get_rhot_sp(s, p, y, guess=[-2, 2.1], alg='hybr'): # in-situ inversion
    if np.isscalar(s):
        sol = root(err_rhot, guess, tol=1e-8, method=alg, args=(s, p, y))
        return sol.x

    rho, t = np.array([get_rhot_sp(s_, p_, y_) for s_, p_, y_ in zip(s, p, y)]).T
    return rho, t

def get_t_rhop(rho, p, y):
    TBOUNDS2 = [0, 4.1]
    if np.isscalar(rho):
        try:
            sol = root_scalar(err_t_rhop, bracket=TBOUNDS2, xtol=XTOL, method='brenth', args=(p, rho, y))
            return sol.root
        except:
            print('rho={}, p={}, y={}'.format(rho, p, y))
            raise
    
    sol = np.array([get_t_rhop(rho_, p_, y_) for rho_, p_, y_ in zip(rho, p, y)])
    return sol

def get_t_sp(s, p, y):
    if np.isscalar(s):
        sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(p, s, y)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
    sol = np.array([get_t_sp(s_, p_, y_) for s_, p_, y_ in zip(s, p, y)])
    return sol

def get_t_srho(s, rho, y):
    if np.isscalar(s):
        try:
            sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(rho, s, y))
            return sol.root
        except:
            print('s={}, rho={}, y={}'.format(s, rho, y))
            raise

    sol = np.array([get_t_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
    return sol

def get_p_srho(s, rho, y):
    t = get_t_srho(s, rho, y)
    return get_p_rhot(rho, t, y)

def get_s_rhop(rho, p, y):
    t = get_t_rhop(rho, p, y)
    #y = cms.n_to_Y(y)
    s = get_s_rhot(rho, t, y)
    return s # in cgs

### derivatives ###

def get_dpdy_srho(s, rho, y, dy=0.1):
    P0 = get_p_srho(s, rho, y)
    P1 = get_p_srho(s, rho, y*(1+dy))

    return (P1 - P0)/(y*dy) #dlogP/dY

def get_dpds_srho(s, rho, y, ds=0.1):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    P0 = get_p_srho(S0*erg_to_kbbar, rho, y)
    P1 = get_p_srho(S1*erg_to_kbbar, rho, y)

    return (P1 - P0)/(S1 - S0)

# def get_dsdy_rhop(rho, p, y, dy=0.1):
#     # S0 = get_s_rhop(rho, p, y)
#     # S1 = get_s_rhop(rho, p, y*(1+dy))

#     return (S1 - S0)/(y*dy)

def get_dsdy_rhop(s, rho, y, dy=0.01, ds=0.1):

    dpdy = get_dpdy_srho(s, rho, y, dy=dy)
    dpds = get_dpds_srho(s, rho, y, ds=ds)

    return -dpdy/dpds
