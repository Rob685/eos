import numpy as np
from scipy.optimize import brenth
from . import cms_newton_raphson as cms
#import cms_tables_rgi as cms_rgi
# import aneos
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
# eos_aneos = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='ice')

erg_to_kbbar = 1.202723550011625e-08

def err(logt, logp, y, s_val, corr):

    #s_ = cms.get_s_mix(logp, logt, y, corr)
    s_ = float(cms.get_s_mix(logp, logt, y, corr))
    s_val /= erg_to_kbbar # in cgs
    #print((s_/s_val) - 1, logt, logp)
    #return (s_/s_val) - 1
    return s_ - s_val

# def get_rho_p_ideal(s, logp, m=15.5):
#     # done from ideal gas
#     # note: 10 is average molecular weight for solar comp
#     # for Y = 0.25, m = 3 * (1 * 1 + 1 * 0) + (1 * 4) / 7 = 1
#     p = 10**logp
#     return np.log10(np.maximum(np.exp((2/5) * (5.096 - s)) * (np.maximum(p, 0) / 1e11)**(3/5) * m**(8/5), np.full_like(p, 1e-10)))

# def get_rho_aneos(logp, logt):
#     return eos_aneos.get_logrho(logp, logt)

# def rho_mix(p, t, y, z, hc_corr, m=15.5):
#     rho_hhe = float(cms.get_rho_mix(p, t, y, hc_corr))
#     try:
#         #t = get_t(s, p, y, z)
#         rho_z = 10**get_rho_id(p, t, m=m)
#         # if ideal:
#         #     rho_z = 10**get_rho_id(p, t)
#         # elif not ideal:
#         #     rho_z = 10**get_rho_aneos(p, t)
#     except:
#         print(p, y, z)

    return np.log10(1/((1 - z)/rho_hhe + z/rho_z))
    # except:
    #     print(s, p, y, z)
    # raise
    # #if z > 0:

    # elif z == 0:
    #      return np.log10(rho_hhe)

def get_t(s, p, y, corr):
    try:
        t_root = brenth(err, 0, 5, args=(p, y, s, corr)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
        return t_root
    except:
        print(s, p, y)
        raise

# def get_rho(s, p, t, y, z, ideal):
#     try:
#         t = get_t(s, p, y, z)
#     except:
#         print(s, p, y, z)
#         raise
#     return rho_mix(s, p, t, y, z, ideal)

def get_rhot(s, p, y, hc_corr):
    try:
        t = get_t(s, p, y, hc_corr)
    except:
        print(s, p, y)
        raise
    #rho = rho_mix(p, t, y, z, hc_corr, m=m)
    rho = np.log10(cms.get_rho_mix(p, t, y, hc_corr))
    return rho, t

###### inverted tables ######

## t, rho (s, p, y) ##
"""To revert to the old version with the HG corrections, uncomment the first line.
All functions should be the same for ease of use."""
s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, gamma1_arr, grada_arr = np.load('%s/cms/cms_hg_thermo.npy' % CURR_DIR)
#s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, grada_arr = np.load('%s/cms/cms_thermo.npy' % CURR_DIR)

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

## p, t (s, rho, y) ##

p_sry, t_sry = np.load('%s/cms/p_sry.npy' % CURR_DIR), np.load('%s/cms/t_sry.npy' % CURR_DIR)

svals = np.arange(5.75, 9.1, 0.01)
logrhovals = np.arange(-4, 1, 0.05)
yvals = np.arange(0.1, 1, 0.01)

get_p_rgi = RGI((svals, logrhovals, yvals), p_sry, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi = RGI((svals, logrhovals, yvals), t_sry, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_sr(s, r, y):
    return get_p_rgi(np.array([s, r, y]).T)

def get_t_sr(s, r, y):
    return get_t_rgi(np.array([s, r, y]).T)


####### P(rho, s, y) #######

#logp_res_rhos, logrho_res_rhos, s_res_rhos = np.load('%s/cms/cms_rho_s.npy' % CURR_DIR)
#yarr2 = np.arange(0.22, 0.46, 0.02)

#get_p_s_r = RGI((yarr2, s_res_rhos[0][:,0], logrho_res_rhos[0][0]), logp_res_rhos, method='linear', bounds_error=False, fill_value=None)

# def get_p_sr(r, s, y):
#     return get_p_s_r(np.array([y, s, r]).T)

#y_arr3 = np.arange(0.22, 0.75, 0.01)

#dlogrho_dY = []
# dlogp_dY = []
# for i, s in enumerate(s_res_rhos[0][:,0]):
#     #drho_dY = []
#     dp_dY = []
#     for j, r in enumerate(logrho_res_rhos[0][0]):
#         logp = get_p_sr(np.full_like(y_arr3, r), np.full_like(y_arr3, s), y_arr3) # at constant rho, s
#         #s = get_s_p_t(np.full_like(y_arr, p), np.full_like(y_arr, t), y_arr)
#         dlogpdy = np.gradient(logp)/np.gradient(y_arr3)
#         #dlogsdY = np.gradient(np.log10(s/erg_to_kbbar))/np.gradient(y_arr)
#         #drho_dY.append(dlogrhodY)
#         dp_dY.append(dlogpdy)
        
#     #dlogrho_dY.append(drho_dY)
#     dlogp_dY.append(dp_dY)
    
# s_arr2 = np.arange(5.6, 10.1, 0.1)

# dlogp_dlogs = []
# for i, y_ in enumerate(yarr2):
#     #drho_dY = []
#     dp_ds = []
#     for j, r in enumerate(logrho_res_rhos[0][0]):
#         logp = get_p_sr(np.full_like(s_arr2, r), s_arr2, np.full_like(s_arr2, y_)) # at constant rho, Y
#         #s = get_s_p_t(np.full_like(y_arr, p), np.full_like(y_arr, t), y_arr)
#         dlogs = np.gradient(np.log10(s_arr2/erg_to_kbbar))
#         dlogpdlogs = np.gradient(logp)/dlogs
#         #dlogsdY = np.gradient(np.log10(s/erg_to_kbbar))/np.gradient(y_arr)
#         #drho_dY.append(dlogrhodY)
#         dp_ds.append(dlogpdlogs)
        
#     #dlogrho_dY.append(drho_dY)
#     dlogp_dlogs.append(dp_ds)

# get_dlogp_dy = RGI((np.array(s_res_rhos)[0][:,0], np.array(logrho_res_rhos)[0][0], y_arr3), dlogp_dY, method='linear', bounds_error=False, fill_value=None)
# get_dlogp_dlogs = RGI((yarr2, np.array(logrho_res_rhos)[0][0], s_arr2), dlogp_dlogs, method='linear', bounds_error=False, fill_value=None)

# def get_dpdy(r, s, y):
#     return get_dlogp_dy(np.array([s, r, y]).T)*(10**get_p_sr(r, s, y))

# def get_dpds(r, s, y):
#     return get_dlogp_dlogs(np.array([y, r, s]).T)*(10**get_p_sr(r, s, y))/(s/erg_to_kbbar)

########### energy ###########

# s_res = np.load('%s/cms/s_ry.npy' % CURR_DIR)

# u_arr = np.linspace(20, 25, 100)
# rho_arr = np.linspace(-4.5, -0.7, 50)
# y_arr4 = np.arange(0.20, 0.42, 0.02)

# get_s_ur = RGI((y_arr4, rho_arr, u_arr), s_res, \
#                 method='linear', bounds_error=False, fill_value=None)

# def get_s_u_r(u, r, y):
#     return get_s_ur(np.array([y, r, u]).T)

# dsdy_ur = np.load('%s/cms/dsdy_ur.npy' % CURR_DIR)

# yvals = np.linspace(0.22, 0.43, 100)

# get_dsdy_ur = RGI((yvals, rho_arr, u_arr), dsdy_ur, \
#                 method='linear', bounds_error=False, fill_value=None)

# def get_dsdy_u_r(u, r, y):
#     return get_dsdy_ur(np.array([y, r, u]).T)


u_res = np.load('%s/cms/u_sry.npy' % CURR_DIR)

s_arr3 = np.linspace(5.75, 9, 30)
rho_arr = np.linspace(-4, 1, 50)
y_arr4 = np.linspace(0.01, 1, 100)

get_u_sr_rgi = RGI((s_arr3, rho_arr, y_arr4), u_res, \
                method='linear', bounds_error=False, fill_value=None)

def get_u_s(s, r, y):
    return get_u_sr_rgi(np.array([s, r, y]).T)

def get_u_t(p, t, y, corr):
    return 10**cms.get_logu_mix(p, t, y, corr)

# dudy_sr = np.load('%s/cms/dudy_sry.npy' % CURR_DIR)

# yvals = np.linspace(0.22, 0.43, 100)

# get_dsdy_ur = RGI((s_arr3, rho_arr, yvals), dudy_sr, \
#                 method='linear', bounds_error=False, fill_value=None)

# def get_dudy_s_r(s, r, y):
#     return get_dsdy_ur(np.array([s, r, y]).T)

# rho_arr = np.linspace(-4, 1, 50)
# y_arr4 = np.arange(0, 1, 0.02)
# t_arr2 = np.linspace(2.1, 5, 100)

# dudy_rt = np.load('eos/cms/dudy_rty.npy')

# get_dudy_rt = RGI((t_arr2, rho_arr, y_arr4), dudy_rt, \
#                 method='linear', bounds_error=False, fill_value=None)

# def get_dudy_r_t(r, t, y):
#     return get_dudy_rt(np.array([t, r, y]).T)

###### inversions ######


### error functions ###
def err_energy_2D(pt_pair, sval, uval, y):
    lgp, lgt = pt_pair
    s, logu = cms.get_s_mix(lgp, lgt, y, hc_corr=True), cms.get_logu_mix(lgp, lgt, y)
    s *= erg_to_kbbar
    return  s/sval - 1, logu/uval -1

def err_energy_2D_rhot(pt_pair, sval, rval, y):
    lgp, lgt = pt_pair
    s, rho = cms.get_s_mix(lgp, lgt, y, hc_corr=True), cms.get_rho_mix(lgp, lgt, y, hc_corr=True)
    sval /= erg_to_kbbar
    logrho = np.log10(rho)
    return  s/sval - 1, logrho/rval -1

#def err_rhos_1D(logp, s)

def err_rhot_1D(lgt, lgp, rhoval, y):
    #lgp, lgt = pt_pair
    logrho = np.log10(cms.get_rho_mix(lgp, lgt, y, hc_corr=True))
    #s *= erg_to_kbbar
    return  logrho/rhoval - 1

def err_rhop_1D(lgp, lgt, rhoval, y):
    #lgp, lgt = pt_pair
    logrho = np.log10(cms.get_rho_mix(lgp, lgt, y, hc_corr=True))
    #s *= erg_to_kbbar
    return  logrho/rhoval - 1


def err_energy_1D(lgp, s, uval, y):
    logt = get_rho_t(s, lgp, y)[-1]
    logu = cms.get_logu_mix(lgp, logt, y, corr=False)
    return logu/uval - 1

def err_rhous_1D(lgr, s, uval, y): #uval in log10
    #logt = get_rho_t(s, lgp, y)[-1]
    logu = np.log10(get_u_sr(s, lgr, y))
    return logu/uval - 1

# def err_rhoup_1D(lgr, uval, pval, y): #uval in log10
#     #logt = get_rho_t(s, lgp, y)[-1]
#     logu = np.log10(get_u_r(s, lgr, y))
#     return logu/uval - 1

def err_ur_1D(lgt, lgr, uval, y):
    logu = float(get_logu_r(lgr, lgt, y))
    return logu/uval - 1

def err_sr_1D(lgt, lgr, sval, y):
    s = get_s_r(lgr, lgt, y)*erg_to_kbbar
    return s/sval - 1

def err_ps_1D(lgp, lgr, sval, y):
    t = get_t_sr(sval, lgr, y)
    sres = cms.get_s_mix(lgp, t, y)*erg_to_kbbar
    return sres/sval - 1

def get_pt_su(s, u, y):
    if u > 12:
        guess = [10, 4]
    else:
        guess = [7, 2.5]
    sol = root(err_energy_2D, guess, args=(s, u, y))
    return sol.x

def get_pt_sr(s, r, y, guess=[7, 2.7], alg='hybr'):
    # if r > 0:
    #     guess = [10, 3]
    # else:
    #guess = [7, 2.7]
    sol = root(err_energy_2D_rhot, guess, tol=1e-8, method=alg, args=(s, r, y))
    return sol.x

# def get_p_sr(s, r, y):
#     sol = root_scalar(err_ps_1D, bracket=[5, 14], method='brenth', args=(r, s, y))
#     return sol.root

### inversion functions ###

def get_p_us(s, u, y):
    sol = root_scalar(err_energy_1D, bracket=[5, 14], method='brenth', args=(s, u, y))
    return sol.root
 
def get_rho_us(s, u, y):
    sol = root_scalar(err_rhous_1D, bracket=[-4, 1], method='brenth', args=(s, u, y))
    return sol.root

def get_t_r(p, r, y):
    if np.isscalar(r):
        sol = root_scalar(err_rhot_1D, bracket=[0, 7], method='brenth', args=(p, r, y))
        return sol.root
    else:
        #guess = np.zeros(len(r))+2.5
        res = []
        for p_, r_, y_ in zip(p, r, y):
            try:
                sol = root_scalar(err_rhot_1D, bracket=[0, 7], method='brenth',args=(p_, r_, y_))
                res.append(sol.root)
            except:
                print('failed at:', p_, r_, y_)
                raise
        return np.array(res)

def get_t_rp(p, r, y):
    sol = root(err_rhot_1D, [2.5], tol=1e-8, method='hybr', args=(p, r, y))
    return sol.x

# def get_t_sr(s, r, y):
#     #if np.isscalar(r):
#         #guess = 2.5
#     sol = root_scalar(err_sr_1D, bracket=[0, 5], method='brenth', args=(r, s, y))
#     return sol.root
    # else:
    #     res = []
    #     for s_, r_, y_ in zip(s, r, y):
    #         sol = root_scalar(err_sr_1D, bracket=[0, 5], method='brenth',args=(r_, s_, y_))
    #         res.append(sol.root)
    #     return np.array(res)

def get_t_ur(u, r, y):
    #y = cms.n_to_Y(x)
    if np.isscalar(r):
        #guess = 2.5
        sol = root_scalar(err_ur_1D, bracket=[0, 5], method='brenth', args=(r, u, y))
        return sol.root
    else:
        res = []
        for u_, r_, y_ in zip(u, r, y):
            
            sol = root_scalar(err_ur_1D, bracket=[0, 5], method='brenth',args=(r_, u_, y_))
            res.append(sol.root)
        return np.array(res)

def get_p_r(r, t, y):
    #y = cms.n_to_Y(x)
    if np.isscalar(r):
        #guess = 7
        sol = root_scalar(err_rhop_1D, bracket=[4, 17], method='brenth', args=(t, r, y))
        return sol.root
    else:
        res = []
        for r_, t_, y_ in zip(r, t, y):
            sol = root_scalar(err_rhop_1D, bracket=[4, 17], method='brenth',args=(t_, r_, y_))
            res.append(sol.root)
        return np.array(res)

#def get_rho_us(u, s, y):


def get_s_r(r, t, y):
    #y = cms.n_to_Y(x)
    p = get_p_r(r, t, y)
    s = cms.get_s_mix(p, t, y, hc_corr=True)
    return s # in cgs

def get_logu_r(r, t, y):
    #y = cms.n_to_Y(x)
    p = get_p_r(r, t, y) 
    logu = cms.get_logu_mix(p, t, y, corr=False)
    return logu

def get_u_sr(s, r, y):
    #y = cms.n_to_Y(x)
    #t = get_t_sr(s, r, y)
    p, t = get_p_sr(s, r, y), get_t_sr(s, r, y)
    #return 10**get_logu_r(r, t, y)
    return 10**cms.get_logu_mix(p, t, y, corr=False)

def get_s_u(u, r, y):
    t = get_t_ur(u, r, y)
    return get_s_r(r, t, y) # in cgs

def get_s_rp(r, p, y):
    t = get_t_r(p, r, y)
    #y = cms.n_to_Y(y)
    s = cms.get_s_mix(p, t, y, hc_corr=True)
    return s # in cgs

############## derivatives ##############


### composition gradients ###

def get_dsdy_rp_new(s, r, y, dy=0.001, ds=0.001):
    P0 = 10**get_p_sr(s, r, y)
    P1 = 10**get_p_sr(s, r, y*(1+dy))
    P2 = 10**get_p_sr(s*(1+ds), r, y)

    S1 = s/erg_to_kbbar
    s2 = s*(1+ds)
    S2 = s2/erg_to_kbbar
    
    dpdy_sr = (P1 - P0)/(y*dy)
    dpds_ry = (P2 - P0)/(S2 - S1)
    return -dpdy_sr/dpds_ry

def get_dsdy_rp(r, p, y, dy=0.01):
    s0 = get_s_rp(r, p, y)
    s1 = get_s_rp(r, p, y*(1+dy))

    dsdy = (s1 - s0)/(y*dy)
    return dsdy

def get_dsdy_rt(r, t, y, dy=0.01):
    s0 = get_s_r(r, t, y)
    s1 = get_s_r(r, t, y*(1+dy))

    dsdy = (s1 - s0)/(y*dy)
    return dsdy

def get_dudy_sr(s, r, y, dy=0.01):
    # u0 = get_u_s(s, r, y)
    # u1 = get_u_s(s, r, y*(1+dy))
    u0 = get_u_sr(s, r, y)
    u1 = get_u_sr(s, r, y*(1+dy))

    return (u1 - u0)/(y*dy)

def get_dtdy_rp(r, p, y, dy=0.01):
    t0 = 10**get_t_r(p, r, y)
    t1 = 10**get_t_r(p, r, y*(1+dy))

    dtdy = (t1 - t0)/(y*dy)
    return dtdy

def get_dlogt_dy_rp(r, p, y, dy=0.01):
    t0 = get_t_r(p, r, y)
    t1 = get_t_r(p, r, y*(1+dy))

    dtdy = (t1 - t0)/(y*dy)
    return dtdy

### entropy gradients ###

def get_dsdu_ry(u, r, y, du = 0.01):
    u1 = 10**u # cgs
    u2 = np.log10(u1*(1+du))
    s0 = get_s_u(u, r, y) # need to be logs
    s1 = get_s_u(u2, r, y)

    return (s1 - s0)/(u1*du)

def get_dlogs_dlogt_rt(r, t, y, dt=0.1):
    s0 = np.log10(get_s_r(r, t, y))
    s1 = np.log10(get_s_r(r, t*(1+dt), y))

    dsdt = (s1 - s0)/(t*dt) # dlogs/dlogt |_rho, x
    return dsdt

### density gradients ###

def get_drhods_py(s, p, y, ds=0.01):
    
    s1_cgs = s/erg_to_kbbar
    s2 = s*(1+ds)
    s2_cgs = s2/erg_to_kbbar 

    #s2_cgs = s1_cgs*(1+ds_cgs)
    #y = cms.n_to_Y(x)
    rho0 = 10**get_rho_t(s, p, y)[0]
    rho1 = 10**get_rho_t(s2, p, y)[0]

    drhods = (rho1 - rho0)/(s2_cgs - s1_cgs)

    return drhods

def get_dlogrho_dlogt_py(p, t, y, dt=0.01):
    #y = cms.n_to_Y(x)
    rho0 = np.log10(cms.get_rho_mix(p, t, y))
    rho1 = np.log10(cms.get_rho_mix(p, t*(1+dt), y))

    drhodt = (rho1 - rho0)/(t*dt)

    return drhodt

def get_dtdy_sp(s, p, y, dy=0.01):
    t0 = 10**get_rho_t(s, p, y)[-1]
    t1 = 10**get_rho_t(s, p, y*(1+dy))[-1]

    dtdy = (t1 - t0)/(y*dy)
    return dtdy



