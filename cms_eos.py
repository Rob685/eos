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


####### composition derivatives #######

y_arr2 = np.arange(0.22, 1.0, 0.01)

dlogrho_dy_pt, dlogs_dy_pt = np.load('%s/cms/rho_s_der_pt.npy' % CURR_DIR)
dlogrho_dy_sp, dlogt_dy_sp = np.load('%s/cms/rho_t_der_sp.npy' % CURR_DIR)

logtvals = np.linspace(2.1, 5, 100)
logpvals = np.linspace(6, 14, 300)

get_dlogs_dy_pt = RGI((logtvals, logpvals, y_arr2), dlogs_dy_pt, method='linear', bounds_error=False, fill_value=None)
get_dlogrho_dy_pt = RGI((logtvals, logpvals, y_arr2), dlogrho_dy_pt, method='linear', bounds_error=False, fill_value=None)

get_dlogt_dy_sp = RGI((s_arr[0][:,0], p_arr[0][0], y_arr2), dlogt_dy_sp, method='linear', bounds_error=False, fill_value=None)
get_dlogrho_dy_sp = RGI((s_arr[0][:,0], p_arr[0][0], y_arr2), dlogrho_dy_sp, method='linear', bounds_error=False, fill_value=None)


def get_dlogsdy_pt(p, t, y):
    return get_dlogs_dy_pt(np.array([t, p, y]).T)

def get_dlogrhody_pt(p, t, y):
    return get_dlogrho_dy_pt(np.array([t, p, y]).T)

def get_dlogtdy_sp(s, p, y):
    return get_dlogt_dy_sp(np.array([s, p, y]).T)

def get_dlogrhody_sp(s, p, y):
    return get_dlogrho_dy_sp(np.array([s, p, y]).T)


####### P(rho, s, y) #######

logp_res_rhos, logrho_res_rhos, s_res_rhos = np.load('%s/cms/cms_rho_s.npy' % CURR_DIR)
yarr = np.arange(0.22, 0.46, 0.02)

get_p_s_r = RGI((yarr, s_res_rhos[0][:,0], logrho_res_rhos[0][0]), logp_res_rhos, method='linear', bounds_error=False, fill_value=None)

def get_p_sr(r, s, y):
    return get_p_s_r(np.array([y, s, r]).T)

y_arr3 = np.arange(0.22, 0.75, 0.01)

#dlogrho_dY = []
dlogp_dY = []
for i, s in enumerate(s_res_rhos[0][:,0]):
    #drho_dY = []
    dp_dY = []
    for j, r in enumerate(logrho_res_rhos[0][0]):
        logp = get_p_sr(np.full_like(y_arr3, r), np.full_like(y_arr3, s), y_arr3) # at constant rho, s
        #s = get_s_p_t(np.full_like(y_arr, p), np.full_like(y_arr, t), y_arr)
        dlogpdy = np.gradient(logp)/np.gradient(y_arr3)
        #dlogsdY = np.gradient(np.log10(s/erg_to_kbbar))/np.gradient(y_arr)
        #drho_dY.append(dlogrhodY)
        dp_dY.append(dlogpdy)
        
    #dlogrho_dY.append(drho_dY)
    dlogp_dY.append(dp_dY)
    
s_arr2 = np.arange(5.6, 10.1, 0.1)

dlogp_dlogs = []
for i, y_ in enumerate(yarr):
    #drho_dY = []
    dp_ds = []
    for j, r in enumerate(logrho_res_rhos[0][0]):
        logp = get_p_sr(np.full_like(s_arr2, r), s_arr2, np.full_like(s_arr2, y_)) # at constant rho, Y
        #s = get_s_p_t(np.full_like(y_arr, p), np.full_like(y_arr, t), y_arr)
        dlogs = np.gradient(np.log10(s_arr2/erg_to_kbbar))
        dlogpdlogs = np.gradient(logp)/dlogs
        #dlogsdY = np.gradient(np.log10(s/erg_to_kbbar))/np.gradient(y_arr)
        #drho_dY.append(dlogrhodY)
        dp_ds.append(dlogpdlogs)
        
    #dlogrho_dY.append(drho_dY)
    dlogp_dlogs.append(dp_ds)

get_dlogp_dy = RGI((np.array(s_res_rhos)[0][:,0], np.array(logrho_res_rhos)[0][0], y_arr3), dlogp_dY, method='linear', bounds_error=False, fill_value=None)
get_dlogp_dlogs = RGI((yarr, np.array(logrho_res_rhos)[0][0], s_arr2), dlogp_dlogs, method='linear', bounds_error=False, fill_value=None)

def get_dpdy(r, s, y):
    return get_dlogp_dy(np.array([s, r, y]).T)*(10**get_p_sr(r, s, y))

def get_dpds(r, s, y):
    return get_dlogp_dlogs(np.array([y, r, s]).T)*(10**get_p_sr(r, s, y))/(s/erg_to_kbbar)

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


# u_res = np.load('%s/cms/u_sry.npy' % CURR_DIR)

# s_arr3 = np.linspace(5.5, 10, 40)
# rho_arr = np.linspace(-3.4, -0.2, 50)
# y_arr4 = np.arange(0.20, 0.42, 0.02)

# get_u_sr = RGI((y_arr4, rho_arr, s_arr3), u_res, \
#                 method='linear', bounds_error=False, fill_value=None)

# def get_u_s_r(s, r, y):
#     return get_u_sr(np.array([y, r, s]).T)

# dudy_sr = np.load('%s/cms/dudy_sry.npy' % CURR_DIR)

# yvals = np.linspace(0.22, 0.43, 100)

# get_dsdy_ur = RGI((s_arr3, rho_arr, yvals), dudy_sr, \
#                 method='linear', bounds_error=False, fill_value=None)

# def get_dudy_s_r(s, r, y):
#     return get_dsdy_ur(np.array([s, r, y]).T)

rho_arr = np.linspace(-4, 1, 50)
y_arr4 = np.arange(0, 1, 0.02)
t_arr2 = np.linspace(2.1, 5, 100)

dudy_rt = np.load('eos/cms/dudy_rty.npy')

get_dudy_rt = RGI((t_arr2, rho_arr, y_arr4), dudy_rt, \
                method='linear', bounds_error=False, fill_value=None)

def get_dudy_r_t(r, t, y):
    return get_dudy_rt(np.array([t, r, y]).T)

###### inversions ######

def err_energy_2D(pt_pair, sval, uval, y):
    lgp, lgt = pt_pair
    s, logu = cms.get_s_mix(lgp, lgt, y, hc_corr=True), cms.get_logu_mix(lgp, lgt, y)
    s *= erg_to_kbbar
    return  s/sval - 1, logu/uval -1

def err_energy_2D_rhot(pt_pair, sval, rval, y):
    lgp, lgt = pt_pair
    s, rho = cms.get_s_mix(lgp, lgt, y, hc_corr=True), cms.get_rho_mix(lgp, lgt, y, hc_corr=True)
    s *= erg_to_kbbar
    logrho = np.log10(rho)
    return  s/sval - 1, logrho/rval -1

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

def err_ur_1D(lgt, lgr, uval, y):
    logu = float(get_logu_r(lgr, lgt, y))
    return logu/uval - 1

def err_sr_1D(lgt, lgr, sval, y):
    s = get_s_r(lgr, lgt, y)*erg_to_kbbar
    return s/sval - 1

def get_pt_su(s, u, y):
    sol = root(err_energy_2D, [7, 2.5], args=(s, u, y))
    return sol.x

def get_pt_sr(s, r, y):
    sol = root(err_energy_2D_rhot, [7, 2.5], args=(s, r, y))
    return sol.x

def get_t_r(p, r, x): # doesn't work very well...
    y = cms.n_to_Y(x)
    sol = root_scalar(err_rhot_1D, bracket=[0, 7], method='brenth', args=(p, r, y))
    return sol.root

def get_t_sr(s, r, x):
    y = cms.n_to_Y(x)
    sol = root_scalar(err_sr_1D, bracket=[0, 7], method='brenth', args=(r, s, y))
    return sol.root

def get_t_ur(u, r, x):
    y = cms.n_to_Y(x)
    sol = root_scalar(err_ur_1D, bracket=[0, 7], method='brenth', args=(r, u, y))
    return sol.root

def get_s_r(r, t, x):
    y = cms.n_to_Y(x)
    p = get_p_r(r, t, y)
    s = cms.get_s_mix(p, t, y, hc_corr=True)
    return s # in cgs

def get_p_r(r, t, x):
    y = cms.n_to_Y(x)
    sol = root_scalar(err_rhop_1D, bracket=[5, 15], method='brenth', args=(t, r, y))
    return sol.root

def get_logu_r(r, t, x):
    y = cms.n_to_Y(x)
    p = get_p_r(r, t, x) 
    logu = float(cms.get_logu_mix(p, t, y)) 
    return logu

def get_u_s(s, r, x):
    #y = cms.n_to_Y(x)
    t = get_t_sr(s, r, x)
    return 10**get_logu_r(r, t, x)

def get_s_u(u, r, x):
    t = get_t_ur(u, r, x)
    return get_s_r(r, t, x) # in cgs

def get_s_rp(r, p, x):
    t = get_t_r(p, r, x)
    y = cms.n_to_Y(x)
    s = cms.get_s_mix(p, t, y, hc_corr=True)
    return s # in cgs

def get_dsdx_rp(r, p, x, dx=0.001):
    s0 = get_s_rp(r, p, x)
    s1 = get_s_rp(r, p, x*(1+dx))

    dsdx = (s1 - s0)/(x*dx)
    return dsdx

def get_dsdx_rt(r, t, x, dx=0.001):
    s0 = get_s_r(r, t, x)
    s1 = get_s_r(r, t, x*(1+dx))

    dsdx = (s1 - s0)/(x*dx)
    return dsdx

def get_dsdu_rx(u, r, x, du = 0.01):
    u1 = 10**u # cgs
    u2 = np.log10(u1*(1+du))
    s0 = get_s_u(u, r, x) # need to be logs
    s1 = get_s_u(u2, r, x)

    return (s1 - s0)/(u1*du)


def get_dudx_sr(s, r, x, dx=0.001):
    u0 = get_u_s(s, r, x)
    u1 = get_u_s(s, r, x*(1+dx))

    return (u1 - u0)/(x*dx)

def get_dlogs_dlogt_rt(r, t, x, dt=0.1):
    s0 = np.log10(get_s_r(r, t, x))
    s1 = np.log10(get_s_r(r, t*(1+dt), x))

    dsdt = (s1 - s0)/(t*dt) # dlogs/dlogt |_rho, x
    return dsdt

def get_nabla_ad(s, p, x, dp=0.1):
    y = cms.n_to_Y(x)
    t1 = get_rho_t(s, p, y)[-1]
    t2 = get_rho_t(s, p*(1+dp), y)[-1]

    return (t2 - t1)/(p*dp)

# def get_dsdt_rx(r, t, x, dt=0.1):
#     s0 = get_s_r(r, t, x)
#     s1 = get_s_r(r, t*(1+dt), x)

#     dsdt = (s1 - s0)/(t*dt)
#     return dsdt

def get_dtdx_rp(r, p, x, dx=0.001):
    t0 = 10**get_t_r(p, r, x)
    t1 = 10**get_t_r(p, r, x*(1+dx))

    dtdx= (t1 - t0)/(x*dx)
    return dtdx

def get_dlogt_dx_rp(r, p, x, dx=0.001):
    t0 = get_t_r(p, r, x)
    t1 = get_t_r(p, r, x*(1+dx))

    dtdx= (t1 - t0)/(x*dx)
    return dtdx

def get_drhods_px(s, p, x, ds=0.01):
    
    s1_cgs = s/erg_to_kbbar
    s2 = s*(1+ds)
    s2_cgs = s2/erg_to_kbbar 

    #s2_cgs = s1_cgs*(1+ds_cgs)
    y = cms.n_to_Y(x)
    rho0 = 10**get_rho_t(s, p, y)[0]
    rho1 = 10**get_rho_t(s2, p, y)[0]

    drhods = (rho1 - rho0)/(s2_cgs - s1_cgs)

    return drhods

def get_dlogrho_dlogt_px(p, t, x, dt=0.01):
    y = cms.n_to_Y(x)
    rho0 = np.log10(cms.get_rho_mix(p, t, y))
    rho1 = np.log10(cms.get_rho_mix(p, t*(1+dt), y))

    drhodt = (rho1 - rho0)/(t*dt)

    return drhodt

# def get_B_grad(r, p, x):
#     dlogT_dlogX_rp = get_dlogt_dx_rp(r, p, x)



