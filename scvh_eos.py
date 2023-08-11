import numpy as np
#import scvh_nr
from scipy.optimize import root, root_scalar
from scipy.interpolate import RegularGridInterpolator as RGI
from eos import aneos, scvh_man
import os
erg_to_kbbar = 1.202723550011625e-08
mh = 1 
mhe = 4.0026

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

eos_aneos = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='serpentine')
eos_scvh = scvh_man.eos(path_to_data='%s/scvh_mesa' % CURR_DIR)

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

def get_s(r, t, yhe):

    if not hasattr(r, '__len__'):
        return float(interp_s((yhe, x(r), y(r, t))))
    else:
        return interp_s(np.array([yhe, x(r), y(r, t)]).T)

def get_p(r, t, yhe):
    if not hasattr(r, '__len__'):
        return float(interp_p((yhe, x(r), y(r, t))))
    else:
        return interp_p(np.array([yhe, x(r), y(r, t)]).T)

def get_sp(r, t, yhe):
    if not hasattr(r, '__len__'):
        return float(interp_s((yhe, x(r), y(r, t)))), float(interp_p((yhe, x(r), y(r, t))))
    else:
        return interp_s(np.array([yhe, x(r), y(r, t)]).T), interp_p(np.array([yhe, x(r), y(r, t)]).T)

def err_scvh(rt_pair, sval, pval, y, z, z_eos):
    rho, temp = rt_pair
    s, p = get_sp(rho, temp, y)
    if z > 0:
        #sz_ideal = sackur_tetrode(p, temp, mz=15.5)
        stot = float(np.log10(get_smix_z(y, z, p, temp, z_eos)*erg_to_kbbar))
        return stot/sval - 1, p/pval -1
    else:
        return  s/sval - 1, p/pval -1

def get_rho_p_ideal(s, logp, m=15.5):
    # done from ideal gas
    # note: 15.5 is average molecular weight for solar comp
    # for Y = 0.25, m = 3 * (1 * 1 + 1 * 0) + (1 * 4) / 7 = 1
    p = 10**logp
    return np.log10(np.maximum(np.exp((2/5) * (5.096 - s)) * (np.maximum(p, 0) / 1e11)**(3/5) * m**(8/5), np.full_like(p, 1e-10)))

def get_rhot(s, p, y, z=0, z_eos='ideal'): # in-situ inversion
    s = np.log10(s)
    sol = root(err_scvh, [-2, 2.1], args=(s, p, y, z, z_eos))

    if z > 0:
        rho_hhe = 10**sol.x[0]
        rho_z = 10**get_rho_p_ideal(s, p)
        return np.log10(1/((1 - z)/rho_hhe + z/rho_z)), sol.x[1]
    else:
        return sol.x

def x_i(Y):
    return ((Y/mhe)/(((1 - Y)/mh) + (Y/mhe)))

##### pressure-temperature #####

#np.load('%s/cms/cms_hg_thermo.npy' % CURR_DIR)

logp_res, logt_res, logrho_res, s_res = np.load('%s/scvh/scvh_pt.npy' % CURR_DIR)
yvals = np.array([0.22, 0.25, 0.28, 0.30])

get_rho_pt = RGI((yvals, logt_res[0][:,0], logp_res[0][0]), logrho_res, method='linear', bounds_error=False, fill_value=None)
get_s_pt = RGI((yvals, logt_res[0][:,0], logp_res[0][0]), s_res, method='linear', bounds_error=False, fill_value=None)

def get_rho_p_t(p, t, y):
    return get_rho_pt(np.array([y, t, p]).T)

def get_s_p_t(p, t, y):
    return get_s_pt(np.array([y, t, p]).T)

def x_i(Y):
    return ((Y/mhe)/(((1 - Y)/mh) + (Y/mhe)))

def x_H(Y, Z, mz):
    Ntot = (1-Y)*(1-Z)/mh + (Y*(1-Z)/mhe) + Z/mz
    return (1-Y)*(1-Z)/mh/Ntot

def x_Z(Y, Z, mz):
    Ntot = (1-Y)*(1-Z)/mh + (Y*(1-Z)/mhe) + Z/mz
    return (Z/mz)/Ntot

def sackur_tetrode(lgp, lgt, mz):
    # lgp must be in log cgs
    # use the P, T basis of SCvH for this?
    return 4.61664 + np.log((10**lgt / 1e3)**(5/2) / (10**lgp / 1e11) * mz**(3/2))

def guarded_log(x):
    if x == 0:
        return 0
    elif x < 0:
        raise ValueError('a')
    return x * np.log(x)

def get_smix_id_y(Y):
    #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
    xhe = x_i(Y)
    xh = 1 - xhe
    return -1*(guarded_log(xh) + guarded_log(xhe))

def get_smix_z(Y, Z, lgp, lgt, z_eos):
    s_xy = get_s_p_t(lgp, lgt, Y) / erg_to_kbbar

    if z_eos == 'ideal':
        mz=15.5
        s_z = sackur_tetrode(lgp, lgt, mz) / erg_to_kbbar

    elif z_eos == 'aneos':
        mz = 18.0 # mean molecular weight of water
        s_z = 10**eos_aneos.get_logs(lgp, lgt) # way too high, dominates over xy mixture

    xhe = x_i(Y)
    xz = x_Z(Y, Z, mz)
    xh = 1 - xhe - xz

    # returning in kb/baryon
    if Z > 0:
        return (s_xy*(1-Z) + s_z*Z) - ((guarded_log(xh) + guarded_log(xhe) + guarded_log(xz)) / erg_to_kbbar)
    elif Z == 0:
        return s_xy 
# logp_res, logt_res, logrho_res, s_res = np.load('%s/scvh/scvh_prho.npy' % CURR_DIR)
# yvals = np.array([0.22, 0.25, 0.28, 0.292])

# get_t_pr = RGI((yvals, logrho_res[0][:,0], logp_res[0][0]), logt_res, method='linear', bounds_error=False, fill_value=None)
# get_s_pr = RGI((yvals, logrho_res[0][:,0], logp_res[0][0]), s_res, method='linear', bounds_error=False, fill_value=None)

# def get_t_rhop(r, p, y):
#     return get_t_pr(np.array([y, r, p]).T)

# def get_s_rhop(r, p, y):
#     return get_s_pr(np.array([y, r, p]).T)


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

def get_rho_z(s, p, y, z, z_eos='ideal'): # y should be scaled outside
    if z_eos == 'ideal':
        rho_z = 10**get_rho_p_ideal(s, p)
    elif z_eos == 'aneos':
        rho_z = 10**eos_aneos.get_logrho(p, get_t(np.array([y/(1-z), s, p]).T))
    rho = 10**get_rho(np.array([y/(1-z), s, p]).T)
    rho_mix = np.log10(1/((1-z)/rho + z/rho_z))
    return rho_mix

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

# def get_grad_a(s, p, y, dp=0.001):
#     delta_logt = -get_t(np.array([y, s, p]).T) + get_t(np.array([y, s, p*(1+dp)]).T)
#     dlogt_dlogP = delta_logt/(p*dp)
#     return dlogt_dlogP

# def get_grad_a(s, p, y, dp=0.001):
#     delta_logt = -get_rhot(s, p, y)[-1] + get_rhot(s, p*(1+dp), y)[-1]
#     dlogt_dlogP = delta_logt/(p*dp)
#     return dlogt_dlogP

def get_gamma1_(s, p, y, z, dp=0.001):
    delta_logrho = -get_rho_z(s, p, y, z) + get_rho_z(s, p*(1+dp), y, z)
    dlogrho_dlogP = delta_logrho/(p*dp)
    return 1/dlogrho_dlogP

def get_c_p_(s, p, y, ds=1e6):
    
    s /= erg_to_kbbar
    s_prime = (s*(1+ds))
    delta_logt = -get_rhot(s*erg_to_kbbar, p, y)[-1] + get_rhot(s_prime*erg_to_kbbar, p, y)[-1]
    dlogs = -np.log10(s) + np.log10(s_prime)
    #ds /= erg_to_kbbar
    dlogs_dlogt = dlogs/delta_logt

    return  s * dlogs_dlogt

def get_c_v_(r, t, y, dt=0.001):
    #s = (10**get_s(r, t, y))*erg_to_kbbar

    s = (10**get_s(r, t, y))/erg_to_kbbar
    dlogs = -np.log10(s) + np.log10((10**get_s(r, t*(1+dt), y))/erg_to_kbbar)

    
    dlogs_dlogt = dlogs/(t*dt)
    return s*dlogs_dlogt

####### composition derivatives #######

y_arr2 = np.arange(0.22, 1.0, 0.01)

dlogrho_dy_pt, dlogs_dy_pt = np.load('%s/scvh/rho_s_der_pt.npy' % CURR_DIR)
dlogrho_dy_sp, dlogt_dy_sp = np.load('%s/scvh/rho_t_der_sp.npy' % CURR_DIR)

get_dlogs_dy_pt = RGI((logt_res[0][:,0], logp_res[0][0], y_arr2), dlogs_dy_pt, method='linear', bounds_error=False, fill_value=None)
get_dlogrho_dy_pt = RGI((logt_res[0][:,0], logp_res[0][0], y_arr2), dlogrho_dy_pt, method='linear', bounds_error=False, fill_value=None)

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

logp_res_rhos, logrho_res_rhos, s_res_rhos = np.load('%s/scvh/scvh_rho_s.npy' % CURR_DIR)

yvals = np.arange(0.22, 0.305, 0.005)
get_p_s_r = RGI((yvals, s_res_rhos[0][:,0], logrho_res_rhos[0][0]), logp_res_rhos, method='linear', bounds_error=False, fill_value=None)

def get_p_sr(r, s, y):
    return get_p_s_r(np.array([y, s, r]).T)

y_arr3 = np.arange(0.22, 1.0, 0.01)

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
    
s_arr2 = np.arange(5.5, 10.01, 0.01)

dlogp_dlogs = []
for i, y_ in enumerate(yvals):
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
get_dlogp_dlogs = RGI((yvals, np.array(logrho_res_rhos)[0][0], s_arr2), dlogp_dlogs, method='linear', bounds_error=False, fill_value=None)

def get_dpdy(r, s, y):
    return get_dlogp_dy(np.array([s, r, y]).T)*(10**get_p_sr(r, s, y))

def get_dpds(r, s, y):
    return get_dlogp_dlogs(np.array([y, r, s]).T)*(10**get_p_sr(r, s, y))/(s/erg_to_kbbar)



# dlogt_dy, dlogs_dy = np.load('%s/scvh/t_s_der_prho.npy' % CURR_DIR)

# # logtvals = np.linspace(2.1, 5, 100)
# # logpvals = np.linspace(5, 14, 300)

# ygrid = np.arange(0.22, 1.0, 0.01)

# get_dlogs_dy = RGI((logrho_res[0][:,0], logp_res[0][0], ygrid), dlogs_dy, method='linear', bounds_error=False, fill_value=None)

# get_dlogt_dy = RGI((logrho_res[0][:,0], logp_res[0][0], ygrid), dlogt_dy, method='linear', bounds_error=False, fill_value=None)


# def get_dlogsdy(r, p , y):
#     dsdy = get_dlogs_dy(np.array([r, p, y]).T)
#     return dsdy

# def get_dlogtdy(r, p , y):
#     dtdy = get_dlogt_dy(np.array([r, p, y]).T)
#     return dtdy
def err_ps_1D(lgp, lgr, sval, y):
    t = get_t_sry(sval, lgr, y)
    sres = get_s_p_t(lgp, t, y)
    return sres/sval - 1

def err_rhot_1D(lgt, lgr, pval, y):
    #lgp, lgt = pt_pair
    logp = get_p(lgr, lgt, y)
    #s *= erg_to_kbbar
    return  logp/pval - 1

def err_energy_2D_rhot(pt_pair, sval, rval, y):
    lgp, lgt = pt_pair
    s, logrho = float(get_s_p_t(lgp, lgt, y)), float(get_rho_p_t(lgp, lgt, y))
    return  s/sval - 1, logrho/rval -1

def err_sr_1D(lgt, lgr, sval, y):
    s = 10**get_s(lgr, lgt, y)
    return s/sval - 1

def get_t_r(p, r, y):
    if np.isscalar(r):
        sol = root_scalar(err_rhot_1D, bracket=[0, 5], method='brenth', args=(r, p, y))
        return sol.root
    else:
        #guess = np.zeros(len(r))+2.5
        res = []
        for p_, r_, y_ in zip(p, r, y):
            try:
                sol = root_scalar(err_rhot_1D, bracket=[0, 5], method='brenth',args=(r_, p_, y_))
                res.append(sol.root)
            except:
                print('failed at:', p_, r_, y_)
                raise
        return np.array(res)

def get_s_rp(r, p, y):
    t = get_t_r(p, r, y)
    #y = cms.n_to_Y(y)
    s = (10**get_s(r, t, y))/erg_to_kbbar
    return s # in cgs

def get_dsdy_rp(r, p, y, dy=0.01):
    s0 = get_s_rp(r, p, y)
    s1 = get_s_rp(r, p, y*(1+dy))

    dsdy = (s1 - s0)/(y*dy)
    return dsdy

def get_pt_sr(s, r, y, guess=[7, 2.7], alg='hybr'):

    sol = root(err_energy_2D_rhot, guess, tol=1e-8, method=alg, args=(s, r, y))
    return sol.x

def get_t_sry(s, r, y):
    if np.isscalar(r):
        #guess = 2.5
        sol = root_scalar(err_sr_1D, bracket=[0, 5], method='brenth', args=(r, s, y))
        return sol.root
    else:
        res = []
        for s_, r_, y_ in zip(s, r, y):
            sol = root_scalar(err_sr_1D, bracket=[0, 5], method='brenth',args=(r_, s_, y_))
            res.append(sol.root)
        return np.array(res)

def get_p_sry(s, r, y):
    # sol = root_scalar(err_ps_1D, bracket=[5, 14], method='brenth', args=(r, s, y))
    # return sol.root

    if np.isscalar(r):
            #guess = 2.5
            sol = root_scalar(err_ps_1D, bracket=[5, 14], method='brenth', args=(r, s, y))
            return sol.root
    else:
        res = []
        for s_, r_, y_ in zip(s, r, y):
            sol = root_scalar(err_ps_1D, bracket=[5, 14], method='brenth',args=(r_, s_, y_))
            res.append(sol.root)
        return np.array(res)

def get_u_sry(s, r, y):
    p, t = get_pt_sr(s, r, y)
    logu = eos_scvh.get_logu(p, t, y)
    return 10**logu