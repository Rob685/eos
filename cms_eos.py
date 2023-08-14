import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import RectBivariateSpline as RBS
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
pd.options.mode.chained_assignment = None

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K

mh = 1 #* amu.value
mhe = 4.0026

###### reading tables ######

def cms_reader(tab_name):

    cols = ['logt','logp','logrho','logu','logs','dlrho/dlT_P','dlrho/dlP_T','dlS/dlT_P',
           'dlS/dlP_T','grad_ad']

    tab = np.loadtxt('%s/cms/DirEOS2019/%s' % (CURR_DIR, tab_name), comments='#')
    tab_df = pd.DataFrame(tab, columns=cols)
    data = tab_df[(tab_df['logt'] <= 5) & (tab_df['logt'] != 2.8)]# ROB: increased to 6.0 to test wider range for brenth
    #data = tab_df

    data['logp'] += 10 # 1 GPa = 1e10 cgs
    data['logu'] += 10 # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g
    data['logs'] += 10 # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g

    return data

def grid_data(df):
    # grids data for interpolation
    twoD = {}
    shape = df['logt'].nunique(), -1
    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    return twoD

cms_hdata = grid_data(cms_reader('TABLE_H_TP_v1'))
cms_hedata = grid_data(cms_reader('TABLE_HE_TP_v1'))

data_hc = pd.read_csv('%s/cms/HG23_Vmix_Smix.csv' % CURR_DIR, delimiter=',')
data_hc = data_hc[(data_hc['LOGT'] <= 5.0) & (data_hc['LOGT'] != 2.8)]
data_hc = data_hc.rename(columns={'LOGT':'logt', 'LOGP':'logp'}).sort_values(by=['logt', 'logp'])

grid_hc = grid_data(data_hc)
svals_hc = grid_hc['Smix']

logpvals_hc = grid_hc['logp'][0]
logtvals_hc = grid_hc['logt'][:,0]

# smix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Smix']) # Smix will be in cgs... not log cgs.
# vmix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Vmix'])

smix_interp_rgi = RGI((logtvals_hc, logpvals_hc), grid_hc['Smix'], method='linear', bounds_error=False, fill_value=None) # Smix will be in cgs... not log cgs.
vmix_interp_rgi = RGI((logtvals_hc, logpvals_hc), grid_hc['Vmix'], method='linear', bounds_error=False, fill_value=None)

def smix_interp(lgt, lgp):
    return smix_interp_rgi(np.array([lgt, lgp]).T)
def vmix_interp(lgt, lgp):
    return vmix_interp_rgi(np.array([lgt, lgp]).T)

# interpolations

logtvals = cms_hdata['logt'][:,0]
logpvals = cms_hdata['logp'][0]

#### H data ####

svals_h = cms_hdata['logs']
rhovals_h = cms_hdata['logrho']
loguvals_h = cms_hdata['logu']

# get_s_h = RBS(logtvals, logpvals, svals_h) # x or y are not changing so can leave out to speed things up
# get_rho_h = RBS(logtvals, logpvals, rhovals_h)
# get_logu_h = RBS(logtvals, logpvals, loguvals_h, s=0)

get_s_h_rgi = RGI((logtvals, logpvals), svals_h, method='linear', bounds_error=False, fill_value=None)
get_rho_h_rgi = RGI((logtvals, logpvals), rhovals_h, method='linear', bounds_error=False, fill_value=None)
get_logu_h_rgi = RGI((logtvals, logpvals), loguvals_h, method='linear', bounds_error=False, fill_value=None)

def get_s_h(lgt, lgp):
    return get_s_h_rgi(np.array([lgt, lgp]).T)
def get_rho_h(lgt, lgp):
    return get_rho_h_rgi(np.array([lgt, lgp]).T)
def get_logu_h(lgt, lgp):
    return get_logu_h_rgi(np.array([lgt, lgp]).T)

# derivatives

# get_rhot_h = RBS(logtvals, logpvals, cms_hdata['dlrho/dlT_P'])
# get_rhop_h = RBS(logtvals, logpvals, cms_hdata['dlrho/dlP_T'])
# get_sp_h = RBS(logtvals, logpvals, cms_hdata['dlS/dlP_T'])
# get_st_h = RBS(logtvals, logpvals, cms_hdata['dlS/dlT_P'])

#### He data ####

svals_he = cms_hedata['logs']
rhovals_he = cms_hedata['logrho']
loguvals_he = cms_hedata['logu']

get_s_he_rgi = RGI((logtvals, logpvals), svals_he, method='linear', bounds_error=False, fill_value=None)
get_rho_he_rgi = RGI((logtvals, logpvals), rhovals_he, method='linear', bounds_error=False, fill_value=None)
get_logu_he_rgi = RGI((logtvals, logpvals), loguvals_he, method='linear', bounds_error=False, fill_value=None)

def get_s_he(lgt, lgp):
    return get_s_he_rgi(np.array([lgt, lgp]).T)
def get_rho_he(lgt, lgp):
    return get_rho_he_rgi(np.array([lgt, lgp]).T)
def get_logu_he(lgt, lgp):
    return get_logu_he_rgi(np.array([lgt, lgp]).T)

###### H-He mixing ######

def Y_to_n(Y):
    return ((Y/mhe)/(((1 - Y)/mh) + (Y/mhe)))
def n_to_Y(x):
    return (mhe * x)/(1 + 3.0026*x)

def guarded_log(x):
    if x == 0:
        return 0
    elif x < 0:
        raise ValueError('a')
    return x * np.log(x)

### isolating the ideal and interacting entropy of mixing terms from HG23 ###

def get_smix_id_y(Y):
    #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
    xhe = Y_to_n(Y)
    xh = 1 - xhe
    return -1*(guarded_log(xh) + guarded_log(xhe))

def get_smix_nd(Y, lgp, lgt):

    # the HG23 \Delta S_mix is the combination of the non-ideal and ideal entropy of mixing
    smix_hg23 = smix_interp(lgt, lgp)*(1 - Y)*Y
    smix_id = get_smix_id_y(Y) / erg_to_kbbar

    return smix_hg23 - smix_id

def get_s_pt(lgp, lgt, y, hc_corr = True):
    s_h = 10 ** get_s_h(lgt, lgp)
    s_he = 10 ** get_s_he(lgt, lgp)
    smix = smix_interp(lgt, lgp)*(1 - y)*y
    if hc_corr==False:
        #smix = get_smix_id_y(y)/erg_to_kbbar
        smix -= get_smix_nd(y, lgp, lgt)
    return (1 - y) * s_h + y * s_he + smix #

def get_rho_pt(lgp, lgt, y, hc_corr = True):
    rho_h = 10 ** get_rho_h(lgt, lgp)
    rho_he = 10 ** get_rho_he(lgt, lgp)
    if hc_corr:
        vmix = vmix_interp(lgt, lgp)
    elif not hc_corr:
        vmix = 0
    return np.log10(1/(((1 - y) / rho_h) + (y / rho_he) + vmix*(1 - y)*y))

def get_u_pt(lgp, lgt, y):
    u_h = 10**get_logu_h(lgt, lgp) # MJ/kg to erg/g
    u_he = 10**get_logu_he(lgt, lgp)
    return np.log10((1 - y)*u_h + y*u_he)

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

def get_c_p_s(s, p, y, ds=0.1):
    S1 = np.log10(s/erg_to_kbbar)
    S2 = np.log10(S1*(1+ds))
    T1 = get_t_sp(S1*erg_to_kbbar, p, y)[-1]
    T2 = get_t_sp(S2*erg_to_kbbar, p, y)[-1]

    return S1*((S2 - S1)/(T2 - T1))

def get_c_v_t(rho, t, y, dt=0.01):
    P1 = get_p_rhot(rho, t, y)
    P2 = get_p_rhot(rho, t*(1+dt), y)

    S1 = np.log10(get_s_pt(P1, t, y))
    S2 = np.log10(get_s_pt(P2, t*(1+dt), y))
    return S1*(S2 - S1)/(t*dt)

def get_c_v_s(s, rho, y, ds=0.1):
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)
    T1 = get_t_srho(S1*erg_to_kbbar, rho, y)
    T2 = get_t_srho(S2*erg_to_kbbar, rho, y)
    return S1*(np.log10(S2) - np.log10(S1))/(T2 - T1)

### error functions ###

def err_p_srho(lgp, lgr, s_val, y):
    t = get_t_srho(s_val, lgr, y)
    s_ = get_s_pt(lgp, t, y)
    s_val /= erg_to_kbbar
    return (s_/s_val) - 1

def err_t_sp(logt, logp, s_val, y):
    s_ = get_s_pt(logp, logt, y)
    s_val /= erg_to_kbbar # in cgs

    return (s_/s_val) - 1

def err_pt_su(pt_pair, sval, uval, y):
    lgp, lgt = pt_pair
    s, logu = get_s_pt(lgp, lgt, y), get_u_pt(lgp, lgt, y)
    s *= erg_to_kbbar
    return  s/sval - 1, logu/uval -1

def err_t_rhop(lgt, lgp, rhoval, y):
    #lgp, lgt = pt_pair
    logrho = get_rho_pt(lgp, lgt, y)
    #s *= erg_to_kbbar
    return  logrho/rhoval - 1

def err_p_rhot(lgp, lgt, rhoval, y):
    #lgp, lgt = pt_pair
    logrho = get_rho_pt(lgp, lgt, y)
    #s *= erg_to_kbbar
    return  logrho/rhoval - 1

# def err_p_su(lgp, s, uval, y):
#     logt = get_rho_t(s, lgp, y)[-1]
#     logu = get_u_pt(lgp, logt, y)
#     return logu/uval - 1

def err_pt_srho(pt_pair, sval, rval, y):
    lgp, lgt = pt_pair
    s, logrho = get_s_pt(lgp, lgt, y), get_rho_pt(lgp, lgt, y)
    sval /= erg_to_kbbar
    #logrho = np.log10(rho)
    return  s/sval - 1, logrho/rval -1

# def err_rho_su(lgr, s, uval, y): #uval in log10
#     #logt = get_rho_t(s, lgp, y)[-1]
#     logu = np.log10(get_u_sr(s, lgr, y))
#     return logu/uval - 1

# def err_t_urho(lgt, lgr, uval, y):
#     logu = get_u_rho(lgr, lgt, y)
#     return logu/uval - 1

def err_t_srho(lgt, lgr, sval, y):
    s = get_s_rhot(lgr, lgt, y)
    sval /= erg_to_kbbar
    return s/sval - 1

### inversion functions ###

TBOUNDS = [2, 7] # s(rho, P, Y) only works for these bounds... [0, 7] even when the top limit of the CMS table is logT<5
PBOUNDS = [0, 15]

XTOL = 1e-8

def get_pt_su(s, u, y):
    if u > 12:
        guess = [10, 4]
    else:
        guess = [7, 2.5]
    sol = root(err_pt_su, guess, args=(s, u, y))
    return sol.x

def get_pt_srho(s, rho, y, guess=[7, 2.7], alg='hybr'):
    if np.isscalar(rho):
        sol = root(err_pt_srho, guess, tol=1e-8, method=alg, args=(s, rho, y))
        return sol.x
    p, t = np.array([get_pt_srho(s_, r_, y_) for s_, r_, y_ in zip(s, rho, y)]).T
    return p, t
    

def get_p_srho(s, rho, y):
    if np.isscalar(s):
            #guess = 2.5
        sol = root_scalar(err_p_srho, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, s, y))
        return sol.root

    sol = np.array([get_p_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
    return sol

def get_t_sp(s, p, y):
    if np.isscalar(s):
        try:
            sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(p, s, y)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
            return sol.root
        except:
            print('s={}, p={}, y={}'.format(s, p, y))
            raise
    sol = np.array([get_t_sp(s_, p_, y_) for s_, p_, y_ in zip(s, p, y)])
    return sol

def get_rhot_sp(s, p, y):
    # t = get_t_sp(s, p, y)
    # rho = get_rho_pt(p, t, y)
    rho, t = get_rho_t(s, p, y)
    return rho, t

def get_t_rhop(rho, p, y):
    if np.isscalar(rho):
        try:
            sol = root_scalar(err_t_rhop, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(p, rho, y))
            return sol.root
        except:
            print('rho={}, p={}, y={}'.format(rho, p, y))
            raise
    sol = np.array([get_t_rhop(rho_, p_, y_) for rho_, p_, y_ in zip(rho, p, y)])
    return sol

def get_t_srho(s, rho, y):
    if np.isscalar(rho):
        #guess = 2.5
        try:
            sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(rho, s, y))
            return sol.root
        except:
            print('s={}, rho={}, y={}'.format(s, rho, y))
            raise
    sol = np.array([get_t_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
    return sol

def get_p_rhot(rho, t, y):
    #y = cms.n_to_Y(x)
    if np.isscalar(rho):
        #guess = 7
        sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', #xtol=1e-6, 
            args=(t, rho, y))
        return sol.root
    sol = np.array([get_p_rhot(rho_, t_, y_) for rho_, t_, y_ in zip(rho, t, y)])
    return sol

def get_s_rhot(rho, t, y):
    #y = cms.n_to_Y(x)
    p = get_p_rhot(rho, t, y)
    s = get_s_pt(p, t, y)
    return s # in cgs

def get_u_rho(rho, t, y):
    #y = cms.n_to_Y(x)
    p = get_p_rhot(rho, t, y) 
    logu = get_u_pt(p, t, y)
    return 10**logu

def get_u_sr(s, rho, y):
    #y = cms.n_to_Y(x)
    #t = get_t_srho(s, rho, y)
    p, t = get_p_srho(s, rho, y), get_t_srho(s, rho, y)
    #return 10**get_logu_r(rho, t, y)
    return 10**get_u_pt(p, t, y)

# def get_s_ur(u, rho, y):
#     t = get_t_ur(u, rho, y)
#     return get_s_rhot(rho, t, y) # in cgs

def get_s_rhop(rho, p, y):
    t = get_t_rhop(rho, p, y)
    #y = cms.n_to_Y(y)
    s = get_s_pt(p, t, y)
    return s # in cgs

############## derivatives ##############


### entropy gradients ###

def get_dsdy_rhop(rho, p, y, dy=0.01):
    S0 = get_s_rhop(rho, p, y)
    S1 = get_s_rhop(rho, p, y*(1+dy))

    return (S1 - S0)/(y*dy)


def get_dsdy_rhot(rho, t, y, dy=0.01):
    s0 = get_s_rhot(rho, t, y)
    s1 = get_s_rhot(rho, t, y*(1+dy))

    dsdy = (s1 - s0)/(y*dy)
    return dsdy

def get_dsdy_pt(p, t, y, dy=0.01):
    S1 = get_s_pt(p, t, y)
    S2 = get_s_pt(p, t, y*(1+dy))

    return (S2 - S1)/(y*dy)

def get_dsdt_ry_rhot(rho, t, y, dt=0.1):
    s0 = get_s_rhot(rho, t, y)
    s1 = get_s_rhot(rho, t*(1+dt), y)

    dsdt = (s1 - s0)/(t*dt)
    return dsdt

### energy gradients ###

# to get chemical potential:
def get_dudy_srho(s, rho, y, dy=0.01):
    # u0 = get_u_s(s, rho, y)
    # u1 = get_u_s(s, rho, y*(1+dy))
    # u0 = np.log10(get_u_sr(s, rho, y))
    # u1 = np.log10(get_u_sr(s, rho, y*(1+dy)))
    P0, T0 = get_pt_srho(s, rho, y)
    P1, T1 = get_pt_srho(s, rho, y*(1+dy))
    U0 = get_u_pt(P0, T0, y)
    U1 = get_u_pt(P1, T1, y*(1+dy))
    return (U1 - U0)/(y*dy)

### density gradients ###

def get_drhods_py(s, p, y, ds=0.01):
    
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)

    rho0 = get_rhot_sp(S1*erg_to_kbbar, p, y)[0]
    rho1 = get_rhot_sp(S2*erg_to_kbbar, p, y)[0]

    drhods = (rho1 - rho0)/(S2 - S1)

    return drhods

def get_drhodt_py(p, t, y, dt=0.1):
    #y = cms.n_to_Y(x)
    rho0 = get_rho_pt(p, t, y)
    rho1 = get_rho_pt(p, t*(1+dt), y)

    drhodt = (rho1 - rho0)/(t*dt)

    return drhodt

### temperature gradients ###

def get_dtdy_sp(s, p, y, dy=0.01):
    # t0 = get_t_sp(s, p, y)
    # t1 = get_t_sp(s, p, y*(1+dy))
    _, t0 = get_rhot_sp(s, p, y)
    _, t1 = get_rhot_sp(s, p, y*(1+dy))

    dtdy = (t1 - t0)/(y*dy)
    return dtdy

def get_dtdy_srho(s, rho, y, dy=0.01):
    t0 = get_t_srho(s, rho, y)
    t1 = get_t_srho(s, rho, y*(1+dy))

    return (t1 - t0)/(y*dy)

def get_dtdy_rhop(rho, p, y, dy=0.01):
    t0 = get_t_rhop(rho, p, y)
    t1 = get_t_rhop(rho, p, y*(1+dy))

    dtdy = (t1 - t0)/(y*dy)
    return dtdy

# def get_dtdy_rp(rho, p, y, dy=0.01):
#     t0 = get_t_pr(p, rho, y)
#     t1 = get_t_pr(p, rho, y*(1+dy))

#     dtdy = (t1 - t0)/(y*dy)
#     return dtdy


### Ledoux terms ###
def get_B1(s, p, y, dy=0.1): # neesd to be multiplied by dY/dP
    #p = get_p_srho(s, rho, y)
    c_p = get_c_p(s, p, y)
    rho, T = get_rhot_sp(s, p, y)
    dsdy_rhop = get_dsdy_rhop(rho, p, y, dy=dy)
    return ((10**p)/c_p)*dsdy_rhop

def get_B2(s, p, y, dy=0.1): # neesd to be multiplied by dY/dP
    t = get_t_sp(s, p, y)
    c_p = get_c_p(s, p, y)
    dsdy_pt = get_dsdy_pt(p, t, y, dy=dy)
    return -((10**p)/c_p)*dsdy_pt

