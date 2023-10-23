import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
#from scipy.interpolate import RectBivariateSpline as RBS
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy.constants import m_p
#from numba import jit
import os
from eos import ideal_eos, aqua_eos
import pdb
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

pd.options.mode.chained_assignment = None
ideal_z = ideal_eos.IdealEOS(m=18)
ideal_xy = ideal_eos.IdealHHeMix()
ideal_x = ideal_eos.IdealEOS(m=2)
mz = 18
mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

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

logpvals = cms_hdata['logp'][0]
logtvals = cms_hdata['logt'][:,0]

logpvals = cms_hedata['logp'][0]
logtvals = cms_hedata['logt'][:,0]

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

def x_H(Y, Z, mz):
    Ntot = (1-Y)*(1-Z)/mh + (Y*(1-Z)/mhe) + Z/mz
    return (1-Y)*(1-Z)/mh/Ntot

def x_Z(Y, Z, mz):
    Ntot = (1-Y)*(1-Z)/mh + (Y*(1-Z)/mhe) + Z/mz
    return (Z/mz)/Ntot

def guarded_log(x):
    if np.isscalar(x):
        if x == 0:
            return 0
        elif x  < 0:
            raise ValueError('a')
        return x * np.log(x)
    # else:
    #     if np.any(x) == 0:
    #         return 0
    #     elif np.any(x)  < 0:
    #         raise ValueError('a')
    #     return x * np.log(x)

    return np.array([guarded_log(x_) for x_ in x])

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

def get_s_pt(lgp, lgt, y, hg = True):
    s_h = 10 ** get_s_h(lgt, lgp)
    s_he = 10 ** get_s_he(lgt, lgp)
    smix = smix_interp(lgt, lgp)*(1 - y)*y
    if not hg:
        #smix = get_smix_id_y(y)/erg_to_kbbar
        smix -= get_smix_nd(y, lgp, lgt)
    return (1 - y) * s_h + y * s_he + smix #

def get_s_ptz(lgp, lgt, y, z, z_eos, mz=18.015):
    s_nid_mix = get_smix_nd(y, lgp, lgt) # in cgs
    s_h = 10 ** get_s_h(lgt, lgp) # in cgs
    s_he = 10 ** get_s_he(lgt, lgp)
    xz = x_Z(y, z, mz)
    xh = x_H(y, z, mz)
    if (z_eos is None): # let's not calculate stuff when z = 0
        xz = 0.0
        s_z = 0.0
    elif z_eos == 'ideal':
        s_z = ideal_z.get_s_pt(lgp, lgt, y) / erg_to_kbbar
    elif z_eos == 'aqua':
        s_z = aqua_eos.get_s_pt(lgp, lgt)
    else:
        raise Exception('z_eos must be either None, ideal, or aqua')

    xhe = 1 - xh - xz
    #xhe = Y_to_n(y)
    xh = 1 - xhe - xz
    s_id_zmix = (guarded_log(xh) + guarded_log(xz) + guarded_log(xhe)) / erg_to_kbbar

    return (1 - y)* (1 - z) * s_h + y * (1 - z) * s_he + s_z * z + s_nid_mix*(1 - z) - s_id_zmix

def get_rho_pt(lgp, lgt, y, hg = True):
    rho_h = 10 ** get_rho_h(lgt, lgp)
    rho_he = 10 ** get_rho_he(lgt, lgp)
    vmix = vmix_interp(lgt, lgp)
    #if hc_corr:
        #vmix = vmix_interp(lgt, lgp)
    if not hg:
        vmix = 0
    return np.log10(1/(((1 - y) / rho_h) + (y / rho_he) + vmix*(1 - y)*y))

def get_rho_ptz(lgp, lgt, y, z, z_eos):
    #if z > 0:
    rho_hhe = 10**get_rho_pt(lgp, lgt, y)
    if z_eos == 'ideal':
        rho_z = 10**ideal_z.get_rho_pt(lgp, lgt, y)
    elif z_eos == 'aqua':
        rho_z = 10**aqua_eos.get_rho_pt(lgp, lgt)

    return np.log10(1/((1 - z)/rho_hhe + z/rho_z))
    #elif z == 0:
        #return get_rho_pt(lgp, lgt, y)

def get_u_pt(lgp, lgt, y):
    u_h = 10**get_logu_h(lgt, lgp) # MJ/kg to erg/g
    u_he = 10**get_logu_he(lgt, lgp)
    return np.log10((1 - y)*u_h + y*u_he)

###### inverted tables ######

## t, rho (s, p, y) ##
"""To revert to the old version with the HG corrections, uncomment the first line.
All functions should be the same for ease of use."""


""" The commented values below are the old S, P basis. While the values are okay, the derivatives are suspect."""
# s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, gamma1_arr, grada_arr = np.load('%s/cms/cms_hg_thermo.npy' % CURR_DIR)
# #s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, grada_arr = np.load('%s/cms/cms_thermo.npy' % CURR_DIR)

# get_rho_ = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)
# get_t_ = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)

# get_cp = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
# get_cv = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

# get_chirho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
# get_chit = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

# get_grada = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
# get_gamma1 = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

# def get_rho_t(s, p, y):
#     return get_rho_(np.array([y, s, p]).T), get_t_(np.array([y, s, p]).T)

# def get_c_p(s, p, y):
#     cp_res = get_cp(np.array([y, s, p]).T)
#     return cp_res

# def get_c_v(s, p, y):
#     cv_res = get_cv(np.array([y, s, p]).T)
#     return cv_res

# def get_chi_rho(s, p, y):
#     chirho_res = get_chirho(np.array([y, s, p]).T)
#     return chirho_res

# def get_chi_t(s, p, y):
#     chit_res = get_chit(np.array([y, s, p]).T)
#     return chit_res

# def get_grad_ad(s, p, y):
#     grada = get_grada(np.array([y, s, p]).T)
#     return grada

logrho_res_sp, logt_res_sp = np.load('%s/cms/sp_base_comb.npy' % CURR_DIR)

svals_sp = np.arange(5.25, 10.1, 0.05)
logpvals_sp = np.arange(5.5, 14, 0.05)
yvals_sp = np.arange(0.05, 1.05, 0.05)
zvals_sp = np.arange(0, 0.98, 0.02)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp), logrho_res_sp, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp), logt_res_sp, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(s, p, y):
    if np.isscalar(s):
        return float(get_rho_rgi_sp(np.array([s, p, y]).T))
    else:
        return get_rho_rgi_sp(np.array([s, p, y]).T)

def get_t_sp_tab(s, p, y):
    if np.isscalar(s):
        return float(get_t_rgi_sp(np.array([s, p, y]).T))
    else:
        return get_t_rgi_sp(np.array([s, p, y]).T)

def get_rhot_sp_tab(s, p, y):
    return get_rho_sp_tab(s, p, y), get_t_sp_tab(s, p, y)

### aqua mixture tables ###

# svals_spz = np.arange(5.0, 10.1, 0.05)
# logpvals_spz = np.arange(5.0, 14, 0.05)
# yvals_spz = np.arange(0.05, 1.0, 0.05)
# zvals_spz = np.arange(0, 0.98, 0.02)

svals_spz = np.arange(5.5, 9.05, 0.05)
logpvals_spz = np.arange(5.5, 14, 0.05)
yvals_spz = np.arange(0.05, 0.55, 0.05)
zvals_spz = np.arange(0, 0.7, 0.1)

logrho_res_spz, logt_res_spz = np.load('%s/cms/sp_base_z_aqua.npy' % CURR_DIR)

get_rho_rgi_spz = RGI((svals_spz, logpvals_spz, yvals_spz, zvals_spz), logrho_res_spz, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_spz = RGI((svals_spz, logpvals_spz, yvals_spz, zvals_spz), logt_res_spz, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_spz_tab(s, p, y, z):
    if np.isscalar(s):
        return float(get_rho_rgi_spz(np.array([s, p, y, z]).T))
    else:
        return get_rho_rgi_spz(np.array([s, p, y, z]).T)

def get_t_spz_tab(s, p, y, z):
    if np.isscalar(s):
        return float(get_t_rgi_spz(np.array([s, p, y, z]).T))
    else:
        return get_t_rgi_spz(np.array([s, p, y, z]).T)

def get_rhot_spz_tab(s, p, y, z):
    return get_rho_spz_tab(s, p, y, z), get_t_spz_tab(s, p, y, z)



### P(s, rho, Y), T(s, rho, Y) tables ###
#p_srho, t_srho = np.load('%s/cms/p_sry.npy' % CURR_DIR), np.load('%s/cms/t_sry.npy' % CURR_DIR)
logp_res_srho, logt_res_srho = np.load('%s/cms/srho_base_comb.npy' % CURR_DIR)

svals_srho = np.arange(5.0, 10.1, 0.05) # new grid
logrhovals_srho = np.arange(-5, 1.5, 0.05)
yvals_srho = np.arange(0.05, 1.05, 0.05)
#zvals_srho = np.arange(0, 0.98, 0.02)

get_p_rgi_srho = RGI((svals_srho, logrhovals_srho, yvals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhovals_srho, yvals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, r, y):
    if np.isscalar(s):
        return float(get_p_rgi_srho(np.array([s, r, y]).T))
    else:
        return get_p_rgi_srho(np.array([s, r, y]).T)

def get_t_srho_tab(s, r, y):
    if np.isscalar(s):
        return float(get_t_rgi_srho(np.array([s, r, y]).T))
    else:
        return get_t_rgi_srho(np.array([s, r, y]).T)

### aqua mixture tables ###
# logp_res_srhoz, logt_res_srhoz = np.load('%s/cms/srho_base_z_aqua.npy' % CURR_DIR)

# svals_srhoz = np.arange(5.5, 10.05, 0.05)
# logrhovals_srhoz = np.arange(-4, 2.5, 0.05)
# yvals_srhoz = np.arange(0.05, 1.0, 0.05)
# zvals_srhoz = np.arange(0, 1.0, 0.1)

# get_p_rgi_srhoz = RGI((svals_srhoz, logrhovals_srhoz, yvals_srhoz, zvals_srhoz), logp_res_srhoz, method='linear', \
#             bounds_error=False, fill_value=None)
# get_t_rgi_srhoz = RGI((svals_srhoz, logrhovals_srhoz, yvals_srhoz, zvals_srhoz), logt_res_srhoz, method='linear', \
#             bounds_error=False, fill_value=None)

# def get_p_srhoz_tab(s, r, y, z):
#     if np.isscalar(s):
#         return float(get_p_rgi_srhoz(np.array([s, r, y, z]).T))
#     else:
#         return get_p_rgi_srhoz(np.array([s, r, y, z]).T)

# def get_t_srhoz_tab(s, r, y, z):
#     if np.isscalar(s):
#         return float(get_t_rgi_srhoz(np.array([s, r, y, z]).T))
#     else:
#         return get_t_rgi_srhoz(np.array([s, r, y, z]).T)


### P(rho, T, Y), s(rho, T, Y) tables ###

logp_res_rhot, s_res_rhot = np.load('%s/cms/rhot_base_comb.npy' % CURR_DIR)

logrhovals_rhot = np.arange(-5, 1.5, 0.05)
logtvals_rhot = np.arange(2.1, 5.1, 0.05)
yvals_rhot = np.arange(0.05, 1.05, 0.05)
zvals_rhot = np.arange(0, 0.98, 0.02)

get_p_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), logp_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), s_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t, y):
    if np.isscalar(rho):
        return float(get_p_rgi_rhot(np.array([rho, t, y]).T))
    else:
        return get_p_rgi_rhot(np.array([rho, t, y]).T)

def get_s_rhot_tab(rho, t, y):
    if np.isscalar(rho):
        return float(get_s_rgi_rhot(np.array([rho, t, y]).T))
    else:
        return get_s_rgi_rhot(np.array([rho, t, y]).T)

### aqua mixture tables ###
# logrhovals_rhotz = np.arange(-5, 2.5, 0.05)
# logtvals_rhotz = np.arange(2.1, 5.1, 0.05)
# yvals_rhotz = np.arange(0.05, 1.0, 0.05)
# zvals_rhotz = np.arange(0, 0.98, 0.02)

logrhovals_rhotz = np.arange(-4.5, 2.0, 0.05)
logtvals_rhotz = np.arange(2.1, 5.1, 0.05)
yvals_rhotz = np.arange(0.05, 0.55, 0.05)
zvals_rhotz = np.arange(0, 0.7, 0.1)

logp_res_rhotz, s_res_rhotz = np.load('%s/cms/rhot_base_z_aqua.npy' % CURR_DIR)

get_p_rgi_rhotz = RGI((logrhovals_rhotz, logtvals_rhotz, yvals_rhotz, zvals_rhotz), logp_res_rhotz, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhotz = RGI((logrhovals_rhotz, logtvals_rhotz, yvals_rhotz, zvals_rhotz), s_res_rhotz, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhotz_tab(rho, t, y, z):
    if np.isscalar(rho):
        return float(get_p_rgi_rhotz(np.array([rho, t, y, z]).T))
    else:
        return get_p_rgi_rhotz(np.array([rho, t, y, z]).T)

def get_s_rhotz_tab(rho, t, y, z):
    if np.isscalar(rho):
        return float(get_s_rgi_rhotz(np.array([rho, t, y, z]).T))
    else:
        return get_s_rgi_rhotz(np.array([rho, t, y, z]).T)

### error functions ###

def err_p_srho(lgp, lgr, s_val, y):
    t = get_t_srho(s_val, lgr, y)
    s_ = get_s_pt(lgp, t, y)
    s_val /= erg_to_kbbar
    return (s_/s_val) - 1

def err_t_sp(logt, logp, s_val, y, z, hg, z_eos):
    #print(logt, logp, s_val, y, z)
    if np.any(z) > 0:
        s_ = get_s_ptz(logp, logt, y, z, z_eos=z_eos)*erg_to_kbbar
        #s_val /= erg_to_kbbar # in cgs
        return (s_/s_val) - 1
    else:
        s_ = get_s_pt(logp, logt, y, hg)*erg_to_kbbar
        #s_val /= erg_to_kbbar # in cgs

        return (s_/s_val) - 1

def err_p_rhot(lgp, rhoval, lgtval, yval, zval, z_eos):
    #if np.any(zval) > 0.0:
        #sval = float(get_s_ptz(float(lgp), lgtval, yval, zval, z_eos = z_eos))*erg_to_kbbar
    logrho = get_rho_ptz(lgp, lgtval, yval, zval, z_eos = z_eos)
    #pdb.set_trace()
    return (logrho/rhoval) - 1

    # elif zval == 0:
    #     logrho = get_rho_pt(lgp, lgtval, yval)
    #     #s *= erg_to_kbbar
    #     if alg == 'root':
    #         return  logrho/rhoval - 1
    #     elif alg == 'brenth':
    #         return float(logrho/rhoval) - 1

def err_t_srho(lgt, sval, rhoval, yval, zval):
    #sval = sval /erg_to_kbbar
    #lgp = get_p_rhot(rval, lgt, y)
    #if np.any(zval) > 0:
    lgp = get_p_rhotz_tab(rhoval, lgt, yval, zval)
    #lgp = get_p_rhotz_tab(rhoval, lgt, yval, zval)
    #s_ = get_s_ptz(lgp, lgt, yval, zval, z_eos)*erg_to_kbbar
    logrho = get_rho_spz_tab(sval, lgp, yval, zval)

    return (logrho/rhoval) - 1
    # else:
    #     # the original err_t_srho was this below, did it in cgs instead of kbbar
    #     if alg == 'root':
    #         lgp = get_p_rhot_tab(rhoval, lgt, yval)
    #         s_ = get_s_pt(lgp, lgt, yval)*erg_to_kbbar
    #         return  s_/sval - 1
    #     elif alg == 'brenth':
    #         lgp = float(get_p_rhot_tab(rhoval, lgt, yval))
    #         s_ = float(get_s_pt(lgp, lgt, yval))*erg_to_kbbar
    #         return float(s_/sval) - 1

### inversion functions ###

TBOUNDS = [2, 7] # s(rho, P, Y) only works for these bounds... [0, 7] even when the top limit of the CMS table is logT<5
PBOUNDS = [0, 15]

XTOL = 1e-4
    
###### Temperature ######
def get_t_sp(s, p, y, hg=True, alg='brenth', z_eos=None):
    if alg == 'root':
        if np.isscalar(s):
            s, p, y = np.array([s]), np.array([p]), np.array([y])
        guess = ideal_xy.get_t_sp(s, p, y)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(p, s, y, 0, hg, z_eos))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(p, s, y, 0, hg, z_eos)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
                return sol.root
            except:
                #print('s={}, p={}, y={}'.format(s, p, y))
                raise
        sol = np.array([get_t_sp(s_, p_, y_, hg) for s_, p_, y_ in zip(s, p, y)])
        return sol

def get_t_spz(s, p, y, z, hg=True, alg='brenth', z_eos=None):
    #print(s, p, y, z)
    if np.any(z) > 0.0 and z_eos is None:
        raise Exception('You gotta chose a z_eos if you want metallicities!')

    if alg == 'root':
        if np.isscalar(s):
            s, p, y, z = np.array([s]), np.array([p]), np.array([y]), np.array([z])
        #print(s, p, y, z)
        #guess = (1 - z)*ideal_xy.get_t_sp(s, p, y) + z*ideal_z.get_t_sp(s, p, y)
        guess = ideal_xy.get_t_sp(s, p, y)
        #guess = get_t_sp_tab(s, p, y)
        #print(s, p, y, z)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(p, s, y, z, hg, z_eos))
        #print(s, p, y, z)
        #pdb.set_trace()
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(p, s, y, z, hg, z_eos)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
                return sol.root
            except:
                print('s={}, p={}, y={}, z={}'.format(s, p, y, z))
                raise
        #sol = np.array([get_t_spz(s_, p_, y_, z_, hg, z_eos) for s_, p_, y_, z_ in zip(s, p, y, z)])
        #return sol

def get_t_srho(s, rho, y, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho, y = np.array([s]), np.array([rho]), np.array([y])
        guess = ideal_xy.get_t_srho(s, rho, y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y, alg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho, y, alg))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise

def get_t_srhoz(s, rho, y, z=0.0, z_eos=None, alg='brenth'):

    # if np.any(z) > 0.0 and z_eos is None:
    #     raise Exception('You gotta chose a z_eos if you want metallicities!')

    if alg == 'root':
        if np.isscalar(s):
            s, rho, y, z = np.array([s]), np.array([rho]), np.array([y]), np.array([z])
        guess = ideal_xy.get_t_srho(s, rho, y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y, z))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho, y, z))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise


###### Density ######
def get_rhot_sp(s, p, y, tab=True, hg=True):
    if not tab:
        t = get_t_sp(s, p, y, hg)
        rho = get_rho_pt(p, t, y, hg)
    else: # tables have hg...
        rho, t = get_rhot_sp_tab(s, p, y)
    return rho, t

def get_rhot_spz(s, p, y, z, z_eos=None, alg='brenth'):
    #if not tab:
    '''This function does not have a table option yet'''
    # mixture temperature
    t = get_t_spz(s, p, y, z, alg=alg, z_eos=z_eos)
    # density components
    rho_hhe = 10**get_rho_sp_tab(s, p, y)
    if z > 0:
        if z_eos == 'ideal':
            rho_z = 10**ideal_z.get_rho_pt(p, t, y) # y is a dummy input, no effect on ideal_z
        elif z_eos == 'aqua':
            rho_z = 10**aqua_eos.get_rho_pt(p, t)
        return float(np.log10(1/((1 - z)/rho_hhe + z/rho_z))), t
    elif z == 0: # no need to calculate rho_z, although if I did, the above would return the right answer
        return get_rhot_sp_tab(s, p, y)
    

###### Pressure ######
def get_p_rhot(rho, t, y, alg='brenth'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t, y = np.array([rho]), np.array([t]), np.array([y])
        guess = ideal_x.get_p_rhot(rho, t, y)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y, alg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, t, y, alg))
                return sol.root
            except:
                #print('rho={}, t={}, y={}'.format(rho, t, y))
                raise
        sol = np.array([get_p_rhot(rho_, t_, y_) for t_, rho_, y_ in zip(t, rho, y)])
        return sol

def get_p_rhotz(rho, t, y, z=0.0, z_eos=None, alg='brenth'):
    # if np.any(z) > 0.0 and z_eos is None:
    #     raise Exception('You gotta chose a z_eos if you want metallicities!')
    if alg == 'root':
        # if np.isscalar(rho):
        #     rho, t, y, z = np.array([rho]), np.array([t]), np.array([y]), np.array([z])
        guess = ideal_xy.get_p_rhot(rho, t, y)
        #pdb.set_trace()
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y, z, z_eos, alg))
        return sol.x
    elif alg == 'brenth':
        #if np.isscalar(rho):
        try:
            sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, t, y, z, z_eos, alg))
            return sol.root
        except:
            #print('rho={}, t={}, y={}'.format(rho, t, y))
            raise
        # sol = np.array([get_p_rhot(rho_, t_, y_, z_) for rho_, t_, y_, z_ in zip(t, rho, y, z)])
        # return sol

def get_p_srho(s, rho, y):
    if np.isscalar(s):
            #guess = 2.5
        sol = root_scalar(err_p_srho, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, s, y))
        return sol.root

    sol = np.array([get_p_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
    return sol


def get_sp_rhot(rho, t, y):
    logp = get_p_rhot_tab(rho, t, y)
    s = get_s_pt(logp, t, y)
    return s, logp

def get_pt_srho(s, rho, y):
    return get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)

# s, rho inversions

        # sol = np.array([get_t_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
        # return sol

# def get_pt_srho(s, rho, y, tab=True, alg='brenth'):
#     if alg == 'brenth':
#         logt = get_t_srho(s, rho, y, alg)
#         if tab:
#             p = get_p_rhot_tab(rho, logt, y)
#             return p, logt
#         else:
#             p = get_p_rhot(rho, t, y)
#     elif alg == 'root':
#         if np.isscalar(rho):
#             #guess = ideal_x.get_pt_srho(s, rho, y) # guess used to be [8,3]
#             sol = root(err_pt_srho, [8, 3], tol=1e-10, method='hybr', args=(s, rho, y))
#             return sol.x
#         p, t = np.array([get_pt_srho(s_, r_, y_, alg='root') for s_, r_, y_ in zip(s, rho, y)]).T
#         return p, t

# ROB (09/18/2023): finish implementing more efficient root function instead of looping root_scalar with ideal gas guesses

# def get_s_rhot(rho, t, y):
#     #y = cms.n_to_Y(x)
#     p = get_p_rhot(rho, t, y)
#     s = get_s_pt(p, t, y)
#     return s # in cgs

def get_u_sp(s, p, y):
    t = get_t_sp(s, p, y)
    return get_u_pt(p, t, y)

def get_u_rhot(rho, t, y):
    #y = cms.n_to_Y(x)
    p = get_p_rhot_tab(rho, t, y) 
    return get_u_pt(p, t, y)

def get_u_srho(s, rho, y):
    p, t = get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)
    return get_u_pt(p, t, y)

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

def get_dsdy_rhop(rho, p, y, dy=0.1):
    S0 = get_s_rhop(rho, p, y)
    S1 = get_s_rhop(rho, p, y*(1+dy))

    return (S1 - S0)/(y*dy)

def get_dsdy_rhop_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    if not tab:
        P0 = 10**get_p_srho(S0*erg_to_kbbar, rho, y)
        P1 = 10**get_p_srho(S1*erg_to_kbbar, rho, y)
        P2 = 10**get_p_srho(S0*erg_to_kbbar, rho, y*(1+dy))
    else:
        P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y)
        P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, rho, y) 
        P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y*(1+dy))   
    
    dpds_rhoy = (P1 - P0)/(S1 - S0)
    dpdy_srho = (P2 - P0)/(y*dy)

    return -dpdy_srho/dpds_rhoy


def get_dsdy_rhot(rho, t, y, dy=0.01):
    S0 = get_s_rhot_tab(rho, t, y)
    S1 = get_s_rhot_tab(rho, t, y*(1+dy))

    dsdy = (S1 - S0)/(y*dy)
    return dsdy

def get_dsdy_pt(p, t, y, dy=0.01):
    S0 = get_s_pt(p, t, y)
    S1 = get_s_pt(p, t, y*(1+dy))

    return (S1 - S0)/(y*dy)

def get_dsdt_ry_rhot(rho, t, y, dt=0.1):
    T0 = 10**t
    T1 = T0*(1+dt)
    S0 = get_s_rhot_tab(rho, np.log10(T0), y)
    S1 = get_s_rhot_tab(rho, np.log10(T1), y)

    return (S1 - S0)/(T1 - T0)

def get_c_s(s, p, y, dp=0.1):
    P0 = 10**p
    P1 = P0*(1+dp)
    R0 = get_rho_sp_tab(s, np.log10(P0), y)
    R1 = get_rho_sp_tab(s, np.log10(P1), y)

    return np.sqrt((P1 - P0)/(10**R1 - 10**R0))

def get_c_v(s, rho, y, ds=0.1):
    # ds/dlogT_{rho, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_srho_tab(S0*erg_to_kbbar, rho, y)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, rho, y)
 
    return (S1 - S0)/(T1 - T0)

def get_c_p(s, p, y, ds=0.1):
    # ds/dlogT_{P, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_sp_tab(S0*erg_to_kbbar, p, y)
    T1 = get_t_sp_tab(S1*erg_to_kbbar, p, y)

    return (S1 - S0)/(T1 - T0)

### pressure gradients ###

def get_dpdt_rhot(rho, t, y, dT=0.01):
    T0 = 10**t
    T1 = T0*(1+dT)
    P0 = get_p_rhot_tab(rho, np.log10(T0), y)
    P1 = get_p_rhot_tab(rho, np.log10(T1), y)
    return (P1 - P0)/(T1 - T0)


### energy gradients ###

# to get chemical potential:
def get_dudy_srho(s, rho, y, dy=0.1):
    U0 = 10**get_u_srho(s, rho, y)
    U1 = 10**get_u_srho(s, rho, y*(1+dy))
    return (U1 - U0)/(y*dy)

# du/ds_(rho, Y) = T test
def get_duds_rhoy_srho(s, rho, y, ds=0.1):
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)
    U0 = 10**get_u_srho(S1*erg_to_kbbar, rho, y)
    U1 = 10**get_u_srho(S2*erg_to_kbbar, rho, y)
    return (U1 - U0)/(S1*ds)

def get_dudrho_sy_srho(s, rho, y, drho=0.1):
    R1 = 10**rho
    R2 = R1*(1+drho)
    #rho1 = np.log10((10**rho)*(1+drho))
    U0 = 10**get_u_srho(s, np.log10(R1), y)
    U1 = 10**get_u_srho(s, np.log10(R2), y)
    #return (U1 - U0)/(R1*drho)
    return (U1 - U0)/((1/R1) - (1/R2))

def get_dudrho_rhot(rho, t, y, drho=0.01):
    R0 = 10**rho
    R1 = R0*(1+drho)
    U0 = 10**get_u_rhot(rho, t, y)
    U1 = 10**get_u_rhot(np.log10(R1), t, y)
    return (U1 - U0)/(R1 - R0)

### density gradients ###

def get_drhods_py(s, p, y, ds=0.01):
    
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)

    rho0 = 10**get_rhot_sp(S1*erg_to_kbbar, p, y)[0]
    rho1 = 10**get_rhot_sp(S2*erg_to_kbbar, p, y)[0]

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
    T0 = 10**get_t_sp_tab(s, p, y) # this was returning dlogT/dY before
    T1 = 10**get_t_sp_tab(s, p, y*(1+dy))

    dtdy = (T1 - T0)/(y*dy)
    return dtdy

def get_dtdy_srho(s, rho, y, dy=0.1, tab=True):
    if not tab:
        T0 = 10**get_t_srho(s, rho, y)
        T1 = 10**get_t_srho(s, rho, y*(1+dy))
    else:
        T0 = 10**get_t_srho_tab(s, rho, y)
        T1 = 10**get_t_srho_tab(s, rho, y*(1+dy)) 

    return (T1 - T0)/(y*dy)

def get_dtdy_rhop(rho, p, y, dy=0.01):
    t0 = 10**get_t_rhop(rho, p, y)
    t1 = 10**get_t_rhop(rho, p, y*(1+dy))

    dtdy = (t1 - t0)/(y*dy)
    return dtdy

def get_dtdrho_sy_srho(s, rho, y, drho = 0.01, tab=True): # dlogT/dlogrho_{s, Y}
    R0 = 10**rho
    R1 = R0*(1+drho)
    if not tab:
        T0 = 10**get_t_srho(s, np.log10(R0), y)
        T1 = 10**get_t_srho(s, np.log10(R1), y)
    else:
        T0 = 10**get_t_srho_tab(s, np.log10(R0), y)
        T1 = 10**get_t_srho_tab(s, np.log10(R1), y)
    return (T1 - T0)/(R1 - R0)

def get_dtds_rhoy_srho(s, rho, y, ds=0.01, tab=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    if not tab:
        T0 = 10**get_t_srho(S0*erg_to_kbbar, rho, y)
        T1 = 10**get_t_srho(S1*erg_to_kbbar, rho, y)
    else:
        T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho, y)
        T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho, y)
    return (T1 - T0)/(S1 - S0)

def get_nabla_ad(s, p, y, dp=0.1):
    t0 = get_t_sp_tab(s, p, y)
    t1 = get_t_sp_tab(s, p*(1+dp), y)
    return (t1 - t0)/(p*dp)


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


### second derivatives ###

def get_d2sdy2_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
    # second derivative of dsdy_rhop
    A0 = get_dsdy_rhop_srho(s, rho, y, ds, dy, tab)
    A1 = get_dsdy_rhop_srho(s, rho, y*(1+dy), ds, dy, tab)
    return (A1 - A0)/(y*dy)

def get_d2sds2_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
    # second derivative of dsdy_rhop
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    A0 = get_dsdy_rhop_srho(S0*erg_to_kbbar, rho, y, ds, dy, tab)
    A1 = get_dsdy_rhop_srho(S1*erg_to_kbbar, rho, y, ds, dy, tab)
    return (A1 - A0)/(S1 - S0)

def get_dcv_ds_srho(s, rho, y, ds=0.1, tab=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    CV0 = get_c_v(S0*erg_to_kbbar, rho, y, ds, tab)
    CV1 = get_c_v(S1*erg_to_kbbar, rho, y, ds, tab)

    return (CV1 - CV0)/(S1 - S0)

def get_dcv_dy_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
    # S0 = s/erg_to_kbbar
    # S1 = S0*(1+ds)
    CV0 = get_c_v(s, rho, y, ds, tab)
    CV1 = get_c_v(s, rho, y*(1+dy), ds, tab)

    return (CV1 - CV0)/(y*dy)

def get_dcp_ds_srho(s, rho, y, ds=0.1, tab=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    CP0 = get_c_p(S0*erg_to_kbbar, rho, y, ds, tab)
    CP1 = get_c_p(S1*erg_to_kbbar, rho, y, ds, tab)

    return (CP1 - CP0)/(S1 - S0)

def get_dcp_dy_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
    # S0 = s/erg_to_kbbar
    # S1 = S0*(1+ds)
    CP0 = get_c_p(s, rho, y, ds, tab)
    CP1 = get_c_p(s, rho, y*(1+dy), ds, tab)

    return (CP1 - CP0)/(y*dy)

##### speed tests #####


#def test():
# import time
# from joblib import Parallel, delayed

# stest = np.zeros(240)+6.2
# rhotest =  np.zeros(240)-4
# ytest = np.zeros(240)+0.25

# start = time.time()
# p, t = get_pt_srho(stest, rhotest, ytest)
# end = time.time()
# print('get_pt_srho test:', end - start)

# start = time.time()
# t = get_t_srho(stest, rhotest, ytest)
# end = time.time()
# print('get_t_srho test:', end - start)

# start = time.time()
# p, t = get_p_srho_tab(stest, rhotest, ytest), get_t_srho_tab(stest, rhotest, ytest)
# end = time.time()
# print('tables:', end - start)

# start = time.time()
# Parallel(n_jobs=2, prefer="threads" )(delayed(get_t_srho)(s_, rho_, y_) for s_, rho_, y_ in zip(stest, rhotest, ytest))
# end = time.time()
# print('parallel code:', end - start)

# from multiprocessing import Pool

# def get_t_srho_par(iter):
#     print(iter)
#     logt = get_t_srho(stest[iter], rhotest[iter], ytest[iter])
#     return logt

# if __name__ == '__main__':
#     pool = Pool(2)
#     start = time.time()
#     results = pool.map(get_t_srho_par, range(len(stest)))
#     end = time.time()
#     print('parallel:', end - start)
#     np.save('%s/cms/parallel_test_tsrho.npy' % CURR_DIR, results)

    



