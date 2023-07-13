import numpy as np
import pandas as pd
from scipy.optimize import newton
#from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from astropy import units as u
from astropy.constants import k_B, m_p
from astropy.constants import u as amu
from tqdm import tqdm
import os
from eos import aneos
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
eos_aneos = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='ice')

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K

pd.options.mode.chained_assignment = None

#erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def cms_reader(tab_name):

    cols = ['logt','logp','logrho','logu','logs','dlrho/dlT_P','dlrho/dlP_T','dlS/dlT_P',
           'dlS/dlP_T','grad_ad']

    tab = np.loadtxt('%s/cms/DirEOS2019/%s' % (CURR_DIR, tab_name), comments='#')
    tab_df = pd.DataFrame(tab, columns=cols)
    data = tab_df[(tab_df['logt'] <= 7.0) & (tab_df['logt'] != 2.8)# ROB: increased to 6.0 to test wider range for brenth
                 ]

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

# USE THIS TO USE THE FULL SCRIPT
#cms_hdata = grid_data(cms_reader('TABLE_H_TP_effective'))
#cms_hedata = grid_data(cms_reader('TABLE_HE_TP_v1'))

# cms_hdata = grid_data(cms_reader('TABLE_H_TP_effective', 2021))
# cms_hedata = grid_data(cms_reader('TABLE_HE_TP_v1', 2021))
# uncomment this to revert
cms_hdata = grid_data(cms_reader('TABLE_H_TP_v1'))
cms_hedata = grid_data(cms_reader('TABLE_HE_TP_v1'))

data_hc = pd.read_csv('%s/cms/HG23_Vmix_Smix.csv' % CURR_DIR, delimiter=',')
data_hc = data_hc[(data_hc['LOGT'] <= 5.0) & (data_hc['LOGT'] != 2.8)]
data_hc = data_hc.rename(columns={'LOGT':'logt', 'LOGP':'logp'}).sort_values(by=['logt', 'logp'])

grid_hc = grid_data(data_hc)
svals_hc = grid_hc['Smix']

logpvals_hc = grid_hc['logp'][0]
logtvals_hc = grid_hc['logt'][:,0]

smix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Smix']) # Smix will be in cgs... not log cgs.
vmix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Vmix'])

# interpolations

logtvals = cms_hdata['logt'][:,0]
logpvals = cms_hdata['logp'][0]

#### H data ####

svals_h = cms_hdata['logs']
rhovals_h = cms_hdata['logrho']
loguvals_h = cms_hdata['logu']

get_s_h = RBS(logtvals, logpvals, svals_h) # x or y are not changing so can leave out to speed things up

get_rho_h = RBS(logtvals, logpvals, rhovals_h)

get_logu_h = RBS(logtvals, logpvals, loguvals_h)

# derivatives

get_rhot_h = RBS(logtvals, logpvals, cms_hdata['dlrho/dlT_P'])
get_rhop_h = RBS(logtvals, logpvals, cms_hdata['dlrho/dlP_T'])
get_sp_h = RBS(logtvals, logpvals, cms_hdata['dlS/dlP_T'])
get_st_h = RBS(logtvals, logpvals, cms_hdata['dlS/dlT_P'])

# thermo

# get_cp = RBS(logtvals, logpvals, cms_hdata['cp'])
# get_chi_rho = RBS(logtvals, logpvals, cms_hdata['chi_rho'])
# get_chi_T = RBS(logtvals, logpvals, cms_hdata['chi_T'])
# get_cv = RBS(logtvals, logpvals, cms_hdata['cv'])

#### He data ####

svals_he = cms_hedata['logs']
rhovals_he = cms_hedata['logrho']
loguvals_he = cms_hedata['logu']

# s(t, rho)
# p(t, rho)
get_s_he = RBS(logtvals, logpvals, svals_he) # x or y are not changing so can leave out to speed things up

get_rho_he = RBS(logtvals, logpvals, rhovals_he)

get_logu_he = RBS(logtvals, logpvals, loguvals_he)

# derivatives

get_rhot_he = RBS(logtvals, logpvals, cms_hedata['dlrho/dlT_P'])
get_rhop_he = RBS(logtvals, logpvals, cms_hedata['dlrho/dlP_T'])
get_sp_he = RBS(logtvals, logpvals, cms_hedata['dlS/dlP_T'])
get_st_he = RBS(logtvals, logpvals,cms_hedata['dlS/dlT_P'])

#### He data ####

svals_he = cms_hedata['logs']
rhovals_he = cms_hedata['logrho']

# s(t, rho)
# p(t, rho)
get_s_he = RBS(logtvals, logpvals, svals_he) # x or y are not changing so can leave out to speed things up

get_rho_he = RBS(logtvals, logpvals, rhovals_he)

# derivatives

get_rhot_he = RBS(logtvals, logpvals, cms_hedata['dlrho/dlT_P'])
get_rhop_he = RBS(logtvals, logpvals, cms_hedata['dlrho/dlP_T'])
get_sp_he = RBS(logtvals, logpvals, cms_hedata['dlS/dlP_T'])
get_st_he = RBS(logtvals, logpvals,cms_hedata['dlS/dlT_P'])


#### Mixtures ####

# def x_i(Y):
#     xi = 2*mp*Y/(4*mp*(1-Y) + mp*Y) # number fraction
#     return xi.value

mh = 1 #* amu.value
mhe = 4.0026


# def x_i(Y):
#     xi = 4*mp*Y/(4*mp*(Y) + mp*(1-Y)) # number fraction
#     return xi.value

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

def get_smix_nd(Y, lgp, lgt):

    # the HG23 \Delta S_mix is the combination of the non-ideal and ideal entropy of mixing
    smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
    smix_id = get_smix_id_y(Y) / erg_to_kbbar

    return smix_hg23 - smix_id

def get_smix_z(Y, Z, lgp, lgt, z_eos='ideal'):
    #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
    #xhe_prime = x_i(Y)
    #s_nid_mix = smix_hg23 + ((guarded_log(1-xhe_prime) + guarded_log(xhe_prime)) / erg_to_kbbar)
    #s_nid_mix = smix_hg23 + get_smix_id_y(Y)
    s_nid_mix = get_smix_nd(Y, lgp, lgt) # in cgs

    s_h = 10 ** get_s_h.ev(lgt, lgp) # in cgs
    s_he = 10 ** get_s_he.ev(lgt, lgp)
    if z_eos=='ideal':
        mz = 15.5
        s_z = sackur_tetrode(lgp, lgt, mz) / erg_to_kbbar
    elif z_eos=='aneos':
        mz = 18
        s_z = 10**eos_aneos.get_logs(lgp, lgt)
    #s_z = 0
    x_He = 1 - x_H(Y, Z, mz) - x_Z(Y, Z, mz)
    xz = x_Z(Y, Z, mz)
    xh = x_H(Y, Z, mz)
    #print(x_He, xh, xz)

    return (1 - Y)* (1 - Z) * s_h + Y * (1 - Z) * s_he + s_z * Z + s_nid_mix*(1 - Z) - ((guarded_log(xh) + guarded_log(xz) + guarded_log(x_He)) / erg_to_kbbar)

# print(get_smix_z(1, 0, 12, 4, 0) * erg_to_kbbar)
# print(get_smix_z(0, 1, 12, 4, 0) * erg_to_kbbar)
# print(get_smix_z(0.5, 0.5, 12, 4, 0) * erg_to_kbbar)

# def s_id_mix(Y):
#     x_i(Y)*np.log(x_i(Y)) + (1 - x_i(Y))*np.log(1 - x_i(Y))

def get_smix_table(y, hc_corr = False):
    s_htab = 10**(svals_h)
    s_hetab = 10**(svals_he)
    if hc_corr:
        smix = svals_hc*(1 - y)*y
    elif not hc_corr:
        smix = get_smix_id_y(y)/erg_to_kbbar # ideal entropy mixing if not HG correction
        #smix = 0
    return np.log10((1 - y) * s_htab + y * s_hetab + smix) # testing ideal entropy mixing

def get_s_mix(lgp, lgt, y, hc_corr = False):
    s_h = 10 ** get_s_h.ev(lgt, lgp)
    s_he = 10 ** get_s_he.ev(lgt, lgp)
    if hc_corr:
        smix = smix_interp.ev(lgt, lgp)*(1 - y)*y
    elif not hc_corr:
        smix = get_smix_id_y(y)/erg_to_kbbar
        #smix = 0

    return (1 - y) * s_h + y * s_he + smix # this was multiplied by XY before, changing to smix if statement to allow for ideal option


def get_rho_mix(lgp, lgt, y, hc_corr = False):
    rho_h = 10 ** get_rho_h.ev(lgt, lgp)
    rho_he = 10 ** get_rho_he.ev(lgt, lgp)
    if hc_corr:
        vmix = vmix_interp.ev(lgt, lgp)
    elif not hc_corr:
        vmix = 0
    return 1/(((1 - y) / rho_h) + (y / rho_he) + vmix*(1 - y)*y)

def get_logu_mix(lgp, lgt, y):
    u_h = (10**get_logu_h(lgt, lgp)) # MJ/kg to erg/g
    u_he = (10**get_logu_he(lgt, lgp))

    return np.log10((1-y)*u_h + y*u_he) # in log cgs


# def get_dsdp_mix(lgp, lgt, y, hc_corr = False):
#     s = get_s_mix(lgp, lgt, y, hc_corr)
#     s_h = 10 ** get_s_h.ev(lgt, lgp)
#     s_he = 10 ** get_s_he.ev(lgt, lgp)

#     sp_h = get_sp_h.ev(lgt, lgp)
#     sp_he = get_sp_he.ev(lgt, lgp)

#     dsdp = (1 - y) * (s_h / s) * sp_h + y * (s_he / s) * sp_he

#     return dsdp

# def get_dsdt_mix(lgp, lgt, y, hc_corr = False):
#     s = get_s_mix(lgp, lgt, y, hc_corr)
#     s_h = 10 ** get_s_h.ev(lgt, lgp)
#     s_he = 10 ** get_s_he.ev(lgt, lgp)

#     st_h = get_st_h.ev(lgt, lgp)
#     st_he = get_st_he.ev(lgt, lgp)

#     dsdt = (1 - y) * (s_h / s) * st_h + y * (s_he / s) * st_he

#     return dsdt

# def get_dsdrho_mix(lgp, lgt, y, hc_corr = False):
#     dsdp = get_dsdp_mix(lgp, lgt, y, hc_corr)
#     dpdrho = get_dpdrho_mix(lgp, lgt, y, hc_corr)

#     return dsdp*dpdrho # dP/dT

# def get_dpdrho_mix(lgp, lgt, y):

#     prho_h = 1/get_rhop_h.ev(lgt, lgp)
#     prho_he = 1/get_rhop_he.ev(lgt, lgp)

#     dpdrho = (1 - y) * prho_h + y * prho_he

#     return dpdrho

# def get_drhodt_mix(lgp, lgt, y, hc_corr=False):

#     rhot_h = get_rhot_h.ev(lgt, lgp)
#     rhot_he = get_rhot_he.ev(lgt, lgp)

#     rho_h = 10 ** get_rho_h.ev(lgt, lgp)
#     rho_he = 10 ** get_rho_he.ev(lgt, lgp)

#     rho_mixed = get_rho_mix(lgt, lgp, y, hc_corr)

#     drhodt = (1 - y) * (rho_mixed / rho_h) * rhot_h + y * (rho_mixed /rho_he) * rhot_he

#     return drhodt

# def get_dpdt_mix(lgp, lgt, y):
#     dpdrho = get_dpdrho_mix(lgp, lgt, y)
#     drhodt = get_drhodt_mix(lgp, lgt, y)

#     return dpdrho*drhodt # dP/dT

# def get_closest(s_val, y):

#     logs_val = np.log10(s_val/erg_to_kbbar) # converting to cgs

#     svals = get_smix_table(y)

#     near_id_s = np.unravel_index((np.abs(svals - logs_val)).argmin(), svals.shape)

#     #return  logrhovals[near_id_s[1]]
#     return  logpvals[near_id_s[1]]

# def err_rbs(logp, logt, y, s_val, corr):

#     s_ = get_s_mix(logp, logt, y, corr)
#     s_val /= erg_to_kbbar # in cgs
#     return (s_/s_val) - 1

# def root_finder(S_val, y, corr=False):

#     svals = get_smix_table(y, corr)
#     p_last = logpvals[np.argmin(np.abs(svals[0,:] - np.log10(S_val/erg_to_kbbar)))]

#     #rho_last = get_closest(S_val, y)

#     logt_root = []
#     logrho_root = []
#     logp_root = []

#     c_v = []
#     c_p = []

#     chirho = []
#     chit = []

#     #for i, t in enumerate(tqdm(logt_grid)):
#     # rob: trying to calculate derivatives directly...
#     eps = 0.1
#     for i, t in enumerate(tqdm(logtvals)):

#         #if t < np.log10(115): continue #put back if there's a discontinuity at low temperatures...
#         root = newton(err_rbs, p_last, args=(t, y, S_val, corr))
#        # print(root)
#         logt_root.append(t)
#         logrho_root.append(np.log10(get_rho_mix(root, t, y, corr)))
#         logp_root.append(root)
#         p_last = root

#         S0, R0 = np.log10(get_s_mix(p_last, t, y, corr)), np.log10(get_rho_mix(p_last, t, y, corr))
#         S1, R1 = np.log10(get_s_mix(p_last*(1+eps), t, y, corr)), np.log10(get_rho_mix(p_last*(1+eps), t, y, corr))
#         S2, R2 = np.log10(get_s_mix(p_last, t*(1+eps), y, corr)), np.log10(get_rho_mix(p_last, t*(1+eps), y, corr))

#         DPDR = p_last*eps/(R1 - R0)
#         DSDP = p_last*eps/(S1 - S0)
#         DRDT = (R2 - R0)/(t*eps)
#         DSDT = (S2 - S0)/(t*eps)

#         DPDT = DSDT/DSDP
#         DSDR = DPDT/DSDT
#         DEN = DPDR*DSDT-DPDT*DSDR

#         #cp = (10**S0) * DEN/DPDR
#         cv = (10**S0)*(DSDT)

#         chirho.append(DPDR)
#         chit.append(DPDT)

#         c_v.append(cv)
#         #c_p.append(cp)

#     return np.array(logp_root), np.array(logrho_root), np.array(logt_root), np.array(c_v), np.array(chirho), np.array(chit)

# logpgrid = np.linspace(5.0, 14.0, 50)

# def get_rho_t(s_val, logp, y, corr=False):
#     lp, lr, lt = root_finder(s_val, y, corr)
#     interp_rho = interp1d(lp, lr, kind='linear', fill_value='extrapolate')
#     interp_t = interp1d(lp, lt, kind='linear', fill_value='extrapolate')
#     return 10**interp_rho(logp), 10**interp_t(logp)
