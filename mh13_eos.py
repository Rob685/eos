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

def kbel_to_kbbar(X):
    # will be useful for MH13 data down the road; multiply their S values by this, given that X = 1-Y
    # ne/nb
    return (1+X)/2

def mh13_reader(tab):
    mh13_ad = np.genfromtxt(tab, dtype=None)

    out_list = []
    for i, row in enumerate(mh13_ad):
        clean_row = []
        for j, val in enumerate(row):
            if j % 2:
                clean_row.append(val)
        out_list.append(clean_row)

    eos_mh13 = pd.DataFrame(np.array(out_list), 
                            columns = ['rs','rho','T','E','P','F','S(kb/el)'])#.sort_values(['T', 'rho'])
    
    #eos_mh13 = eos_mh13[eos_mh13['S(kb/el)'] > 0.0]
    eos_mh13['S(kb/bar)'] = eos_mh13['S(kb/el)']*kbel_to_kbbar((1-0.245))
    eos_mh13['logrho'] = np.log10(eos_mh13['rho'])
    eos_mh13['logp'] = np.log10(eos_mh13['P'])+10 # in cgs
    eos_mh13['logt'] = np.log10(eos_mh13['T'])
    #eos_mh13['logs'] = np.log10(eos_mh13['S(kb/bar)']*erg_to_kbbar)#log cgs for interpolation
    return eos_mh13

def grid_data_mh13(tab):
    
    df = mh13_reader(tab)
    twoD = {}
    shape = df['logrho'].nunique(), -1
    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    
    #s_table = twoD['S(kb/bar)'] # 2-D array with entropies
    #logtvals = np.log10(twoD['T'][:, 0]) # 1-D array with unique temps
    #logrhovals = np.log10(twoD['rho'][0, :]) # 1-D array with unique densities
    
    return twoD

### Y = 0.245 mixture ###
data_mix = grid_data_mh13('%s/mh13_eos.out' % CURR_DIR)
logtvals_mix = data_mix['logt'][0]
logrhovals_mix = data_mix['logrho'][:,0]

stab_mix = data_mix['S(kb/bar)']
ptab_mix = data_mix['logp']

def get_s_rhot_mix(logt, logrho): # in cgs
    return RBS(logrhovals_mix, logtvals_mix, stab_mix).ev(logrho, logt)/erg_to_kbbar

def get_p_rhot_mix(logt, logrho):
    return RBS(logrhovals_mix, logtvals_mix, ptab_mix).ev(logrho, logt)

### helium tables ###
data_he = grid_data_mh13('%s/mh13_he_eos.out' % CURR_DIR)
logtvals_he = data_he['logt'][0]
logrhovals_he = data_he['logrho'][:,0]

stab_he = data_he['S(kb/bar)']
ptab_he = data_he['logp']

def get_s_rhot_he(logt, logrho): # in cgs
    return RBS(logrhovals_he, logtvals_he, stab_he).ev(logrho, logt)/erg_to_kbbar

def get_p_rhot_he(logt, logrho):
    return RBS(logrhovals_he, logtvals_he, ptab_he).ev(logrho, logt)

### inversions ###
def err_rhot(rt_pair, sval, pval):
    rho, t = rt_pair
    sval /= erg_to_kbbar
    s, p = get_s_rhot_mix(t, rho), get_p_rhot_mix(t, rho)
    return s/sval - 1, p/pval - 1

def err_t_rhop(lgt, rho, pval):
    logp = get_p_rhot_mix(lgt, rho)
    return logp/pval - 1

# def err_t_sp(lgt, rho, sval):
#     logp = get_p_rhot_mix(lgt, rho)
#     return logp/pval - 1

def get_rhot_sp(s, p):
    if np.isscalar(s):
        sol = root(err_rhot, [-0.5, 2.5], args=(s, p))
        return sol.x
    sol = np.array([get_rhot_sp(s_, p_) for s_, p_ in zip(s, p)])
    return sol

def get_t_rhop(rho, p):
    if np.isscalar(p):
        sol = root_scalar(err_t_rhop, bracket=[2.7, 5.06], method='brenth', args=(rho, p))
        return sol.root
    sol = np.array([get_t_rhop(r_, p_) for r_, p_ in zip(rho, p)])
    return sol

def get_s_rhop(rho, p):
    t = get_t_rhop(rho, p)
    #y = cms.n_to_Y(y)
    s = get_s_rhot_mix(rho, t)
    return s # in cgs

### entropy gradients ###

# def get_dsdy_rhop(rho, p, y, dy=0.01):
#     # P0, T0 = get_pt_srho(s, rho, y)
#     # P1, T1 = get_pt_srho(s, rho, y*(1+dy))
#     #S1 = get_s_pt(P0, T0, y)
#     #S2 = get_s_pt(P1, T1, y*(1+dy))
#     S0 = get_s_rhop(rho, p, y)
#     S1 = get_s_rhop(rho, p, y*(1+dy))

#     return (S1 - S0)/(y*dy)