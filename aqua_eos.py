import numpy as np
import pandas as pd
import os
from astropy import units as u
from astropy.constants import k_B, m_p
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from eos import ideal_eos

mp = amu.to('g')
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
J_to_erg = (u.J / (u.kg * u.K)).to('erg/(K*g)')
J_to_cgs = (u.J/u.kg).to('erg/g')

ideal_z = ideal_eos.IdealEOS(m=18)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def aqua_reader(basis):
    cols = ['press', 'temp', 'rho', 'grada', 's', 'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase']
    tab = np.loadtxt('%s/aqua/aqua_eos_%s_v1_0.dat' % (CURR_DIR, basis))
    tab_df = pd.DataFrame(tab, columns=cols)

    tab_df['logp'] = np.log10(tab_df['press']*10)
    tab_df['logrho'] = np.log10(tab_df['rho'])
    tab_df['logt'] = np.log10(tab_df['temp'])
    tab_df['s'] = tab_df['s']*J_to_erg
    #tab_df['u_cgs'] = tab_df['u']*J_to_cgs

    return tab_df


def grid_data(df):
    # grids data for interpolation
    twoD = {}
    shape = df['logp'].nunique(), -1
    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    return twoD

aqua_data = grid_data(aqua_reader('pt'))

logpvals = aqua_data['logp'][:,0]
logtvals = aqua_data['logt'][0]

svals = aqua_data['s']
logrhovals = aqua_data['logrho']

s_rgi = RGI((logpvals, logtvals), svals, method='linear', \
            bounds_error=False, fill_value=None)

rho_rgi = RGI((logpvals, logtvals), logrhovals, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_pt(lgp, lgt):
    return s_rgi(np.array([lgp, lgt]).T)

def get_rho_pt(lgp, lgt):
    return rho_rgi(np.array([lgp, lgt]).T)

####### Inverted Functions #######

### rho(s, p), t(s, p) ###

logrho_res_sp, logt_res_sp = np.load('%s/aqua/sp_base.npy' % CURR_DIR)

svals_sp = np.arange(0.1, 10.05, 0.05)
logpvals_sp = np.arange(5, 16, 0.1)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp), logrho_res_sp, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp), logt_res_sp, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(s, p, y):
    if np.isscalar(s):
        return float(get_rho_rgi_sp(np.array([s, p]).T))
    else:
        return get_rho_rgi_sp(np.array([s, p]).T)

def get_t_sp_tab(s, p, y):
    if np.isscalar(s):
        return float(get_t_rgi_sp(np.array([s, p]).T))
    else:
        return get_t_rgi_sp(np.array([s, p]).T)

def get_rhot_sp_tab(s, p):
    return get_rho_sp_tab(s, p), get_t_sp_tab(s, p)

### p(rho, t), s(rho, t) ###

logp_res_rhot, s_res_rhot = np.load('%s/aqua/rhot_base.npy' % CURR_DIR)

logrhovals_rhot = np.arange(-4, 4.55, 0.05)

get_p_rgi_rhot = RGI((logrhovals_rhot, logtvals), logp_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot = RGI((logrhovals_rhot, logtvals), s_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t):
    if np.isscalar(rho):
        return float(get_p_rgi_rhot(np.array([rho, t]).T))
    else:
        return get_p_rgi_rhot(np.array([rho, t]).T)

def get_s_rhot_tab(rho, t):
    if np.isscalar(rho):
        return float(get_s_rgi_rhot(np.array([rho, t]).T))
    else:
        return get_s_rgi_rhot(np.array([rho, t]).T)

### p(s, rho), T(s, rho) ###

logp_res_srho, logt_res_srho = np.load('%s/aqua/srho_base.npy' % CURR_DIR)

svals_srho = np.arange(0.1, 10.05, 0.05)
logrhovals_srho= np.arange(-4, 4.55, 0.05)

get_p_rgi_srho = RGI((svals_srho, logrhovals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhovals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_p_rgi_srho(np.array([s, r]).T))
    else:
        return get_p_rgi_srho(np.array([s, r]).T)

def get_t_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_t_rgi_srho(np.array([s, r]).T))
    else:
        return get_t_rgi_srho(np.array([s, r]).T)

####### inversion functions #######

TBOUNDS = [2, 5]
PBOUNDS = [0, 16]
XTOL = 1e-8

def err_t_sp(logt, s_val, logp):
    #print(logt, logp, s_val, y, z)
    s_ = get_s_pt(logp, logt)*erg_to_kbbar
    #s_val /= erg_to_kbbar # in cgs

    return (s_/s_val) - 1

def err_p_rhot(lgp, rhoval, lgtval):
    #if zval > 0.0:
    logrho = get_rho_pt(lgp, lgtval)
    #pdb.set_trace()
    return float(logrho/rhoval) - 1

def err_t_srho(lgt, sval, rhoval):
    lgp = get_p_rhot(rhoval, lgt)
    s_ = get_s_pt(lgp, lgt)*erg_to_kbbar
    return  s_/sval - 1

def get_p_rhot(rho, t ,alg='brenth'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t = np.array([rho]), np.array([t])
        guess = ideal_z.get_p_rhot(rho, t, 0)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, t))
                return sol.root
            except:
                #print('rho={}, t={}, y={}'.format(rho, t, y))
                raise
        sol = np.array([get_p_rhot(rho_, t_) for rho_, t_ in zip(rho, t)])
        return sol

def get_t_sp(s, p, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, p = np.array([s]), np.array([p])
        guess = ideal_z.get_t_sp(s, p, 0)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, p)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
                return sol.root
            except:
                #print('s={}, p={}, y={}'.format(s, p, y))
                raise
        sol = np.array([get_t_sp(s_, p_) for s_, p_ in zip(s, p)])
        return sol

def get_t_srho(s, rho, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho = np.array([s]), np.array([rho])
        guess = ideal_z.get_t_srho(s, rho)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise
        sol = np.array([get_t_srho(s_, rho_) for s_, rho_ in zip(s, rho)])
        return sol


############## derivatives ##############

def get_dtdrho_srho(s, rho, drho=0.01):
    R0 = 10**rho
    R1 = R0*(1+drho)
    T0 = 10**get_t_srho_tab(s, np.log10(R0))
    T1 = 10**get_t_srho_tab(s, np.log10(R1))

    return (T1 - T0)/(R1 - R0)

def get_dtds_srho(s, rho, ds=0.01):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho)

    return (T1 - T0)/(S1 - S0)
