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
from eos import ideal_eos, aqua_eos, ppv_eos
import pdb
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

pd.options.mode.chained_assignment = None

ideal_xy = ideal_eos.IdealHHeMix()
ideal_x = ideal_eos.IdealEOS(m=2)

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

    tab = np.loadtxt('%s/cms/DirEOS2021/%s' % (CURR_DIR, tab_name), comments='#')
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

cms_hdata = grid_data(cms_reader('TABLE_H_TP_effective'))
cms_hedata = grid_data(cms_reader('TABLE_HE_TP_v1'))

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
    return np.array([guarded_log(x_) for x_ in x])

def get_smix_id_y(Y):
    xhe = Y_to_n(Y)
    xh = 1 - xhe
    return -1*(guarded_log(xh) + guarded_log(xhe))

def get_s_pt(lgp, lgt, y, z=0.0):
    s_h = 10 ** get_s_h(lgt, lgp)
    s_he = 10 ** get_s_he(lgt, lgp)
    smix = get_smix_id_y(y)
    return (1 - y) * s_h + y * s_he + smix

def get_rho_pt(lgp, lgt, y, z=0.0):
    rho_h = 10 ** get_rho_h(lgt, lgp)
    rho_he = 10 ** get_rho_he(lgt, lgp)
    return np.log10(1/(((1 - y) / rho_h) + (y / rho_he)))

def get_u_pt(lgp, lgt, y, z=0.0):
    u_h = 10**get_logu_h(lgt, lgp) # MJ/kg to erg/g
    u_he = 10**get_logu_he(lgt, lgp)
    return np.log10((1 - y)*u_h + y*u_he)

### error functions ###

def err_p_srho(lgp, lgr, s_val, y):
    t = get_t_srho(s_val, lgr, y)
    s_ = get_s_pt(lgp, t, y)
    s_val /= erg_to_kbbar
    return (s_/s_val) - 1

def err_t_sp(logt, s_val, logp, y):
    s_ = get_s_pt(logp, logt, y)*erg_to_kbbar
    return (s_/s_val) - 1

def err_p_rhot(lgp, rhoval, lgtval, yval):
    logrho = get_rho_pt(lgp, lgtval, yval)
    return (logrho/rhoval) - 1

def err_p_srho(lgp, sval, rhoval, yval):
    logrho = get_rho_sp_tab(sval, lgp, yval) # circumvents temperature
    return (logrho/rhoval) - 1

def err_t_srho(lgt, sval, rhoval, yval):
    #logp = get_p_rhot(rhoval, lgt, yval, alg='root')
    s_test = get_s_rhot_tab(rhoval, lgt, yval)*erg_to_kbbar
    return (s_test/sval) - 1

def err_t_rhop(_lgt, _lgrho, _lgp, _y):
    logrho_test = get_rho_pt(_lgp, _lgt, _y)
    return (logrho_test/_lgrho) - 1

### inversion functions ###

TBOUNDS = [2, 7]
PBOUNDS = [0, 15]

XTOL = 1e-8
    
###### Temperature ######
def get_t_sp(s, p, y, hg=True, alg='root'):
    if alg == 'root':
        if np.isscalar(s):
            s, p, y = np.array([s]), np.array([p]), np.array([y])
            guess = ideal_xy.get_t_sp(s, p, y)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, y))
            return float(sol.x)
        guess = ideal_xy.get_t_sp(s, p, y)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, p, y)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
                return sol.root
            except:
                raise
        sol = np.array([get_t_sp(s_, p_, y_) for s_, p_, y_ in zip(s, p, y)])
        return sol

def get_t_srho(s, rho, y, alg='root'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho, y = np.array([s]), np.array([rho]), np.array([y])
            guess = ideal_xy.get_t_srho(s, rho, y)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y))
            return float(sol.x)
        guess = ideal_xy.get_t_srho(s, rho, y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho, y))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise
        sol = np.array([get_t_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
        return sol

def get_t_rhop(_lgrho, _lgp, _y, alg='root'):
    if alg == 'root':
        if np.isscalar(_lgrho):
            _lgrho, _lgp, _y = np.array([_lgrho]), np.array([_lgp]), np.array([_y])
            guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
            sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(_lgrho, _lgp, _y))
            return float(sol.x)

        guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
        sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(_lgrho, _lgp, _y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(_lgrho):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_rhop, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(_lgrho, _lgp, _y))
                return sol.root
            except:
                raise
        sol = np.array([get_t_rhop(rho_, p_, y_) for rho_, p_, y_ in zip(_lgrho, _lgp, _y)])
        return sol

###### Density ######
def get_rhot_sp(s, p, y, z=0.0):
    rho, t = get_rhot_sp_tab(s, p, y)
    return rho, t
    

###### Pressure ######
def get_p_rhot(rho, t, y, alg='root'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t, y = np.array([rho]), np.array([t]), np.array([y])
            guess = ideal_x.get_p_rhot(rho, t, y)
            sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y))
            return float(sol.x)
        guess = ideal_x.get_p_rhot(rho, t, y)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, t, y))
                return sol.root
            except:
                #print('rho={}, t={}, y={}'.format(rho, t, y))
                raise
        sol = np.array([get_p_rhot(rho_, t_, y_) for rho_, t_, y_ in zip(rho, t, y)])
        return sol

def get_p_srho(s, rho, y, alg='root'):
    logt = get_t_srho(s, rho, y, alg)
    return get_p_rhot_tab(rho, logt, y)


def get_sp_rhot(rho, t, y):
    logp = get_p_rhot_tab(rho, t, y)
    s = get_s_pt(logp, t, y)
    return s, logp

def get_pt_srho(s, rho, y):
    return get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)

def get_u_sp(s, p, y):
    t = get_t_sp(s, p, y)
    return get_u_pt(p, t, y)

def get_u_rhot(rho, t, y):
    p = get_p_rhot_tab(rho, t, y) 
    return get_u_pt(p, t, y)

def get_u_srho(s, rho, y, z=0.0):
    p, t = get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)
    return get_u_pt(p, t, y)

def get_s_rhop(rho, p, y):
    t = get_t_rhop(rho, p, y)
    #y = cms.n_to_Y(y)
    return get_s_pt(p, t, y)