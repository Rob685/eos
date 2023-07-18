from eos import cms_newton_raphson as cms
from eos import cms_eos
import numpy as np
from scipy.optimize import root, root_scalar
erg_to_kbbar = 1.202723550011625e-08

def err_energy_2D(pt_pair, sval, uval, y):
    lgp, lgt = pt_pair
    s, logu = cms.get_s_mix(lgp, lgt, y, hc_corr=True), cms.get_logu_mix(lgp, lgt, y)
    s *= erg_to_kbbar
    return  s/sval - 1, logu/uval -1

def get_pt(s, u, y):
    sol = root(err_energy_2D, [7, 2.5], args=(s, u, y))
    return sol.x

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

def err_st_1D(lgt, lgp, sval, y):
    #lgp, lgt = pt_pair
    s = np.cms.get_s_mix(lgp, lgt, y, hc_corr=True)
    s *= erg_to_kbbar
    return  s/sval - 1
def err_sp_1D(lgp, lgt, sval, y):
    #lgp, lgt = pt_pair
    s = np.cms.get_s_mix(lgp, lgt, y, hc_corr=True)
    s *= erg_to_kbbar
    return  s/sval - 1

def err_up_1D(lgp, lgt, uval, y):
    logu = cms.get_logu_mix(lgp, lgt, y)
    return logu/uval - 1
def err_ut_1D(lgt, lgp, uval, y):
    logu = cms.get_logu_mix(lgp, lgt, y)
    return logu/uval - 1

def get_t_r(p, r, y): # doesn't work very well...
    sol = root_scalar(err_rhot_1D, bracket=[0, 7], method='brenth', args=(p, r, y))
    return sol.root
def get_t_s(s, p, y):
    sol = root_scalar(err_st_1D, bracket=[0, 7], method='brenth', args=(p, s, y))
    return sol.root
def get_t_u(u, p, y):
    sol = root_scalar(err_ut_1D, bracket=[0, 7], method='brenth', args=(p, u, y))
    return sol.root

def get_p_r(r, t, y):
    sol = root_scalar(err_rhop_1D, bracket=[5, 15], method='brenth', args=(t, r, y))
    return sol.root

def get_p_u(u, t, y):
    sol = root_scalar(err_up_1D, bracket=[5, 15], method='brenth', args=(t, u, y))
    return sol.root

def get_p_s(s, t, y):
    sol = root_scalar(err_sp_1D, bracket=[5, 15], method='brenth', args=(t, s, y))
    return sol.root

def get_s_r(r, t, y):
    p = get_p_r(r, t, y)
    s = cms.get_s_mix(p, t, y, hc_corr=True)
    return s # in cgs

def get_logu_r(r, t, y):
    p = get_p_r(r, t, y) 
    logu = float(cms.get_logu_mix(p, t, y)) 
    return logu

def err_ur_1D(lgt, lgr, uval, y):
    logu = float(get_logu_r(lgr, lgt, y))
    return logu/uval - 1

def err_sr_1D(lgt, lgr, sval, y):
    s = get_s_r(lgr, lgt, y)*erg_to_kbbar
    return s/sval - 1

def get_t_ur(u, r, y):
    sol = root_scalar(err_ur_1D, bracket=[0, 7], method='brenth', args=(r, u, y))
    return sol.root

def get_t_sr(s, r, y):
    sol = root_scalar(err_sr_1D, bracket=[0, 7], method='brenth', args=(r, s, y))
    return sol.root

def get_s_u(u, r, y):
    t = get_t_ur(u, r, y)
    return get_s_r(r, t, y)

def get_u_s(s, r, y):
    t = get_t_sr(s, r, y)
    return get_logu_r(r, t, y)

def get_u_rt(r, t, y):
    p = get_p_r(r, t, y)
    return cms.get_logu_mix(p, t, y)

def get_s_rp(r, p, y):
    t = get_t_r(p, r, y)
    s = cms.get_s_mix(p, t, y, hc_corr=True)
    return s # in cgs