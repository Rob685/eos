import numpy as np
from scipy.optimize import brenth, brentq
from scipy.interpolate import RegularGridInterpolator as RGI
import scvh_thermo as scvh

logtvals = np.linspace(2.1, 5, 100)
logpvals = np.linspace(6, 15, 300)

rhoarr, sarr = np.load('inverted_eos_data/eos_data/scvh_pt.npy')
yarr = np.array([0.22, 0.25, 0.28, 0.30, 0.32, 0.35])

get_s_p_t = RGI((yarr, logtvals, logpvals), sarr, method='linear', bounds_error=False, fill_value=None)
get_rho_p_t = RGI((yarr, logtvals, logpvals), rhoarr, method='linear', bounds_error=False, fill_value=None)

def get_s_pt(p, t, y):
    return get_s_p_t(np.array([y, t, p]).T)

def get_rho_pt(p, t, y):
    return get_rho_p_t(np.array([y, t, p]).T)

def err_scvh(logt, logp, y, s_val):
    s_ = 10**float(get_s_pt(logp, logt, y)) # build a get_smix_z later
    return (s_/s_val) - 1

def get_rho_p_ideal(s, logp, m=15.5):
    # done from ideal gas
    # note: 15.5 is average molecular weight for solar comp
    # for Y = 0.25, m = 3 * (1 * 1 + 1 * 0) + (1 * 4) / 7 = 1
    p = 10**logp
    return np.log10(np.maximum(np.exp((2/5) * (5.096 - s)) * (np.maximum(p, 0) / 1e11)**(3/5) * m**(8/5), np.full_like(p, 1e-10)))

def rho_mix(p, t, y):
    rho_hhe = 10**float(get_rho_pt(p, t, y))
    #rho_z = 10**get_rho_p_ideal(s, p)
    #return np.log10(1/((1 - z)/rho_hhe + z/rho_z))
    return np.log10(rho_hhe)

def get_t(s, p, y):
    t_root = brenth(err_scvh, 2, 5, args=(p, y, s))
    return t_root

def get_rho(s, p, y, z=0):
    t = get_t(s, p, y)
    return rho_mix(p, t, y)

def get_rho_t(s, p, y, z, ideal=False):
    t = get_t(s, p, y)
    rho = rho_mix(p, t, y)
    return rho, t

# print(get_rho(6.21, 6, 0.25))
# print(float(get_rho_pt(6,2.2229707972712047, 0.25 )))
# print(get_t(6.21, 6, 0.25))
# print(10**float(get_s_pt(6, 2.2229707972712047, 0.25)))


######## derivatives ########

