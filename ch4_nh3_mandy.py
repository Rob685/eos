import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
from tqdm import tqdm

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
erg_to_kJ = (u.erg/u.gram).to('kJ/g')

# np.log10((u.MJ/u.kg).to('erg/g'))

data_methane = np.load('eos/methane_ammonia/methane_eos_rhot.npz')
data_ammonia = np.load('eos/methane_ammonia/ammonia_eos_rhot.npz')

rgi_args = {'method': 'linear', 'bounds_error': False, 'fill_value': None}

u_rhot_rgi_methane = RGI((data_methane['T'][:,0], data_methane['rho'][0]), data_methane['u_cgs'], **rgi_args)
p_rhot_rgi_methane = RGI((data_methane['T'][:,0], data_methane['rho'][0]), data_methane['p_cgs'], **rgi_args)
s_rhot_rgi_methane = RGI((data_methane['T'][:,0], data_methane['rho'][0]), data_methane['s_cgs'], **rgi_args)
a_rhot_rgi_methane = RGI((data_methane['T'][:,0], data_methane['rho'][0]), data_methane['a_cgs'], **rgi_args)

u_rhot_rgi_ammonia = RGI((data_ammonia['T'][:,0], data_ammonia['rho'][0]), data_ammonia['u_cgs'], **rgi_args)
p_rhot_rgi_ammonia = RGI((data_ammonia['T'][:,0], data_ammonia['rho'][0]), data_ammonia['p_cgs'], **rgi_args)
s_rhot_rgi_ammonia = RGI((data_ammonia['T'][:,0], data_ammonia['rho'][0]), data_ammonia['s_cgs'], **rgi_args)
a_rhot_rgi_ammonia = RGI((data_ammonia['T'][:,0], data_ammonia['rho'][0]), data_ammonia['a_cgs'], **rgi_args)

def get_p_rhot(_rho, _T, molecule='methane'):

    args = (_T, _rho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    if molecule=='methane':
        result = p_rhot_rgi_methane(pts)
    elif molecule=='ammonia':
        result = p_rhot_rgi_ammonia(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_rhot(_rho, _T, molecule='methane'):

    args = (_T, _rho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    if molecule=='methane':
        result = s_rhot_rgi_methane(pts)
    elif molecule=='ammonia':
        result = s_rhot_rgi_ammonia(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_u_rhot(_rho, _T, molecule='methane'):

    args = (_T, _rho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    if molecule=='methane':
        result = u_rhot_rgi_methane(pts)
    elif molecule=='ammonia':
        result = u_rhot_rgi_ammonia(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_a_rhot(_rho, _T, molecule='methane'):

    args = (_T, _rho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    if molecule=='methane':
        result = a_rhot_rgi_methane(pts)
    elif molecule=='ammonia':
        result = a_rhot_rgi_ammonia(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result
