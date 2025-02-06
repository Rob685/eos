from eos import ideal_eos
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
from tqdm import tqdm

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
J_to_erg = (u.J / (u.kg * u.K)).to('erg/(K*g)')
J_to_kbbar = (u.J / (u.kg * u.K)).to(k_B/mp)

mg = 24.305
si = 28.085
o3 = 48.000

mgsio3 = mg+si+o3 # molecular weight of post-perovskite

# for guesses
ideal_z = ideal_eos.IdealEOS(m=mgsio3)

### S, P ###
s_grid = np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/s_grid.txt')*J_to_kbbar
logpgrid = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/P_grid.txt')*10)

logtvals = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/T_table_ppv.txt'))
logrhovals = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/rho_table_ppv.txt')*1e-3)
loguvals = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/E_table_ppv.txt')*J_to_erg)

rho_rgi = RGI((s_grid, logpgrid), logrhovals, method='linear', bounds_error=False, fill_value=None)
t_rgi = RGI((s_grid, logpgrid), logtvals, method='linear', bounds_error=False, fill_value=None)
u_rgi = RGI((s_grid, logpgrid), loguvals, method='linear', bounds_error=False, fill_value=None)

### P, T ###

pt_data = np.load('eos/zhang_eos/zhang_multiphase/ppv2/zhang_ppv_2024_pt.npz')

logpvals_pt = pt_data['logpvals'] # log g/cm^3
logtvals_pt = pt_data['logtvals'] # log K
logrho_grid_pt = pt_data['logrhovals'] # in dyn/cm2
s_grid_pt = pt_data['svals'] # in erg/g/K
logu_grid_pt = pt_data['loguvals'] # in erg/g

rho_rgi_pt = RGI((logpvals_pt, logtvals_pt), logrho_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
s_rgi_pt = RGI((logpvals_pt, logtvals_pt), s_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
u_rgi_pt = RGI((logpvals_pt, logtvals_pt), logu_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)

def get_logrho_pt_tab(_lgp, _lgt): 
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt_tab(_lgp, _lgt): # returns in erg/g/K
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_pt_tab(_lgp, _lgt): # returns in log10 erg/g
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_sp_tab(_s, _lgp):
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_rgi(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result
def get_logt_sp_tab(_s, _lgp):
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = t_rgi(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result
        
def get_logu_sp_tab(_s, _lgp):
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_rgi(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result