import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from eos import aneos
import os

from astropy import units as u
#from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu

"""
    This module borrows from the alice repository from Christopher 
    Mankovich (https://github.com/chkvch/alice) to provide access
    to the ANEOS Serpentine EOS in the same format as all other EOSes.

    Author: Roberto Tejada Arevalo

"""

mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

aneos_ice = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='ice')

def get_s_pt_tab(p, t):
    if np.isscalar(p):
        return float((10**aneos_ice.get_logs(p, t)/11605.333))
    return 10**aneos_ice.get_logs(p, t)/11605.333

def get_rho_pt_tab(p, t):
    if np.isscalar(p):
        return float(aneos_ice.get_logrho(p, t))
    return aneos_ice.get_logrho(p, t)

def get_u_pt_tab(p, t):
    if np.isscalar(p):
        return float(aneos_ice.get_logu(p, t))
    return aneos_ice.get_logu(p, t)

