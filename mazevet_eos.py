import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

from eos import ideal_eos

ideal_z = ideal_eos.IdealEOS(m=18) # ideal EOS for water

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

rhot_data = np.load('eos/mazevet_eos/mazevet_water_2021_rhot.npz')
pt_data = np.load('eos/mazevet_eos/mazevet_water_2021_pt.npz')

pressure_grid = rhot_data['pvals'] # in dyn/cm2
entropy_grid = rhot_data[]