import numpy as np
import pandas as pd
import os
from astropy import units as u
from astropy.constants import k_B, m_p
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
J_to_erg = (u.J / (u.kg * u.K)).to('erg/(K*g)')
J_to_cgs = (u.J/u.kg).to('erg/g')

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

logrho_rgi = RGI((logpvals, logtvals), logrhovals, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_pt(lgp, lgt):
    return s_rgi(np.array([lgp, lgt]).T)

def get_rho_pt(lgp, lgt):
    return logrho_rgi(np.array([lgp, lgt]).T)