import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
import const
import pdb
import pandas as pd
from tqdm import tqdm
from numba import njit
from eos import ideal_eos, metals_eos
from scipy.optimize import root, newton, brentq, brenth, minimize
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ideal_xy = ideal_eos.IdealHHeMix()

mh = 1
mhe = 4.0026

##### useful unit conversions #####

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

log10_to_loge = np.log(10)

class hhe:
    def __init__(self, hhe_eos):
        self.hhe_eos = hhe_eos

        if self.hhe_eos == 'cms':
            self.hdata = self.grid_data(self.table_reader('TABLE_H_TP_v1'))
        elif self.hhe_eos == 'cd':
            self.hdata = self.grid_data(self.table_reader('TABLE_H_TP_effective'))

        self.hedata = self.grid_data(self.table_reader('TABLE_HE_TP_v1'))   

        self.logpvals = self.hdata['logp'][0]
        self.logtvals = self.hdata['logt'][:,0]

        self.svals_h = self.hdata['logs']
        self.logrhovals_h = self.hdata['logrho']
        self.loguvals_h = self.hdata['logu']

        self.svals_he = self.hedata['logs']
        self.logrhovals_he = self.hedata['logrho']
        self.loguvals_he = self.hedata['logu']

        self.data_hc = pd.read_csv(f'{CURR_DIR}/cms/HG23_Vmix_Smix_Umix.csv', delimiter=',')
        self.data_hc = self.data_hc[(self.data_hc['LOGT'] <= 6.0) & (self.data_hc['LOGT'] != 2.8)].copy()
        self.data_hc = self.data_hc.rename(columns={'LOGT': 'logt', 'LOGP': 'logp'}).sort_values(by=['logt', 'logp'])

        self.grid_hc = self.grid_data(self.data_hc)
        self.svals_hc = self.grid_hc['Smix']
        self.uvals_hc = self.grid_hc['Umix']

        self.logpvals_hc = self.grid_hc['logp'][0]
        self.logtvals_hc = self.grid_hc['logt'][:,0]

        #### H data ####

        self.get_s_h_rgi = RGI((self.logtvals, self.logpvals), self.svals_h, method='linear', bounds_error=False, fill_value=None)
        self.get_logrho_h_rgi = RGI((self.logtvals, self.logpvals), self.logrhovals_h, method='linear', bounds_error=False, fill_value=None)
        self.get_logu_h_rgi = RGI((self.logtvals, self.logpvals), self.loguvals_h, method='linear', bounds_error=False, fill_value=None)

        #### He data ####

        self.get_s_he_rgi = RGI((self.logtvals, self.logpvals), self.svals_he, method='linear', bounds_error=False, fill_value=None)
        self.get_logrho_he_rgi = RGI((self.logtvals, self.logpvals), self.logrhovals_he, method='linear', bounds_error=False, fill_value=None)
        self.get_logu_he_rgi = RGI((self.logtvals, self.logpvals), self.loguvals_he, method='linear', bounds_error=False, fill_value=None)


        #### Non-ideal mixing terms for VAL ####
        self.smix_interp_rgi = RGI((self.logtvals_hc, self.logpvals_hc), self.grid_hc['Smix'], method='linear', bounds_error=False, fill_value=None) # Smix will be in cgs... not log cgs.
        self.vmix_interp_rgi = RGI((self.logtvals_hc, self.logpvals_hc), self.grid_hc['Vmix'], method='linear', bounds_error=False, fill_value=None)
        self.umix_interp_rgi = RGI((self.logtvals_hc, self.logpvals_hc), self.grid_hc['Umix'], method='linear', bounds_error=False, fill_value=None)


    def table_reader(self, tab_name):

        cols = ['logt', 'logp', 'logrho', 'logu', 'logs', 'dlrho/dlT_P', 'dlrho/dlP_T',
                'dlS/dlT_P', 'dlS/dlP_T', 'grad_ad']

        if self.hhe_eos == 'cms':
            tab = np.loadtxt(f'{CURR_DIR}/cms/DirEOS2019/{tab_name}', comments='#')
        elif self.hhe_eos == 'cd':
            tab = np.loadtxt(f'{CURR_DIR}/cms/DirEOS2021/{tab_name}', comments='#')
        else:
            raise ValueError(f"Invalid hhe_eos value '{self.hhe_eos}'. Only 'cms' and 'cd' are supported.")

        tab_df = pd.DataFrame(tab, columns=cols)
        # Explicitly make a copy of the slice
        data = tab_df[(tab_df['logt'] <= 6) & (tab_df['logt'] != 2.8)].copy()

        data['logp'] += 10  # 1 GPa = 1e10 cgs
        data['logu'] += 10  # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g
        data['logs'] += 10  # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g

        return data

    def grid_data(self, df):
    # grids data for interpolation
        twoD = {}
        shape = df['logt'].nunique(), -1
        for i in df.keys():
            twoD[i] = np.reshape(np.array(df[i]), shape)
        return twoD

    
    def _interpolate(self, interpolator, _lgp, _lgt):
        args = (_lgt, _lgp)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = interpolator(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def smix_interp(self, _lgp, _lgt):
        return self._interpolate(self.smix_interp_rgi, _lgp, _lgt)

    def vmix_interp(self, _lgp, _lgt):
        return self._interpolate(self.vmix_interp_rgi, _lgp, _lgt)

    def umix_interp(self, _lgp, _lgt):
        return self._interpolate(self.umix_interp_rgi, _lgp, _lgt)

    def get_s_h(self, _lgp, _lgt):
        return self._interpolate(self.get_s_h_rgi, _lgp, _lgt)

    def get_logrho_h(self, _lgp, _lgt):
        return self._interpolate(self.get_logrho_h_rgi, _lgp, _lgt)

    def get_logu_h(self, _lgp, _lgt):
        return self._interpolate(self.get_logu_h_rgi, _lgp, _lgt)

    def get_s_he(self, _lgp, _lgt):
        return self._interpolate(self.get_s_he_rgi, _lgp, _lgt)

    def get_logrho_he(self, _lgp, _lgt):
        return self._interpolate(self.get_logrho_he_rgi, _lgp, _lgt)

    def get_logu_he(self, _lgp, _lgt):
        return self._interpolate(self.get_logu_he_rgi, _lgp, _lgt)


class mixtures(hhe):
    def __init__(self, hhe_eos, z_eos, hg=True, y_prime=False, interp_method='linear'):

        super().__init__(hhe_eos=hhe_eos)
        
        self.y_prime = y_prime
        self.hg = hg
        self.z_eos = z_eos
        self.interp_method = interp_method

        if hhe_eos == 'cms':
            if self.hg:
                self.pt_data = np.load('eos/cms/{}_hg_{}_pt_compressed.npz'.format(hhe_eos, z_eos))
                self.rhot_data = np.load('eos/cms/{}_hg_{}_rhot.npz'.format(hhe_eos, z_eos))
                self.sp_data = np.load('eos/cms/{}_hg_{}_sp.npz'.format(hhe_eos, z_eos))
                self.rhop_data = np.load('eos/cms/{}_hg_{}_rhop.npz'.format(hhe_eos, z_eos))
                self.srho_data = np.load('eos/cms/{}_hg_{}_srho.npz'.format(hhe_eos, z_eos))
            else:
                self.pt_data = np.load('eos/cms/{}_{}_pt_compressed.npz'.format(hhe_eos, z_eos))
                self.rhot_data = np.load('eos/cms/{}_{}_rhot.npz'.format(hhe_eos, z_eos))
                self.sp_data = np.load('eos/cms/{}_{}_sp.npz'.format(hhe_eos, z_eos))
                self.rhop_data = np.load('eos/cms/{}_{}_rhop.npz'.format(hhe_eos, z_eos))
                self.srho_data = np.load('eos/cms/{}_{}_srho.npz'.format(hhe_eos, z_eos))
        else:
            self.pt_data = np.load('eos/{}/{}_{}_pt_compressed.npz'.format(hhe_eos, hhe_eos, z_eos))
            self.rhot_data = np.load('eos/{}/{}_{}_rhot.npz'.format(hhe_eos, hhe_eos, z_eos))
            self.sp_data = np.load('eos/{}/{}_{}_sp.npz'.format(hhe_eos, hhe_eos, z_eos))
            self.rhop_data = np.load('eos/{}/{}_{}_rhop.npz'.format(hhe_eos, hhe_eos, z_eos))
            self.srho_data = np.load('eos/{}/{}_{}_srho.npz'.format(hhe_eos, hhe_eos, z_eos))
    
        # 1-D independent grids
        self.logpvals = self.pt_data['logpvals'] # these are shared. Units: log10 dyn/cm^2
        self.logpvals_sp = self.sp_data['logpvals']
        self.logpvals_rhop = self.rhop_data['logpvals']

        self.logtvals = self.pt_data['logtvals'] # log10 K
        self.logtvals_rhot = self.rhot_data['logtvals']

        self.logrhovals_rhot = self.rhot_data['logrhovals'] # log10 g/cc
        self.logrhovals_rhop = self.rhop_data['logrhovals'] # log10 g/cc -- rho, P table range
        self.logrhovals_srho = self.srho_data['logrhovals'] # log10 g/cc -- rho, P table range

        self.svals_sp = self.sp_data['s_vals'] # kb/baryon
        self.svals_srho = self.srho_data['s_vals'] # kb/baryon

        self.yvals_pt = self.pt_data['yvals'] # mass fraction -- yprime
        self.zvals_pt = self.pt_data['zvals'] # mass fraction

        self.yvals_rhot = self.rhot_data['yvals']
        self.zvals_rhot = self.rhot_data['zvals']

        self.yvals_sp = self.sp_data['yvals']
        self.zvals_sp = self.sp_data['zvals']

        self.yvals_rhop = self.rhop_data['yvals']
        self.zvals_rhop = self.rhop_data['zvals']

        self.yvals_srho = self.srho_data['yvals']
        self.zvals_srho = self.srho_data['zvals']

        # 4-D dependent grids
        self.s_pt_tab = self.pt_data['s_pt'] # erg/g/K
        self.logrho_pt_tab = self.pt_data['logrho_pt'] # log10 g/cc
        self.logu_pt_tab = self.pt_data['logu_pt'] # log10 erg/g

        self.s_rhot_tab = self.rhot_data['s_rhot'] # erg/g/K
        self.logp_rhot_tab = self.rhot_data['logp_rhot']

        self.logt_sp_tab = self.sp_data['logt_sp']
        self.logrho_sp_tab = self.sp_data['logrho_sp']

        self.s_rhop_tab = self.rhop_data['s_rhop'] # erg/g/K
        self.logt_rhop_tab = self.rhop_data['logt_rhop']

        self.logp_srho_tab = self.srho_data['logp_srho']
        self.logt_srho_tab = self.srho_data['logt_srho']

        # RGI interpolation functions
        rgi_args = {'method': self.interp_method, 'bounds_error': False, 'fill_value': None}

        self.s_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt), self.s_pt_tab, **rgi_args)
        self.logrho_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt), self.logrho_pt_tab, **rgi_args)
        self.logu_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt), self.logu_pt_tab, **rgi_args)

        self.s_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot, self.yvals_rhot, self.zvals_rhot), self.s_rhot_tab, **rgi_args)
        self.logp_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot, self.yvals_rhot, self.zvals_rhot), self.logp_rhot_tab, **rgi_args)

        # self.logt_sp_rgi = RGI((self.svals_sp, self.logpvals_sp, self.yvals_sp, self.zvals_sp), self.logt_sp_tab, **rgi_args)
        # self.logrho_sp_rgi = RGI((self.svals_sp, self.logpvals_sp, self.yvals_sp, self.zvals_sp), self.logrho_sp_tab, **rgi_args)

        # self.s_rhop_rgi = RGI((self.logrhovals_rhop, self.logpvals_rhop, self.yvals_rhop, self.zvals_rhop), self.s_rhop_tab, **rgi_args)
        # self.logt_rhop_rgi = RGI((self.logrhovals_rhop, self.logpvals_rhop, self.yvals_rhop, self.zvals_rhop), self.logt_rhop_tab, **rgi_args)

        # self.logp_srho_rgi = RGI((self.svals_srho, self.logrhovals_srho, self.yvals_srho, self.zvals_srho), self.logp_srho_tab, **rgi_args)
        # self.logt_srho_rgi = RGI((self.svals_srho, self.logrhovals_srho, self.yvals_srho, self.zvals_srho), self.logt_srho_tab, **rgi_args)


    def Y_to_n(self, _y):
        ''' Change between mass and number fraction OF HELIUM'''
        return ((_y/mhe)/(((1 - _y)/mh) + (_y/mhe)))

    def n_to_Y(self, x):
        ''' Change between number and mass fraction OF HELIUM'''
        return (mhe * x)/(1 + 3.0026*x)

    def x_H(self, _y, _z, mz):
        yeff = _y#/(1 - _z)
        Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
        return (1-yeff)*(1-_z)/mh/Ntot

    def x_Z(self, _y, _z, mz):
        yeff = _y#/(1 - _z)
        Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
        return (_z/mz)/Ntot

    def guarded_log(self, x):
        ''' Used to calculate ideal enetropy of mixing: xlogx'''
        if np.isscalar(x):
            if x == 0:
                return 0
            elif x  < 0:
                raise ValueError('Number fraction went negative.')
            return x * np.log(x)
        return np.array([self.guarded_log(x_) for x_ in x])

    def get_smix_id_y(self, Y):
        #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
        xhe = self.Y_to_n(Y)
        xh = 1 - xhe
        q = mh*xh + mhe*xhe
        return -1*(self.guarded_log(xh) + self.guarded_log(xhe)) / q

    def get_smix_id_yz(self, Y, Z, mz):
        #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
        xh = self.x_H(Y, Z, mz)
        xz = self.x_Z(Y, Z, mz)
        xhe = 1 - xh - xz
        q = mh*xh + mhe*xhe + mz*xz
        return -1*(self.guarded_log(xh) + self.guarded_log(xhe) + self.guarded_log(xz)) / q


    ####### Volume-Addition Law #######


    # def get_s_pt(self, _lgp, _lgt, _y_prime, _z):

    #     """
    #     This calculates the entropy for a metallicity mixture using the volume addition law.
    #     These terms contain the ideal entropy of mixing, so
    #     for metal mixures, we subtract the H-He ideal entropy of mixing and
    #     add back the metal mixture entropy of mixing plus the non-ideal
    #     correction from Howard & Guillot (2023a).

    #     The _y_prime parameter is the Y in a pure H-He EOS. Therefore, it
    #     is Y/(1 - Z). So the y value that should be
    #     used to calculate the entropy of mixing should be Y*(1 - Z).
    #     """

    #     _y = _y_prime*(1 - _z)

    #     if (
    #         (np.isscalar(_y_prime) and _y_prime > 1.0)
    #         or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
    #         or (np.isscalar(_z) and _z > 1.0)
    #         or ((not np.isscalar(_z)) and np.any(_z > 1.0))
    #     ):
    #         raise Exception('Invalid mass fractions: X + Y + Z > 1.')

    #     smix_xy_ideal =  self.get_smix_id_y(_y_prime) / erg_to_kbbar 
    #     if self.hg:
    #         smix_xy_nonideal =  self.smix_interp(_lgp, _lgt)*(1 - _y_prime)*_y_prime - smix_xy_ideal if self.hhe_eos == 'cms' else 0.0
    #     else: 
    #         smix_xy_nonideal = 0.0

    #     s_x = 10**self.get_s_h(_lgp, _lgt)
    #     s_y = 10**self.get_s_he(_lgp, _lgt)
    #     s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos)

    #     if self.z_eos == 'aqua': 
    #         mz = 18.015
    #     elif self.z_eos == 'ppv': 
    #         mz = 100.3887
    #     elif self.z_eos == 'iron': 
    #         mz = 55.845

    #     else: 
    #         raise ValueError('Only aqua and ppv supported for now.')

    #     smix_xyz_ideal = self.get_smix_id_yz(_y, _z, mz) / erg_to_kbbar

    #     return s_x * (1 - _y_prime) * (1 - _z) + s_y * _y_prime * (1 - _z) + s_z * _z + smix_xyz_ideal + smix_xy_nonideal*(1 - _z)

    def get_s_pt(self, _lgp, _lgt, _y_prime, _z):
        """
        This calculates the entropy for a metallicity mixture using the volume addition law.
        These terms contain the ideal entropy of mixing, so
        for metal mixtures, we subtract the H-He ideal entropy of mixing and
        add back the metal mixture entropy of mixing plus the non-ideal
        correction from Howard & Guillot (2023a).

        The _y_prime parameter is the Y in a pure H-He EOS. Therefore, it
        is Y/(1 - Z). So the y value that should be
        used to calculate the entropy of mixing should be Y*(1 - Z).
        """

        def validate_mass_fractions(_y_prime, _z):
            if (
                (np.isscalar(_y_prime) and _y_prime > 1.0)
                or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
                or (np.isscalar(_z) and _z > 1.0)
                or ((not np.isscalar(_z)) and np.any(_z > 1.0))
            ):
                raise ValueError('Invalid mass fractions: X + Y + Z > 1.')

        def get_mz(z_eos):
            if z_eos == 'aqua':
                return 18.015
            elif z_eos == 'ppv':
                return 100.3887
            elif z_eos == 'iron':
                return 55.845
            else:
                raise ValueError('Only aqua, ppv, and iron supported for now.')

        _y = _y_prime * (1 - _z)
        validate_mass_fractions(_y_prime, _z)

        smix_xy_ideal = self.get_smix_id_y(_y_prime) / erg_to_kbbar
        smix_xy_nonideal = 0.0
        if self.hg:
            if self.hhe_eos == 'cms':
                smix_xy_nonideal = self.smix_interp(_lgp, _lgt) * (1 - _y_prime) * _y_prime - smix_xy_ideal

        s_x = 10 ** self.get_s_h(_lgp, _lgt)
        s_y = 10 ** self.get_s_he(_lgp, _lgt)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos)

        mz = get_mz(self.z_eos)
        smix_xyz_ideal = self.get_smix_id_yz(_y, _z, mz) / erg_to_kbbar

        return (
            s_x * (1 - _y_prime) * (1 - _z)
            + s_y * _y_prime * (1 - _z)
            + s_z * _z
            + smix_xyz_ideal
            + smix_xy_nonideal * (1 - _z)
        )

    # def get_logrho_pt(self, _lgp, _lgt, _y_prime, _z):

    #     """
    #     This function calculates the density of a H-He-Z mixture using the volume addition law.
    #     When including the non-ideal corrections, this function adds the volume of mixing from Howard & Guillot (2023a)
    #     """

    #     _y = _y_prime*(1 - _z)

    #     if (
    #         (np.isscalar(_y_prime) and _y_prime > 1.0)
    #         or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
    #         or (np.isscalar(_z) and _z > 1.0)
    #         or ((not np.isscalar(_z)) and np.any(_z > 1.0))
    #     ):
    #         raise Exception('Invalid mass fractions: X + Y + Z > 1.')
        
    #     if self.hg:
    #         vmix = self.vmix_interp(_lgp, _lgt)*(1 - _y_prime)*_y_prime if self.hhe_eos=='cms' else 0.0
    #     else:
    #         vmix = 0.0

    #     rho_h = 10**self.get_logrho_h(_lgp, _lgt)
    #     rho_he = 10**self.get_logrho_he(_lgp, _lgt)
    #     rho_z = 10**metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos)

    #     return np.log10(1/(((1 - _y_prime) * (1 - _z) / rho_h) + (_y_prime * (1 - _z) / rho_he) + vmix*(1 - _z) + _z/rho_z))

    def get_logrho_pt(self, _lgp, _lgt, _y_prime, _z):
        """
        This function calculates the density of a H-He-Z mixture using the volume addition law.
        When including the non-ideal corrections, this function adds the volume of mixing from Howard & Guillot (2023a).

        Parameters:
            _lgp (float): Logarithm of pressure.
            _lgt (float): Logarithm of temperature.
            _y_prime (float): Helium mass fraction in a pure H-He EOS.
            _z (float): Metallicity.

        Returns:
            float: Logarithm of the density.
        """

        def validate_mass_fractions(_y_prime, _z):
            if (
                (np.isscalar(_y_prime) and _y_prime > 1.0)
                or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
                or (np.isscalar(_z) and _z > 1.0)
                or ((not np.isscalar(_z)) and np.any(_z > 1.0))
            ):
                raise ValueError('Invalid mass fractions: X + Y + Z > 1.')

        def calculate_vmix(_lgp, _lgt, _y_prime):
            if self.hg and self.hhe_eos == 'cms':
                return self.vmix_interp(_lgp, _lgt) * (1 - _y_prime) * _y_prime
            return 0.0

        _y = _y_prime * (1 - _z)
        validate_mass_fractions(_y_prime, _z)

        vmix = calculate_vmix(_lgp, _lgt, _y_prime)

        rho_h = 10 ** self.get_logrho_h(_lgp, _lgt)
        rho_he = 10 ** self.get_logrho_he(_lgp, _lgt)
        rho_z = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos)

        mixture_density = (1 - _y_prime) * (1 - _z) / rho_h + _y_prime * (1 - _z) / rho_he + vmix * (1 - _z) + _z / rho_z

        return np.log10(1 / mixture_density)

    # def get_logu_pt(self, _lgp, _lgt, _y_prime, _z):

    #     """
    #     This function calculates the internal energy per unit mass of a H-He-Z mixture using the volume addition law.
    #     When including the non-ideal corrections, this function adds the volume of mixing from Howard & Guillot (2023a).    
    #     """
        
    #     if self.hg:
    #         umix = self.umix_interp(_lgp, _lgt) * (1 - _y_prime) * _y_prime if self.hhe_eos=='cms' else 0.0
    #     else:
    #         umix = 0.0

    #     u_h = 10**self.get_logu_h(_lgp, _lgt)
    #     u_he = 10**self.get_logu_he(_lgp, _lgt)
    #     u_z = 10**metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos)

    #     return np.log10(u_h * (1 - _y_prime) * (1 - _z) + u_he * _y_prime * (1 - _z) + umix * (1 - _z) + u_z * _z)

    def get_logu_pt(self, _lgp, _lgt, _y_prime, _z):
        """
        This function calculates the internal energy per unit mass of a H-He-Z mixture using the volume addition law.
        When including the non-ideal corrections, this function adds the volume of mixing from Howard & Guillot (2023a).

        Parameters:
            _lgp (float): Logarithm of pressure.
            _lgt (float): Logarithm of temperature.
            _y_prime (float): Helium mass fraction in a pure H-He EOS.
            _z (float): Metallicity.

        Returns:
            float: Logarithm of the internal energy per unit mass.
        """

        def calculate_umix(_lgp, _lgt, _y_prime):
            if self.hg and self.hhe_eos == 'cms':
                return self.umix_interp(_lgp, _lgt) * (1 - _y_prime) * _y_prime
            return 0.0

        umix = calculate_umix(_lgp, _lgt, _y_prime)

        u_h = 10 ** self.get_logu_h(_lgp, _lgt)
        u_he = 10 ** self.get_logu_he(_lgp, _lgt)
        u_z = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos)

        mixture_energy = (
            u_h * (1 - _y_prime) * (1 - _z)
            + u_he * _y_prime * (1 - _z)
            + umix * (1 - _z)
            + u_z * _z
        )

        return np.log10(mixture_energy)


    ####### EOS table calls #######

    # logp, logt tables
    def get_s_pt_tab(self, _lgp, _lgt, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.s_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logrho_pt_tab(self, _lgp, _lgt, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logrho_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logu_pt_tab(self, _lgp, _lgt, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)
        
        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logu_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # logrho, logt tables
    def get_s_rhot_tab(self, _lgrho, _lgt, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_lgrho, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.s_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logp_rhot_tab(self, _lgrho, _lgt, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_lgrho, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logp_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # S, logp tables
    def get_logt_sp_tab(self, _s, _lgp, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_s, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_sp_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logrho_sp_tab(self, _s, _lgp, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_s, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logrho_sp_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # logrho, logp tables

    def get_logt_rhop_tab(self, _lgrho, _lgp, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_lgrho, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_rhop_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_s_rhop_tab(self, _lgrho, _lgp, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_lgrho, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.s_rhop_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result
        r#eturn self.get_s_pt_tab(_lgp, self.get_logt_rhop_tab(*args), _y, _z)

    # S, logrho tables
    def get_logp_srho_tab(self, _s, _lgrho, _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_s, _lgrho, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logp_srho_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logt_srho_tab(self, _s, _lgrho,  _y, _z):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_s, _lgrho, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_srho_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result


    ### Inversion Functions ###

    def get_logt_sp_inv(self, _s, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the temperature given entropy, pressure, helium abundance, and metallicity.
    
        Parameters:
            _s (array_like): Entropy values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
    
        _s = np.atleast_1d(_s)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # _y = _y if self.y_prime else _y * (1 - _z)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgp, _y, _z = np.broadcast_arrays(_s, _lgp, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_t_sp(_s, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            guess = arr_guess

    # Define a function to compute root and capture convergence
        def root_func(s_i, lgp_i, y_i, z_i, guess_i):
            def err(_lgt):
                # Error function for logt(S, logp)
                s_test = self.get_s_pt_tab(lgp_i, _lgt, y_i, z_i) * erg_to_kbbar
                return (s_test/s_i) - 1

            if method == 'root':
                sol = root(err, guess_i, tol=1e-8)
                if sol.success:
                    return sol.x[0], True
                else:
                    return np.nan, False  # Assign np.nan to non-converged elements

            elif method == 'newton':
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    #Convergence failed
                    return np.nan, False
                except Exception as e:
                    #Handle other exceptions
                    return np.nan, False

            elif method == 'brentq':
                # Define an initial interval around the guess
                delta = 0.1  # Initial interval half-width
                a = guess_i - delta
                b = guess_i + delta

                # Try to find a valid interval where the function changes sign
                max_attempts = 5
                factor = 2.0  # Factor to expand the interval if needed

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")

                        if fa * fb < 0:
                            # Valid interval found
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            # Expand the interval and try again
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor  # Increase delta for next iteration
                    except ValueError:
                        # If err() cannot be evaluated, expand the interval
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor

                # If no valid interval is found after max_attempts
                return np.nan, False

            elif method == 'newton_brentq':
                # Try the Newton method first
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    # Fall back to the Brentq method if Newton fails
                    delta = 0.1
                    a = guess_i - delta
                    b = guess_i + delta
                    max_attempts = 5
                    factor = 2.0

                    for attempt in range(max_attempts):
                        try:
                            fa = err(a)
                            fb = err(b)
                            if np.isnan(fa) or np.isnan(fb):
                                raise ValueError("Function returned NaN.")
                            if fa * fb < 0:
                                sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                                return sol_root, True
                            else:
                                a -= delta * factor
                                b += delta * factor
                                delta *= factor
                        except ValueError:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    return np.nan, False

                except OverflowError:
                    print('Failed at s={}, logp={}, y={}, z={}'.format(s_i, lgp_i, y_i, z_i))
                    raise
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgp, _y, _z, guess)

        return temperatures, converged

    def get_logrho_sp_inv(self, _s, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        logt, conv = self.get_logt_sp_inv( _s, _lgp, _y, _z, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        return self.get_logrho_pt_tab(_lgp, logt, _y, _z)

    def get_logp_rhot_inv(self, _lgrho, _lgt, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the pressure given density, temperature, helium abundance, and metallicity.
    
        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgt (array_like): Log10 temperature values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
    
        _lgrho = np.atleast_1d(_lgrho)
        _lgt = np.atleast_1d(_lgt)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgt, _y, _z = np.broadcast_arrays(_lgrho, _lgt, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
       # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgt_i, y_i, z_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                _y_call = y_i if self.y_prime else y_i / (1 - z_i)
                logrho_test = self.get_logrho_pt_tab(_lgp, lgt_i, _y_call, z_i)
                return (logrho_test/lgrho_i) - 1

            if method == 'root':
                sol = root(err, guess_i, tol=1e-8)
                if sol.success:
                    return sol.x[0], True
                else:
                    return np.nan, False  # Assign np.nan to non-converged elements

            elif method == 'newton':
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    #Convergence failed
                    return np.nan, False
                except Exception as e:
                    #Handle other exceptions
                    return np.nan, False

            elif method == 'brentq':
                # Define an initial interval around the guess
                delta = 0.1  # Initial interval half-width
                a = guess_i - delta
                b = guess_i + delta

                # Try to find a valid interval where the function changes sign
                max_attempts = 5
                factor = 2.0  # Factor to expand the interval if needed

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")

                        if fa * fb < 0:
                            # Valid interval found
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            # Expand the interval and try again
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor  # Increase delta for next iteration
                    except ValueError:
                        # If err() cannot be evaluated, expand the interval
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor

            elif method == 'newton_brentq':
                # Try the Newton method first
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    # Fall back to the Brentq method if Newton fails
                    delta = 0.1
                    a = guess_i - delta
                    b = guess_i + delta
                    max_attempts = 5
                    factor = 2.0

                    for attempt in range(max_attempts):
                        try:
                            fa = err(a)
                            fb = err(b)
                            if np.isnan(fa) or np.isnan(fb):
                                raise ValueError("Function returned NaN.")
                            if fa * fb < 0:
                                sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                                return sol_root, True
                            else:
                                a -= delta * factor
                                b += delta * factor
                                delta *= factor
                        except ValueError:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    return np.nan, False
                # If no valid interval is found after max_attempts
                return np.nan, False
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        pressure, converged = vectorized_root_func(_lgrho, _lgt, _y, _z, guess)

        return pressure, converged

    def get_s_rhot_inv(self, _lgrho, _lgt, _y, _z, ideal_guess=True, arr_guess=None):
        logp, conv = self.get_logp_rhot_inv(_lgrho, _lgt, _y, _z, ideal_guess=ideal_guess, arr_guess=arr_guess)
        return self.get_s_pt_tab(logp, _lgt, _y, _z)

    def get_logp_srho_inv(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the pressure given entropy, density, helium abundance, and metallicity.
    
        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
        
        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_p_srho(_s, _lgrho, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
    # Define a function to compute root and capture convergence
        def root_func(s_i, lgrho_i, y_i, z_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                logrho_test = self.get_logrho_sp_tab(s_i, _lgp, y_i, z_i)
                return (logrho_test/lgrho_i) - 1

            if method == 'root':
                sol = root(err, guess_i, tol=1e-8)
                if sol.success:
                    return sol.x[0], True
                else:
                    return np.nan, False  # Assign np.nan to non-converged elements

            elif method == 'newton':
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    #Convergence failed
                    return np.nan, False
                except Exception as e:
                    #Handle other exceptions
                    return np.nan, False

            elif method == 'brentq':
                # Define an initial interval around the guess
                delta = 0.1  # Initial interval half-width
                a = guess_i - delta
                b = guess_i + delta

                # Try to find a valid interval where the function changes sign
                max_attempts = 5
                factor = 2.0  # Factor to expand the interval if needed

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")

                        if fa * fb < 0:
                            # Valid interval found
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            # Expand the interval and try again
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor  # Increase delta for next iteration
                    except ValueError:
                        # If err() cannot be evaluated, expand the interval
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor

                # If no valid interval is found after max_attempts
                return np.nan, False

            elif method == 'newton_brentq':
                # Try the Newton method first
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    # Fall back to the Brentq method if Newton fails
                    delta = 0.1
                    a = guess_i - delta
                    b = guess_i + delta
                    max_attempts = 5
                    factor = 2.0

                    for attempt in range(max_attempts):
                        try:
                            fa = err(a)
                            fb = err(b)
                            if np.isnan(fa) or np.isnan(fb):
                                raise ValueError("Function returned NaN.")
                            if fa * fb < 0:
                                sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                                return sol_root, True
                            else:
                                a -= delta * factor
                                b += delta * factor
                                delta *= factor
                        except ValueError:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    return np.nan, False
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgrho, _y, _z, guess)

        return temperatures, converged

    def get_logt_srho_inv(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the temperature given entropy, density, helium abundance, and metallicity.
    
        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
        
        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

       # _y = _y if self.y_prime else _y / (1 - _z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_t_srho(_s, _lgrho, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
    # Define a function to compute root and capture convergence
        def root_func(s_i, lgrho_i, y_i, z_i, guess_i):
            def err(_lgt):
                # Error function for logt(S, logp)
                s_test = self.get_s_rhot_tab(lgrho_i, _lgt, y_i, z_i) * erg_to_kbbar
                return (s_test/s_i) - 1

            if method == 'root':
                sol = root(err, guess_i, tol=1e-8)
                if sol.success:
                    return sol.x[0], True
                else:
                    return np.nan, False  # Assign np.nan to non-converged elements

            elif method == 'newton':
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    #Convergence failed
                    return np.nan, False
                except Exception as e:
                    #Handle other exceptions
                    return np.nan, False

            elif method == 'brentq':
                # Define an initial interval around the guess
                delta = 0.1  # Initial interval half-width
                a = guess_i - delta
                b = guess_i + delta

                # Try to find a valid interval where the function changes sign
                max_attempts = 5
                factor = 2.0  # Factor to expand the interval if needed

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")

                        if fa * fb < 0:
                            # Valid interval found
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            # Expand the interval and try again
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor  # Increase delta for next iteration
                    except ValueError:
                        # If err() cannot be evaluated, expand the interval
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor

                # If no valid interval is found after max_attempts
                return np.nan, False

            elif method == 'newton_brentq':
                # Try the Newton method first
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    # Fall back to the Brentq method if Newton fails
                    delta = 0.1
                    a = guess_i - delta
                    b = guess_i + delta
                    max_attempts = 5
                    factor = 2.0

                    for attempt in range(max_attempts):
                        try:
                            fa = err(a)
                            fb = err(b)
                            if np.isnan(fa) or np.isnan(fb):
                                raise ValueError("Function returned NaN.")
                            if fa * fb < 0:
                                sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                                return sol_root, True
                            else:
                                a -= delta * factor
                                b += delta * factor
                                delta *= factor
                        except ValueError:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    return np.nan, False
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgrho, _y, _z, guess)

        return temperatures, converged

    def get_logp_logt_srho_2Dinv(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='root'):
        """
        Compute temperature and pressure given entropy, density, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            arr_guess (tuple of array_like, optional): User-provided initial guesses for log temperature and log pressure when `ideal_guess` is False.

        Returns:
            logt_values (ndarray): Computed log10 temperature values.
            logp_values (ndarray): Computed log10 pressure values.
            converged (ndarray): Boolean array indicating convergence for each point.
        """

        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        # Prepare output arrays
        shape = _s.shape
        logt_values = np.empty(shape)
        logp_values = np.empty(shape)
        converged = np.zeros(shape, dtype=bool)

        # Initial guesses for log temperature and log pressure
        if ideal_guess:
            # Use the ideal EOS for the initial guesses
            #pdb.set_trace()
            guess_lgp, guess_lgt = ideal_xy.get_pt_srho(_s, _lgrho, _y).T
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            else:
                guess_lgp, guess_lgt = arr_guess

        # Flatten arrays for iteration
        lgrho_flat = _lgrho.flatten()
        s_flat = _s.flatten()
        y_flat = _y.flatten()
        z_flat = _z.flatten()
        guess_lgp_flat = guess_lgp.flatten()
        guess_lgt_flat = guess_lgt.flatten()

        # Iterate over each element
        for idx in range(len(s_flat)):
            lgrho_i = lgrho_flat[idx]
            s_i = s_flat[idx]
            y_i = y_flat[idx]
            z_i = z_flat[idx]
            guess_lgp_i = guess_lgp_flat[idx]
            guess_lgt_i = guess_lgt_flat[idx]
            if method == 'root':

                def opt(vars):
                    lgp, lgt = vars
                    s_calc = self.get_s_pt_tab(lgp, lgt, y_i, z_i) * erg_to_kbbar
                    lgrho_calc = self.get_logrho_pt_tab(lgp, lgt, y_i, z_i)
                    
                    # Convert s_calc and lgrho_calc to scalars if they are arrays
                    if isinstance(s_calc, np.ndarray):
                        s_calc = s_calc.item()
                    if isinstance(lgrho_calc, np.ndarray):
                        lgrho_calc = lgrho_calc.item()
                    
                    err1 = (s_calc/s_i) - 1
                    err2 = (lgrho_calc/lgrho_i) - 1
                    return np.array([err1, err2]) 

                try:
                    sol = root(
                        opt, [guess_lgp_i, guess_lgt_i], method='hybr', tol=1e-6
                    )
                    if sol.success:
                        logp_values.flat[idx], logt_values.flat[idx] = sol.x
                        converged.flat[idx] = True
                    else:
                        logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                        converged.flat[idx] = False
                except Exception as e:
                    logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                    converged.flat[idx] = False

            elif method == 'nelder-mead':
                def opt(vars):

                    lgp, lgt = vars
                    s_calc = self.get_s_pt_tab(lgp, lgt, y_i, z_i) * erg_to_kbbar
                    lgrho_calc = self.get_logrho_pt_tab(lgp, lgt, y_i, z_i)
                    
                    # Convert s_calc and lgrho_calc to scalars if they are arrays
                    if isinstance(s_calc, np.ndarray):
                        s_calc = s_calc.item()
                    if isinstance(lgrho_calc, np.ndarray):
                        lgrho_calc = lgrho_calc.item()
                    
                    err1 = (s_calc / s_i) - 1
                    err2 = (lgrho_calc / lgrho_i) - 1
                    return err1**2 + err2**2  # Return a scalar

                try:
                    sol = minimize(
                        opt, [guess_lgp_i, guess_lgt_i], method='nelder-mead'
                    )
                    if sol.success:
                        logp_values.flat[idx], logt_values.flat[idx] = sol.x
                        converged.flat[idx] = True
                    else:
                        logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                        converged.flat[idx] = False
                except Exception as e:
                    logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                    converged.flat[idx] = False


        # Reshape output arrays to original shape
        logp_values = logp_values.reshape(shape)
        logt_values = logt_values.reshape(shape)
        converged = converged.reshape(shape)

        return logp_values, logt_values, converged


    def get_logp_logt_srho_inv(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        logp, convp = self.get_logp_srho_inv(_s, _lgrho, _y, _z, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        logt, convt = self.get_logt_sp_inv(_s, logp, _y, _z, ideal_guess=True, arr_guess=None, method=method)
        return logp, logt

    def get_logu_srho(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):

        _y = _y if self.y_prime else _y / (1 - _z)

        args = (_s, _lgrho, _y, _z)
        if tab:
            logp, logt = self.get_logp_srho_tab(*args), self.get_logt_srho_tab(*args)
        else:
            logp, logt = self.get_logp_logt_srho_inv( _s, _lgrho, _y, _z, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        return self.get_logu_pt_tab(logp, logt, _y, _z)


    def get_logt_rhop_inv(self, _lgrho, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the temperature given density, pressure, helium abundance, and metallicity.
    
        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.
    
        Returns:
            ndarray: Computed temperature values.
        """
        
        _lgrho = np.atleast_1d(_lgrho)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgp, _y, _z = np.broadcast_arrays(_lgrho, _lgp, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            else:
                guess = arr_guess
        # sol = root(err, guess, tol=1e-6)
        # return sol.x, sol.success

    # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgp_i, y_i, z_i, guess_i):
            def err(_lgt):
                logrho_test = self.get_logrho_pt_tab(lgp_i, _lgt, y_i, z_i)
                return (logrho_test / lgrho_i) - 1


            if method == 'root':
                sol = root(err, guess_i, tol=1e-8)
                if sol.success:
                    return sol.x[0], True
                else:
                    return np.nan, False  # Assign np.nan to non-converged elements

            elif method == 'newton':
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    #Convergence failed
                    return np.nan, False
                except Exception as e:
                    #Handle other exceptions
                    return np.nan, False

            elif method == 'brentq':
                # Define an initial interval around the guess
                delta = 0.1  # Initial interval half-width
                a = guess_i - delta
                b = guess_i + delta

                # Try to find a valid interval where the function changes sign
                max_attempts = 5
                factor = 2.0  # Factor to expand the interval if needed

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")

                        if fa * fb < 0:
                            # Valid interval found
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            # Expand the interval and try again
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor  # Increase delta for next iteration
                    except ValueError:
                        # If err() cannot be evaluated, expand the interval
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor

                # If no valid interval is found after max_attempts
                return np.nan, False

            elif method == 'newton_brentq':
                # Try the Newton method first
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    # Fall back to the Brentq method if Newton fails
                    delta = 0.1
                    a = guess_i - delta
                    b = guess_i + delta
                    max_attempts = 5
                    factor = 2.0

                    for attempt in range(max_attempts):
                        try:
                            fa = err(a)
                            fb = err(b)
                            if np.isnan(fa) or np.isnan(fb):
                                raise ValueError("Function returned NaN.")
                            if fa * fb < 0:
                                sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                                return sol_root, True
                            else:
                                a -= delta * factor
                                b += delta * factor
                                delta *= factor
                        except ValueError:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    return np.nan, False
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_lgrho, _lgp, _y, _z, guess)

        return temperatures, converged


    def get_s_rhop_inv(self, _lgrho, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        """
        Compute the entropy given density, pressure, helium abundance, and metallicity.

        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            arr_guess (array_like, optional): User-provided initial guess for log entropy when `ideal_guess` is False.
            method (str, optional): Method to use for root finding ('root', 'newton', or 'brentq').

        Returns:
            ndarray: Computed entropy values.
            ndarray: Convergence status for each element.
        """
        
        _lgrho = np.atleast_1d(_lgrho)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgp, _y, _z = np.broadcast_arrays(_lgrho, _lgp, _y, _z)

        if ideal_guess:
            #y_call = _y if self.y_prime else _y / (1 - _z)
            guess = ideal_xy.get_s_rhop(_lgrho, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            else:
                guess = arr_guess

        def root_func(lgrho_i, lgp_i, y_i, z_i, guess_i):
            def err(_s):
                logrho_test = self.get_logrho_sp_tab(_s, lgp_i, y_i, z_i)
                return (logrho_test / lgrho_i) - 1

            if method == 'root':
                sol = root(err, guess_i, tol=1e-8)
                if sol.success:
                    return sol.x[0], True
                else:
                    return np.nan, False  # Assign np.nan to non-converged elements

            elif method == 'newton':
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    # Convergence failed
                    return np.nan, False
                except Exception as e:
                    # Handle other exceptions
                    return np.nan, False

            elif method == 'brentq':
                try:
                    a, b = guess_i - 1, guess_i + 1  # Initial bracket
                    fa, fb = err(a), err(b)
                    factor = 1.5
                    delta = 0.1
                    while fa * fb > 0:
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor
                        fa, fb = err(a), err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")
                    sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                    return sol_root, True
                except ValueError:
                    return np.nan, False

            elif method == 'newton_brentq':
                # Try the Newton method first
                try:
                    sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                    return sol_root, True
                except RuntimeError:
                    # Fall back to the Brentq method if Newton fails
                    delta = 0.1
                    a = guess_i - delta
                    b = guess_i + delta
                    max_attempts = 5
                    factor = 2.0

                    for attempt in range(max_attempts):
                        try:
                            fa = err(a)
                            fb = err(b)
                            if np.isnan(fa) or np.isnan(fb):
                                raise ValueError("Function returned NaN.")
                            if fa * fb < 0:
                                sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                                return sol_root, True
                            else:
                                a -= delta * factor
                                b += delta * factor
                                delta *= factor
                        except ValueError:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    return np.nan, False

            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")

        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        entropies, converged = vectorized_root_func(_lgrho, _lgp, _y, _z, guess)

        return entropies / erg_to_kbbar, converged
        
    # def get_s_rhop_inv(self, _lgrho, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq'):
    #     logt, conv = self.get_logt_rhop_inv(_lgrho, _lgp, _y, _z, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
    #     return self.get_s_pt_tab(_lgp, logt, _y, _z)

    # adaptive delta function for z and y derivatives
    def adaptive_dx(self, x_profile, initial_dx=0.1, tolerance=1e-2):
        # Initialize dx as an array with an initial value
        dx = np.full_like(x_profile, initial_dx, dtype=float)

        # Adjust each dz based on z_profile constraints
        for i in range(len(x_profile)):
            # Adjust dz so z_profile[i] - dz[i] >= 0
            if x_profile[i] - dx[i] < 0:
                dx[i] = x_profile[i]  # Set dz to the maximum allowed value to keep z_profile - dz non-negative
                
            # Adjust dz so z_profile[i] + dz[i] <= 1
            elif x_profile[i] + dx[i] > 1:
                dx[i] = 1 - x_profile[i]  # Set dz to the maximum allowed value to keep z_profile + dz <= 1
                
            # Add a tolerance check to prevent overshooting the bounds
            if x_profile[i] - dx[i] < 0:
                dx[i] = max(dx[i] - tolerance, 0)
            if x_profile[i] + dx[i] > 1:
                dx[i] = min(dx[i] - tolerance, 1 - x_profile[i])

        return dx


    def interpolate_non_converged_temperatures_1d(self, _lgrho, temperatures, converged, interp_kind='linear'):

        # Get converged and non-converged indices
        converged_indices = np.where(converged)
        non_converged_indices = np.where(~converged)

        # Extract converged data
        lgrho_converged = _lgrho[converged_indices]
        temperatures_converged = temperatures[converged_indices]

        # Sort data for interpolation
        sorted_indices = np.argsort(lgrho_converged)
        lgrho_converged_sorted = lgrho_converged[sorted_indices]
        temperatures_converged_sorted = temperatures_converged[sorted_indices]

        # Create interpolation function
        interp_func = interp1d(
            lgrho_converged_sorted, temperatures_converged_sorted, kind=interp_kind, fill_value="extrapolate"
        )

        # Interpolate temperatures for non-converged points
        temperatures_interpolated = temperatures.copy()
        temperatures_interpolated[non_converged_indices] = interp_func(_lgrho[non_converged_indices])

        return temperatures_interpolated


    def adaptive_hampel_filter(self, y, min_window=3, max_window=15, n_sigmas=3):
        y = np.array(y)
        n = len(y)
        y_filtered = y.copy()
        outlier_indices = []

        for i in range(n):
            # Determine the optimal window size at position i
            local_window_size = self.determine_optimal_window(y, i, min_window, max_window)
            window_range = range(max(0, i - local_window_size), min(n, i + local_window_size + 1))
            window = y[window_range]
            median = np.median(window)
            mad = 1.4826 * np.median(np.abs(window - median))
            deviation = np.abs(y[i] - median)
            if deviation > n_sigmas * mad:
                y_filtered[i] = median
                outlier_indices.append(i)
        return y_filtered, outlier_indices

    def determine_optimal_window(self, y, index, min_window, max_window):
        # Custom logic to determine window size based on local data properties
        # Placeholder for actual implementation
        return min_window  # Or any logic to vary window size

    def remove_outliers(self, x, y, outlier_indices):
        """
        Removes outliers from the data arrays.
        
        Parameters:
            x (array-like): The independent variable array.
            y (array-like): The dependent variable array.
            outlier_indices (list): Indices of the outliers to remove.
        
        Returns:
            x_clean (np.array): The x array without outliers.
            y_clean (np.array): The y array without outliers.
        """
        x_clean = np.delete(x, outlier_indices)
        y_clean = np.delete(y, outlier_indices)
        return x_clean, y_clean

    def interpolate_missing(self, x_clean, y_clean, x_original, kind='linear'):
        """
        Interpolates the missing data points.
        
        Parameters:
            x_clean (array-like): The x array without outliers.
            y_clean (array-like): The y array without outliers.
            x_original (array-like): The original x array (including outlier positions).
            kind (str): Type of interpolation ('linear', 'quadratic', 'cubic', etc.).
        
        Returns:
            y_interpolated (np.array): The y array with interpolated values at missing points.
        """
        interp_func = interp1d(x_clean, y_clean, kind=kind, fill_value='extrapolate')
        y_interpolated = interp_func(x_original)
        return y_interpolated

    def return_noglitch(self, x, y):

        y_filtered, outlier_indices = self.adaptive_hampel_filter(y, min_window=3, max_window=15, n_sigmas=3)
        x_clean, y_clean = self.remove_outliers(x, y, outlier_indices)
        y_interpolated = self.interpolate_missing(x_clean, y_clean, x, kind='linear')

        return y_interpolated

    # def inversion(self, a_arr, b_arr, y_arr, z_arr, basis, inversion_method='newton_brentq', twoD_inv=False, calc_derivatives=False):

    #     """
    #     Invert the EOS table to compute new thermodynamic quantities.

    #     Parameters:
    #         a_arr (array_like): Array of 'a' values (e.g., entropy).
    #         b_arr (array_like): Array of 'b' values (e.g., pressure).
    #         y_arr (array_like): Array of helium mass fraction values.
    #         z_arr (array_like): Array of metallicity values.
    #         basis (str): The basis for inversion ('sp', 'rhot', 'srho', 'rhop').

    #     Returns:
    #         Two arrays of the inverted quantities.
    #     """

    #     res1_list = []
    #     res2_list = []

    #     if calc_derivatives:
    #         der1_list = []
    #         der2_list = []
    #         der3_list = []
    #         der4_list = []

    #     for a_ in tqdm(a_arr):
    #         res1_b = []
    #         res2_b = []
    #         if calc_derivatives:
    #             der1_b = []
    #             der2_b = []
    #             der3_b = []
    #             der4_b = []
    #         for b_ in b_arr:
    #             res1_y = []
    #             res2_y = []
    #             if calc_derivatives:
    #                 der1_y = []
    #                 der2_y = []
    #                 der3_y = []
    #                 der4_y = []
    #             prev_res1_temp = None  # Initialize previous res1_temp to None
    #             prev_res2_temp = None # For double inversion
    #             for y_ in y_arr:
    #                 a_const = np.full_like(z_arr, a_)
    #                 b_const = np.full_like(z_arr, b_)
    #                 y_const = np.full_like(z_arr, y_)
    #                 # der1 = np.full_like(z_arr, y_)
    #                 # der2 = np.full_like(z_arr, y_)
    #                 # der3 = np.full_like(z_arr, y_)
    #                 # der4 = np.full_like(z_arr, y_)                    
    #                 if basis == 'sp':
    #                     try:
    #                         if prev_res1_temp is None:
    #                             res1_temp, conv = self.get_logt_sp_inv(
    #                                 a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                                 )
    #                             # if calc_derivatives:
    #                             #     res1_temp2, conv2 = self.get_logt_sp_inv(
    #                             #         a_const, b_const*1.1, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                             #         )
    #                             #     res1_temp3, conv3 = self.get_logt_sp_inv(
    #                             #         a_const, b_const*0.9, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                             #         )
    #                         else:
    #                             res1_temp, conv = self.get_logt_sp_inv(
    #                                 a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                 )
    #                             # if calc_derivatives:
    #                             #     res1_temp2, conv2 = self.get_logt_sp_inv(
    #                             #         a_const, b_const*1.1, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                             #         )
    #                             #     res1_temp3, conv3 = self.get_logt_sp_inv(
    #                             #         a_const, b_const*0.9, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                             #         )
    #                     except:
    #                         print('Failed at s={}, logp={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
    #                         raise

    #                     # if calc_derivatives:
                        
    #                     #     res2_temp = self.get_logrho_pt_tab(b_const, res1_temp, y_const, z_arr)
    #                     #     res2_temp2 = self.get_logrho_pt_tab(b_const*1.1, res1_temp2, y_const, z_arr)
    #                     #     res2_temp3 = self.get_logrho_pt_tab(b_const*0.9, res1_temp3, y_const, z_arr)

    #                     #     der1_temp = (res1_temp2 - res1_temp3)/(b_const*0.2) # grad_ad
    #                     #     der2_temp = (b_const*0.2)/(res2_temp2 - res2_temp3) # Gamma_1
    #                     #     der3_temp = (res1_temp2 - res1_temp3)/(res2_temp2 - res2_temp3) # Gruneisen parameter
    #                     #     der4_temp = np.sqrt((10**(b_const*1.1) - 10**(b_const*0.9))/(10**res2_temp2 - 10**res2_temp3)) # c_s

    #                     #     # interpolates quadratically over any non-converged values (NaN returns from inversion)
    #                     #     res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, res1_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     res2_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, res2_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     der1_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der1_temp, conv2, interp_kind='quadratic'
    #                     #     )

    #                     #     der2_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der2_temp, conv2, interp_kind='quadratic'
    #                     #     )

    #                     #     der3_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der3_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     der4_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der4_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     # two passes through no-glitch filter... some have more than two glitches that the first pass does
    #                     #     # not catch.
    #                     #     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                     #     res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                     #     res2_noglitch = self.return_noglitch(z_arr, res2_interp)
    #                     #     res2 = self.return_noglitch(z_arr, res2_noglitch)

    #                     #     der1_noglitch = self.return_noglitch(z_arr, der1_temp)
    #                     #     der1 = self.return_noglitch(z_arr, der1_noglitch)

    #                     #     der2_noglitch = self.return_noglitch(z_arr, der2_temp)
    #                     #     der2 = self.return_noglitch(z_arr, der2_noglitch)

    #                     #     der3_noglitch = self.return_noglitch(z_arr, der3_temp)
    #                     #     der3 = self.return_noglitch(z_arr, der3_noglitch)

    #                     #     der4_noglitch = self.return_noglitch(z_arr, der4_temp)
    #                     #     der4 = self.return_noglitch(z_arr, der4_noglitch)

    #                     # else:

    #                     res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                         z_arr, res1_temp, conv, interp_kind='quadratic'
    #                     )

    #                     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                     res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                     res2 = self.get_logrho_pt_tab(b_const, res1, y_const, z_arr)


    #                     prev_res1_temp = res1 # Update prev_res1_temp for the next iteration

    #                 elif basis == 'rhot':

    #                     try:
    #                         if prev_res1_temp is None:

    #                             res1_temp, conv = self.get_logp_rhot_inv(
    #                                 a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                                 )
    #                         else:
    #                             res1_temp, conv = self.get_logp_rhot_inv(
    #                                 a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                 )

    #                     except:
    #                         print('Failed at rho={}, logt={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
    #                         raise

    #                     res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                         z_arr, res1_temp, conv, interp_kind='quadratic'
    #                     )

    #                     # two passes through no-glitch filter... some have more than two glitches that the first pass does
    #                     # not catch.
    #                     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                     res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                     res2 = self.get_s_pt(res1, b_const, y_const, z_arr)

    #                     prev_res1_temp = res1

    #                 elif basis == 'srho':

    #                     if twoD_inv:
    #                         try:
    #                             if prev_res1_temp is None:
    #                                 res1_temp, res2_temp, conv = self.get_logp_logt_srho_2Dinv(
    #                                     a_const, b_const, y_const, z_arr, ideal_guess=True, method='root'
    #                                     )
    #                                 if calc_derivatives:
    #                                     res1_temp2, res2_temp2, conv2 = self.get_logp_logt_srho_2Dinv(
    #                                         a_const, b_const, y_const*1.1, z_arr, ideal_guess=True, method='root'
    #                                         )
    #                                     res1_temp3, res2_temp3, conv3 = self.get_logp_logt_srho_2Dinv(
    #                                         a_const, b_const, y_const*0.9, z_arr, ideal_guess=True, method='root'
    #                                         )
    #                             else:
    #                                 res1_temp, res2_temp, conv = self.get_logp_logt_srho_2Dinv(
    #                                     a_const, b_const, y_const, z_arr, ideal_guess=False, method='root', arr_guess=[prev_res1_temp, prev_res2_temp]
    #                                     )
    #                                 if calc_derivatives:
    #                                     res1_temp2, res2_temp2, conv2 = self.get_logp_logt_srho_2Dinv(
    #                                         a_const, b_const, y_const*1.1, z_arr, ideal_guess=False, method='root', arr_guess=[prev_res1_temp, prev_res2_temp]
    #                                         )
    #                                     res1_temp3, res2_temp3, conv3 = self.get_logp_logt_srho_2Dinv(
    #                                         a_const, b_const, y_const*0.9, z_arr, ideal_guess=False, method='root', arr_guess=[prev_res1_temp, prev_res2_temp]
    #                                         )                                                                                
    #                         except:
    #                             print('Failed at s={}, rho={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
    #                             raise

    #                         res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                             z_arr, res1_temp, conv, interp_kind='quadratic'
    #                         )

    #                         res2_interp = self.interpolate_non_converged_temperatures_1d(
    #                             z_arr, res2_temp, conv, interp_kind='quadratic'
    #                         )

    #                         res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                         res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                         res2_noglitch = self.return_noglitch(z_arr, res2_interp)
    #                         res2 = self.return_noglitch(z_arr, res2_noglitch)

    #                         # if calc_derivatives:

    #                         #     res3_temp = self.get_logu_pt_tab(res1_temp, res2_temp, y_const, z_arr)
    #                         #     res3_temp2 = self.get_logu_pt_tab(res1_temp2, res2_temp2, y_const*1.1, z_arr)
    #                         #     res3_temp3 = self.get_logu_pt_tab(res1_temp3, res2_temp3, y_const*0.9, z_arr)

    #                         #     der1_temp = np.gradient(10**res3_temp)/np.gradient(z_arr) # dUdz_srho
    #                         #     der2_temp = (10**res3_temp2 - 10**res3_temp3)/(y_const*0.2) # dUdy_srho

    #                         #     der3_temp = np.gradient(10**res1_temp)/np.gradient(z_arr) # dTdz_srho
    #                         #     der4_temp = (10**res2_temp2 - 10**res2_temp3)/(y_const*0.2) # dTdy_srho

    #                         #     # res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                         #     #     z_arr, res1_temp, conv, interp_kind='quadratic'
    #                         #     # )

    #                         #     # res2_interp = self.interpolate_non_converged_temperatures_1d(
    #                         #     #     z_arr, res2_temp, conv, interp_kind='quadratic'
    #                         #     # )

    #                         #     der1_interp = self.interpolate_non_converged_temperatures_1d(
    #                         #         z_arr, der1_temp, conv, interp_kind='quadratic'
    #                         #     )

    #                         #     der2_interp = self.interpolate_non_converged_temperatures_1d(
    #                         #         z_arr, der2_temp, conv2, interp_kind='quadratic'
    #                         #     )

    #                         #     der3_interp = self.interpolate_non_converged_temperatures_1d(
    #                         #         z_arr, der3_temp, conv, interp_kind='quadratic'
    #                         #     )

    #                         #     der4_interp = self.interpolate_non_converged_temperatures_1d(
    #                         #         z_arr, der4_temp, conv2, interp_kind='quadratic'
    #                         #     )

    #                         #     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                         #     res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                         #     res2_noglitch = self.return_noglitch(z_arr, res2_interp)
    #                         #     res2 = self.return_noglitch(z_arr, res2_noglitch)

    #                         #     der1_noglitch = self.return_noglitch(z_arr, der1_temp)
    #                         #     der1 = self.return_noglitch(z_arr, der1_noglitch)

    #                         #     der2_noglitch = self.return_noglitch(z_arr, der2_temp)
    #                         #     der2 = self.return_noglitch(z_arr, der2_noglitch)

    #                         #     der3_noglitch = self.return_noglitch(z_arr, der3_temp)
    #                         #     der3 = self.return_noglitch(z_arr, der3_noglitch)

    #                         #     der4_noglitch = self.return_noglitch(z_arr, der4_temp)
    #                         #     der4 = self.return_noglitch(z_arr, der4_noglitch)

    #                         prev_res1_temp = res1
    #                         prev_res2_temp = res2

    #                     else: # uses 1-D inversion via SP inverted table

    #                         try:
    #                             if prev_res1_temp is None:

    #                                 res1_temp, conv = self.get_logp_srho_inv(
    #                                     a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                                     )
    #                                 res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                                     z_arr, res1_temp, conv, interp_kind='quadratic'
    #                                     )

    #                                 res2_temp, conv2_1 = self.get_logt_sp_inv(
    #                                     a_const, res1_interp, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                                     )
    #                                 res2_interp = self.interpolate_non_converged_temperatures_1d(
    #                                     z_arr, res2_temp, conv, interp_kind='quadratic'
    #                                     )

    #                                 # if calc_derivatives:
    #                                 #     res1_temp2, conv2 = self.get_logp_srho_inv(
    #                                 #         a_const, b_const, y_const*1.1, z_arr, method=inversion_method, ideal_guess=True
    #                                 #         )   

    #                                 #     res1_temp3, conv3 = self.get_logp_srho_inv(
    #                                 #         a_const, b_const, y_const*0.9, z_arr, method=inversion_method, ideal_guess=True
    #                                 #         )   

    #                                 #     res1_interp2 = self.interpolate_non_converged_temperatures_1d(
    #                                 #         z_arr, res1_temp2, conv2, interp_kind='quadratic'
    #                                 #         )
    #                                 #     res1_interp3 = self.interpolate_non_converged_temperatures_1d(
    #                                 #         z_arr, res1_temp3, conv3, interp_kind='quadratic'
    #                                 #         )     
                                   
    #                                 #     res2_temp2, conv2_2 = self.get_logt_sp_inv(
    #                                 #         a_const, res1_interp2, y_const*1.1, z_arr, method=inversion_method, ideal_guess=True
    #                                 #         )
    #                                 #     res2_temp3, conv2_3 = self.get_logt_sp_inv(
    #                                 #         a_const, res1_interp3, y_const*0.9, z_arr, method=inversion_method, ideal_guess=True
    #                                 #         )   


    #                             else:
    #                                 res1_temp, conv = self.get_logp_srho_inv(
    #                                     a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                     )
    #                                 res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                                     z_arr, res1_temp, conv, interp_kind='quadratic'
    #                                     )
    #                                 res2_temp, conv2_1 = self.get_logt_sp_inv(
    #                                     a_const, res1_interp, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res2_temp
    #                                     )
    #                                 res2_interp = self.interpolate_non_converged_temperatures_1d(
    #                                     z_arr, res2_temp, conv2_1, interp_kind='quadratic'
    #                                     )


    #                                 # if calc_derivatives:
    #                                 #     res1_temp2, conv2 = self.get_logp_srho_inv(
    #                                 #         a_const, b_const, y_const*1.1, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                 #         )                                        
    #                                 #     res1_temp3, conv3 = self.get_logp_srho_inv(
    #                                 #         a_const, b_const, y_const*0.9, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                 #         )


    #                                 #     res1_interp2 = self.interpolate_non_converged_temperatures_1d(
    #                                 #         z_arr, res1_temp2, conv2, interp_kind='quadratic'
    #                                 #         )
    #                                 #     res1_interp3 = self.interpolate_non_converged_temperatures_1d(
    #                                 #         z_arr, res1_temp3, conv3, interp_kind='quadratic'
    #                                 #         )   

    #                                 #res2_temp, conv2_1 = self.get_logt_sp_inv(a_const, res1_interp, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res2_temp)
    #                                     # res2_temp2, conv2_2 = self.get_logt_sp_inv(a_const, res1_interp2, y_const*1.1, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res2_temp)
    #                                     # res2_temp3, conv2_3 = self.get_logt_sp_inv(a_const, res1_interp3, y_const*0.9, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res2_temp)
        


    #                         except:
    #                             print('Failed at s={}, rho={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
    #                             raise
    #                         # else:
    #                         #     res1_temp, conv = self.get_logp_srho_inv(
    #                         #         a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                         #         )

    #                         # if calc_derivatives:
    #                         #     res2_interp2 = self.interpolate_non_converged_temperatures_1d(
    #                         #         z_arr, res2_temp2, conv2_2, interp_kind='quadratic'
    #                         #         )
    #                         #     res2_interp3 = self.interpolate_non_converged_temperatures_1d(
    #                         #         z_arr, res2_temp3, conv2_3, interp_kind='quadratic'
    #                         #         )  
                            
    #                         #     res3_temp = self.get_logu_pt_tab(res1_interp, res2_interp, y_const, z_arr)
    #                         #     res3_temp2 = self.get_logu_pt_tab(res1_interp2, res2_interp2, y_const*1.1, z_arr)
    #                         #     res3_temp3 = self.get_logu_pt_tab(res1_interp3, res2_interp3, y_const*0.9, z_arr)

    #                         #     der1_temp = np.gradient(10**res3_temp)/np.gradient(z_arr) # dUdz_srho
    #                         #     der2_temp = (10**res3_temp2 - 10**res3_temp3)/(y_const*0.2) # dUdy_srho

    #                         #     der3_temp = np.gradient(10**res2_interp)/np.gradient(z_arr) # dTdz_srho
    #                         #     der4_temp = (10**res2_interp2 - 10**res2_interp3)/(y_const*0.2) # dTdy_srho

    #                         #     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                         #     res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                         #     res2_noglitch = self.return_noglitch(z_arr, res2_interp)
    #                         #     res2 = self.return_noglitch(z_arr, res2_noglitch)

    #                         #     der1_noglitch = self.return_noglitch(z_arr, der1_temp)
    #                         #     der1 = self.return_noglitch(z_arr, der1_noglitch)

    #                         #     der2_noglitch = self.return_noglitch(z_arr, der2_temp)
    #                         #     der2 = self.return_noglitch(z_arr, der2_noglitch)

    #                         #     der3_noglitch = self.return_noglitch(z_arr, der3_temp)
    #                         #     der3 = self.return_noglitch(z_arr, der3_noglitch)

    #                         #     der4_noglitch = self.return_noglitch(z_arr, der4_temp)
    #                         #     der4 = self.return_noglitch(z_arr, der4_noglitch)

    #                         # else:
    #                         res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                         res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                         res2_noglitch = self.return_noglitch(z_arr, res2_interp)
    #                         res2 = self.return_noglitch(z_arr, res2_noglitch)

    #                         prev_res1_temp = res1
    #                         prev_res2_temp = res2

    #                 elif basis == 'rhop':
    #                     try:
    #                         if prev_res1_temp is None:

    #                             # res1_temp, conv = self.get_logt_rhop_inv(
    #                             #     a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                             #     )
    #                             # res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                             # z_arr, res1_temp, conv, interp_kind='quadratic'
    #                             #     )
    #                             # if calc_derivatives:
    #                             #     res1_temp2, conv2 = self.get_logt_rhop_inv(
    #                             #         a_const, b_const, y_const*1.1, z_arr, method=inversion_method, ideal_guess=True
    #                             #         )

    #                             #     res1_temp3, conv3 = self.get_logt_rhop_inv(
    #                             #         a_const, b_const, y_const*0.9, z_arr, method=inversion_method, ideal_guess=True
    #                             #         )
                                
    #                             # inverting the table along entropy instead of temperature instead....
    #                             res1_temp, conv = self.get_s_rhop_inv(
    #                                 a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
    #                                 )
    #                             res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                             z_arr, res1_temp, conv, interp_kind='quadratic'
    #                                 )
    #                             # if calc_derivatives:
    #                             #     res1_temp2, conv2 = self.get_logt_rhop_inv(
    #                             #         a_const, b_const, y_const*1.1, z_arr, method=inversion_method, ideal_guess=True
    #                             #         )

    #                             #     res1_temp3, conv3 = self.get_logt_rhop_inv(
    #                             #         a_const, b_const, y_const*0.9, z_arr, method=inversion_method, ideal_guess=True
    #                             #         )
    #                         else:
    #                             res1_temp, conv = self.get_s_rhop_inv(
    #                                 a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                 )
    #                             res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                             z_arr, res1_temp, conv, interp_kind='quadratic'
    #                                 )
    #                             if calc_derivatives:
    #                                 res1_temp2, conv2 = self.get_logt_rhop_inv(
    #                                     a_const, b_const, y_const*1.1, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                     )

    #                                 res1_temp3, conv3 = self.get_logt_rhop_inv(
    #                                     a_const, b_const, y_const*0.9, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
    #                                     )  

    #                     except:
    #                         print('Failed at rho={}, logp={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
    #                         raise

    #                     # if calc_derivatives:

    #                     #     res2_temp = self.get_s_pt_tab(b_const, res1_temp, y_const, z_arr)
    #                     #     res2_temp2 = self.get_s_pt_tab(b_const, res1_temp2, y_const*1.1, z_arr)
    #                     #     res2_temp3 = self.get_s_pt_tab(b_const, res1_temp3, y_const*0.9, z_arr)

    #                     #     der1_temp = np.gradient(res2_temp)/np.gradient(z_arr) # dsdz_rhop
    #                     #     der2_temp = (res2_temp2 - res2_temp3)/(y_const*0.2) # dsdy_rhop

    #                     #     der3_temp = np.gradient(10**res1_temp)/np.gradient(z_arr) # dTdz_rhop
    #                     #     der4_temp = (10**res1_temp2 - 10**res1_temp3)/(y_const*0.2) # dTdy_rhop

    #                     #     res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, res1_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     res2_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, res2_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     der1_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der1_temp, conv2, interp_kind='quadratic'
    #                     #     )

    #                     #     der2_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der2_temp, conv2, interp_kind='quadratic'
    #                     #     )

    #                     #     der3_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der3_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     der4_interp = self.interpolate_non_converged_temperatures_1d(
    #                     #         z_arr, der4_temp, conv, interp_kind='quadratic'
    #                     #     )

    #                     #     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                     #     res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                     #     res2_noglitch = self.return_noglitch(z_arr, res2_interp)
    #                     #     res2 = self.return_noglitch(z_arr, res2_noglitch)

    #                     #     der1_noglitch = self.return_noglitch(z_arr, der1_temp)
    #                     #     der1 = self.return_noglitch(z_arr, der1_noglitch)

    #                     #     der2_noglitch = self.return_noglitch(z_arr, der2_temp)
    #                     #     der2 = self.return_noglitch(z_arr, der2_noglitch)

    #                     #     der3_noglitch = self.return_noglitch(z_arr, der3_temp)
    #                     #     der3 = self.return_noglitch(z_arr, der3_noglitch)

    #                     #     der4_noglitch = self.return_noglitch(z_arr, der4_temp)
    #                     #     der4 = self.return_noglitch(z_arr, der4_noglitch)

    #                     # else:

    #                         # res1_interp = self.interpolate_non_converged_temperatures_1d(
    #                         #     z_arr, res1_temp, conv, interp_kind='quadratic'
    #                         # )

    #                     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
    #                     res1 = self.return_noglitch(z_arr, res1_noglitch)

    #                     res2 = self.get_logt_sp_tab(res1*erg_to_kbbar, b_const, y_const, z_arr)
                        

    #                     prev_res1_temp = res1
    #                     prev_res2_temp = res2


    #                 else:
    #                     raise ValueError('Unknown inversion basis. Please choose sp, rhot, srho, or rhop')

    #                 res1_y.append(res1)
    #                 res2_y.append(res2)
    #                 # if calc_derivatives:
    #                 #     der1_y.append(der1)
    #                 #     der2_y.append(der2)
    #                 #     der3_y.append(der3)
    #                 #     der4_y.append(der4)

    #             res1_b.append(res1_y)
    #             res2_b.append(res2_y)
    #             # if calc_derivatives:
    #             #     der1_b.append(der1_y)
    #             #     der2_b.append(der2_y)
    #             #     der3_b.append(der3_y)
    #             #     der4_b.append(der4_y)

    #         res1_list.append(res1_b)
    #         res2_list.append(res2_b)
    #         # if calc_derivatives:
    #         #     der1_list.append(der1_b)
    #         #     der2_list.append(der2_b)
    #         #     der3_list.append(der3_b)
    #         #     der4_list.append(der4_b)

    #     # if calc_derivatives:
    #     #     return np.array(res1_list), np.array(res2_list), np.array(der2_list), np.array(der1_list), np.array(der4_list), np.array(der3_list)
    #     # else:
    #      return np.array(res1_list), np.array(res2_list)

    def inversion(self, a_arr, b_arr, y_arr, z_arr, basis, inversion_method='newton_brentq', twoD_inv=False, calc_derivatives=False):

        """
        Invert the EOS table to compute new thermodynamic quantities.

        Parameters:
            a_arr (array_like): Array of 'a' values (e.g., entropy).
            b_arr (array_like): Array of 'b' values (e.g., pressure).
            y_arr (array_like): Array of helium mass fraction values.
            z_arr (array_like): Array of metallicity values.
            basis (str): The basis for inversion ('sp', 'rhot', 'srho', 'rhop').

        Returns:
            Two arrays of the inverted quantities.
        """

        res1_list = []
        res2_list = []

        if calc_derivatives:
            der1_list = []
            der2_list = []
            der3_list = []
            der4_list = []

        for a_ in tqdm(a_arr):
            res1_b = []
            res2_b = []
            if calc_derivatives:
                der1_b = []
                der2_b = []
                der3_b = []
                der4_b = []
            for b_ in b_arr:
                res1_y = []
                res2_y = []
                if calc_derivatives:
                    der1_y = []
                    der2_y = []
                    der3_y = []
                    der4_y = []
                prev_res1_temp = None  # Initialize previous res1_temp to None
                prev_res2_temp = None # For double inversion
                for y_ in y_arr:
                    a_const = np.full_like(z_arr, a_)
                    b_const = np.full_like(z_arr, b_)
                    y_const = np.full_like(z_arr, y_)                
                    if basis == 'sp':
                        try:
                            if prev_res1_temp is None:
                                res1_temp, conv = self.get_logt_sp_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                    )
                            else:
                                res1_temp, conv = self.get_logt_sp_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                    )
                        except:
                            print('Failed at s={}, logp={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                            raise

                        res1_interp = self.interpolate_non_converged_temperatures_1d(
                            z_arr, res1_temp, conv, interp_kind='quadratic'
                        )

                        res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        res1 = self.return_noglitch(z_arr, res1_noglitch)

                        res2 = self.get_logrho_pt_tab(b_const, res1, y_const, z_arr)


                        prev_res1_temp = res1 # Update prev_res1_temp for the next iteration

                    elif basis == 'rhot':

                        try:
                            if prev_res1_temp is None:

                                res1_temp, conv = self.get_logp_rhot_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                    )
                            else:
                                res1_temp, conv = self.get_logp_rhot_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                    )

                        except:
                            print('Failed at rho={}, logt={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                            raise

                        res1_interp = self.interpolate_non_converged_temperatures_1d(
                            z_arr, res1_temp, conv, interp_kind='quadratic'
                        )

                        # two passes through no-glitch filter... some have more than two glitches that the first pass does
                        # not catch.
                        res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        res1 = self.return_noglitch(z_arr, res1_noglitch)

                        res2 = self.get_s_pt(res1, b_const, y_const, z_arr)

                        prev_res1_temp = res1

                    elif basis == 'srho':

                        if twoD_inv:
                            try:
                                if prev_res1_temp is None:
                                    res1_temp, res2_temp, conv = self.get_logp_logt_srho_2Dinv(
                                        a_const, b_const, y_const, z_arr, ideal_guess=True, method='root'
                                        )
                                    if calc_derivatives:
                                        res1_temp2, res2_temp2, conv2 = self.get_logp_logt_srho_2Dinv(
                                            a_const, b_const, y_const*1.1, z_arr, ideal_guess=True, method='root'
                                            )
                                        res1_temp3, res2_temp3, conv3 = self.get_logp_logt_srho_2Dinv(
                                            a_const, b_const, y_const*0.9, z_arr, ideal_guess=True, method='root'
                                            )
                                else:
                                    res1_temp, res2_temp, conv = self.get_logp_logt_srho_2Dinv(
                                        a_const, b_const, y_const, z_arr, ideal_guess=False, method='root', arr_guess=[prev_res1_temp, prev_res2_temp]
                                        )
                                    if calc_derivatives:
                                        res1_temp2, res2_temp2, conv2 = self.get_logp_logt_srho_2Dinv(
                                            a_const, b_const, y_const*1.1, z_arr, ideal_guess=False, method='root', arr_guess=[prev_res1_temp, prev_res2_temp]
                                            )
                                        res1_temp3, res2_temp3, conv3 = self.get_logp_logt_srho_2Dinv(
                                            a_const, b_const, y_const*0.9, z_arr, ideal_guess=False, method='root', arr_guess=[prev_res1_temp, prev_res2_temp]
                                            )                                                                                
                            except:
                                print('Failed at s={}, rho={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                                raise

                            res1_interp = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='quadratic'
                            )

                            res2_interp = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res2_temp, conv, interp_kind='quadratic'
                            )

                            res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                            res1 = self.return_noglitch(z_arr, res1_noglitch)

                            res2_noglitch = self.return_noglitch(z_arr, res2_interp)
                            res2 = self.return_noglitch(z_arr, res2_noglitch)

                            prev_res1_temp = res1
                            prev_res2_temp = res2

                        else: # uses 1-D inversion via SP inverted table

                            try:
                                if prev_res1_temp is None:

                                    res1_temp, conv = self.get_logp_srho_inv(
                                        a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                        )
                                    res1_interp = self.interpolate_non_converged_temperatures_1d(
                                        z_arr, res1_temp, conv, interp_kind='quadratic'
                                        )

                                    res2_temp, conv2_1 = self.get_logt_sp_inv(
                                        a_const, res1_interp, y_const, z_arr, method=inversion_method, ideal_guess=True
                                        )
                                    res2_interp = self.interpolate_non_converged_temperatures_1d(
                                        z_arr, res2_temp, conv, interp_kind='quadratic'
                                        )


                                else:
                                    res1_temp, conv = self.get_logp_srho_inv(
                                        a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                        )
                                    res1_interp = self.interpolate_non_converged_temperatures_1d(
                                        z_arr, res1_temp, conv, interp_kind='quadratic'
                                        )
                                    res2_temp, conv2_1 = self.get_logt_sp_inv(
                                        a_const, res1_interp, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res2_temp
                                        )
                                    res2_interp = self.interpolate_non_converged_temperatures_1d(
                                        z_arr, res2_temp, conv2_1, interp_kind='quadratic'
                                        )

                            except:
                                print('Failed at s={}, rho={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                                raise

                            res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                            res1 = self.return_noglitch(z_arr, res1_noglitch)

                            res2_noglitch = self.return_noglitch(z_arr, res2_interp)
                            res2 = self.return_noglitch(z_arr, res2_noglitch)

                            prev_res1_temp = res1
                            prev_res2_temp = res2

                    elif basis == 'rhop':
                        try:
                            if prev_res1_temp is None:

                                
                                # inverting the table along entropy instead of temperature instead....
                                res1_temp, conv = self.get_s_rhop_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                    )
                                res1_interp = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='quadratic'
                                    )
                            else:
                                res1_temp, conv = self.get_s_rhop_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                    )
                                res1_interp = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='quadratic'
                                    )
                                if calc_derivatives:
                                    res1_temp2, conv2 = self.get_logt_rhop_inv(
                                        a_const, b_const, y_const*1.1, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                        )

                                    res1_temp3, conv3 = self.get_logt_rhop_inv(
                                        a_const, b_const, y_const*0.9, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                        )  

                        except:
                            print('Failed at rho={}, logp={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                            raise

                        res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        res1 = self.return_noglitch(z_arr, res1_noglitch)

                        res2 = self.get_logt_sp_tab(res1*erg_to_kbbar, b_const, y_const, z_arr)
                        

                        prev_res1_temp = res1
                        prev_res2_temp = res2


                    else:
                        raise ValueError('Unknown inversion basis. Please choose sp, rhot, srho, or rhop')

                    res1_y.append(res1)
                    res2_y.append(res2)

                res1_b.append(res1_y)
                res2_b.append(res2_y)

            res1_list.append(res1_b)
            res2_list.append(res2_b)


        return np.array(res1_list), np.array(res2_list)


    ################################################ Wrapper Functions ################################################

    def get_logt_sp(self, _s, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logt_sp_tab(_s, _lgp, _y, _z)

        else:
            return self.get_logt_sp_inv(_s, _lgp, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)[0]

    def get_logrho_sp(self, _s, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logrho_sp_tab(_s, _lgp, _y, _z)

        else:
            return self.get_logrho_sp_inv(_s, _lgp, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)

    def get_logp_rhot(self, _lgrho, _lgt, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logp_rhot_tab(_lgrho, _lgt, _y, _z)

        else:
            return self.get_logp_rhot_inv(_lgrho, _lgt, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)[0]

    def get_s_rhot(self, _lgrho, _lgt, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_s_rhot_tab(_lgrho, _lgt, _y, _z)

        else:
            return self.get_s_rhot_inv(_lgrho, _lgt, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)

    def get_logt_rhop(self, _lgrho, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logt_rhop_tab(_lgrho, _lgp, _y, _z)

        else:
            return self.get_logt_rhop_inv(_lgrho, _lgp, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)[0]

    def get_s_rhop(self, _lgrho, _lgp, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_s_rhop_tab(_lgrho, _lgp, _y, _z)

        else:
            return self.get_s_rhop_inv(_lgrho, _lgp, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)[0]

    def get_logp_srho(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logp_srho_tab(_s, _lgrho, _y, _z)

        else:
            return self.get_logp_srho_inv(_s, _lgrho, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)[0]

    def get_logt_srho(self, _s, _lgrho, _y, _z, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logt_srho_tab(_s, _lgrho, _y, _z)

        else:
            return self.get_logp_logt_srho_inv(_s, _lgrho, _y, _z, ideal_guess=ideal_guess, 
                                        arr_guess=arr_guess, method=method)[-1]

    ################################################ Derivatives ################################################

    ########### Convection Derivatives ###########

    # Specific heat at constant pressure
    def get_cp_sp(self, _s, _lgp, _y, _z, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        ds = _s*0.1 if ds is None else ds

        lgt1 = self.get_logt_sp(_s - ds, _lgp, _y, _z, **kwargs)
        lgt2 = self.get_logt_sp(_s + ds, _lgp, _y, _z, **kwargs)

        return (2 * ds / erg_to_kbbar) / ((lgt2 - lgt1) * log10_to_loge)

    def get_cp_pt(self, _lgp, _lgt, _y, _z, dt=0.1):

        s1 = self.get_s_pt_tab(_lgp, _lgt - dt, _y, _z)
        s2 = self.get_s_pt_tab(_lgp, _lgt + dt, _y, _z)

        return (s2 - s1) / (2 * dt * log10_to_loge)

    # Specific heat at constant volume
    def get_cv_srho(self, _s, _lgrho, _y, _z, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        ds = _s*0.1 if ds is None else ds

        lgt1 = self.get_logt_srho(_s - ds, _lgrho, _y, _z, **kwargs)
        lgt2 = self.get_logt_srho(_s + ds, _lgrho, _y, _z, **kwargs)

        return (2 * ds / erg_to_kbbar) / ((lgt2 - lgt1) * log10_to_loge)

    def get_cv_rhot(self, _lgrho, _lgt, _y, _z, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        s1 = self.get_s_rhot(_lgrho, _lgt - dt, _y, _z, **kwargs)
        s2 = self.get_s_rhot(_lgrho, _lgt + dt, _y, _z, **kwargs)

        return (s2 - s1) / (2 * dt * log10_to_loge)

    # Adiabatic temperature gradient
    def get_nabla_ad(self, _s, _lgp, _y, _z, dp=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        lgt1 = self.get_logt_sp(_s, _lgp - dp, _y, _z, **kwargs)
        lgt2 = self.get_logt_sp(_s, _lgp + dp, _y, _z, **kwargs)
        return (lgt2 - lgt1)/(2 * dp)

    # DS/DX|_P, T - DERIVATIVES NECESSARY FOR THE SCHWARZSCHILD CONDITION
    def get_dsdy_pt(self, _lgp, _lgt, _y, _z, dy=0.1):
        dy = _y*0.1 if dy is None else dy
        s1 = self.get_s_pt_tab(_lgp, _lgt, _y - dy, _z)
        s2 = self.get_s_pt_tab(_lgp, _lgt, _y + dy, _z)
        return (s2 - s1)/(2 * dy)

    def get_dsdz_pt(self, _lgp, _lgt, _y, _z, dz=0.1):
        dz = _z*0.1 if dz is None else dz
        s1 = self.get_s_pt_tab(_lgp, _lgt, _y, _z - dz)
        s2 = self.get_s_pt_tab(_lgp, _lgt, _y, _z + dz)
        return (s2 - s1)/(2 * dz)

    # DS/DX|_rho, P - DERIVATIVES NECESSARY FOR THE LEDOUX CONDITION
    def get_dsdy_rhop(self, _lgrho, _lgp, _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        s1 = self.get_s_rhop(_lgrho, _lgp, _y - dy, _z, **kwargs)
        s2 = self.get_s_rhop(_lgrho, _lgp, _y + dy, _z, **kwargs)
        return (s2 - s1)/(2 * dy)

    def get_dsdz_rhop(self, _lgrho, _lgp, _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        s1 = self.get_s_rhop(_lgrho, _lgp, _y, _z - dz, **kwargs)
        s2 = self.get_s_rhop(_lgrho, _lgp, _y, _z + dz, **kwargs)
        return (s2 - s1)/(2 * dz)

    def get_gamma1(self, _s, _lgp, _y, _z, dp=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # dlnP/dlnrho_S, Y, Z = dlogP/dlogrho_S, Y, Z
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgrho1 = self.get_logrho_sp(_s, _lgp - dp, _y, _z, **kwargs)
        lgrho2 = self.get_logrho_sp(_s, _lgp + dp, _y, _z, **kwargs)
        return (2*dp)/(lgrho2 - lgrho1)

    # Brunt coefficient when computing in drho space
    def get_dlogrho_ds_py(self, _s, _lgp, _y, _z, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgrho2 = self.get_logrho_sp(_s + ds, _lgp, _y, _z, **kwargs)
        lgrho1 = self.get_logrho_sp(_s - ds, _lgp, _y, _z, **kwargs)
        return ((lgrho2 - lgrho1) * log10_to_loge) / (2 * ds / erg_to_kbbar)

    # Chi_T/Chi_rho
    # aka "delta" in MLT flux
    def get_dlogrho_dlogt_py(self, _lgp, _lgt, _y, _z, dt=1e-2):
        
        lgrho1 = self.get_logrho_pt_tab(_lgp, _lgt - dt, _y, _z)
        lgrho2 = self.get_logrho_pt_tab(_lgp, _lgt + dt, _y, _z)

        return (lgrho2 - lgrho1)/(2 * dt)

    def get_dlogp_dy_rhot(self, _lgrho, _lgt,  _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_Y
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot(_lgrho, _lgt, _y - dy, _z, **kwargs)
        lgp2 = self.get_logp_rhot(_lgrho, _lgt, _y + dy, _z, **kwargs)

        return ((lgp2 - lgp1) * log10_to_loge)/(2 * dy)

    def get_dlogp_dz_rhot(self, _lgrho, _lgt,  _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_Z
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot_tab(_lgrho, _lgt, _y, _z - dz, **kwargs)
        lgp2 = self.get_logp_rhot_tab(_lgrho, _lgt, _y, _z + dz, **kwargs)

        return ((lgp2 - lgp1) * log10_to_loge)/(2 * dz)

    def get_dlogp_dlogt_rhoy_rhot(self, _lgrho, _lgt,  _y, _z, dt=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_T
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot_tab(_lgrho, _lgt - dt, _y, _z, **kwargs)
        lgp2 = self.get_logp_rhot_tab(_lgrho, _lgt + dt, _y, _z, **kwargs)

        return (lgp2 - lgp1)/(2 * dt)

    # Chi_Y/Chi_T
    def get_dlogt_dy_rhop_rhot(self, _lgrho, _lgt,  _y, _z, dy=0.1, dt=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        Chi_Y = self.get_dlogp_dy_rhot(_lgrho, _lgt,  _y, _z, dy=dy, **kwargs)
        Chi_T = self.get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, dt=dt, **kwargs)

        return Chi_Y/Chi_T

    # Chi_Z/Chi_T
    def get_dlogt_dz_rhop_rhot(self, _lgrho, _lgt,  _y, _z, dz=0.1, dt=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        Chi_Z = self.get_dlogp_dz_rhot(_lgrho, _lgt,  _y, _z, dz=dz, **kwargs)
        Chi_T = self.get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, dt=dt, **kwargs)

        return Chi_Z/Chi_T

    #### Triple Product Rule Derivatives ###*


    def get_dpds_rhoy_srho(self, _s, _lgrho, _y, _z, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        ds = _s*0.1 if ds is None else ds
        p1 = 10**self.get_logp_srho(_s - ds, _lgrho, _y, _z, **kwargs)
        p2 = 10**self.get_logp_srho(_s + ds, _lgrho, _y, _z, **kwargs)

        return (p2 - p1) / (2 * ds / erg_to_kbbar)

    def get_dpdy_srho(self, _s, _lgrho, _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        p1 = 10**self.get_logp_srho(_s, _lgrho, _y - dy, _z, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, _y + dy, _z, **kwargs)

        return (p2 - p1) / (2 * dy)


    def get_dpdz_srho(self, _s, _lgrho, _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        p1 = 10**self.get_logp_srho(_s, _lgrho, _y, _z - dz, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, _y, _z + dz, **kwargs)

        return (p2 - p1) / (2 * dz)

    ########### Triple product rule dsdx_rhop version ###########
    def get_dsdy_rhop_srho(self, _s, _lgrho, _y, _z, ds=0.1, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, ds=ds, **kwargs)
        #dPdY|{S, rho, Y}:
        dpdy_srho = self.get_dpdy_srho(_s, _lgrho, _y, _z, dy=dy, **kwargs)

        #dSdY|{rho, P, Z} = -dPdY|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdy_rhopy = -dpdy_srho/dpds_rhoy_srho # triple product rule

        return dsdy_rhopy


    def get_dsdz_rhop_srho(self, _s, _lgrho, _y, _z, ds=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, ds=ds, **kwargs)
        #dPdY|{S, rho, Y}:
        dpdz_srho = self.get_dpdz_srho(_s, _lgrho, _y, _z, dz=dz, **kwargs)

        #dSdZ|{rho, P, Z} = -dPdZ|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdz_rhopy = -dpdz_srho/dpds_rhoy_srho # triple product rule

        return dsdz_rhopy


    def get_drhods_rhoy_sp(self, _s, _lgp, _y, _z, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        ds = _s*0.1 if ds is None else ds
        rho1 = 10**self.get_logrho_sp(_s - ds, _lgp, _y, _z, **kwargs)
        rho2 = 10**self.get_logrho_sp(_s + ds, _lgp, _y, _z, **kwargs)

        return (rho2 - rho1) / (2 * ds / erg_to_kbbar)

    def get_drhody_sp(self, _s, _lgp, _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        rho1 = 10**self.get_logrho_sp(_s, _lgp, _y - dy, _z, **kwargs)
        rho2 = 10**self.get_logrho_sp(_s, _lgp, _y + dy, _z, **kwargs)

        return (rho2 - rho1) / (2 * dy)


    def get_drhodz_sp(self, _s, _lgp, _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        rho1 = 10**self.get_logrho_sp(_s, _lgp, _y, _z - dz, **kwargs)
        rho2 = 10**self.get_logrho_sp(_s, _lgp, _y, _z + dz, **kwargs)

        return (rho2 - rho1) / (2 * dz)

    def get_dsdy_rhop_sp(self, _s, _lgp, _y, _z, ds=0.1, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dPdS|{rho, Y, Z}:
        drhods_rhoy_srho = self.get_drhods_rhoy_sp(_s, _lgp, _y, _z, ds=ds, **kwargs)
        #dPdY|{S, rho, Y}:
        drhody_sp = self.get_drhody_sp(_s, _lgp, _y, _z, dy=dy, **kwargs)

        #dSdY|{rho, P, Z} = -dPdY|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdy_rhop = -drhody_sp/drhods_rhoy_srho # triple product rule

        return dsdy_rhop


    def get_dsdz_rhop_sp(self, _s, _lgp, _y, _z, ds=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #drhodS|{P, Y, Z}:
        drhods_rhoy_srho = self.get_drhods_rhoy_sp(_s, _lgp, _y, _z, ds=ds, **kwargs)
        #drhodZ|{P, rho, Y}:
        drhodz_sp = self.get_drhodz_sp(_s, _lgp, _y, _z, dz=dz, **kwargs)

        #dSdZ|{rho, P, Z} = -drhodZ|{S, rho, Y} / drhodS|{rho, Y, Z}
        dsdz_rhop = -drhodz_sp/drhods_rhoy_srho # triple product rule

        return dsdz_rhop


    ########### Chemical Potential Terms ###########

    def get_dudy_srho(self, _s, _lgrho, _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        u1 = 10**self.get_logu_srho(_s, _lgrho, _y - dy, _z, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, _y + dy, _z, **kwargs)

        return (u2 - u1)/(2 * dy)

    def get_dudz_srho(self, _s, _lgrho, _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        u1 = 10**self.get_logu_srho(_s, _lgrho, _y, _z - dz, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, _y, _z + dz, **kwargs)

        return (u2 - u1)/(2 * dz)

    ########### Conductive Flux Terms ###########

    def get_dtdy_srho(self, _s, _lgrho, _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        func = self.get_logt_srho_tab if tab else self.get_logt_srho
        t1 = 10**self.func(_s, _lgrho, _y - dy, _z, **kwargs)
        t2 = 10**self.func(_s, _lgrho, _y + dy, _z, **kwargs)

        return (t2 - t1)/(2 * dy)

    def get_dtdz_srho(self, _s, _lgrho, _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_srho(_s, _lgrho, _y, _z - dz, **kwargs)
        t2 = 10**self.get_logt_srho(_s, _lgrho, _y, _z + dz, **kwargs)

        return (t2 - t1)/(2 * dz)

    ########## Thermodynamic Consistency Test ###########

    # du/ds_(rho, Y) = T 
    def get_duds_rhoy_srho(self, _s, _lgrho, _y, _z, ds=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        u1 = 10**self.get_logu_srho(_s - ds, _lgrho, _y, _z, **kwargs)
        u2 = 10**self.get_logu_srho(_s + ds, _lgrho, _y, _z, **kwargs)

        return (u2 - u1)/(2 * ds / erg_to_kbbar)

    # -du/dV_(S, Y) = P 
    def get_duds_rhoy_srho(self, _s, _lgrho, _y, _z, drho=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        R0 = 10 **_lgrho
        R1 = R0*(1-drho)
        R2 = R0*(1+drho)

        u1 = 10**self.get_logu_srho(_s, np.log10(R1), _y, _z, **kwargs)
        u2 = 10**self.get_logu_srho(_s, np.log10(R2), _y, _z, **kwargs)

        return (u2 - u1)/((1/R1) - (1/R2))

    ########## Atmospheric update derivative ###########

    def get_dtds_sp(self, _s, _lgp, _y, _z, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_sp(_s - ds, _lgp, _y, _z, **kwargs)
        t2 = 10**self.get_logt_sp(_s + ds, _lgp, _y, _z, **kwargs)

        return (t2 - t1)/(2 * ds / erg_to_kbbar)
    

