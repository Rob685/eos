import numpy as np
from eos import const
from scipy.interpolate import RegularGridInterpolator, splrep, splev
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.optimize import brentq
import os
import pickle
from astropy.constants import k_B
from astropy import units as u
from astropy.constants import u as amu
from eos.cms_eos import get_smix_id_y
import pdb

# kb = k_B.to('erg/K').value
erg_to_kbbar = 1.202723550011625e-08

"""
    This file comes directly from Christopher Mankovich's alice repository (https://github.com/chkvch/alice).
    It is only used to compute the internal energy when called in scvh_eos.py.

    This is necessary because the entropy of mixing calculations in this file are different from our precomputed
    tables used in scvh_eos.py, but those tables do not have the interal energy of mixtures.

"""

class eos:
    def __init__(self, path_to_data=None, fac_for_numerical_partials=1e-10):
        '''
        load the Saumon, Chabrier, van Horn 1995 EOS tables for H and He.
        the eos tables were pulled from mesa-r8845/eos/eosDT_builder/eos_input_data/scvh/.

        the only user-facing method you should need is eos.get.

        to see all dependent variables available, check the attributes eos.h_names and eos.he_names.
        '''

        self.fac_for_numerical_partials = fac_for_numerical_partials

        if not path_to_data: path_to_data = os.environ['ongp_data_path']

        # not using these at present, just making them available for reference
        self.logtmin, self.logtmax = 2.10, 7.06

        self.path_to_h_data = '{}/scvh_h_tab.dat'.format(path_to_data)
        self.path_to_he_data = '{}/scvh_he_tab.dat'.format(path_to_data)

        if os.path.exists('{}/scvh_h.dat.pkl'.format(path_to_data)) and os.path.exists('{}/scvh_he.dat.pkl'.format(path_to_data)):
            with open('{}/scvh_h.dat.pkl'.format(path_to_data), 'rb') as f:
                self.h_data = pickle.load(f)
            with open('{}/scvh_he.dat.pkl'.format(path_to_data), 'rb') as f:
                self.he_data = pickle.load(f)
            assert list(self.h_data) == list(self.he_data)
            self.logtvals = list(self.h_data)
            self.h_names = list(self.h_data[self.logtvals[0]])
            self.he_names = list(self.he_data[self.logtvals[0]])
        else:
            self.load()

        # set up reasonable rectangular grid in logP for the purposes of modelling Jupiter and Saturn-mass planets.
        # points not in the original tables will just return nans.
        self.logpvals = np.union1d(self.h_data[2.1]['logp'], self.h_data[5.06]['logp'])
        self.logpmin, self.logpmax = 5.0, 17. # january 17 2017: had logpmax=17 for daniel
        self.logpvals = self.logpvals[self.logpvals >= self.logpmin]
        self.logpvals = self.logpvals[self.logpvals <= self.logpmax]

        npts_t = len(self.logtvals)
        npts_p = len(self.logpvals)
        basis_shape = (npts_p, npts_t)

        self.h_data_rect = {}
        self.he_data_rect = {}

        def value_on_node(table, var, logp, logt):
            d = {'h':self.h_data, 'he':self.he_data}[table]
            match = d[logt][var][d[logt]['logp'] == logp]
            if len(match) == 1:
                return match[0]
            else:
                return 0

        for name in list(self.h_names):
            if name == 'logp': continue
            self.h_data_rect[name] = np.zeros(basis_shape)
        for name in list(self.he_names):
            if name == 'logp': continue
            self.he_data_rect[name] = np.zeros(basis_shape)

        for ip, logp in enumerate(self.logpvals):
            for it, logt in enumerate(self.logtvals):
                for name in list(self.h_names):
                    if name == 'logp': continue
                    self.h_data_rect[name][ip, it] = value_on_node('h', name, logp, logt)
                for name in list(self.he_names):
                    if name == 'logp': continue
                    self.he_data_rect[name][ip, it] = value_on_node('he', name, logp, logt)

        self.get_h = {}
        self.get_he = {}
        for name in list(self.h_names):
            if name == 'logp': continue
            self.get_h[name] = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_data_rect[name],bounds_error=False, fill_value=None)
            # self.get_h[name] = rbs(self.logpvals, self.logtvals, self.h_data_rect[name])
        for name in list(self.he_names):
            if name == 'logp': continue
            self.get_he[name] = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_data_rect[name],bounds_error=False, fill_value=None)
            # self.get_he[name] = rbs(self.logpvals, self.logtvals, self.he_data_rect[name])

        del(self.h_data)
        del(self.he_data)
        del(self.h_data_rect)
        del(self.he_data_rect)

    def load(self):
        '''
        read the ascii scvh tables into handier python dicts. this is called by __init__ unless
        those dicts are already saved as pickles scvh_*.dat.pkl in the data path. loading the
        pickles involves less io overhead.
        '''

        # for logt = 3.38, 3.46, 3.54, extrapolate up to logp = 11.6, 11.8, 12.0
        # on isotherms. necessary to have these points for some cold Saturn models.
        logpvals_to_fill = np.array([11.6, 11.8, 12.0, 12.2, 12.4])
        logtvals_to_fill = np.array([3.38, 3.46, 3.54])
        npts_extrap = 5

        self.h_names = 'logp', 'xh2', 'xh', 'logrho', 'logs', 'logu', 'rhot', 'rhop', 'st', 'sp', 'grada'
        self.h_data = {}
        logtvals_h = np.array([])
        with open(self.path_to_h_data) as fr:
            for i, line in enumerate(fr.readlines()):
                if len(line.split()) == 2:
                    logt, nrows = line.split()
                    logt = float(logt)
                    nrows = int(nrows)
                    # _data = np.genfromtxt(path_to_h_data, skip_header=i+1, max_rows=nrows, names=self.h_names)
                    with open(self.path_to_h_data, "rb") as f:
                        from itertools import islice
                        _data = np.genfromtxt(islice(f, i+1, i+nrows+1), names=self.h_names)

                    # this ndarray is too annoying, convert data for this logt to a dict
                    data = {}
                    for name in self.h_names:
                        data[name] = _data[name]

                    if logt in logtvals_to_fill:
                        for name in self.h_names:
                            if name == 'logp': continue
                            tck = splrep(data['logp'][-npts_extrap:], data[name][-npts_extrap:], k=1)
                            new = splev(logpvals_to_fill, tck)
                            data[name] = np.append(data[name], new)
                        data['logp'] = np.append(data['logp'], logpvals_to_fill)
                    self.h_data[logt] = data

                    logtvals_h = np.append(logtvals_h, logt)

        self.he_names = 'logp', 'xhe', 'xhep', 'logrho', 'logs', 'logu', 'rhot', 'rhop', 'st', 'sp', 'grada'
        self.he_data = {}
        logtvals_he = np.array([])
        with open(self.path_to_he_data) as fr:
            for i, line in enumerate(fr.readlines()):
                if len(line.split()) == 2:
                    logt, nrows = line.split()
                    logt = float(logt)
                    nrows = int(nrows)
                    # _data = np.genfromtxt(path_to_he_data, skip_header=i+1, max_rows=nrows, names=self.he_names)
                    with open(self.path_to_he_data, "rb") as f:
                        from itertools import islice
                        _data = np.genfromtxt(islice(f, i+1, i+nrows+1), names=self.he_names)

                    # this ndarray is too annoying, convert data for this logt to a dict
                    data = {}
                    for name in self.he_names:
                        data[name] = _data[name]

                    if logt in logtvals_to_fill:
                        for name in self.he_names:
                            if name == 'logp': continue
                            tck = splrep(data['logp'][-npts_extrap:], data[name][-npts_extrap:], k=1)
                            new = splev(logpvals_to_fill, tck)
                            data[name] = np.append(data[name], new)
                        data['logp'] = np.append(data['logp'], logpvals_to_fill)
                    self.he_data[logt] = data

                    logtvals_he = np.append(logtvals_he, logt)

        assert np.all(logtvals_h == logtvals_he) # verify H and He are on the same temperature grid
        self.logtvals = logtvals_h

        with open('{}.pkl'.format(self.path_to_h_data), 'wb') as f:
            pickle.dump(self.h_data, f)
            print('wrote cache to {}.pkl'.format(self.path_to_h_data))
        with open('{}.pkl'.format(self.path_to_he_data), 'wb') as f:
            pickle.dump(self.he_data, f)
            print('wrote cache to {}.pkl'.format(self.path_to_he_data))

    # these wrapper functions are the ones meant to be called externally.
    def get(self, logp, logt, y):
        '''return all eos results for a (logp, logt) pair at any H-He mixture.'''
        if type(logp) is np.float64 or type(logp) is float: logp = np.array([logp])
        if type(logt) is np.float64 or type(logt) is float: logt = np.array([logt])
        if type(y) is np.float64 or type(y) is float: y = np.array([y])
        pair = (logp, logt)
        res = {}
        if type(y) is np.ndarray:
            if not np.all(0. <= y) and np.all(y <= 1.):
                raise ValueError('invalid helium mass fraction(s)')
        elif type(y) is np.float64:
            if not 0. <= y <= 1.:
                raise ValueError('invalid helium mass fraction %f' % y)
        try:
            res = self.get_hhe(pair, y)
        except ValueError:
            raise ValueError('probably out of bounds in logP, logT, or Y -- did you accidentally pass P, T? (or loglogP, loglogT?)')
            # print(logp)
            # print(logt)
            # raise

        res['logp'] = logp
        res['logt'] = logt

        return res

    # convenience routines for essential quantities

    def get_logrho(self, logp, logt, y):
        return self.get(logp, logt, y)['logrho']

    def get_logs(self, logp, logt, y):
        return self.get(logp, logt, y)['logs']

    def get_logsmix(self, logp, logt, y):
        return self.get(logp, logt, y)['logsmix']

    def get_logu(self, logp, logt, y):
        return self.get(logp, logt, y)['logu']

    def get_grada(self, logp, logt, y):
        return self.get(logp, logt, y)['grada']

    def get_gamma1(self, logp, logt, y):
        return self.get(logp, logt, y)['gamma1']

    def get_chirho(self, logp, logt, y):
        return self.get(logp, logt, y)['chirho']

    def get_chit(self, logp, logt, y):
        return self.get(logp, logt, y)['chit']

    def get_cv(self, logp, logt, y):
        return self.get(logp, logt, y)['cv']

    def get_cp(self, logp, logt, y):
        return self.get(logp, logt, y)['cp']

    def rhot_get(self, logrho, logt, y, logp_guess=None):
        # if want to use (rho, t, y) as basis.
        # comes at the expense of doing root finds; looks to take an order of magnitude
        # more cpu time than a call to self.get (30 ms versus 2.5 ms).
        # this is prohibitively slow -- 30 ms per zone times 10^3 zones is 30 seconds.
        # then a 100-step evolutionary model is taking close to an hour.

        if type(logrho) is float or type(logrho) is np.float64: logrho = np.array([logrho])
        if type(logt) is float or type(logt) is np.float64: logt = np.array([logt])
        if type(y) is float or type(y) is np.float64: y = np.array([y])

        assert len(logrho) == 1 and len(logt) == 1 and len(y) == 1, 'rhot_get only works for length-1 arrays at present.'
        def zero_me(logpval):
            return self.get(np.array([logpval]), logt, y)['logrho'] - logrho

        if logp_guess:
            # a good starting guess only helps marginally with time for root find (order unity)
            logpmin = logp_guess * (1. - 5e-4)
            logpmax = logp_guess * (1. + 5e-4)
        else:
            logt_lower = self.logtvals[np.where(np.sign(self.logtvals - logt) > 0)[0][0] - 1]
            logt_upper = self.logtvals[np.where(np.sign(self.logtvals - logt) > 0)[0][0]]
            logpmax_lot = max(self.h_data[logt_lower]['logp'])
            logpmax_hit = max(self.h_data[logt_upper]['logp'])
            logpmax = min(logpmax_lot, logpmax_hit)
            logpmax = min(self.logpmax, logpmax)
            logpmin = self.logpmin


        # print 'logp brackets for root find: ', self.logpmin, logpmax
        logp, solve_details = brentq(zero_me, logpmin, logpmax, full_output=True)
        # print 'brentq found root (logp = %f) in %i iterations' % (logp, solve_details.iterations)
        res = self.get(logp, logt, y)

        res['logp'] = logp
        res['logt'] = logt

        return res


    # aka chi_y. since it's just additive volume, it's simple analytically
    def get_dlogrho_dlogy(self, logp, logt, y):
        rho = 10 ** self.get_logrho(logp, logt, y)
        rho_h = 10 ** self.get_h['logrho']((logp, logt))
        rho_he = 10 ** self.get_he['logrho']((logp, logt))
        return -1. * rho * y * (1. / rho_he - 1. / rho_h)

    def get_hhe(self, pair, y):
        '''combines the results of the hydrogen and helium equations of state for an arbitrary
        mixture of the two. takes the helium mass fraction Y as input. makes use of the equations
        in SCvH 1995, namely equations 39, 41, 45-47, and 53-56, with typos corrected as per
        Baraffe et al. 2008 (footnote 4).
        pair is just the tuple (logp, logt).'''

        # can't be bothered to sort out handling of pure H or He for now
        if np.any(y == 0.):
            raise ValueError('get_hhe cannot handle pure H for the time being. try a mixture.')
        if np.any(y == 1.):
            raise ValueError('get_hhe cannot handle pure He for the time being. try a mixture.')

        def get_beta(y):
            '''eq. 54 of SCvH95.
            y is the helium mass fraction.'''
            try:
                return (const.mh / const.mhe) * (y / (1. - y))
            except ZeroDivisionError:
                print('tried divide by zero in beta')
                return np.nan

        def get_gamma(xh, xh2, xhe, xhep):
            '''eq. 55 of SCvH95.
            xh is the number fraction of atomic H (relative to all particles, inc. electrons)
            xh2 is that of molecular H
            xhe is that of neutral helium
            xhep is that of singly ionized helium
            '''
            return 1.5 * (1. + xh + 3 * xh2) / (1. + 2 * xhe + xhep)

        def get_delta(y, xh, xh2, xhe, xhep):
            '''eq. 56 of SCvH95.
            input parameters are as for the functions beta and gamma.prefactor is corrected as pointed out in
            footnote 4 of Baraffe et al. 2008, namely it is flipped relative to how it appears in SCvH95 eq. 56.'''

            species_num = (2. - 2. * xhe - xhep) # proportional to the abundance of free electrons assuming pure He
            species_den = (1. - xh2 - xh) # proportional to the abundance of free electrons assuming pure H

            # print 'type xh, xh2, xhe, xhep:', type(xh), type(xh2), type(xhe), type(xhep)
            # print 'xh, xh2, xhe, xhep:', xh, xh2, xhe, xhep
            # print type(species_num)
            # print type(species_den)

            if type(xh) is np.ndarray:
                assert type(species_num) is np.ndarray, 'species are ndarray, but delta numerator is not.'
                assert type(species_den) is np.ndarray, 'species are ndarray, but delta denominator is not.'
                # number density of free e- for one of the pure species is sometimes a tiny negative number.
                # in cases where there are no free electrons, delta does not matter (prefactor vanishes);
                # it's just crucial that it's > 0 and not a nan.
                species_num[species_num <= 0.] = 1.
                species_den[species_den <= 0.] = 1.
            elif type(xh) is np.float64:
                if species_num <= 0.: species_num = 1.
                if species_den <= 0.: species_den = 1.
            else:
                raise TypeError('input type %s not recognized in get_delta' % str(type(xh)))

            return 1.5 * (species_num / species_den) * get_beta(y) * get_gamma(xh, xh2, xhe, xhep)

        def get_smix(y, xh, xh2, xhe, xhep):
            '''ideal entropy of mixing for H/He, same units as H and He tables -- erg K^-1 g^-1.
            eq. 53 of SCvH95.
            input parameters as above.'''

            beta = get_beta(y)
            gamma = get_gamma(xh, xh2, xhe, xhep)
            delta = get_delta(y, xh, xh2, xhe, xhep)

            # assert not np.any(np.isnan(beta)), 'beta nan'
            # assert not np.any(np.isnan(gamma)), 'gamma nan'
            # assert not np.any(np.isnan(delta)), 'delta nan'

            # print d
            xeh = (1. / 2) * (1. - xh2 - xh) # number fraction of e-, for pure H -- eq. 34.
            xehe = (1. / 3) * (2. - 2. * xhe - xhep) # number fraction of e- for pure He -- eq. 35.
            return const.kb * (1. - y) / const.mh * 2. / (1. + xh + 3. * xh2) * \
                (np.log(1. + beta * gamma) \
                - xeh * np.log(1. + delta) \
                + beta * gamma * (np.log(1. + 1. / beta / gamma) \
                - xehe * np.log(1. + 1. / delta)))

        def species_partials(pair, y, f):
            # f is a dimensionless factor by which we perturb logp, logt to compute centered differences for numerical partials.
            # returns pair (d*_dlogp, d*_dlogt) of quadruples (dxh2_dlog*, dxh_dlog*, dhe_dlog*, dhep_dlog*)

            logp, logt = pair

            if np.any(logp * (1. + f) > self.logpmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))
            if np.any(logp * (1. - f) < self.logpmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))
            if np.any(logt * (1. + f) > self.logtmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))
            if np.any(logt * (1. - f) < self.logtmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))

            pair_p_plus = logp * (1. + f), logt
            pair_p_minus = logp * (1. - f), logt
            xh2_p_plus = self.get_h['xh2'](pair_p_plus)
            xh2_p_minus = self.get_h['xh2'](pair_p_minus)
            xh_p_plus = self.get_h['xh'](pair_p_plus)
            xh_p_minus = self.get_h['xh'](pair_p_minus)
            xhe_p_plus = self.get_he['xhe'](pair_p_plus)
            xhe_p_minus = self.get_he['xhe'](pair_p_minus)
            xhep_p_plus = self.get_he['xhep'](pair_p_plus)
            xhep_p_minus = self.get_he['xhep'](pair_p_minus)

            pair_t_plus = logp, logt * (1. + f)
            pair_t_minus = logp, logt * (1. - f)
            xh2_t_plus = self.get_h['xh2'](pair_t_plus)
            xh2_t_minus = self.get_h['xh2'](pair_t_minus)
            xh_t_plus = self.get_h['xh'](pair_t_plus)
            xh_t_minus = self.get_h['xh'](pair_t_minus)
            xhe_t_plus = self.get_he['xhe'](pair_t_plus)
            xhe_t_minus = self.get_he['xhe'](pair_t_minus)
            xhep_t_plus = self.get_he['xhep'](pair_t_plus)
            xhep_t_minus = self.get_he['xhep'](pair_t_minus)

            dxh2_dlogp = (xh2_p_plus - xh2_p_minus) / (2. * f)
            dxh_dlogp = (xh_p_plus - xh_p_minus) / (2. * f)
            dxhe_dlogp = (xhe_p_plus - xhe_p_minus) / (2. * f)
            dxhep_dlogp = (xhep_p_plus - xhep_p_minus) / (2. * f)

            dxh2_dlogt = (xh2_t_plus - xh2_t_minus) / (2. * f)
            dxh_dlogt = (xh_t_plus - xh_t_minus) / (2. * f)
            dxhe_dlogt = (xhe_t_plus - xhe_t_minus) / (2. * f)
            dxhep_dlogt = (xhep_t_plus - xhep_t_minus) / (2. * f)

            d_dlogp = (dxh2_dlogp, dxh_dlogp, dxhe_dlogp, dxhep_dlogp)
            d_dlogt = (dxh2_dlogt, dxh_dlogt, dxhe_dlogt, dxhep_dlogt)

            return d_dlogp, d_dlogt

        res_h = {}
        res_h['xh2'] = self.get_h['xh2'](pair)
        res_h['xh'] = self.get_h['xh'](pair)
        res_h['xhe'] = np.zeros_like(pair[0])
        res_h['xhep'] = np.zeros_like(pair[0])
        res_h['logrho'] = self.get_h['logrho'](pair)
        res_h['logs'] = self.get_h['logs'](pair)
        res_h['logu'] = self.get_h['logu'](pair)
        res_h['rhot'] = self.get_h['rhot'](pair)
        res_h['rhop'] = self.get_h['rhop'](pair)
        res_h['st'] = self.get_h['st'](pair)
        res_h['sp'] = self.get_h['sp'](pair)
        res_h['grada'] = self.get_h['grada'](pair)

        res_he = {}
        res_he['xh2'] = np.zeros_like(pair[0])
        res_he['xh'] = np.zeros_like(pair[0])
        res_he['xhe'] = self.get_he['xhe'](pair)
        res_he['xhep'] = self.get_he['xhep'](pair)
        res_he['logrho'] = self.get_he['logrho'](pair)
        res_he['logs'] = self.get_he['logs'](pair)
        res_he['logu'] = self.get_he['logu'](pair)
        res_he['rhot'] = self.get_he['rhot'](pair)
        res_he['rhop'] = self.get_he['rhop'](pair)
        res_he['st'] = self.get_he['st'](pair)
        res_he['sp'] = self.get_he['sp'](pair)
        res_he['grada'] = self.get_he['grada'](pair)

        # assert not np.any(np.isnan(res_h['xh2'])), 'got nan in h eos call within overall P-T limits. probably off the original tables.'
        # assert not np.any(np.isnan(res_he['xhe'])), 'got nan in he eos call within overall P-T limits. probably off the original tables.'

        res = {}
        rho_h = res['rho_h'] = 10 ** res_h['logrho']
        rho_he = res['rho_he'] = 10 ** res_he['logrho']
        rho =  1 / (((1. - y) / rho_h) + (y / rho_he)) # additive volume rule -- eq. 39.
        #rho = rhoinv ** -1.
        res['logrho'] = np.log10(rho)

        u = (1. - y) * 10 ** res_h['logu'] + y * 10 ** res_he['logu'] # also additive volume -- eq. 40.
        res['logu'] = np.log10(u)

        res['rhot'] = (1. - y) * rho / rho_h * res_h['rhot'] + y * rho / rho_he * res_he['rhot']
        res['rhop'] = (1. - y) * rho / rho_h * res_h['rhop'] + y * rho / rho_he * res_he['rhop']

        # note-- additive volume approximation for internal energy and density means if you do the energy
        # equation with du, drho, you're leaving out the contribution from d(entropy of mixing).
        # this is included if you're differencing the entropy, which includes s_mix.

        s_h = 10 ** res_h['logs']
        s_he = 10 ** res_he['logs']
        xh = res_h['xh']
        xh2 = res_h['xh2']
        xhe = res_he['xhe']
        xhep = res_he['xhep']
        smix_old = get_smix(y, xh, xh2, xhe, xhep)
        try:
            smix = get_smix_id_y(y)/erg_to_kbbar #+ smix_old
            #pdb.set_trace()
        except:
            pdb.set_trace()
            raise
        s = (1. - y) * s_h + y * s_he - smix # entropy for an ideal (noninteracting) mixture -- eq. 41. ROB: commented out smix
        #print(s)
        res['logs'] = np.log10(s)
        res['logsmix'] = np.log10(smix)
        res['xh'] = xh
        res['xh2'] = xh2
        res['xhe'] = xhe
        res['xhep'] = xhep

        # the bits to compute derivatives of entropy, and thus grad_ad, make use of analytic derivatives of SCvH 1995 eq. 53 with respect to abundances
        # of the four independent species (see CM notes 11/29/2016). the derivatives of each abundance with respect to logp and logt are computed
        # numerically with centered finite differences. equations with alphanumeric labels (A*) are in the handwritten notes.

        dxi_dlogp, dxi_dlogt = species_partials(pair, y, f=self.fac_for_numerical_partials)
        dxh2_dlogp, dxh_dlogp, dxhe_dlogp, dxhep_dlogp = dxi_dlogp
        dxh2_dlogt, dxh_dlogt, dxhe_dlogt, dxhep_dlogt = dxi_dlogt

        # prefactor defined such that smix = smix_prefactor * s_tilde, where s_tilde is the dimensionless entropy I work with in the handwritten notes. (in code below i'll refer to s_tilde as ss)
        smix_prefactor = 2. * const.kb * (1. - y) / const.mh

        beta = get_beta(y)
        gamma = get_gamma(xh, xh2, xhe, xhep)
        delta = get_delta(y, xh, xh2, xhe, xhep)

        # eqs. (A5-A8)
        dgamma_dxh2 = 9. / 2 * (1. + 2 * xhe + xhep) ** -1
        dgamma_dxh = dgamma_dxh2 / 3.
        dgamma_dxhe = -3. * (1. + xh + 3 * xh2) / (1. + 2 * xhe + xhep) ** 2
        dgamma_dxhep = dgamma_dxhe / 2.

        # eqs. (A9-A12)
        num = (2. - 2 * xhe - xhep)
        num[num < 0.] = 0
        den = (1. - xh2 - xh)

        # special handling is required for cases where hydrogen (and thus helium) is totally neutral, or else dividing by zero
        hydrogen_is_neutral = den == 0.

        if type(xh) is np.ndarray:
            den[hydrogen_is_neutral] = 1. # kludge to guarantee that delta derivs are calculable. we'll zero them in the neutral case afterward.
        elif type(xh) is np.float64:
            if hydrogen_is_neutral: den = 1.
        else:
            raise TypeError('type %s not recognized in get_hhe' % str(type(xh)))

        ddelta_dxh2 = 2. / 3 * num / den ** 2 * beta * gamma + delta / gamma * dgamma_dxh2
        ddelta_dxh = 2. / 3 * num / den ** 2 * beta * gamma + delta / gamma * dgamma_dxh
        ddelta_dxhe = - 4. / 3 * den ** -1 * beta * gamma + delta / gamma * dgamma_dxhe
        ddelta_dxhep = -2. / 3 * den ** -1 * beta * gamma + delta / gamma * dgamma_dxhep

        ddelta_dxh2[hydrogen_is_neutral] = 0.
        ddelta_dxh[hydrogen_is_neutral] = 0.
        ddelta_dxhe[hydrogen_is_neutral] = 0.
        ddelta_dxhep[hydrogen_is_neutral] = 0.

        in_square_brackets = np.log(1. + 1. / beta / gamma) - 1. / 3 * (2. - 2 * xhe - xhep) * np.log(1. + 1. / delta)
        in_curly_brackets = np.log(1. + beta * gamma) - 1. / 2 * (1. - xh2 - xh) * np.log(1. + delta) + \
                            beta * gamma * in_square_brackets

        dss_dxh2 = -1. * (1. + xh + 3 * xh2) ** -2 * 3 * in_curly_brackets + \
                    (1. + xh + 3 * xh2) ** -1 * ((1. + beta * gamma) ** -1 * beta * dgamma_dxh2 + \
                    1. / 2 * np.log(1. + delta) - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxh2 + \
                    beta * dgamma_dxh2 * in_square_brackets + \
                    beta * gamma * ((1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxh2)) # eq. (A1)
        dss_dxh = -1. * (1. + xh * 3 * xh2) ** -2 * in_curly_brackets + \
                    (1. + xh + 3 * xh2) ** -1 * ((1. + beta * gamma) ** -1 * beta * dgamma_dxh + \
                    1. / 2 * np.log(1. + delta) - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxh + \
                    beta * dgamma_dxh * in_square_brackets + \
                    beta * gamma * ((1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxh)) # eq. (A2)
        dss_dxhe = (1. + xh + 3 * xh2) ** -1 * ( \
                    (1. + beta * gamma) ** -1 * beta * dgamma_dxhe - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxhe + \
                    beta * dgamma_dxhe * in_square_brackets + beta * gamma * ( \
                    (1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxhe + 2. / 3 * np.log(1. + 1. / delta) - \
                    1. / 3 * (2. - 2 * xhe - xhep) * (1. + 1. / delta) ** -1 * (-1.) * delta ** -2 * ddelta_dxhe)) # eq. (A3)
        dss_dxhep = (1. + xh + 3 * xh2) ** -1 * ( \
                    (1. + beta * gamma) ** -1 * beta * dgamma_dxhep - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxhep + \
                    beta * dgamma_dxhep * in_square_brackets + \
                    beta * gamma * ((1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxhep + \
                    1. / 3 * np.log(1. + 1. / delta) - 1. / 3 * (2. - 2 * xhe - xhep) * (1. + 1. / delta) ** -1 * (-1.) / delta ** 2 * ddelta_dxhep))

        dsmix_dxh2 = dss_dxh2 * smix_prefactor
        dsmix_dxh = dss_dxh * smix_prefactor
        dsmix_dxhe = dss_dxhe * smix_prefactor
        dsmix_dxhep = dss_dxhep * smix_prefactor

        dsmix_dlogt = dsmix_dxh2 * dxh2_dlogt + dsmix_dxh * dxh_dlogt + dsmix_dxhe * dxhe_dlogt + dsmix_dxhep * dxhep_dlogt
        dsmix_dlogp = dsmix_dxh2 * dxh2_dlogp + dsmix_dxh * dxh_dlogp + dsmix_dxhe * dxhe_dlogp + dsmix_dxhep * dxhep_dlogp

        dlogsmix_dlogt = dsmix_dlogt / smix
        dlogsmix_dlogp = dsmix_dlogp / smix

        res['st'] = (1. - y) * s / s_h * res_h['st'] + y * s / s_he * res_he['st'] + smix / s * dlogsmix_dlogt
        res['sp'] = (1. - y) * s / s_h * res_h['sp'] + y * s / s_he * res_he['sp'] + smix / s * dlogsmix_dlogp

        res['grada'] = -1. * res['sp'] / res['st']

        logp, logt = pair
        # dpdt_const_rho = - 10 ** logp / 10 ** logt * res['rhot'] / res['rhop']
        # dudt_const_rho = s * (res['st'] - res['sp'] * res['rhot'] / res['rhop'])
        # dpdu_const_rho = dpdt_const_rho / dudt_const_rho # had a 1. / rho for some reason?
        # gamma3 = 1. + dpdu_const_rho # cox and giuli 9.93a
        # gamma1 = (gamma3 - 1.) / res['grada']
        # res['gamma3'] = gamma3
        # res['gamma1'] = gamma1
        # res['chirho'] = res['rhop'] ** -1 # rhop = dlogrho/dlogp|t
        # res['chit'] = dpdt_const_rho * 10 ** logt / 10 ** logp
        res['chiy'] = -1. * 10 ** res['logrho'] * y * (1. / 10 ** res_he['logrho'] - 1. / 10 ** res_h['logrho']) # dlnrho/dlnY|P,T
        res['chirho'] = 1. / res['rhop']
        res['chit'] = - res['rhot'] / res['rhop']
        res['gamma1'] = res['chirho'] / (1. - res['chit'] * res['grada'])
        res['gamma3'] = 1. + res['gamma1'] * res['grada']
        res['csound'] = np.sqrt(10 ** logp / 10 ** res['logrho'] * res['gamma1'])

        # from mesa's scvh in mesa/eos/eosPT_builder/src/scvh_eval.f
        # 1005:      Cv = chiT * P / (rho * T * (gamma3 - 1)) ! C&G 9.93
        # 1006:      Cp = Cv + P * chiT**2 / (Rho * T * chiRho) ! C&G 9.86
        res['cv_alt'] = res['chit'] * 10 ** logp / (10 ** res['logrho'] * 10 ** logt * (res['gamma3'] - 1.)) # erg g^-1 K^-1
        res['cp_alt'] = res['cv_alt'] + 10 ** logp * res['chit'] ** 2 / (10 ** res['logrho'] * 10 ** logt * res['chirho']) # erg g^-1 K^-1
        res['cp'] = 10 ** res['logs'] * res['st']
        res['cv'] = res['cp'] * res['chirho'] / res['gamma1'] # Unno 13.87

        return res

    def plot_pt_coverage(self, ax=None, symbol='.', **kwargs):
        if not ax:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        for logt in self.logtvals:
            logp = self.h_data[logt]['logp']
            ax.plot(10**logp, np.ones_like(logp) * 10**logt, symbol, **kwargs)

        # ax.set_xlim(-1.5, 19.5)
        # ax.set_xlabel(r'$\log\ P$')
        # ax.set_ylabel(r'$\log\ T$')

    def plot_rhot_coverage(self, ax=None, symbol='.', **kwargs):
        if not ax:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        for logt in self.logtvals:
            logrho = self.h_data[logt]['logrho']
            ax.plot(10 ** logrho, np.ones_like(logrho) * 10 ** logt, symbol, **kwargs)

        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim(3e-2, 5e19)
        ax.set_ylim(ymin=3e1)
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r'$T$')
