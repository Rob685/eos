import numpy as np
from astropy import units as u
from astropy.constants import u as amu
from astropy.constants import k_B
from scipy.interpolate import RectBivariateSpline as RBS
import cms_newton_raphson as cms
import pandas as pd
import main_eos as eos
import pdb
import os
import re
import scvh_nr

'''This module calculates the derivatives at constant temperature/pressure or temperature/density.'''
'''Call the thermo_sp.py module for derivatives at constant entropy or pressure'''

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

# class thermo_sp:
#     def __init__(self, Y):
#         #self.Y = Y
#         #self.tab = tab

#         # in the inverted tables, s is in kb/baryon. Need to be in log cgs for later.
#         self.s, self.p, self.t, self.r = eos.mixture(Y, 'cms_hc_lowes')
#         #self.s, self.p = mixture.s_arr[0][:,0], mixture.p_arr[0,:][0]
#         self.s /= erg_to_kbbar # converting to cgs

#         # s should be in log of cgs for later
#         self.rho_interp = RBS(np.log10(self.s[:,0]), self.p[0,:], self.r)
#         self.t_interp = RBS(np.log10(self.s[:,0]), self.p[0,:], self.t)

class scvh_thermo_tr:
    def __init__(self, Y):
        self.Y = Y
        if 0.22 <= self.Y <= 0.25:
            bounds_1, self.stab1 = scvh_nr.scvh_reader('s22scz.dat')
            _, self.ptab1 = scvh_nr.scvh_reader('p22scz.dat')
        
            self.Y1, INDEX, self.R1, self.R2, self.T1, self.T2, self.T12, self.T22 = bounds_1

            self.logrhovals1 = np.linspace(self.R1, self.R2, self.stab1.shape[0])

            bounds_2, self.stab2 = scvh_nr.scvh_reader('s25scz.dat')
            _, self.ptab2 = scvh_nr.scvh_reader('p25scz.dat')
        
            self.Y2, INDEX, self.R1, self.R2, self.T1, self.T2, self.T12, self.T22 = bounds_2

            self.logrhovals2 = np.linspace(self.R1, self.R2, self.stab2.shape[0])
        
    def deltaT(self, R):
        eta = (R - self.R1)/(self.R2 - self.R1)
        return eta*(self.T2 - self.T1) + (1-eta)*(self.T22 - self.T12)

    def get_tarr(self, R):
        m1 = (self.T12 - self.T1)/(self.R2 - self.R1)
        b = self.T1 - m1*self.R1
        T1p = R*m1 + b
        #T1p = (R - R1)*m1 + b
        T2p = T1p + self.deltaT(R)

        return np.linspace(T1p, T2p, 100)

    def s_interp(self, R, logrho_array, SL):
        logt_array = self.get_tarr(R)
        interp_s = RBS(logrho_array, logt_array, SL)
        #interp_p = RBS(logrho_array, logt_array, PL)

        return interp_s

    def p_interp(self, R, logrho_array, PL):
        logt_array = self.get_tarr(R)
        #interp_s = RBS(logrho_array, logt_array, SL)
        interp_p = RBS(logrho_array, logt_array, PL)

        return interp_p

    # def get_s(self, R, T):
    #     return self.s_interp.ev(R, self.logrhovals1, self.stab1).ev(R, T)

    # def get_p(self, R, T):
    #     return self.p_interp.ev(R, self.logrhovals1, self.stab1).ev(R, T)

    def get_smix(self, R, T):
        s_1 = (10**self.s_interp(R, self.logrhovals1, self.stab1).ev(R, T))/erg_to_kbbar
        s_2 = (10**self.s_interp(R, self.logrhovals2, self.stab2).ev(R, T))/erg_to_kbbar

        eta1 = (self.Y2 - self.Y)/(self.Y2 - self.Y1)
        eta2 = 1-eta1
        smix = eta1*s_1 + eta2*s_2
        return smix

    def get_pmix(self, R, T):
        p_1 = (10**self.p_interp(R, self.logrhovals1, self.ptab1).ev(R, T))
        p_2 = (10**self.p_interp(R, self.logrhovals2, self.ptab2).ev(R, T))

        eta1 = (self.Y2 - self.Y)/(self.Y2 - self.Y1)
        eta2 = 1-eta1
        pmix = eta1*p_1 + eta2*p_2
        return pmix
                
        # self.y_arr = np.array([y for y in sorted(yvals)])
        # # sorted according to increasing y:
        # self.s_arr = np.array([x for _, x in sorted(zip(self.y_arr, s_arr))])
        # self.p_arr = np.array([x for _, x in sorted(zip(self.y_arr, p_arr))])
        # self.t_arr = np.array([x for _, x in sorted(zip(self.y_arr, t_arr))])
        # self.r_arr = np.array([x for _, x in sorted(zip(self.y_arr, r_arr))])


class cms_thermo_tp:
    def __init__(self, tab):
        #self.Y = Y
        hename = 'TABLE_HE_TP_v1'
        if tab == 2019:
            hname = 'TABLE_H_TP_v1'
        elif tab == 2021:
            hname = 'TABLE_H_TP_effective'

        self.hdata = cms.grid_data(cms.cms_reader(hname, tab))
        self.hedata = cms.grid_data(cms.cms_reader(hename, tab)) 

        # reading the smix correction values form Howard & Guillot (2023)

        data_hc = pd.read_csv('data/HG23_Vmix_Smix.csv', delimiter=',')
        data_hc = data_hc[(data_hc['LOGT'] <= 5.0) & (data_hc['LOGT'] != 2.8)]
        data_hc = data_hc.rename(columns={'LOGT':'logt', 'LOGP':'logp'}).sort_values(by=['logt', 'logp'])

        grid_hc = cms.grid_data(data_hc)
        self.svals_hc = grid_hc['Smix']

        logpvals_hc = grid_hc['logp'][0]
        logtvals_hc = grid_hc['logt'][:,0]

        self.smix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Smix']) # Smix will be in cgs... not log cgs.
        self.vmix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Vmix'])


        self.logtvals = self.hdata['logt'][:,0]
        self.logpvals = self.hdata['logp'][0]

        ### H ###

        svals_h = self.hdata['logs']
        rhovals_h = self.hdata['logrho']
        loguvals_h = self.hdata['logu']

        self.get_s_h = RBS(self.logtvals, self.logpvals, svals_h) # x or y are not changing so can leave out to speed things up
        self.get_rho_h = RBS(self.logtvals, self.logpvals, rhovals_h)
        self.get_u_h = RBS(self.logtvals, self.logpvals, loguvals_h)

        # derivatives

        self.get_rhot_h = RBS(self.logtvals, self.logpvals, self.hdata['dlrho/dlT_P'])
        self.get_rhop_h = RBS(self.logtvals, self.logpvals, self.hdata['dlrho/dlP_T'])
        self.get_sp_h = RBS(self.logtvals, self.logpvals, self.hdata['dlS/dlP_T'])
        self.get_st_h = RBS(self.logtvals, self.logpvals, self.hdata['dlS/dlT_P'])
        self.get_grada_h = RBS(self.logtvals, self.logpvals, self.hdata['grad_ad'])

        #### He ####

        svals_he = self.hedata['logs']
        rhovals_he = self.hedata['logrho']
        loguvals_he = self.hedata['logu']

        # s(t, rho)
        # p(t, rho)
        self.get_s_he = RBS(self.logtvals, self.logpvals, svals_he) 
        self.get_rho_he = RBS(self.logtvals, self.logpvals, rhovals_he)
        self.get_u_he = RBS(self.logtvals, self.logpvals, loguvals_he)

        # derivatives

        self.get_rhot_he = RBS(self.logtvals, self.logpvals, self.hedata['dlrho/dlT_P'])
        self.get_rhop_he = RBS(self.logtvals, self.logpvals, self.hedata['dlrho/dlP_T'])
        self.get_sp_he = RBS(self.logtvals, self.logpvals, self.hedata['dlS/dlP_T'])
        self.get_st_he = RBS(self.logtvals, self.logpvals,self.hedata['dlS/dlT_P'])
        self.get_grada_he = RBS(self.logtvals, self.logpvals, self.hedata['grad_ad'])

    # def get_st_h(self, lgt, lgp): # at a constant pressure
    #     return self.get_s_h.ev(lgt, lgp, dx=1)
    # def get_sp_h(self, lgt, lgp): # at a constant temperatre
    #     return self.get_s_h.ev(lgt, lgp, dy=1)
    # def get_rhot_h(self, lgt, lgp): # at a constant pressure
    #     return self.get_rho_h.ev(lgt, lgp, dx=1)
    # def get_rhop_h(self, lgt, lgp): # at a constant temperatre
    #     return self.get_rho_h.ev(lgt, lgp, dy=1)
    # def get_ut_h(self, lgt, lgp): # at a constant pressure
    #     return self.get_u_h.ev(lgt, lgp, dx=1)
    # def get_urho_h(self, lgt, lgp): # at a constant temperatre
    #     return self.get_u_h.ev(lgt, lgp, dy=1)
    

    #    # derivatives: He

    # def get_st_he(self, lgt, lgp): # at a constant pressure
    #     return self.get_s_he.ev(lgt, lgp, dx=1)
    # def get_sp_he(self, lgt, lgp): # at a constant temperatre
    #     return self.get_s_he.ev(lgt, lgp, dy=1)
    # def get_rhot_he(self, lgt, lgp): # at a constant pressure
    #     return self.get_rho_he.ev(lgt, lgp, dx=1)
    # def get_rhop_he(self, lgt, lgp): # at a constant temperatre
    #     return self.get_rho_he.ev(lgt, lgp, dy=1)
    # def get_ut_he(self, lgt, lgp): # at a constant pressure
    #     return self.get_u_he.ev(lgt, lgp, dx=1)
    # def get_urho_he(self, lgt, lgp): # at a constant temperatre
    #     return self.get_u_he.ev(lgt, lgp, dy=1)

class cms_thermo_tr:
    def __init__(self, tab):
        #self.Y = Y
        hename = 'TABLE_HE_Trho_v1'
        if tab == 2019:
            hname = 'TABLE_H_Trho_v1'
        elif tab == 2021:
            hname = 'TABLE_H_Trho_v1'

        self.hdata = cms.grid_data(cms.cms_reader(hname, tab))
        self.hedata = cms.grid_data(cms.cms_reader(hename, tab)) 

        #### H ####

        svals_h = self.hdata['logs']
        pvals_h = self.hdata['logp']
        loguvals_h = self.hdata['logu']

        self.logtvals = self.hdata['logt'][:,0]
        self.logrhovals = self.hdata['logrho'][0]

        self.get_s_h = RBS(self.logtvals, self.logrhovals, svals_h) # x or y are not changing so can leave out to speed things up
        self.get_p_h = RBS(self.logtvals, self.logrhovals, pvals_h)
        self.get_u_h = RBS(self.logtvals, self.logrhovals, loguvals_h)

        #### He  ####

        svals_he = self.hedata['logs']
        pvals_he = self.hedata['logp']
        loguvals_he = self.hedata['logu']

        # s(t, rho)
        # p(t, rho)
        self.get_s_he = RBS(self.logtvals, self.logrhovals, svals_he) # x or y are not changing so can leave out to speed things up
        self.get_p_he = RBS(self.logtvals, self.logrhovals, pvals_he)
        self.get_u_he = RBS(self.logtvals, self.logrhovals, loguvals_he)

        # derivatives: H

    def get_st_h(self, lgt, lgr): # at a constant density
        return self.get_s_h.ev(lgt, lgr, dx=1)
    def get_srho_h(self, lgt, lgr): # at a constant temperatre
        return self.get_s_h.ev(lgt, lgr, dy=1)
    def get_pt_h(self, lgt, lgr): # at a constant density
        return self.get_p_h.ev(lgt, lgr, dx=1)
    def get_prho_h(self, lgt, lgr): # at a constant temperatre
        return self.get_p_h.ev(lgt, lgr, dy=1)
    def get_ut_h(self, lgt, lgr): # at a constant density
        return self.get_u_h.ev(lgt, lgr, dx=1)
    def get_urho_h(self, lgt, lgr): # at a constant temperatre
        return self.get_u_h.ev(lgt, lgr, dy=1)
    

       # derivatives: He

    def get_st_he(self, lgt, lgr): # at a constant density
        return self.get_s_he.ev(lgt, lgr, dx=1)
    def get_srho_he(self, lgt, lgr): # at a constant temperatre
        return self.get_s_he.ev(lgt, lgr, dy=1)
    def get_pt_he(self, lgt, lgr): # at a constant density
        return self.get_p_he.ev(lgt, lgr, dx=1)
    def get_prho_he(self, lgt, lgr): # at a constant temperatre
        return self.get_p_he.ev(lgt, lgr, dy=1)
    def get_ut_he(self, lgt, lgr): # at a constant density
        return self.get_u_he.ev(lgt, lgr, dx=1)
    def get_urho_he(self, lgt, lgr): # at a constant temperatre
        return self.get_u_he.ev(lgt, lgr, dy=1)