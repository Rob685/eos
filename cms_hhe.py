import numpy as np
from astropy import units as u
from astropy.constants import u as amu
from astropy.constants import k_B
from scipy.interpolate import RectBivariateSpline as RBS
from eos import cms_newton_raphson as cms
import pandas as pd
import main_eos as eos
import pdb
import os
import re
#import scvh_nr

'''This module calculates the derivatives at constant temperature/pressure or temperature/density.'''
'''Call the thermo_sp.py module for derivatives at constant entropy or pressure'''

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
class cms_thermo_tp:
    def __init__(self, tab):
        #self.Y = Y
        hename = 'TABLE_HE_TP_v1'
        if tab == 2019:
            hname = 'TABLE_H_TP_v1'
        elif tab == 2021:
            hname = 'TABLE_H_TP_effective'

        self.hdata = cms.grid_data(cms.cms_reader(hname))
        self.hedata = cms.grid_data(cms.cms_reader(hename)) 

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

        self.hdata = cms.grid_data(cms.cms_reader(hname))
        self.hedata = cms.grid_data(cms.cms_reader(hename)) 

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