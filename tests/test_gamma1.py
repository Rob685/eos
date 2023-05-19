'''
seems like this is known; AVL doesn't work for derivatives WRT extrinsic vars
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
plt.rc('figure', figsize=(8.0, 8.0), dpi=300)
import cms_eos
import cms_newton_raphson as cms

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
erg_to_kbbar = 1.202723550011625e-08

#######################################################
#### copied from original code with minimal changes ###
#######################################################
def orig_get_gamma1_calc(s, p, y, z, dp=0.001, t0=None, t1=None):
    # I kept the original code, but fwiw, I think this just should be
    # rho(p + dp) - rho(p), no? not rho(p * (1 + dp))? since p is already logp
    # here
    delta_logrho = (
        cms_eos.get_logrho_mix(s, p*(1+dp), y, z, t=t1)
        -cms_eos.get_logrho_mix(s, p, y, z, t=t0)
    )
    dlogrho_dlogP = delta_logrho/(p*dp)
    return 1/dlogrho_dlogP

def orig_get_gamma_1(s, p, y, z, ideal):
    if not np.isscalar(p):
        return np.array([orig_get_gamma1(si, pi, yi, zi, ideal)
                         for si, pi, yi, zi in zip(s, p, y, z)])


    t = cms_eos.get_t(s, p, y, z)
    s_hhe = cms.get_smix_z(y, 0, p, t, mz=15.5)*erg_to_kbbar

    gamma1_hhe = orig_get_gamma1_calc(s_hhe, p, y, 0)
    rho_tot = 10**cms_eos.rho_mix(p, t, y, z, ideal)
    rho_hhe = 10**cms_eos.rho_mix(p, t, y, 0, ideal)
    rho_z = 10**cms_eos.get_rho_id(p, t)
    return 1/((1-z)*(rho_tot/rho_hhe) * (1/gamma1_hhe) + z*(rho_tot/rho_z) * (3/5))
#######################################################
#### end copy #########################################
#######################################################

def get_gamma1_id(p, t0, t1, dp=1e-3):
    lgrho2 = cms_eos.get_rho_id(p * (1 + dp), t1)
    lgrho1 = cms_eos.get_rho_id(p, t0)
    return dp * p / (lgrho2 - lgrho1)

def example_fixed_get_gamma_1(s, p, y, z, ideal, dp=1e-3):
    if not np.isscalar(p):
        return np.array([orig_get_gamma1(si, pi, yi, zi, ideal)
                         for si, pi, yi, zi in zip(s, p, y, z)])


    t = cms_eos.get_t(s, p, y, z)
    t1 = cms_eos.get_t(s, p * (1 + dp), y, z)

    # with temperatures, don't need s_hhe
    gamma1_hhe = orig_get_gamma1_calc(-1, p, y, 0, t0=t, t1=t1)
    gamma1_id = get_gamma1_id(p, t, t1)
    rho_tot = 10**cms_eos.rho_mix(p, t, y, z, ideal)
    rho_hhe = 10**cms_eos.rho_mix(p, t, y, 0, ideal)
    rho_z = 10**cms_eos.get_rho_id(p, t)
    return 1/((1-z)*(rho_tot/rho_hhe) * (1/gamma1_hhe)
              + z*(rho_tot/rho_z) * (1 / gamma1_id))

def base_test(s=9, p=13, y=0.4, dp=1e-3):
    '''
    This is the test that we need to pass:
    '''
    # z = 0.6
    # orig_get_gamma1_calc(s, p, y, z, dp=dp)
    # # orig_get_gamma_1(s, p, y, z, False)
    # example_fixed_get_gamma_1(s, p, y, z, False, dp=dp)
    # return

    g1_fd = [] # from finite differencing
    g1_composite = [] # from combining the metals & HHe fraction
    g1_fixed = []
    z_arr = np.linspace(0, 1, 301)
    for z in z_arr:
        g1_fd.append(orig_get_gamma1_calc(s, p, y, z, dp=dp))
        g1_composite.append(orig_get_gamma_1(s, p, y, z, False))
        g1_fixed.append(example_fixed_get_gamma_1(s, p, y, z, False, dp=dp))
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True)
    ax1.plot(z_arr, g1_fd, 'k:', label='finite diff')
    ax1.plot(z_arr, g1_composite, 'g--', label='get_gamma_1')
    ax1.plot(z_arr, g1_fixed, 'b', label='fixed')
    ax2.plot(z_arr, np.array(g1_composite) - np.array(g1_fd), 'g--')
    ax2.plot(z_arr, np.array(g1_fixed) - np.array(g1_fd), 'b')
    ax2.set_xlabel('Z')
    ax1.set_ylabel(r'$\Gamma_1$')
    # residuals measured WRT finite difference
    ax2.set_ylabel(r'$\Gamma_1 - \Gamma_{\rm 1, fd}$')
    ax2.set_ylim(-0.01, 0.01)
    ax1.legend()
    ax1.set_title('s=%.1f, p=%.1f, y=%.1f' % (s, p, y))
    plt.tight_layout()
    plt.savefig('%s/test_gamma1_base' % CURR_DIR, bbox_inches='tight')

if __name__ == '__main__':
    base_test()
