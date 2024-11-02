'''
every function should:

- accept scalars or vectors as args (but not python lists)
- same naming convention as SCvH (eventually)
- argument order is: two of (S, logrho, logP, logT) [in that order], Y, and
  others

Proposed naming convention:
Getters for getting q1 as a function of q2, q3):
    f"get_{q2}_{q2}{q3}(q2, q3, Y))"
Getters for partials of q1 WRT q2 leaving q3q4 constant as functions of q5q6:
    f"d{q1}d{q2}_{q3}{q4}_{q5}{q6}(q5, q6, Y)"
Ordering on quantities follows the ordering above
'''
import numpy as np
from scipy.optimize import brenth, minimize, root_scalar, root
from astropy import units as u
from astropy.constants import k_B, m_p
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI

"""
    This file provides access to the ideal EOS and accepts
    the same format as the other non-ideal EOSes. 
    
    This file does not use precomputed tables and it can be
    used to compare to the other EOSes. The ideal EOS is also 
    used to provided initial guesses to the inversion optimization
    functions used in all other EOS files.

    Authors: Yubo Su, Roberto Tejada Arevalo
    
"""

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

kB = 1.380649e-16
mp = 1.67262192369e-24
#erg_to_kbbar = 1.2114751277768644e-08

S_UNIT = 1 #kB / mp
U_UNIT = kB / amu.to('g').value
Rideal = 8.314e7

class IdealEOS(object):
    """
    ideal eos with proton mass m
    """
    def __init__(self, m):
        super(IdealEOS, self).__init__()
        self.m = m

    ## S getters
    def get_s_pt(self, logp, logt, _y):
        p = 10**logp
        t = 10**logt
        return S_UNIT * (
            4.6166 + np.log((t / 1e3)**(5/2) / (p / 1e11) * self.m**(3/2))
        ) / self.m

    def get_s_rhot(self, logrho, logt, _y):
        rho = 10**logrho
        t = 10**logt
        return S_UNIT * (
            2.818087 + np.log((t / 1e2)**(3/2) / rho * self.m**(5/2))
        ) / self.m

    def get_s_rhop(self, logrho, logp, _y):
        p = 10**logp
        rho = 10**logrho
        return S_UNIT * (
            5.096 + np.log((p / 1e11)**(3/2) * self.m**4 / (rho**(5/2)))
        ) / self.m

    ## rho getters
    def get_rho_pt(self, logp, logt, _y):
        p = 10**logp
        t = 10**logt
        return np.log10(p * self.m * mp / (kB * t))

    def get_rho_sp(self, s, logp, _y):
        p = 10**logp
        # rho is in units of g/cm^3 already
        return np.log10((
            (p / 1e11)**(3/2)
            * self.m**4
            / np.exp(s * self.m - 5.096))**(2/5))

    ## P getters
    def get_p_rhot(self, logrho, logt, _y):
        rho = 10**logrho
        t = 10**logt
        return np.log10(rho * kB * t / (self.m * mp))

    def get_p_srho(self, s, logrho, _y):
        rho = 10**logrho
        return np.log10((
            (rho)**(5/2)
            * np.exp(s * self.m - 5.096)
            / self.m**4)**(2/3))

    ## T getters
    def get_t_rhop(self, logrho, logp, _y):
        p = 10**logp
        rho = 10**logrho
        return np.log10(p * self.m * mp / (rho * kB))

    def get_t_sp(self, s, logp, _y):
        # same as
        # return self.get_t_rhop(self.get_rho_sp(s, logp), logp)
        p = 10**logp
        return np.log10((
            np.exp(s * self.m - 4.6166)
            * (p / 1e11)
            / self.m**(3/2))**(2/5) * 1000)

    def get_t_srho(self, s, logrho, _y):
        rho = 10**logrho
        return np.log10((
            np.exp(s * self.m - 2.8181)
            * rho
            / self.m**(5/2))**(2/3))

    ## U getters
    def get_u_pt(self, logp, logt, _y):
        return np.log10(U_UNIT * 3/2 * 10**logt / self.m)

    def get_u_srho(self, s, logrho, _y):
        logp, logt = self.get_pt_srho(s, logrho, _y)
        return self.get_u_pt(logp, logt, _y)

    ## combined getters
    def get_sp_rhot(self, logrho, logt, _y):
        return self.get_s_rhot(logrho, logt), self.get_p_rhot(logrho, logt)

    def get_rhot_sp(self, s, logp, _y):
        return self.get_rho_sp(s, logp, _y), self.get_t_sp(s, logp, _y)

    def get_pt_srho(self, s, logrho, _y):
        return self.get_p_srho(s, logrho, _y), self.get_t_srho(s, logrho, _y)

    ## analytic derivatives
    def get_chirho_sp(self, s, _logp, _y):
        # idiomatic (?) way of returning the same type as s
        return 0 * s + 1

    def get_grad_ad(self, s, _logp, _y):
        return 0 * s + 5/3

    ## misc
    def get_c_p(self, s, _logp, _y):
        return 0 * s + 5/2 * Rideal
    def get_c_v(self, s, _logp, _y):
        return 0 * s + 3/2 * Rideal

def get_number_fracs(y, m_h, m_he):
    # vector-compatible expressions
    f_h = (1 - y) / m_h
    f_he = y / m_he
    f_tot = f_h + f_he
    return f_h / f_tot, f_he / f_tot
def get_smix(y, m_h, m_he):
    f_h, f_he = get_number_fracs(y, m_h, m_he)
    # entropy of mixing in units of kB / baryon
    smix = -(f_h * np.log(f_h) + f_he * np.log(f_he)) / (
        f_h * m_h + f_he * m_he)
    return smix

TBOUNDS = (-100, 100)
PBOUNDS = (-100, 100)
class IdealHHeMix(object):
    """
    ideal eos with proton mass m
    """
    def __init__(self, m_h=2.016, m_he=4.0026):
        super(IdealHHeMix, self).__init__()
        self.m_h = m_h
        self.m_he = m_he
        self.eos_h = IdealEOS(m_h)
        self.eos_he = IdealEOS(m_he)

    ## S getters
    def get_s_pt(self, logp, logt, y):
        smix = get_smix(y, self.m_h, self.m_he)
        return (
            (1 - y) * self.eos_h.get_s_pt(logp, logt, y)
            + y * self.eos_he.get_s_pt(logp, logt, y)
            + smix)

    def get_s_rhot(self, logrho, logt, y):
        logp = self.get_p_rhot(logrho, logt, y)
        return self.get_s_pt(logp, logt, y)

    def get_s_rhop(self, logrho, logp, y):
        logt = self.get_t_rhop(logrho, logp, y)
        return self.get_s_pt(logp, logt, y)

    ## rho getters
    def get_rho_pt(self, logp, logt, y):
        return np.log10(1 / (
            (1 - y) / 10**self.eos_h.get_rho_pt(logp, logt, y)
            + y / 10**self.eos_he.get_rho_pt(logp, logt, y)))

    def get_rho_sp(self, s, logp, y):
        if not np.isscalar(y):
            rets = [self.get_rho_sp(*el)
                    for el in zip(s, logp, y)]
            return np.array(rets)
        def obj(logt):
            return self.get_s_pt(logp, logt, y) / s - 1
        opt_logt = brenth(obj, *TBOUNDS)
        return self.get_rho_pt(logp, opt_logt, y)

    ## P getters
    def get_p_rhot(self, logrho, logt, y):
        if not np.isscalar(y):
            rets = [self.get_p_rhot(*el)
                    for el in zip(logrho, logt, y)]
            return np.array(rets)
        def obj(logp):
            return self.get_rho_pt(logp, logt, y) / logrho - 1
        return brenth(obj, *PBOUNDS)

    def get_p_srho(self, s, logrho, y):
        return self.get_pt_srho(s, logrho, y)[:,0]

    ## T getters
    def get_t_rhop(self, logrho, logp, y):
        if not np.isscalar(y):
            rets = [self.get_t_rhop(*el)
                    for el in zip(logrho, logp, y)]
            return np.array(rets)
        # def obj(logt):
        #     return self.get_rho_pt(logp, logt, y) / logrho - 1
        # return brenth(obj, *TBOUNDS)

        def obj(logt):
            return self.get_rho_pt(logp, logt, y) / logrho - 1
        return root_scalar(obj, method='brenth', bracket=TBOUNDS).root

    def get_t_sp(self, s, logp, y):
        if not np.isscalar(y):
            rets = [self.get_t_sp(*el)
                    for el in zip(s, logp, y)]
            return np.array(rets)
        # def obj(logt):
        #     return self.get_s_pt(logp, logt, y) / s - 1
        # return brenth(obj, *TBOUNDS)

        def obj(logt):
            return self.get_s_pt(logp, logt, y) / s - 1
        return root_scalar(obj, method='brenth', bracket=TBOUNDS).root

    def get_t_srho(self, s, logrho, y):
        if not np.isscalar(s):
            return self.get_pt_srho(s, logrho, y)[:,1]
        else:
            return self.get_pt_srho(s, logrho, y)[1]

    ## U getters
    def get_u_pt(self, logp, logt, y):
        return (
            (1 - y) * self.eos_h.get_u_pt(logp, logt, y)
            + y * self.eos_he.get_u_pt(logp, logt, y)
        )
    def get_u_srho(self, s, logrho, y):

        logp, logt = self.get_pt_srho(s, logrho, y)
        return self.get_u_pt(logp, logt, y)
        # return (
        #     (1 - y) * self.eos_h.get_u_srho(s, logrho, y)
        #     + y * self.eos_he.get_u_srho(s, logrho, y)
        # )

    ## combined getters
    def get_sp_rhot(self, logrho, logt, _y):
        return self.get_s_rhot(logrho, logt), self.get_p_rhot(logrho, logt)

    def get_rhot_sp(self, s, logp, _y):
        return self.get_rho_sp(s, logp, _y), self.get_t_sp(s, logp, _y)

    def get_pt_srho(self, s, logrho, y):
        # 2D inversion...
        if not np.isscalar(s):
            return np.array([self.get_pt_srho(*el)
                             for el in zip(s, logrho, y)])
        def opt(v):
            logp, logt = v
            return (
                (self.get_s_pt(logp, logt, y) / (s+1e-15) - 1)**2 # avoiding zero divisions
                + (self.get_rho_pt(logp, logt, y) / (logrho+1e-15) - 1)**2
            )

        def print_res(v):
            logp, logt = v
            print(
                (self.get_s_pt(logp, logt, y) / s - 1)**2,
                (self.get_rho_pt(logp, logt, y) / logrho - 1)**2)
        sol = minimize(opt, [8, 3],
                       bounds=(PBOUNDS, TBOUNDS),
                       method='nelder-mead')
        # print_res(sol.x)
        return sol.x

    ## analytic derivatives
    def get_chirho_sp(self, s, _logp, _y):
        # idiomatic (?) way of returning the same type as s
        return 0 * s + 1

    def get_grad_ad(self, s, _logp, _y):
        return 0 * s + 5/3

    ## misc
    def get_c_p(self, s, _logp, _y):
        return 0 * s + 5/2
    def get_c_v(self, s, _logp, _y):
        return 0 * s + 3/2

class EOSFiniteDs(object):
    """
    wraps an eos and attaches finite difference functions

    any unknown function names call through to eos!
    """
    def __init__(self, eos):
        super(EOSFiniteDs, self).__init__()
        self.eos = eos
        self.d = 1e-3

    def __getattr__(self, attr):
        '''
        https://stackoverflow.com/questions/57091503/catch-all-method-in-class-that-passes-all-unknown-functions-to-instance-in-class
        '''
        return getattr(self.eos, attr)


    def get_chirho_sp(self, s, logp, y):
        logrho = self.eos.get_rho_sp(s, logp, y)
        logt = self.eos.get_t_sp(s, logp, y)
        return (
            self.eos.get_p_rhot(logrho * (1 + self.d), logt)
            - self.eos.get_p_rhot(logrho, logt)
        ) / (logrho * self.d)
    def get_grad_ad(self, s, logp, y):
        logt1 = self.eos.get_t_sp(s, logp, y)
        logt2 = self.eos.get_t_sp(s, logp * (1 + self.d), y)
        return (logt2 - logt1) / (logp * self.d)

    ## dQ/dY
    def get_dsdy_rhop_pt(self, logp, logt, y):
        logrho = self.eos.get_rho_pt(logp, logt, y)
        logt2 = self.eos.get_t_rhop(logrho, logp, y * (1 + self.d))
        s1 = self.eos.get_s_pt(logp, logt, y)
        s2 = self.eos.get_s_pt(logp, logt2, y * (1 + self.d))
        return (s2 - s1) / (y * self.d)
    def get_dsdy_rhop(self, logrho, logp, y):
        logt1 = self.eos.get_t_rhop(logrho, logp, y)
        logt2 = self.eos.get_t_rhop(logrho, logp, y * (1 + self.d))
        s1 = self.eos.get_s_pt(logp, logt1, y)
        s2 = self.eos.get_s_pt(logp, logt2, y * (1 + self.d))
        return (s2 - s1) / (y * self.d)
    def get_dsdy_pt(self, logp, logt, y):
        s1 = self.eos.get_s_pt(logp, logt, y)
        s2 = self.eos.get_s_pt(logp, logt, y * (1 + self.d))
        return (s2 - s1) / (y * self.d)
    def get_drhody_pt(self, logp, logt, y):
        rho1 = self.eos.get_rho_pt(logp, logt, y)
        rho2 = self.eos.get_rho_pt(logp, logt, y * (1 + self.d))
        return (rho2 - rho1) / (y * self.d)
    def get_dtdy_sp(self, s, logp, y):
        t1 = self.eos.get_t_sp(s, logp, y)
        t2 = self.eos.get_t_sp(s, logp, y * (1 + self.d))
        return (t2 - t1) / (y * self.d)
    def get_drhody_sp(self, s, logp, y):
        rho1 = self.eos.get_rho_sp(s, logp, y)
        rho2 = self.eos.get_rho_sp(s, logp, y * (1 + self.d))
        return (rho2 - rho1) / (y * self.d)

    def get_dudy_srho(self, s, logrho, y):
        logp1, logt1 = self.eos.get_pt_srho(s, logrho, y)
        logp2, logt2 = self.eos.get_pt_srho(s, logrho, y * (1 + self.d))
        u1 = self.eos.get_u_pt(logp1, logt1, y)
        u2 = self.eos.get_u_pt(logp2, logt2, y * (1 + self.d))
        return (u2 - u1) / (y * self.d)

    def get_duds_rhoy_srho(self, s, rho, y, ds=0.001):
        S1 = s/erg_to_kbbar
        S2 = S1*(1+ds)
        U0 = 10**self.get_u_srho(S1*erg_to_kbbar, rho, y)
        U1 = 10**self.get_u_srho(S2*erg_to_kbbar, rho, y)
        return (U1 - U0)/(S1*ds)

    def get_dudrho_sy_srho(self, s, rho, y, drho=0.1):
        R1 = 10**rho
        R2 = R1*(1+drho)
        #rho1 = np.log10((10**rho)*(1+drho))
        U0 = 10**self.get_u_srho(s, np.log10(R1), y)
        U1 = 10**self.get_u_srho(s, np.log10(R2), y)
        #return (U1 - U0)/(R1*drho)
        return (U1 - U0)/((1/R1) - (1/R2))

    def get_dtdy_srho(self, s, logrho, y):
        _, logt1 = self.eos.get_pt_srho(s, logrho, y)
        _, logt2 = self.eos.get_pt_srho(s, logrho, y * (1 + self.d))
        return (logt2 - logt1) / (y * self.d)

if __name__ == '__main__':
    # quick test: made the dsdy plots from my previous script
    import numpy as np
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.rc('lines', lw=2.5)
    plt.rc('xtick', direction='in', top=True, bottom=True)
    plt.rc('ytick', direction='in', left=True, right=True)
    plt.rc('figure', figsize=(8.0, 8.0), dpi=300)

    hhe_eos = IdealHHeMix()
    rho = -2
    Y = 0.25
    s_grid = np.linspace(6, 9, 301) * S_UNIT
    ds = np.diff(s_grid)[0]

    p0, t0 = hhe_eos.get_pt_srho(s_grid[0], rho, Y)
    u0 = hhe_eos.get_u_pt(p0, t0, Y)
    u_lst = [u0]
    u_int = [u0]
    u_prev = u0
    for si in s_grid[1: ]:
        pi, ti = hhe_eos.get_pt_srho(si, rho, Y)
        u_lst.append(hhe_eos.get_u_pt(pi, ti, Y))
        du = 10**ti * ds
        u_int.append(u_prev + du)
        u_prev += du
    plt.semilogy(s_grid, u_lst)
    plt.plot(s_grid, u_int)
    plt.show()
    #plt.savefig('/tmp/foo')
    #plt.close()
