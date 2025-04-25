import numpy as np

# ---------------------------------------------------------------------
# CONSTANTS & REFERENCE VALUES  (CODATA 2014 unless noted otherwise)
# ---------------------------------------------------------------------
R_J_PER_MOLK = 8.3144598          # J mol⁻¹ K⁻¹
J2ERG        = 1.0e7              # erg/J

Tc   = 405.56                     # K          (critical T)
rhoc = 13.696                     # mol L⁻¹    (critical ρ)
M    = 17.03052                   # g mol⁻¹    (molar mass NH₃)

# Reference state (triple point, Gao et al.)
T0  = 195.49                      # K
h0  = 28989.53846026596           # J mol⁻¹
s0  = 159.53986538192434          # J mol⁻¹ K⁻¹


# ---------------------------------------------------------------------
# IDEAL‑GAS PART  COEFFICIENTS (Table S19)
# ---------------------------------------------------------------------
c0 = 4.0
# c1 = -(h0) / (R_J_PER_MOLK * T0)          # ‑17.8353943629…
# cII = -(s0) /  R_J_PER_MOLK               # ‑19.1882418364…
a1 = -6.59406093943886
a2 = 5.60101151987913

m_i     = np.array([2.224,  3.148, 0.9579])
theta_i = np.array([1646.0, 3965.0, 7231.0])        # K


# ---------------------------------------------------------------------
# RESIDUAL PART  COEFFICIENTS (Table S20)
# ---------------------------------------------------------------------
n = np.array([
     6.132232e-03,  1.7395866e+00, -2.2261792e+00, -3.0127553e-01,
     8.967023e-02, -7.6387037e-02, -8.4063963e-01, -2.7026327e-01,
     6.212578e+00, -5.7844357e+00,  2.4817542e+00, -2.3739168e+00,
     1.493697e-02,  -3.7749264e+00,  6.254348e-04,   -1.7359e-05,
    -1.3462033e-01,  7.749073e-02, -1.6909858e+00,  9.3739074e-01])

t = np.array([
    1.0, 0.3820, 1.0,   1.0,
    0.6770, 2.9150, 3.5100, 1.0630,
    0.6550, 1.3000, 3.1000, 1.4395,
    1.6230, 0.6430, 1.1300, 4.5000,
    1.0, 4.0, 4.3315, 4.0150])

d = np.array([
    4, 1, 1, 2,
    3, 3, 2, 3,
    1, 1, 1, 2,
    2, 1, 3, 3,
    1, 1, 1, 1])

# Special exponents l_i (only terms 6‑8, i = 6…8)
l = np.array([2, 2, 1])                       # length 3

# Gaussian / exponential modifiers (i = 9…20  →  length 12)
eta     = np.array([ 0.42776, 0.6424, 0.8175, 0.7995, 0.91,
                     0.3574, 1.21,   4.14,  22.56, 22.68, 2.8452, 2.8342])

beta    = np.array([ 1.708,   1.4865, 2.0915, 2.43,  0.488,
                     1.1,     0.85,   1.14, 945.64, 993.85, 0.3696, 0.2962])

gamma_i = np.array([ 1.036, 1.2777, 1.083, 1.2906, 0.928,
                     0.934, 0.919, 1.852, 1.05897, 1.05277, 1.108, 1.313])

epsilon = np.array([-0.0726, -0.1274, 0.7527, 0.57,  2.2,
                    -0.243,   2.96,   3.02,  0.9574, 0.9576, 0.4478, 0.44689])

# Additional denominator term b_i for i = 19,20  (length 2)
b = np.array([1.244, 0.6826])

#log1mexp = np.log1p   # shorthand (safer numerically)

# ---------------------------------------------------------------------
# CORE ROUTINES  -------------------------------------------------------
# ---------------------------------------------------------------------
def _alpha_ideal(delta, tau):
    """
    Ideal‑gas contribution α°(δ,τ).
    """
    #exponent = - (theta_i / Tc) * tau[..., None]        # shape (N,3)
    # planck = np.sum(m_i * np.log(1.0 - np.exp(-theta_i * tau/ Tc[..., None])),
    #                 axis=-1)
    # planck = np.sum(np.array([m_i[i] * np.log(1.0 - np.exp(-theta_i[i] * tau / Tc)) for i in range(3)]))
    # #planck   = (m_i * np.log(1.0 - np.exp(exponent))).sum(axis=-1)
    # return a1 + a2 * tau + np.log(delta) + (c0 - 1) * np.log(tau) + planck
    exponent = -np.outer(tau, theta_i) / Tc    # shape (N,T) if T is array‑like
    planck   = (m_i * np.log(1 - np.exp(exponent))).sum(axis=-1)
    return a1 + a2 * tau + np.log(delta) + (c0 - 1.0) * np.log(tau) + planck

# ------------------------------------------------------------------
# 2.  residual contribution  αʳ(τ,δ) -------------------------------
# ------------------------------------------------------------------
#  index ranges (paper’s i):
#  1–5   : simple power terms
#  6–8   : power terms × exp(−δ^{ℓᵢ})
#  9–18  : Gaussian terms in (δ,τ)
#  19–20 : “exponential‑plus‑Lorentzian” tails

def _alpha_residual(delta, tau):
    """
    Residual Helmholtz contribution αʳ(τ,δ) — Gao et al. (2023) Eq. (9).

    Parameters
    ----------
    delta, tau : array‑like or scalar
        Must be broadcast‑compatible (same rules as NumPy arithmetic).

    Returns
    -------
    αʳ with the same shape as the broadcast of (delta, tau)
    """
    # --- make delta and tau broadcast‑compatible --------------------------------
    delta, tau = np.broadcast_arrays(delta, tau)          # common shape
    shp   = delta.shape                                   # remember final shape
    N     = delta.size                                    # total points
    δ     = delta.ravel()                                 # 1‑D views (N,)
    τ     = tau.ravel()

    αr = np.zeros_like(δ, dtype=float)                    # accumulate on 1‑D

    # convenience: add a trailing axis for term‑wise broadcasting
    δ2 = δ[np.newaxis, :]                                # (1, N)
    τ2 = τ[np.newaxis, :]

    # ---- block (1)  i = 1 … 5 --------------------------------------------------
    i1 = slice(0, 5)
    n1, d1, t1 = n[i1, None], d[i1, None], t[i1, None]    # (5,1)
    αr += (n1 * δ2**d1 * τ2**t1).sum(axis=0)

    # ---- block (2)  i = 6 … 8 --------------------------------------------------
    i2 = slice(5, 8)                                      # three “ℓ” terms
    n2, d2, t2 = n[i2, None], d[i2, None], t[i2, None]
    li  = l[:, None]                                       # (3,1)
    αr += (n2 * δ2**d2 * τ2**t2 * np.exp(-δ2**li)).sum(axis=0)

    # ---- block (3)  i = 9 … 18  (10 Gaussian terms) ---------------------------
    i3 = slice(8, 18)
    j3 = np.arange(10)                                    # 0…9
    n3, d3, t3 = n[i3, None], d[i3, None], t[i3, None]
    αr += (
        n3 * δ2**d3 * τ2**t3 *
        np.exp(-eta[j3, None]*(δ2 - epsilon[j3, None])**2
               -beta[j3, None]*(τ2 - gamma_i[j3, None])**2)
    ).sum(axis=0)

    # ---- block (4)  i = 19 … 20  (2 tail terms) -------------------------------
    i4 = slice(18, 20)
    j4 = np.arange(10, 12)                                # 10,11
    n4, d4, t4 = n[i4, None], d[i4, None], t[i4, None]
    αr += (
        n4 * δ2**d4 * τ2**t4 *
        np.exp(-eta[j4, None]*(δ2 - epsilon[j4, None])**2
               + 1.0 / (beta[j4, None]*(τ2 - gamma_i[j4, None])**2 + b[:, None]))
    ).sum(axis=0)

    return αr.reshape(shp)                                # restore original shape


# ---------------------------------------------------------------------
# UPDATED API  ---------------------------------------------------------
# ---------------------------------------------------------------------
def free_energy(_rho, _T):
    """
    Specific Helmholtz free energy of NH₃ in **erg g⁻¹**.

    Parameters
    ----------
    rho : float or ndarray
        Mass density ρ  [g cm⁻³].
    T   : float or ndarray
        Temperature [K].

    Returns
    -------
    a_specific : float or ndarray
        Free energy per unit mass (erg g⁻¹).
    """
    rho = np.asarray(_rho, dtype=float)
    T   = np.asarray(_T,   dtype=float)

    # -----------------------------------------------------------------
    # convert ρ [g cm⁻³]  →  molar density ρ_mol [mol L⁻¹]
    # 1 cm³ = 1 × 10⁻³ L  ⇒  ρ_mol = ρ * 1000 / M
    # -----------------------------------------------------------------
    rho_mol = rho * 1000.0 / M        # mol L⁻¹
    #rho_mol = rho

    delta = rho_mol / rhoc            # reduced density
    tau   = Tc / T                    # reduced temperature

    α = _alpha_ideal(delta, tau) + _alpha_residual(delta, tau)

    a_molar_J  = α * R_J_PER_MOLK * T   # J mol⁻¹
    a_specific = a_molar_J * J2ERG / M  # erg g⁻¹
    return a_specific

# ---------------------------------------------------------------------
# PRESSURE from numerical derivative  p(ρ,T) = ρ² ∂f/∂ρ |_T
# ---------------------------------------------------------------------

def get_p_rhot(_rho, _T, *, drho_rel=1e-6, drho_min=1e-9):
    """
    ρ in g cm⁻³, T in K; returns p in erg cm⁻³ (dyne cm⁻²).

    drho = max(drho_rel * ρ, drho_min)  ensures that very low densities
    still get a usable absolute step.
    """
    rho = np.asarray(_rho, dtype=float)
    T   = np.asarray(_T,   dtype=float)

    if np.any(rho <= 0):
        raise ValueError("rho must be positive")

    drho = np.maximum(drho_rel * rho, drho_min)

    f_plus  = free_energy(rho + drho, T)
    f_minus = free_energy(rho - drho, T)

    df_drho = (f_plus - f_minus) / (2.0 * drho)
    return rho**2 * df_drho          # erg cm⁻³ ≡ dyne cm⁻²

# ---------------------------------------------------------------------
# INTERNAL ENERGY   u(ρ,T) = -T² ∂(f/T)/∂T |_ρ
# ---------------------------------------------------------------------
def get_u_rhot(_rho, _T, *, dT_rel=1e-5):
    """
    Specific internal energy of NH₃, in **erg g⁻¹**.

    Parameters
    ----------
    rho : float or ndarray
        Mass density [g cm⁻³]  (same convention as free_energy).
    T   : float or ndarray
        Temperature [K].
    dT_rel : float, optional
        Relative temperature step ΔT / T for the central finite difference.
        Default 1×10⁻⁵ is a good compromise for 64‑bit precision.

    Returns
    -------
    u : float or ndarray
        Internal energy (erg g⁻¹).
    """
    rho = np.asarray(_rho, dtype=float)
    T   = np.asarray(_T,   dtype=float)

    # -----------------------------------------------------------------
    # central finite difference on  g(T) = f(ρ,T)/T
    # -----------------------------------------------------------------
    dT = dT_rel * T
    dT = np.where(dT == 0.0, dT_rel, dT)     # ensure non‑zero step

    g_plus  = free_energy(rho, T + dT) / (T + dT)
    g_minus = free_energy(rho, T - dT) / (T - dT)

    dg_dT = (g_plus - g_minus) / (2.0 * dT)

    return -T**2 * dg_dT

# ---------------------------------------------------------------------
# ENTROPY  S(ρ,T) = -∂f/∂T |_ρ
# ---------------------------------------------------------------------

def get_s_rhot(_rho, _T, *, dT_rel=1e-5):
    """
    Specific entropy of NH₃, in **erg g⁻¹**.

    Parameters
    ----------
    rho : float or ndarray
        Mass density [g cm⁻³]  (same convention as free_energy).
    T   : float or ndarray
        Temperature [K].
    dT_rel : float, optional
        Relative temperature step ΔT / T for the central finite difference.
        Default 1×10⁻⁵ is a good compromise for 64‑bit precision.

    Returns
    -------
    u : float or ndarray
        Internal energy (erg g⁻¹).
    """
    rho = np.asarray(_rho, dtype=float)
    T   = np.asarray(_T,   dtype=float)

    # -----------------------------------------------------------------
    # central finite difference on  g(T) = f(ρ,T)/T
    # -----------------------------------------------------------------
    dT = dT_rel * T
    dT = np.where(dT == 0.0, dT_rel, dT)     # ensure non‑zero step

    f_plus  = free_energy(rho, T + dT)
    f_minus = free_energy(rho, T - dT)

    df_dT = (f_plus - f_minus) / (2.0 * dT)

    return -df_dT