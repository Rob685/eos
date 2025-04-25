import numpy as np

# ---------------------------------------------------------------------
# CH4 – critical parameters (Setzmann & Wagner, 1991)
# ---------------------------------------------------------------------
Tc   = 190.564        # K
rhoc = 162.66        # kg m⁻³            (mass basis)
R_kJ_PER_KG = 0.5182705 # kJ kg⁻¹ K⁻¹
kJ2ERG = 1e7          # erg / kJ

# ---------------------------------------------------------------------
# Ideal-gas Helmholtz coefficients, Eq. (5.2)
# ---------------------------------------------------------------------
a_i = np.array([
     9.91243972,   # a1
    -6.33270087,   # a2
     3.0016,       # a3
     0.008449,     # a4
     4.6942,       # a5
     3.4865,       # a6
     1.6572,       # a7
     1.4115        # a8
])

theta_i = np.array([                 # Θi [K]  – only i = 4 … 8 are non-zero
         0.0, 0.0, 0.0,              # Θ1 … Θ3 not used in ideal part
         3.40043240,
        10.26951575,
        20.43932747,
        29.93744884,
        79.13351945
])

# ---------------------------------------------------------------------
# 1.  core numeric data  (Setzmann & Wagner, 1991 – Table 35)
# ---------------------------------------------------------------------
# --- block A : i = 1 … 13 -------------------------------------------
n_A = np.array([
            0.4367901028e-1,  0.6709236199,  -0.1765577859e1,  0.8582330241,
            -0.1206513052e1, 0.5120467220,  -0.4000010791e-3,
            -0.1247842423e-1, 0.3100269701e-1, 0.1754748522e-2,
            -0.3171921605e-5, -0.2240346840e-5, 0.2947056156e-6
])
d_A = np.array([1,1,1,2,
                2,2,2,
                3,4,4,
                8,9,10])
t_A = np.array([-0.5,0.5,1.0,0.5,
                1.0,1.5,4.5,
                0.0, 1.0,3.0,
                1.0,3.0,3.0])

# --- block B : i = 14 … 36  (power × exp(−δ^c_i)) -------------------
n_B = np.array([
            0.1830487909,   0.1511883679,  -0.4289363877,   0.6894002446e-1,
            -0.1408313996e-1, -0.3063054830e-1, -0.2969906708e-1, -0.1932040831e-1,
            -0.1105739959,   0.9952548995e-1,  0.8548437825e-2, -0.6150555662e-1,
            -0.4291792423e-1, -0.1813207290e-1,  0.3445904760e-1, -0.2385919450e-2,
            -0.1159094939e-1,  0.6641693602e-1, -0.2371549590e-1, -0.3961624905e-1,
            -0.1387292044e-1,  0.3389489599e-1, -0.2927378753e-2
])
d_B = np.array([
                1,1,1,2,
                4,5,6,1,
                2,3,4,4,
                3,5,5,8,
                2,3,4,4,
                4,5,6
                ])
t_B = np.array([
                0.0,1.0,2.0,0.0,
                0.0,2.0,2.0,5.0,
                5.0,5.0,2.0,4.0,
                12.0,8.0,10.0,10.0,
                10.0,14.0,12.0,18.0,
                22.0,18.0,14.0
                ])
c_B = np.array([
                1,1,1,1,
                1,1,1,2,
                2,2,2,2,
                3,3,3,3,
                4,4,4,4,
                4,4,4
                ])

# --- block C : i = 37 … 40  (Gaussian) ------------------------------
n_C     = np.array([
                    0.9324799946e-4, -0.6287171518e1,
                    0.1271069467e2, -0.6423953466e1
                    ])

d_C     = np.array([2,0,0,0])
t_C     = np.array([2.0,0.0,1.0,2.0])
alpha_C = np.array([20., 40., 40., 40.])
beta_C  = np.array([200., 250., 250., 250.])
gamma_C = np.array([1.07, 1.11, 1.11, 1.11])
Delta_C = np.array([1.,   1.,   1.,   1.])

# ---------------------------------------------------------------------
# Ideal-gas contribution α°(δ,τ)   (dimensionless A/RT)
# ---------------------------------------------------------------------
def _alpha_ideal(delta, tau):
    """
    Ideal-gas part of the dimensionless Helmholtz free energy for CH₄
    after Setzmann & Wagner (1991), Eq. (5.2).

    Parameters
    ----------
    delta : float or ndarray
        Reduced density  δ = ρ / ρ_c  (ρ in kg m⁻³ on the *same* basis as ρ_c).
    tau   : float or ndarray
        Inverse reduced temperature  τ = T_c / T.

    Returns
    -------
    α° : float or ndarray
        Dimensionless ideal-gas Helmholtz energy.
    """
    delta = np.asarray(delta, dtype=float)
    tau   = np.asarray(tau,   dtype=float)

    # “Planck–Einstein” sum  Σ a_i ln(1 − exp(−Θ_i τ)),  i = 4 … 8
    exp_term = -np.outer(tau, theta_i[3:])              # shape (N,5)
    planck   = (a_i[3:] * np.log1p(-np.exp(exp_term))).sum(axis=-1)

    return (
        np.log(delta)
        + a_i[0]                         # a1
        + a_i[1] * tau                  # a2 τ
        + a_i[2] * np.log(tau)          # a3 ln τ
        + planck
    )

# ---------------------------------------------------------------------
# 2.  residual Helmholtz contribution αʳ(δ,τ)
# ---------------------------------------------------------------------
def _alpha_residual(delta, tau):
    """
    Residual part of the CH₄ Helmholtz free-energy, Eq. (5.3).

    Parameters
    ----------
    delta : float or ndarray
        Reduced density  δ = ρ / ρ_c.
    tau   : float or ndarray
        Reduced inverse temperature τ = T_c / T.

    Returns
    -------
    αʳ : float or ndarray  (broadcast shape of delta & tau)
    """
    delta, tau = np.broadcast_arrays(delta, tau)
    δ = delta.ravel(); τ = tau.ravel()
    N = δ.size

    # ----- block A ----------------------------------------------------
    αr = (n_A[:,None] * δ**d_A[:,None] * τ**t_A[:,None]).sum(axis=0)

    # ----- block B ----------------------------------------------------
    αr += (n_B[:,None] * δ**d_B[:,None] * τ**t_B[:,None] *
           np.exp(-δ**c_B[:,None])).sum(axis=0)

    # ----- block C ----------------------------------------------------
    αr += (n_C[:,None] * δ**d_C[:,None] * τ**t_C[:,None] *
           np.exp(-alpha_C[:,None]*(δ-Delta_C[:,None])**2
                  -beta_C[:,None]*(τ-gamma_C[:,None])**2)
          ).sum(axis=0)

    return αr.reshape(delta.shape)



# ---------------------------------------------------------------------
# UPDATED API  ---------------------------------------------------------
# ---------------------------------------------------------------------
def free_energy(_rho, _T):
    """
    Specific Helmholtz free energy of CH₄ in **erg g⁻¹**.

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
    # convert ρ [g cm⁻³]  →  ρ [kg m⁻³]
    # -----------------------------------------------------------------
    rho_SI = rho * 1000.0             # kg m⁻³

    delta = rho_SI / rhoc            # reduced density
    tau   = Tc / T                    # reduced temperature

    α = _alpha_ideal(delta, tau) + _alpha_residual(delta, tau)

    a_SI_kJ  = α * R_kJ_PER_KG * T   # kJ kg⁻¹
    a_specific = a_SI_kJ * kJ2ERG  # erg g⁻¹
    return a_specific

# ---------------------------------------------------------------------
# PRESSURE from numerical derivative  p(ρ,T) = ρ² ∂f/∂ρ |_T
# ---------------------------------------------------------------------

def get_p_rhot(_rho, _T, *, drho_rel=1e-2, drho_min=1e-9):
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
def get_u_rhot(_rho, _T, *, dT_rel=1e-2):
    """
    Specific internal energy of CH₄, in **erg g⁻¹**.

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

def get_s_rhot(_rho, _T, *, dT_rel=1e-2):
    """
    Specific entropy of CH₄, in **erg g⁻¹**.

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