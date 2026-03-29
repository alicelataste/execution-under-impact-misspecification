"""
AFS Model — Alfonsi, Fruth & Schied (2010)
Implementation based on Hey, Bouchaud, Mastromatteo, Muhle-Karbe & Webster (2023)
"The Cost of Misspecifying Price Impact"
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AFSParams:
    """
    Parameters of the AFS price impact model.

    c     : impact concavity (0 < c <= 1). c=0.5 = square-root law, c=1 = linear
    tau   : impact decay timescale (in days)
    lam   : impact scale lambda (normalized via ADV and volatility)
    sigma : asset volatility
    V     : average daily volume (ADV)
    """
    c: float = 0.48
    tau: float = 0.2
    lam: float = 1.0
    sigma: float = 1.0
    V: float = 1.0


def optimal_impact(alpha: float | np.ndarray,
                   mu_alpha: float | np.ndarray,
                   params: AFSParams) -> np.ndarray:
    """
    Optimal impact state I* from Hey et al. Eq (2.4).

    I*_t = 1/(1+c) * (alpha_t - tau * mu_alpha_t)

    Parameters
    ----------
    alpha     : alpha signal level at time t
    mu_alpha  : drift of alpha (decay rate), mu_alpha = alpha'_t
    params    : AFS model parameters

    Returns
    -------
    I_star : optimal impact state
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    mu_alpha = np.atleast_1d(np.asarray(mu_alpha, dtype=float))

    I_star = (alpha - params.tau * mu_alpha) / (1 + params.c)
    return I_star


def pnl_optimal(alpha: float, params: AFSParams, T: float, tau_grid: np.ndarray) -> float:
    """
    Expected P&L of the optimal policy for constant alpha.
    Hey et al. Section 4.2, formula for U(J(c); c).

    Parameters
    ----------
    alpha    : constant alpha signal (signal Sharpe = alpha/sigma)
    params   : AFS model parameters (true parameters)
    T        : trading horizon (in days)
    tau_grid : impact decay timescale grid for integration

    Returns
    -------
    U_opt : expected P&L of optimal policy (normalized)
    """
    c = params.c
    tau = params.tau
    sigma = params.sigma
    V = params.V
    lam = params.lam

    g = _prefactor_g(c, tau)
    alpha_sr = alpha / sigma  # Sharpe ratio of alpha

    U_opt = (sigma * V / g ** (1/c)) * (alpha_sr ** (1 + 1/c)) * (
        c / (1 + c)
    ) * (T / (tau * (1 + c)) ** (1/c) + 1)

    return U_opt


def pnl_misspecified(alpha: float,
                     params_true: AFSParams,
                     c_hat: float,
                     T: float) -> float:
    """
    Expected P&L of the misspecified policy (wrong concavity c_hat)
    under the true model with concavity c.
    Hey et al. Section 4.2.

    Parameters
    ----------
    alpha       : constant alpha signal
    params_true : true AFS parameters
    c_hat       : misspecified concavity parameter
    T           : trading horizon

    Returns
    -------
    U_misspec : expected P&L under misspecified policy
    """
    c = params_true.c
    tau = params_true.tau
    sigma = params_true.sigma
    V = params_true.V

    g_hat = _prefactor_g(c_hat, tau)
    g_true = _prefactor_g(c, tau)
    alpha_sr = alpha / sigma

    term1 = (alpha_sr ** (1 + 1/c_hat)) * (
        T / (tau * (1 + c_hat)) ** (1/c_hat) + 1
    )

    term2 = (g_true / g_hat) ** (c / c_hat) * (alpha_sr ** ((1 + c) / c_hat)) * (
        T / (tau * (1 + c_hat)) ** ((1 + c) / c_hat) + 1 / (1 + c)
    )

    U_misspec = (sigma * V / g_hat ** (1/c_hat)) * (term1 - term2)

    return U_misspec


def profit_ratio_concavity(alpha: float,
                           params_true: AFSParams,
                           c_hat_grid: np.ndarray,
                           T: float) -> np.ndarray:
    """
    Profit ratio U(J(c_hat); c) / U(J(c); c) across misspecified concavities.
    Reproduces Figure 4 (right panel) of Hey et al.

    Parameters
    ----------
    alpha        : constant alpha signal
    params_true  : true AFS parameters
    c_hat_grid   : array of misspecified concavity values to evaluate
    T            : trading horizon

    Returns
    -------
    ratios : profit ratio for each c_hat in c_hat_grid
    """
    U_opt = pnl_optimal(alpha, params_true, T, tau_grid=None)

    ratios = np.array([
        pnl_misspecified(alpha, params_true, c_hat, T) / U_opt
        for c_hat in c_hat_grid
    ])

    return ratios


def _prefactor_g(c: float, tau: float) -> float:
    """
    Normalized prefactor g(c, tau) from Hey et al. Eq (2.6).
    Placeholder — in practice calibrated from data.
    For simulation purposes we use g=1 as a normalization baseline.
    """
    # En pratique calibré depuis les données (Section 3 de Hey et al.)
    # Pour la reproduction des figures on normalise à 1
    return 1.0