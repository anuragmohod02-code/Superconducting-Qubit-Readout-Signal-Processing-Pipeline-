"""
readout_optimization.py — Advanced Readout Optimization Techniques
===================================================================

Implements four optimization techniques for superconducting qubit readout:

1. ROC-based optimal threshold (Youden's J statistic)
   ───────────────────────────────────────────────────
   The standard LDA discriminator uses P(|1⟩)=0.5 as the decision
   threshold.  When T1 relaxation creates an asymmetric error matrix
   (P(0|1) >> P(1|0)), shifting the threshold to maximise Youden's J
   = TPR − FPR improves the assignment fidelity.

2. 2-D integration window optimisation
   ─────────────────────────────────────
   Independently sweeps the integration start time (t_on) and end time
   (t_off) to find the window that maximises assignment fidelity.
   The optimal window avoids:
     - Ring-up transient (t < 2/κ) — low signal amplitude, poor SNR
     - Late-window T1 contamination — |1⟩ shots decay partway through
       the readout, producing intermediate IQ values that smear the
       |1⟩ cluster toward |0⟩

3. Quantum efficiency sweep
   ──────────────────────────
   Real amplification chains have quantum efficiency η < 1 due to
   insertion losses and finite amplifier noise temperature.
   Effective noise: σ_eff = σ_ql / sqrt(η)
     η ≈ 0.3–0.6  : JPA / TWPA + HEMT chain
     η ≈ 0.05–0.1 : HEMT-only chain
     η = 1.0       : quantum limit (ideal zero-added-noise amplifier)

4. Active reset convergence
   ──────────────────────────
   Simulates measurement-feedback:  measure → if P(|1⟩) > threshold,
   apply π pulse (after latency τ).  Computes residual |1⟩ population
   after N rounds as a function of feedback latency τ.  Compares the
   default 0.5 threshold against the Youden-optimal threshold.

References
----------
Reed et al., PRL 105, 173601 (2010)          — dispersive readout fidelity
Johnson et al., PRL 109, 050506 (2012)        — active reset
Heinsoo et al., PRApplied 10, 034040 (2018)   — optimal integration window
Krantz et al., Appl. Phys. Rev. 6, 021318 (2019) — readout overview
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.transmon    import TransmonParams, simulate_shots
from src.discriminator import (LDADiscriminator, GMMDiscriminator,
                                assignment_matrix, readout_fidelity,
                                compute_roc, iq_to_xy)


# ── Integration helpers ───────────────────────────────────────────────────────────

def _box_integrate(shots: np.ndarray,
                   t:     np.ndarray,
                   t_on:  float,
                   t_off: float) -> np.ndarray:
    """
    Box-car integrate complex shots over [t_on, t_off].

    Parameters
    ----------
    shots : (n_shots, n_time) complex — raw or decimated IQ traces
    t     : (n_time,) time axis in seconds
    t_on  : integration start (seconds)
    t_off : integration end (seconds)

    Returns
    -------
    (n_shots,) complex — mean IQ point per shot
    """
    idx_on  = int(np.searchsorted(t, t_on))
    idx_off = int(np.searchsorted(t, t_off))
    idx_off = max(idx_off, idx_on + 2)        # at least 2 samples
    idx_off = min(idx_off, shots.shape[-1])
    return shots[:, idx_on:idx_off].mean(axis=-1)


def _fit_lda_fidelity(iq_0: np.ndarray, iq_1: np.ndarray) -> float:
    """Fit LDA on in-sample data; return assignment fidelity."""
    try:
        disc = LDADiscriminator().fit(iq_0, iq_1)
        M    = assignment_matrix(disc, iq_0, iq_1)
        return readout_fidelity(M)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────────
# 1. ROC-based optimal threshold
# ─────────────────────────────────────────────────────────────────────────────────

def youden_threshold(fpr: np.ndarray,
                     tpr: np.ndarray,
                     thresholds: np.ndarray) -> tuple:
    """
    Find the Youden-optimal operating point on the ROC curve.

    Youden's J = TPR − FPR.  The threshold maximising J gives the best
    balance between sensitivity and specificity without assuming prior
    class costs.

    Returns
    -------
    (thresh_opt, fpr_opt, tpr_opt)
    """
    J    = tpr - fpr
    idx  = int(np.argmax(J))
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def assignment_matrix_at_threshold(predictor,
                                   iq_0:      np.ndarray,
                                   iq_1:      np.ndarray,
                                   threshold: float = 0.5) -> np.ndarray:
    """
    Compute the 2×2 assignment matrix using a fixed P(|1⟩) threshold.

    Parameters
    ----------
    predictor : object with predict_proba() returning P(|1⟩) per shot
    threshold : shots with P(|1⟩) ≥ threshold are assigned to |1⟩

    Returns
    -------
    M : (2, 2) assignment matrix  M[i, j] = P(readout=i | state=j)
    """
    def _predict(iq):
        proba = predictor.predict_proba(iq)
        return (proba >= threshold).astype(int)

    pred_0 = _predict(iq_0)
    pred_1 = _predict(iq_1)
    p00 = np.mean(pred_0 == 0);  p10 = np.mean(pred_0 == 1)
    p01 = np.mean(pred_1 == 0);  p11 = np.mean(pred_1 == 1)
    return np.array([[p00, p01],
                     [p10, p11]])


def roc_threshold_analysis(iq_0: np.ndarray,
                            iq_1: np.ndarray) -> dict:
    """
    Full ROC threshold analysis: compare default (0.5) vs Youden-optimal.

    Both use the same fitted LDA discriminator; only the decision
    threshold differs.

    Returns
    -------
    dict with keys:
        disc            — fitted LDADiscriminator
        fpr, tpr, thresholds, auc_score — full ROC curve
        thresh_youden, fpr_youden, tpr_youden — Youden-optimal point
        M_default, M_youden  — assignment matrices
        F_default, F_youden  — assignment fidelities
        F_gain               — improvement in percentage points
        thresh_sweep, F_sweep — fidelity vs threshold (fine sweep)
    """
    disc = LDADiscriminator().fit(iq_0, iq_1)
    fpr, tpr, thr, auc_score = compute_roc(disc, iq_0, iq_1)

    thresh_y, fpr_y, tpr_y = youden_threshold(fpr, tpr, thr)

    M_def  = assignment_matrix_at_threshold(disc, iq_0, iq_1, 0.5)
    M_you  = assignment_matrix_at_threshold(disc, iq_0, iq_1, thresh_y)
    F_def  = readout_fidelity(M_def)
    F_you  = readout_fidelity(M_you)

    # Fine sweep of threshold 0→1 for fidelity vs threshold plot
    thresh_sweep = np.linspace(0.02, 0.98, 200)
    F_sweep = np.array([
        readout_fidelity(assignment_matrix_at_threshold(disc, iq_0, iq_1, t))
        for t in thresh_sweep
    ])

    return {
        'disc':          disc,
        'fpr':           fpr,
        'tpr':           tpr,
        'thresholds':    thr,
        'auc_score':     auc_score,
        'thresh_youden': thresh_y,
        'fpr_youden':    fpr_y,
        'tpr_youden':    tpr_y,
        'M_default':     M_def,
        'M_youden':      M_you,
        'F_default':     F_def,
        'F_youden':      F_you,
        'F_gain':        (F_you - F_def) * 100,      # percentage points
        'thresh_sweep':  thresh_sweep,
        'F_sweep':       F_sweep,
    }


# ─────────────────────────────────────────────────────────────────────────────────
# 2. 2-D integration window optimisation
# ─────────────────────────────────────────────────────────────────────────────────

def integration_window_2d(shots_0:   np.ndarray,
                           shots_1:   np.ndarray,
                           t:         np.ndarray,
                           n_start:   int   = 16,
                           n_end:     int   = 16,
                           min_frac:  float = 0.06) -> dict:
    """
    Sweep integration window start (t_on) and end (t_off) independently.

    For each valid (t_on, t_off) pair:
      1. Box-car integrate shots over [t_on, t_off]
      2. Fit LDA discriminator
      3. Compute assignment fidelity

    Parameters
    ----------
    shots_0  : (n_shots, n_time) complex — |0⟩ shots
    shots_1  : (n_shots, n_time) complex — |1⟩ shots
    t        : (n_time,) time axis in seconds
    n_start  : number of t_on grid points
    n_end    : number of t_off grid points
    min_frac : minimum window width as fraction of T_R

    Returns
    -------
    dict with:
        t_on_axis   : (n_start,) in µs
        t_off_axis  : (n_end,) in µs
        fidelity    : (n_start, n_end) — NaN where invalid
        t_on_opt    : optimal t_on (µs)
        t_off_opt   : optimal t_off (µs)
        F_opt       : optimal fidelity
        F_full      : fidelity for full window [0, T_R]
        F_gain_pp   : gain vs full window (percentage points)
        F_1d_start  : (n_start,) fidelity vs t_on with t_off fixed at T_R
        F_1d_end    : (n_end,) fidelity vs t_off with t_on fixed at 0
    """
    T_R     = t[-1]
    min_win = min_frac * T_R

    t_on_ax  = np.linspace(0.0,        0.75 * T_R, n_start)
    t_off_ax = np.linspace(0.25 * T_R, T_R,        n_end)

    fid_map = np.full((n_start, n_end), np.nan)
    for i, t_on in enumerate(t_on_ax):
        for j, t_off in enumerate(t_off_ax):
            if t_off - t_on < min_win:
                continue
            iq_0 = _box_integrate(shots_0, t, t_on, t_off)
            iq_1 = _box_integrate(shots_1, t, t_on, t_off)
            fid_map[i, j] = _fit_lda_fidelity(iq_0, iq_1)

    # Full window baseline
    iq_0_full = shots_0.mean(axis=-1)
    iq_1_full = shots_1.mean(axis=-1)
    F_full    = _fit_lda_fidelity(iq_0_full, iq_1_full)

    # 1-D slices
    F_1d_start = np.array([
        _fit_lda_fidelity(
            _box_integrate(shots_0, t, t_on, T_R),
            _box_integrate(shots_1, t, t_on, T_R))
        for t_on in t_on_ax])

    F_1d_end = np.array([
        _fit_lda_fidelity(
            _box_integrate(shots_0, t, 0.0, t_off),
            _box_integrate(shots_1, t, 0.0, t_off))
        for t_off in t_off_ax])

    flat_idx  = np.nanargmax(fid_map)
    i_opt, j_opt = np.unravel_index(flat_idx, fid_map.shape)

    return {
        't_on_axis':  t_on_ax * 1e6,
        't_off_axis': t_off_ax * 1e6,
        'fidelity':   fid_map,
        't_on_opt':   float(t_on_ax[i_opt] * 1e6),
        't_off_opt':  float(t_off_ax[j_opt] * 1e6),
        'F_opt':      float(np.nanmax(fid_map)),
        'F_full':     F_full,
        'F_gain_pp':  float((np.nanmax(fid_map) - F_full) * 100),
        'i_opt':      i_opt,
        'j_opt':      j_opt,
        'F_1d_start': F_1d_start,
        'F_1d_end':   F_1d_end,
    }


# ─────────────────────────────────────────────────────────────────────────────────
# 3. Quantum efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────────

def quantum_efficiency_sweep(base_params: TransmonParams,
                              eta_values:  np.ndarray | None = None,
                              n_shots:     int = 1000,
                              sigma_ql:    float | None = None,
                              n_tail:      int = 1) -> dict:
    """
    Sweep quantum efficiency η and record assignment fidelity.

    σ_eff = σ_ql / sqrt(η)  where σ_ql is the quantum-limited noise floor.

    Physical reference points:
      η = 1.0        — quantum limit (ideal JPA, zero added noise)
      η = 0.3–0.5    — JPA/TWPA + HEMT chain (state-of-the-art)
      η = 0.05–0.1   — HEMT-only chain (older setups)
      η = 0.01–0.03  — room-temperature preamplifier only

    Returns
    -------
    dict with:
        eta          : array of η values
        sigma_eff    : effective per-sample noise σ
        fidelity_lda : LDA assignment fidelity
        fidelity_gmm : GMM assignment fidelity
        snr_db       : single-shot SNR at steady state [dB]

    Notes
    -----
    n_tail controls how many steady-state samples are averaged per shot:
      n_tail=1  — single-sample regime; per-shot SNR = |Δα|²/(4σ²)
      n_tail>1  — n_tail-sample average; per-shot SNR scales as n_tail
    For demonstrations where η variation should drive visible fidelity
    changes, use n_tail=1 and sigma_ql=0.15 (quantum limit).
    """
    if eta_values is None:
        eta_values = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20,
                                0.30, 0.40, 0.60, 0.80, 1.00])
    if sigma_ql is None:
        sigma_ql = base_params.noise_sigma

    fid_lda      = np.zeros(len(eta_values))
    fid_gmm      = np.zeros(len(eta_values))
    snr_arr      = np.zeros(len(eta_values))
    sigma_eff_arr= np.zeros(len(eta_values))

    for k, eta in enumerate(eta_values):
        sigma_eff = sigma_ql / np.sqrt(eta)
        p = TransmonParams(
            omega_q=base_params.omega_q, alpha_q=base_params.alpha_q,
            omega_r=base_params.omega_r, kappa=base_params.kappa,
            chi=base_params.chi, omega_d=base_params.omega_d,
            epsilon=base_params.epsilon,
            noise_sigma=sigma_eff,
            t1_us=base_params.t1_us, n_bar_th=base_params.n_bar_th,
            t_end=base_params.t_end, n_time=base_params.n_time,
        )
        data   = simulate_shots(n_shots=n_shots, params=p, rng_seed=7 + k)
        # Use last n_tail samples (steady-state region)
        iq_0   = data['shots_0'][:, -n_tail:].mean(axis=-1)
        iq_1   = data['shots_1'][:, -n_tail:].mean(axis=-1)
        a0     = data['alpha_0'];  a1 = data['alpha_1']

        fid_lda[k] = _fit_lda_fidelity(iq_0, iq_1)

        try:
            disc_g = GMMDiscriminator().fit(iq_0, iq_1)
            M_g    = assignment_matrix(disc_g, iq_0, iq_1)
            fid_gmm[k] = readout_fidelity(M_g)
        except Exception:
            fid_gmm[k] = np.nan

        # SNR from steady-state separation
        n     = len(a0)
        tail  = slice(int(0.9 * n), n)
        sig   = float(np.mean(np.abs(a0[tail] - a1[tail]) ** 2))
        snr_arr[k]       = 10 * np.log10(sig / (4 * sigma_eff ** 2))
        sigma_eff_arr[k] = sigma_eff

    return {
        'eta':          eta_values,
        'sigma_eff':    sigma_eff_arr,
        'fidelity_lda': fid_lda,
        'fidelity_gmm': fid_gmm,
        'snr_db':       snr_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────────
# 4. Active reset model
# ─────────────────────────────────────────────────────────────────────────────────

def active_reset_model(F0:          float,
                       F1:          float,
                       n_bar_th:    float,
                       t1_us:       float,
                       latency_us:  np.ndarray | None = None,
                       n_rounds:    int = 4) -> dict:
    """
    Analytical active reset model.

    Protocol per round:
      1. Measure qubit (fidelities F0 = P(correct|0⟩), F1 = P(correct|1⟩))
      2. If outcome is |1⟩: wait τ (feedback latency), then apply π pulse

    State update per round:
      p1_new = p1 * ε10(τ) + (1-p1) * ε01

    where:
      ε10(τ) = P(stay in |1⟩ | was |1⟩, one reset attempted)
             = (1-F1)              # missed detection → no π, stays |1⟩
             + F1*(1-exp(-τ/T1))   # detected, but decayed before π → π takes |0⟩→|1⟩
      ε01    = 1-F0                 # false alarm → π takes |0⟩→|1⟩

    Steady state: p1_ss = ε01 / (ε01 + F1·exp(-τ/T1))

    Parameters
    ----------
    F0, F1     : P(correct readout) per state  (from assignment matrix diagonal)
    n_bar_th   : initial thermal qubit occupation
    t1_us      : T1 in µs
    latency_us : feedback latency sweep axis (µs); default 0→2×T1
    n_rounds   : maximum reset rounds to simulate

    Returns
    -------
    dict with:
        latency_us  : (n_lat,) latency sweep axis
        p1_vs_lat   : (n_rounds, n_lat) residual |1⟩ per round vs latency
        p1_no_reset : initial thermal population (no reset)
        p1_ss       : (n_lat,) steady-state residual
        lat_opt_us  : latency minimising p1 after round 1
        epsilon01   : false-alarm contribution ε01
    """
    if latency_us is None:
        latency_us = np.linspace(0.0, 2.0 * t1_us, 60)

    epsilon01 = 1.0 - F0

    p1_vs_lat = np.zeros((n_rounds, len(latency_us)))
    p1_ss_arr = np.zeros(len(latency_us))

    for li, tau in enumerate(latency_us):
        decay_p   = 1.0 - np.exp(-tau / t1_us)
        eps10_tau = (1.0 - F1) + F1 * decay_p

        p1 = float(n_bar_th)
        for r in range(n_rounds):
            p1 = p1 * eps10_tau + (1.0 - p1) * epsilon01
            p1_vs_lat[r, li] = p1

        # Steady state
        denom = epsilon01 + F1 * (1.0 - decay_p)
        p1_ss_arr[li] = epsilon01 / denom if denom > 0 else 1.0

    lat_opt_idx = int(np.argmin(p1_vs_lat[0, :]))

    return {
        'latency_us':  latency_us,
        'p1_vs_lat':   p1_vs_lat,
        'p1_no_reset': float(n_bar_th),
        'p1_ss':       p1_ss_arr,
        'lat_opt_us':  float(latency_us[lat_opt_idx]),
        'epsilon01':   float(epsilon01),
        'F0':          F0,
        'F1':          F1,
    }


def active_reset_compare_thresholds(iq_0_full:  np.ndarray,
                                    iq_1_full:  np.ndarray,
                                    n_bar_th:   float,
                                    t1_us:      float) -> dict:
    """
    Compare active reset: default (0.5) vs Youden-optimal threshold.

    Runs roc_threshold_analysis to get both thresholds, extracts
    (F0, F1) for each, and calls active_reset_model for both.

    Returns
    -------
    dict with 'default' and 'youden' active_reset_model dicts, plus
    threshold metadata and fidelity comparison.
    """
    disc = LDADiscriminator().fit(iq_0_full, iq_1_full)
    fpr_arr, tpr_arr, thr_arr, auc_val = compute_roc(disc, iq_0_full, iq_1_full)
    thresh_y, fpr_y, tpr_y = youden_threshold(fpr_arr, tpr_arr, thr_arr)

    lat = np.linspace(0.0, 3.0 * t1_us, 80)

    def _get_model(thresh):
        M = assignment_matrix_at_threshold(disc, iq_0_full, iq_1_full, thresh)
        F0 = float(M[0, 0]);  F1 = float(M[1, 1])
        return active_reset_model(F0, F1, n_bar_th, t1_us, lat), M, F0, F1

    res_def, M_def, F0_def, F1_def = _get_model(0.5)
    res_you, M_you, F0_you, F1_you = _get_model(thresh_y)

    return {
        'auc':             auc_val,
        'thresh_default':  0.5,
        'thresh_youden':   thresh_y,
        'fpr_youden':      fpr_y,
        'tpr_youden':      tpr_y,
        'default':         res_def,
        'youden':          res_you,
        'M_default':       M_def,
        'M_youden':        M_you,
        'F_default':       readout_fidelity(M_def),
        'F_youden':        readout_fidelity(M_you),
        'F0_def': F0_def, 'F1_def': F1_def,
        'F0_you': F0_you, 'F1_you': F1_you,
    }
