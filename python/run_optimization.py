#!/usr/bin/env python3
"""
run_optimization.py — Readout Optimisation Demo
=================================================

Generates four publication-quality figures demonstrating advanced
single-shot readout optimisation techniques for superconducting qubits.

Demo scenario
-------------
  κ/2π = 2 MHz,  χ/2π = 1 MHz,  T1 = 3 µs,  T_readout = 1 µs
  noise_sigma = 0.8  →  η ≈ (0.15/0.8)² ≈ 3.5%  (HEMT-only chain)
  n_bar_th = 0.05   →  5% residual thermal population at base temperature

  T1 decay probability during 1 µs window: 1 − exp(−1/3) ≈ 28%
  This creates an asymmetric error matrix (P(0|1) >> P(1|0)) that
  Youden threshold optimisation and optimal integration windows can
  exploit.

Output files (outputs/)
------------------------
  07_roc_threshold.png       — ROC curve + Youden point + confusion matrices
  08_integration_window.png  — 2-D fidelity heatmap + 1-D slices + ring-up
  09_quantum_efficiency.png  — Fidelity vs η + SNR vs η
  10_active_reset.png        — Residual |1⟩ vs rounds and vs latency

Usage
-----
  cd d:\\Resume_Projects\\Project2_Qubit_Readout
  python python/run_optimization.py
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from src.transmon import (TransmonParams, simulate_shots,
                           simulate_cavity, DEFAULT_PARAMS)
from src.readout_optimization import (
    roc_threshold_analysis, integration_window_2d,
    quantum_efficiency_sweep, active_reset_compare_thresholds,
    _box_integrate,
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':       150,
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'legend.fontsize':  9,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'lines.linewidth':  2.0,
})

OUT = Path(ROOT) / 'outputs'
OUT.mkdir(exist_ok=True)
SEP = "=" * 60

# ── Demo parameters ───────────────────────────────────────────────────────────
# σ=0.8 → η=(0.15/0.8)²≈3.5% (HEMT-only), T1=3µs (28% decay in 1µs window)
DEMO_PARAMS = TransmonParams(
    noise_sigma=0.8,
    t1_us=3.0,
    n_bar_th=0.05,
)
SIGMA_QL = 0.15    # quantum-limited noise floor (used in η sweep)
N_SHOTS  = 2000

print(SEP)
print("  Readout Optimisation Demo")
print(SEP)
p = DEMO_PARAMS
print(f"\n  χ/2π = {p.chi/(2*np.pi)/1e6:.1f} MHz  "
      f"κ/2π = {p.kappa/(2*np.pi)/1e6:.1f} MHz  "
      f"T1 = {p.t1_us:.0f} µs  T_R = {p.t_end*1e6:.1f} µs")
print(f"  σ = {p.noise_sigma:.2f}  (η ≈ {(SIGMA_QL/p.noise_sigma)**2:.3f})  "
      f"n_bar = {p.n_bar_th:.2f}")
p_decay = 1 - np.exp(-p.t_end / (p.t1_us * 1e-6))
print(f"  P(T1 decay in readout window) = {p_decay*100:.1f}%")

# ── Simulate shots ────────────────────────────────────────────────────────────
print(f"\n[1/5] Simulating {N_SHOTS} shots per state…", flush=True)
data    = simulate_shots(n_shots=N_SHOTS, params=DEMO_PARAMS, rng_seed=42)
t       = data['t']
shots_0 = data['shots_0']
shots_1 = data['shots_1']
alpha_0 = data['alpha_0']
alpha_1 = data['alpha_1']

# Full-window integrated IQ (used for ROC and active reset)
iq_0_full = shots_0.mean(axis=-1)
iq_1_full = shots_1.mean(axis=-1)

print(f"   T1 decay events : {data['n_relaxed']}/{N_SHOTS}")
print(f"   Thermal flips   : {data['n_thermal']}/{N_SHOTS}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ROC + Youden threshold + confusion matrices
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] ROC threshold analysis…", flush=True)

roc = roc_threshold_analysis(iq_0_full, iq_1_full)

print(f"   AUC          = {roc['auc_score']:.4f}")
print(f"   Default thr  = 0.5  →  F = {roc['F_default']*100:.2f}%")
print(f"   Youden thr   = {roc['thresh_youden']:.3f}  →  F = {roc['F_youden']*100:.2f}%"
      f"  (+{roc['F_gain']:.2f} pp)")

fig = plt.figure(figsize=(12, 9))
fig.suptitle("ROC-Based Threshold Optimisation for Dispersive Readout",
             fontsize=13, fontweight='bold')
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

ax_roc   = fig.add_subplot(gs[:, 0])
ax_fthr  = fig.add_subplot(gs[0, 1])
ax_iqdis = fig.add_subplot(gs[1, 1])
ax_m_def = fig.add_subplot(gs[0, 2])
ax_m_you = fig.add_subplot(gs[1, 2])

# ROC curve
ax_roc.plot(roc['fpr'], roc['tpr'], color='#2980B9', linewidth=2.5,
            label=f"LDA  AUC = {roc['auc_score']:.4f}")
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)

# Default threshold point (P=0.5 → not always on ROC grid, mark nearest)
# Find ROC point closest to (FPR_default, TPR_default)
M_d = roc['M_default']
fpr_def_pt = float(M_d[1, 0])   # P(readout=1|state=0)
tpr_def_pt = float(M_d[1, 1])   # P(readout=1|state=1)
ax_roc.scatter([fpr_def_pt], [tpr_def_pt], s=120, color='#E67E22', zorder=5,
               label=f"thr=0.50  F={roc['F_default']*100:.1f}%")
ax_roc.scatter([roc['fpr_youden']], [roc['tpr_youden']], s=120,
               color='#E74C3C', marker='*', zorder=5,
               label=f"Youden={roc['thresh_youden']:.2f}  F={roc['F_youden']*100:.1f}%")
# Annotate Youden's J
ax_roc.annotate(
    f"ΔF = +{roc['F_gain']:.2f} pp",
    xy=(roc['fpr_youden'], roc['tpr_youden']),
    xytext=(roc['fpr_youden'] + 0.12, roc['tpr_youden'] - 0.12),
    fontsize=9, color='#E74C3C',
    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.3),
)
ax_roc.set_xlabel("False Positive Rate (FPR)"); ax_roc.set_ylabel("True Positive Rate (TPR)")
ax_roc.set_title("ROC Curve", fontweight='bold')
ax_roc.legend(fontsize=8.5); ax_roc.set_aspect('equal')
ax_roc.grid(alpha=0.25)

# Fidelity vs threshold
ax_fthr.plot(roc['thresh_sweep'], roc['F_sweep'] * 100, color='#2980B9', linewidth=2)
ax_fthr.axvline(0.5, color='#E67E22', linestyle='--', linewidth=1.5, label='thr = 0.5')
ax_fthr.axvline(roc['thresh_youden'], color='#E74C3C', linestyle='--',
                linewidth=1.5, label=f"Youden = {roc['thresh_youden']:.2f}")
ax_fthr.axhline(roc['F_default'] * 100, color='#E67E22', alpha=0.35, linewidth=1)
ax_fthr.axhline(roc['F_youden']  * 100, color='#E74C3C', alpha=0.35, linewidth=1)
ax_fthr.set_xlabel("Decision threshold"); ax_fthr.set_ylabel("Assignment fidelity (%)")
ax_fthr.set_title("Fidelity vs Threshold", fontweight='bold')
ax_fthr.legend(fontsize=8.5); ax_fthr.grid(alpha=0.25)

# IQ scatter (subsample)
ns = 300
iq0_plot = iq_0_full[:ns]
iq1_plot = iq_1_full[:ns]
ax_iqdis.scatter(iq0_plot.real, iq0_plot.imag, s=6, alpha=0.4, color='#3498DB', label='|0⟩')
ax_iqdis.scatter(iq1_plot.real, iq1_plot.imag, s=6, alpha=0.4, color='#E74C3C', label='|1⟩')
# Draw LDA decision line (vertical in Q direction at LDA boundary)
disc = roc['disc']
xlim = ax_iqdis.get_xlim() if ax_iqdis.get_xlim() != (0.0, 1.0) else (-1.5, 1.5)
ax_iqdis.set_xlabel("I (arb. units)"); ax_iqdis.set_ylabel("Q (arb. units)")
ax_iqdis.set_title("IQ Scatter (full window)", fontweight='bold')
ax_iqdis.legend(fontsize=8.5, markerscale=3); ax_iqdis.grid(alpha=0.25)

# Confusion matrices
def _plot_confusion(ax, M, title, f):
    im = ax.imshow(M * 100, cmap='Blues', vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            v = M[i, j] * 100
            ax.text(j, i, f"{v:.1f}%", ha='center', va='center',
                    fontsize=10, color='white' if v > 50 else 'black', fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['|0⟩ prep', '|1⟩ prep']); ax.set_yticklabels(['|0⟩ read', '|1⟩ read'])
    ax.set_title(f"{title}\nF = {f*100:.2f}%", fontweight='bold', fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('%')

_plot_confusion(ax_m_def, roc['M_default'], "Default  (thr=0.50)", roc['F_default'])
_plot_confusion(ax_m_you, roc['M_youden'],  f"Youden  (thr={roc['thresh_youden']:.2f})", roc['F_youden'])

fig.savefig(OUT / '07_roc_threshold.png', dpi=150, bbox_inches='tight')
print(f"   Saved → {OUT/'07_roc_threshold.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — 2-D integration window optimisation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] 2-D integration window sweep…", flush=True)

win = integration_window_2d(shots_0, shots_1, t, n_start=20, n_end=20)

print(f"   Full-window F     = {win['F_full']*100:.2f}%")
print(f"   Optimal F         = {win['F_opt']*100:.2f}%  "
      f"(+{win['F_gain_pp']:.2f} pp)")
print(f"   Optimal window    : [{win['t_on_opt']:.3f}, {win['t_off_opt']:.3f}] µs")

fig, axes = plt.subplots(2, 2, figsize=(11, 9),
                          gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2]})
fig.suptitle("2-D Integration Window Optimisation", fontsize=13, fontweight='bold')

ax_top   = axes[0, 0]   # 1-D: fidelity vs t_off (fixed t_on=0)
ax_empty = axes[0, 1]
ax_heat  = axes[1, 0]   # 2-D heatmap
ax_right = axes[1, 1]   # 1-D: fidelity vs t_on (fixed t_off=T_R)

# Ring-up curve on top panel
t_us      = t * 1e6
alpha_mag = np.abs(alpha_1)   # |1⟩ field magnitude (representative)
alpha_mag_norm = alpha_mag / alpha_mag[-1]
ax_top2 = ax_top.twinx()
ax_top2.fill_between(t_us, 0, alpha_mag_norm, alpha=0.15, color='#9B59B6', label='|α(t)|')
ax_top2.set_ylabel('|α(t)| / |α_ss|', color='#9B59B6', fontsize=9)
ax_top2.tick_params(axis='y', labelcolor='#9B59B6', labelsize=8)
ax_top2.set_ylim(0, 1.4)

# τ_ring annotation
kappa = DEMO_PARAMS.kappa
tau_ring_us = 2 / kappa * 1e6   # 2/κ in µs
ax_top.axvline(tau_ring_us, color='#9B59B6', linestyle=':', alpha=0.7, linewidth=1.5)
ax_top.text(tau_ring_us + 0.02, win['F_full'] * 100 - 0.5,
            f"2/κ={tau_ring_us:.2f}µs", fontsize=8, color='#9B59B6')

ax_top.plot(win['t_off_axis'], win['F_1d_end'] * 100,
            color='#2980B9', linewidth=2, label='F vs t_off (t_on=0)')
ax_top.axvline(win['t_off_opt'], color='#E74C3C', linestyle='--', linewidth=1.5)
ax_top.axhline(win['F_full'] * 100, color='gray', linestyle=':', linewidth=1, alpha=0.6)
ax_top.set_xlabel("Integration end t_off (µs)"); ax_top.set_ylabel("Fidelity (%)")
ax_top.set_title("Effect of T1 contamination\n(vary t_off, fix t_on=0)", fontweight='bold')
ax_top.legend(fontsize=8.5, loc='lower left'); ax_top.grid(alpha=0.25)

# 2-D heatmap
fid_pct = win['fidelity'] * 100
im = ax_heat.pcolormesh(win['t_off_axis'], win['t_on_axis'], fid_pct,
                         cmap='RdYlGn', vmin=np.nanmin(fid_pct),
                         vmax=np.nanmax(fid_pct), shading='auto')
plt.colorbar(im, ax=ax_heat, label='Assignment fidelity (%)')
ax_heat.scatter([win['t_off_opt']], [win['t_on_opt']], s=150,
                color='white', edgecolors='black', zorder=5, marker='*',
                label=f"Optimal: [{win['t_on_opt']:.2f}, {win['t_off_opt']:.2f}] µs")
# Full-window point
ax_heat.scatter([t[-1] * 1e6], [0.0], s=80, color='#3498DB',
                edgecolors='black', zorder=5, label='Full window')
# Contour at F_full level
cs = ax_heat.contour(win['t_off_axis'], win['t_on_axis'], fid_pct,
                      levels=[win['F_full'] * 100], colors='white',
                      linewidths=1.5, linestyles='--')
ax_heat.clabel(cs, fmt=f"F_full={win['F_full']*100:.1f}%%", fontsize=8)
ax_heat.set_xlabel("Integration end t_off (µs)")
ax_heat.set_ylabel("Integration start t_on (µs)")
ax_heat.set_title("Assignment Fidelity vs Integration Window", fontweight='bold')
ax_heat.legend(fontsize=8, loc='upper left'); ax_heat.grid(alpha=0.15, color='white')

# Right 1-D panel: fidelity vs t_on (fixed t_off = T_R)
ax_right.plot(win['F_1d_start'] * 100, win['t_on_axis'],
              color='#E67E22', linewidth=2)
ax_right.axhline(win['t_on_opt'], color='#E74C3C', linestyle='--', linewidth=1.5)
ax_right.axhline(tau_ring_us, color='#9B59B6', linestyle=':', linewidth=1.5)
ax_right.text(win['F_full'] * 100 - 0.15, tau_ring_us + 0.02,
              f"2/κ", fontsize=8, color='#9B59B6')
ax_right.set_xlabel("Fidelity (%)")
ax_right.set_ylabel("Integration start t_on (µs)")
ax_right.set_title("Ring-up transient\n(vary t_on, fix t_off=T_R)", fontweight='bold')
ax_right.grid(alpha=0.25)

ax_empty.axis('off')
ax_empty.text(0.5, 0.5,
    f"Full window:\nF = {win['F_full']*100:.2f}%\n\n"
    f"Optimal window:\n[{win['t_on_opt']:.2f}, {win['t_off_opt']:.2f}] µs\n"
    f"F = {win['F_opt']*100:.2f}%\n\n"
    f"Gain: +{win['F_gain_pp']:.2f} pp\n\n"
    f"T1 decay ≈ 28%\nof |1⟩ shots in\nfull 1 µs window",
    transform=ax_empty.transAxes, ha='center', va='center',
    fontsize=10, bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8))

plt.tight_layout()
fig.savefig(OUT / '08_integration_window.png', dpi=150, bbox_inches='tight')
print(f"   Saved → {OUT/'08_integration_window.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Quantum efficiency sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Quantum efficiency sweep…", flush=True)

# Use T1=50µs (no T1 floor) and single last-sample fidelity so that η
# is the dominant driver of fidelity variation.  This models a single-mode
# heterodyne measurement (bandwidth ≈ κ/2π = 2 MHz) where one complex
# sample is captured per shot at steady state.
QE_PARAMS = TransmonParams(
    noise_sigma = SIGMA_QL,    # σ_ql = 0.15 is the η=1 reference
    t1_us       = 50.0,        # negligible T1 decay in 1 µs window
    n_bar_th    = 0.0,
)

eta_vals = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
                     0.15, 0.20, 0.30, 0.40, 0.60, 1.00])

qe = quantum_efficiency_sweep(
    QE_PARAMS,
    eta_values = eta_vals,
    sigma_ql   = SIGMA_QL,
    n_shots    = 1500,
    n_tail     = 1,      # single steady-state sample → per-shot SNR = |Δα|²/(4σ²)
)

print(f"   η=1.0 (quantum limit) : F_LDA = {qe['fidelity_lda'][-1]*100:.2f}%  "
      f"SNR = {qe['snr_db'][-1]:.1f} dB")
idx_hemt = int(np.searchsorted(qe['eta'], 0.05))
print(f"   η=0.05 (HEMT only)    : F_LDA = {qe['fidelity_lda'][idx_hemt]*100:.2f}%  "
      f"SNR = {qe['snr_db'][idx_hemt]:.1f} dB")
print(f"   Single-sample regime (n_tail=1): η directly sets per-shot SNR")

fig, (ax_fid, ax_snr) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Quantum Efficiency vs Assignment Fidelity and SNR",
             fontsize=13, fontweight='bold')

# Reference lines (η_ql for common hardware)
refs = [('JPA/TWPA', 0.40, '#2ECC71'),
        ('JPA+HEMT', 0.15, '#E67E22'),
        ('HEMT only', 0.035, '#E74C3C')]

for label, eta_ref, col in refs:
    ax_fid.axvline(eta_ref, color=col, linestyle=':', linewidth=1.5, alpha=0.7)
    ax_fid.text(eta_ref, 51, label, rotation=90, fontsize=7.5,
                color=col, va='bottom', ha='right')

ax_fid.semilogx(qe['eta'], qe['fidelity_lda'] * 100, 'o-', color='#2980B9',
                 linewidth=2, markersize=7, label='LDA discriminator')
ax_fid.semilogx(qe['eta'], qe['fidelity_gmm'] * 100, 's--', color='#E74C3C',
                 linewidth=2, markersize=6, label='GMM discriminator')
ax_fid.set_xlabel("Quantum efficiency η")
ax_fid.set_ylabel("Assignment fidelity (%)")
ax_fid.set_title("Readout Fidelity vs η", fontweight='bold')
ax_fid.legend(); ax_fid.grid(alpha=0.3, which='both')
ax_fid.set_xlim([qe['eta'][0] * 0.8, 1.2])

for label, eta_ref, col in refs:
    ax_snr.axvline(eta_ref, color=col, linestyle=':', linewidth=1.5, alpha=0.7)

ax_snr.semilogx(qe['eta'], qe['snr_db'], 'o-', color='#9B59B6',
                 linewidth=2, markersize=7, label='Steady-state SNR')

# Overlay theoretical slope: SNR ∝ η → 10log10(η) = 10dB per decade
eta_th = np.logspace(np.log10(qe['eta'][0]), 0, 50)
snr_th_line = qe['snr_db'][-1] + 10 * np.log10(eta_th)
ax_snr.semilogx(eta_th, snr_th_line, 'k--', linewidth=1.3, alpha=0.5,
                 label='SNR ∝ η  (10 dB/decade)')

ax_snr.set_xlabel("Quantum efficiency η")
ax_snr.set_ylabel("Single-shot SNR (dB)")
ax_snr.set_title("SNR vs η", fontweight='bold')
ax_snr.legend(); ax_snr.grid(alpha=0.3, which='both')
ax_snr.set_xlim([qe['eta'][0] * 0.8, 1.2])

plt.tight_layout()
fig.savefig(OUT / '09_quantum_efficiency.png', dpi=150, bbox_inches='tight')
print(f"   Saved → {OUT/'09_quantum_efficiency.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Active reset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Active reset analysis…", flush=True)

# Generate CLEAN shots (n_bar_th=0) to obtain F0/F1 from pure states.
# This separates intrinsic readout noise (AWGN + T1) from thermal
# excitations: thermal shots in iq_0_full inflate the apparent false-alarm
# rate, making the reset look ineffective.  The clean shots give the true
# per-shot classification quality, while n_bar_th=0.05 sets the initial
# state we are trying to reset.
CLEAN_PARAMS = TransmonParams(
    noise_sigma = DEMO_PARAMS.noise_sigma,
    t1_us       = DEMO_PARAMS.t1_us,
    n_bar_th    = 0.0,     # no thermal excitations in the training set
)
clean_data   = simulate_shots(n_shots=N_SHOTS, params=CLEAN_PARAMS, rng_seed=44)
iq_0_clean   = clean_data['shots_0'].mean(axis=-1)
iq_1_clean   = clean_data['shots_1'].mean(axis=-1)

ar = active_reset_compare_thresholds(
    iq_0_clean, iq_1_clean,
    n_bar_th = DEMO_PARAMS.n_bar_th,     # initial thermal population to reset
    t1_us    = DEMO_PARAMS.t1_us,
)

print(f"   Assignment fidelities (default thr=0.5):")
print(f"     F0 = {ar['F0_def']*100:.2f}%  F1 = {ar['F1_def']*100:.2f}%  "
      f"F_avg = {ar['F_default']*100:.2f}%")
print(f"   Assignment fidelities (Youden thr={ar['thresh_youden']:.3f}):")
print(f"     F0 = {ar['F0_you']*100:.2f}%  F1 = {ar['F1_you']*100:.2f}%  "
      f"F_avg = {ar['F_youden']*100:.2f}%")

rd = ar['default'];  ry = ar['youden']

print(f"\n   After round 1 (τ→0):")
print(f"     Default : p1 = {rd['p1_vs_lat'][0, 0]*100:.3f}%")
print(f"     Youden  : p1 = {ry['p1_vs_lat'][0, 0]*100:.3f}%")
print(f"   Thermal baseline: {DEMO_PARAMS.n_bar_th*100:.1f}%")

lat_us = ar['default']['latency_us']

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Active Reset — Residual |1⟩ Population vs Rounds and Latency",
             fontsize=13, fontweight='bold')

ax_rounds = axes[0]
ax_lat    = axes[1]

round_colors = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C']
tau_demo_idx = min(10, len(lat_us) - 1)    # pick a moderate latency

# Panel A: residual vs latency for each round
for r in range(4):
    col = round_colors[r]
    ax_lat.semilogy(lat_us, rd['p1_vs_lat'][r] * 100,
                    linestyle='-', color=col, linewidth=2,
                    label=f"Default, round {r+1}")
    ax_lat.semilogy(lat_us, ry['p1_vs_lat'][r] * 100,
                    linestyle='--', color=col, linewidth=1.8,
                    label=f"Youden, round {r+1}")

ax_lat.axhline(DEMO_PARAMS.n_bar_th * 100, color='black',
               linestyle=':', linewidth=1.5, label=f"Thermal n̄={DEMO_PARAMS.n_bar_th:.2f}")
ax_lat.axvline(DEMO_PARAMS.t1_us, color='gray', linestyle=':', linewidth=1.3, alpha=0.7)
ax_lat.text(DEMO_PARAMS.t1_us + 0.05, DEMO_PARAMS.n_bar_th * 100 * 1.5,
            f"T1={DEMO_PARAMS.t1_us:.0f} µs", fontsize=8.5, color='gray')
ax_lat.set_xlabel("Feedback latency τ (µs)")
ax_lat.set_ylabel("Residual |1⟩ population (%)")
ax_lat.set_title("Residual |1⟩ vs Feedback Latency\n(solid=default thr, dashed=Youden)",
                 fontweight='bold')
ax_lat.legend(fontsize=7.5, ncol=2, loc='upper left'); ax_lat.grid(alpha=0.3, which='both')

# Panel B: convergence curve across rounds (at near-zero latency)
rounds = np.arange(0, 5)
p1_default_rounds = np.concatenate([[DEMO_PARAMS.n_bar_th],
                                     rd['p1_vs_lat'][:, 0]])
p1_youden_rounds  = np.concatenate([[DEMO_PARAMS.n_bar_th],
                                     ry['p1_vs_lat'][:, 0]])
ax_rounds.semilogy(rounds, p1_default_rounds * 100, 'o-', color='#E67E22',
                   linewidth=2, markersize=8, label=f"Default thr=0.5")
ax_rounds.semilogy(rounds, p1_youden_rounds  * 100, 's-', color='#E74C3C',
                   linewidth=2, markersize=8, label=f"Youden thr={ar['thresh_youden']:.2f}")
ax_rounds.axhline(DEMO_PARAMS.n_bar_th * 100, color='black',
                  linestyle=':', linewidth=1.5, label=f"Thermal n̄={DEMO_PARAMS.n_bar_th:.2f}")

# Annotate suppression factor after round 3
sup_def = DEMO_PARAMS.n_bar_th / rd['p1_vs_lat'][2, 0]
sup_you = DEMO_PARAMS.n_bar_th / ry['p1_vs_lat'][2, 0]
ax_rounds.text(3.05, rd['p1_vs_lat'][2, 0] * 100 * 1.2,
               f"{sup_def:.0f}× suppression", fontsize=8.5, color='#E67E22')
ax_rounds.text(3.05, ry['p1_vs_lat'][2, 0] * 100 * 0.7,
               f"{sup_you:.0f}× suppression", fontsize=8.5, color='#E74C3C')

ax_rounds.set_xlabel("Number of reset rounds")
ax_rounds.set_ylabel("Residual |1⟩ population (%)")
ax_rounds.set_title(f"Convergence vs N_rounds  (τ → 0)\n"
                    f"n_bar={DEMO_PARAMS.n_bar_th:.2f}  "
                    f"F1_def={ar['F1_def']*100:.1f}%  "
                    f"F1_you={ar['F1_you']*100:.1f}%",
                    fontweight='bold')
ax_rounds.set_xticks(rounds)
ax_rounds.legend(fontsize=9); ax_rounds.grid(alpha=0.3, which='both')

plt.subplots_adjust(top=0.88, bottom=0.12, left=0.09, right=0.97)
fig.savefig(OUT / '10_active_reset.png', dpi=150, bbox_inches='tight')
print(f"   Saved → {OUT/'10_active_reset.png'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Summary")
print(SEP)
p1_after3_def = rd['p1_vs_lat'][2, 0] * 100
p1_after3_you = ry['p1_vs_lat'][2, 0] * 100
sup_def_3 = DEMO_PARAMS.n_bar_th / rd['p1_vs_lat'][2, 0]
sup_you_3 = DEMO_PARAMS.n_bar_th / ry['p1_vs_lat'][2, 0]
print(f"  Baseline (full window, thr=0.5)        : F = {roc['F_default']*100:.2f}%  (T1+noise limited)")
print(f"  Youden threshold gain                  : +{roc['F_gain']:.2f} pp  "
      f"→ F = {roc['F_youden']*100:.2f}%")
print(f"  Optimal integration window gain        : +{win['F_gain_pp']:.2f} pp  "
      f"→ F = {win['F_opt']*100:.2f}%")
print(f"  QE sweep: η=1 (QL)  → F = {qe['fidelity_lda'][-1]*100:.1f}%  "
      f"SNR = {qe['snr_db'][-1]:.1f} dB")
print(f"           η=0.035 (HEMT) → F = {qe['fidelity_lda'][0]*100:.1f}%  "
      f"SNR = {qe['snr_db'][0]:.1f} dB")
print(f"  Active reset (thr=0.5,  3 rds, τ→0)   : "
      f"{DEMO_PARAMS.n_bar_th*100:.1f}% → {p1_after3_def:.3f}%  ({sup_def_3:.0f}× suppression)")
print(f"  Active reset (Youden, 3 rds, τ→0)     : "
      f"{DEMO_PARAMS.n_bar_th*100:.1f}% → {p1_after3_you:.3f}%  ({sup_you_3:.0f}× suppression)")
print(SEP)
print("  All figures saved to outputs/")
