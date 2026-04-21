#!/usr/bin/env python3
"""
run_pipeline.py — Standalone Qubit Readout Pipeline
=====================================================

Runs the full dispersive readout simulation end-to-end and generates
all publication-quality output plots.  No Jupyter or Vivado required.

Usage
-----
    cd d:\\Resume_Projects\\Project2_Qubit_Readout
    python python/run_pipeline.py

Output files (saved to outputs/)
---------------------------------
    01_cavity_field_buildup.png   — Cavity field for |0⟩ vs |1⟩
    02_ddc_pipeline.png           — 5-stage processing chain waterfall
    03_iq_scatter_gmm.png         — IQ scatter with GMM + LDA boundaries
    04_roc_confusion.png          — ROC curve + confusion matrix
    05_fidelity_vs_time.png       — Fidelity as function of integration window
    06_snr_vs_detuning.png        — Analytical SNR vs drive detuning
    readout_summary.csv           — Numerical results table
"""

import sys
import os

# Allow imports from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from pathlib import Path

from src.transmon    import simulate_shots, snr_db, steady_state_analytical, DEFAULT_PARAMS
from src.readout_chain import process_shots_batch, DEFAULT_CHAIN
from src.discriminator import (GMMDiscriminator, LDADiscriminator,
                                assignment_matrix, readout_fidelity,
                                compute_roc, fidelity_vs_integration_time,
                                snr_vs_detuning, iq_to_xy)

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        150,
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'legend.fontsize':   9,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   1.5,
})

OUT = Path(ROOT) / 'outputs'
OUT.mkdir(exist_ok=True)

SEP = "=" * 60

# ============================================================================
# Step 1 — Simulate qubit physics
# ============================================================================
print(SEP)
print("Project 2 — Superconducting Qubit Readout Pipeline")
print(SEP)
print("\n[1/6] Simulating dispersive readout  (1000 shots × 2 states)...")

data    = simulate_shots(n_shots=1000, rng_seed=42)
t       = data['t']           # (512,) seconds
alpha_0 = data['alpha_0']     # (512,) complex  noise-free
alpha_1 = data['alpha_1']
shots_0 = data['shots_0']     # (1000, 512) complex
shots_1 = data['shots_1']
p       = data['params']

ss0, ss1 = steady_state_analytical(p)
snr      = snr_db(alpha_0, alpha_1, p.noise_sigma)

print(f"   χ/2π  = {p.chi/(2*np.pi)/1e6:.1f} MHz   "
      f"κ/2π = {p.kappa/(2*np.pi)/1e6:.1f} MHz   "
      f"ε/2π = {p.epsilon/(2*np.pi)/1e6:.1f} MHz")
print(f"   Steady-state  |0⟩: {ss0:.4f}   |1⟩: {ss1:.4f}")
print(f"   Single-shot SNR   : {snr:.1f} dB")
print(f"   T1 decay events   : {data['n_relaxed']}/1000 |1⟩ shots  "
      f"(T1={p.t1_us:.0f} µs,  P≈{(1-np.exp(-p.t_end/(p.t1_us*1e-6)))*100:.1f}%)")
print(f"   Thermal flip events: {data['n_thermal']}/1000 |0⟩ shots  "
      f"(n_bar={p.n_bar_th:.2f},  P≈{p.n_bar_th/(p.n_bar_th+1)*100:.1f}%)")

# ── Figure 1: Cavity field buildup ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
fig.suptitle('Dispersive Readout — Cavity Field Buildup', fontsize=13, fontweight='bold')

t_us     = t * 1e6
n_ex     = 8        # example shots to overlay

labels_rc = [('Re(α)  |0⟩', alpha_0.real, shots_0, '#2196F3'),
             ('Im(α)  |0⟩', alpha_0.imag, shots_0, '#2196F3'),
             ('Re(α)  |1⟩', alpha_1.real, shots_1, '#E91E63'),
             ('Im(α)  |1⟩', alpha_1.imag, shots_1, '#E91E63')]

for ax, (title, mean_trace, shots, colour) in zip(axes.flat, labels_rc):
    comp = 'real' if 'Re' in title else 'imag'
    shot_traces = getattr(shots[:n_ex], comp)
    ax.plot(t_us, shot_traces.T, color=colour, alpha=0.25, linewidth=0.8)
    ax.plot(t_us, mean_trace, color=colour, linewidth=2.0, label='Mean')
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.set_title(title)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(loc='lower right')

fig.savefig(OUT / '01_cavity_field_buildup.png', bbox_inches='tight')
plt.close(fig)
print("   → outputs/01_cavity_field_buildup.png")

# ============================================================================
# Step 2 — Digital readout chain
# ============================================================================
print("\n[2/6] Running DDC signal processing chain...")

# Build matched-filter template from difference of noise-free fields
template = alpha_0 - alpha_1   # (n_time,)

# Process full batches
chain_0 = process_shots_batch(shots_0, t, template=template)
chain_1 = process_shots_batch(shots_1, t, template=template)

# Pipeline snapshots for single-shot waterfall plot
shot_idx  = 0
t_us_plot = t * 1e6

# ── Figure 2: DDC pipeline waterfall ─────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), constrained_layout=True)
fig.suptitle('Heterodyne Readout — Digital Signal Processing Chain',
             fontsize=13, fontweight='bold')

gs    = gridspec.GridSpec(5, 1, figure=fig, hspace=0.08)
axes2 = [fig.add_subplot(gs[i]) for i in range(5)]

# Stage labels and data for one |0⟩ shot
stage_data = [
    ('Stage 1 — Raw IF signal  (f_IF = 10 MHz)',
     chain_0['raw_rf'][shot_idx],    '#455A64', False),
    ('Stage 2 — After DDC mixing',
     chain_0['after_ddc'][shot_idx].real, '#1565C0', False),
    ('Stage 3 — After FIR LPF  (cutoff 4 MHz)',
     chain_0['after_lpf'][shot_idx].real, '#6A1B9A', False),
    ('Stage 4 — After decimation  (÷8)',
     chain_0['after_dec'][shot_idx].real,  '#2E7D32', True),
    ('Stage 5 — Box-car integration window',
     chain_0['after_dec'][shot_idx].real,  '#BF360C', True),
]

n_dec    = chain_0['after_dec'].shape[-1]
t_dec_us = np.linspace(0, t[-1] * 1e6, n_dec)

for i, (ax, (label, sig, col, is_dec)) in enumerate(zip(axes2, stage_data)):
    tax = t_dec_us if is_dec else t_us_plot
    ax.plot(tax, sig, color=col, linewidth=0.9)
    ax.set_ylabel('Amp.', fontsize=8)
    ax.set_title(label, fontsize=9, loc='left', pad=2)
    if i < 4:
        ax.set_xticklabels([])
    if i == 4:
        # Shade integration window
        n     = len(sig)
        start = int(0.5 * n)
        ax.axvspan(t_dec_us[start], t_dec_us[-1], color='#BF360C', alpha=0.15,
                   label='50% integration window')
        ax.legend(loc='upper left', fontsize=8)

axes2[-1].set_xlabel('Time (µs)')

fig.savefig(OUT / '02_ddc_pipeline.png', bbox_inches='tight')
plt.close(fig)
print("   → outputs/02_ddc_pipeline.png")

# ============================================================================
# Step 3 — State discrimination
# ============================================================================
print("\n[3/6] Fitting GMM and LDA discriminators...")

iq_bc_0 = chain_0['boxcar_iq']   # (n_shots,) complex
iq_bc_1 = chain_1['boxcar_iq']
iq_mf_0 = chain_0['mf_iq']       # (n_shots,) real
iq_mf_1 = chain_1['mf_iq']

gmm = GMMDiscriminator().fit(iq_bc_0, iq_bc_1)
lda = LDADiscriminator().fit(iq_bc_0, iq_bc_1)

M_gmm = assignment_matrix(gmm, iq_bc_0, iq_bc_1)
M_lda = assignment_matrix(lda, iq_bc_0, iq_bc_1)
F_gmm = readout_fidelity(M_gmm)
F_lda = readout_fidelity(M_lda)

print(f"   GMM fidelity  : {F_gmm*100:.2f}%   "
      f"(P(0|1)={M_gmm[0,1]*100:.2f}%,  P(1|0)={M_gmm[1,0]*100:.2f}%)")
print(f"   LDA fidelity  : {F_lda*100:.2f}%   "
      f"(P(0|1)={M_lda[0,1]*100:.2f}%,  P(1|0)={M_lda[1,0]*100:.2f}%)")

# ── Figure 3: IQ scatter + GMM + LDA ─────────────────────────────────────────
fig, axes3 = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
fig.suptitle('IQ Plane — State Discrimination', fontsize=13, fontweight='bold')

def _draw_gmm_ellipses(ax, gmm_disc, n_std=2.0):
    for k, (mean, cov) in enumerate(zip(gmm_disc.means, gmm_disc.covariances)):
        eigvals, eigvecs = np.linalg.eigh(cov)
        order   = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        angle   = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        w, h    = 2 * n_std * np.sqrt(eigvals)
        colour  = '#2196F3' if k == 0 else '#E91E63'
        ell     = Ellipse(xy=mean, width=w, height=h, angle=angle,
                          edgecolor=colour, facecolor='none', linewidth=2,
                          linestyle='--', zorder=5)
        ax.add_patch(ell)

# Left: scatter + GMM ellipses
ax = axes3[0]
ax.scatter(iq_bc_0.real, iq_bc_0.imag, s=4, alpha=0.4,
           color='#2196F3', label='|0⟩', rasterized=True)
ax.scatter(iq_bc_1.real, iq_bc_1.imag, s=4, alpha=0.4,
           color='#E91E63', label='|1⟩', rasterized=True)
_draw_gmm_ellipses(ax, gmm, n_std=2.0)
ax.set_xlabel('I (a.u.)')
ax.set_ylabel('Q (a.u.)')
ax.set_title(f'GMM discriminator  (F = {F_gmm*100:.2f}%)')
ax.legend(markerscale=3)

# Right: scatter + LDA boundary
ax = axes3[1]
ax.scatter(iq_bc_0.real, iq_bc_0.imag, s=4, alpha=0.4,
           color='#2196F3', label='|0⟩', rasterized=True)
ax.scatter(iq_bc_1.real, iq_bc_1.imag, s=4, alpha=0.4,
           color='#E91E63', label='|1⟩', rasterized=True)
all_I = np.concatenate([iq_bc_0.real, iq_bc_1.real])
xlim  = (all_I.min() - 0.05, all_I.max() + 0.05)
xs, ys = lda.decision_line_points(xlim)
ax.plot(xs, ys, 'k--', linewidth=1.5, label='LDA boundary')
ax.set_xlim(xlim)
ax.set_xlabel('I (a.u.)')
ax.set_ylabel('Q (a.u.)')
ax.set_title(f'LDA discriminator  (F = {F_lda*100:.2f}%)')
ax.legend(markerscale=3)

fig.savefig(OUT / '03_iq_scatter_gmm.png', bbox_inches='tight')
plt.close(fig)
print("   → outputs/03_iq_scatter_gmm.png")

# ============================================================================
# Step 4 — ROC curve + confusion matrix
# ============================================================================
print("\n[4/6] Computing ROC curve and confusion matrix...")

fpr, tpr, _, auc_lda = compute_roc(lda, iq_bc_0, iq_bc_1)
print(f"   LDA ROC AUC : {auc_lda:.5f}")

fig, axes4 = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
fig.suptitle('Readout Performance Metrics', fontsize=13, fontweight='bold')

# ROC curve
ax = axes4[0]
ax.plot(fpr, tpr, color='#1565C0', linewidth=2, label=f'LDA  (AUC = {auc_lda:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax.set_xlabel('False Positive Rate  P(1|0)')
ax.set_ylabel('True Positive Rate   P(1|1)')
ax.set_title('ROC Curve')
ax.legend()
ax.set_aspect('equal')

# Confusion matrix (LDA)
ax = axes4[1]
import matplotlib.cm as cm
im = ax.imshow(M_lda, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(['|0⟩', '|1⟩'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Readout 0', 'Readout 1'])
ax.set_xlabel('Prepared state')
ax.set_title(f'Assignment Matrix (LDA)\nFidelity = {F_lda*100:.2f}%')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{M_lda[i,j]:.4f}',
                ha='center', va='center',
                color='white' if M_lda[i, j] > 0.5 else 'black',
                fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.savefig(OUT / '04_roc_confusion.png', bbox_inches='tight')
plt.close(fig)
print("   → outputs/04_roc_confusion.png")

# ============================================================================
# Step 5 — Fidelity vs integration time + SNR vs detuning
# ============================================================================
print("\n[5/6] Sweeping fidelity vs integration window...")

fracs, fidels = fidelity_vs_integration_time(
    chain_0['after_dec'], chain_1['after_dec'], n_fractions=30)
det_MHz, snr_norm = snr_vs_detuning()

fig, axes5 = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
fig.suptitle('Readout Optimisation', fontsize=13, fontweight='bold')

ax = axes5[0]
ax.plot(fracs * 100, fidels * 100, color='#2E7D32', marker='o',
        markersize=4, label='LDA fidelity')
ax.axhline(y=F_lda * 100, color='gray', linestyle='--', linewidth=1,
           label=f'Full-window (50% box-car): {F_lda*100:.2f}%')
ax.set_xlabel('Readout duration (% of pulse window,  t = 0 → T)')
ax.set_ylabel('Readout Fidelity (%)')
ax.set_title('Fidelity vs. Integration Duration')
ax.set_ylim([max(50, fidels.min() * 100 - 2), 101])
ax.legend()

ax = axes5[1]
peak_det = det_MHz[np.argmax(snr_norm)]
ax.plot(det_MHz, snr_norm, color='#6A1B9A', linewidth=2)
ax.axvline(x=peak_det, color='gray', linestyle='--', linewidth=1,
           label=f'Peak at Δ = {peak_det:.1f} MHz')
ax.set_xlabel('Drive detuning from bare cavity (MHz)')
ax.set_ylabel('Normalised SNR (linear)')
ax.set_title('SNR vs. Drive Frequency Detuning')
ax.legend()

fig.savefig(OUT / '05_fidelity_vs_time.png', bbox_inches='tight')
plt.close(fig)
print("   → outputs/05_fidelity_vs_time.png")

# ============================================================================
# Step 6 — Save results table
# ============================================================================
print("\n[6/6] Saving results table...")

import csv
rows = [
    ['Metric',                   'Value', 'Unit'],
    ['chi/2pi',                  f'{p.chi/(2*np.pi)/1e6:.2f}',  'MHz'],
    ['kappa/2pi',                f'{p.kappa/(2*np.pi)/1e6:.2f}', 'MHz'],
    ['epsilon/2pi',              f'{p.epsilon/(2*np.pi)/1e6:.2f}', 'MHz'],
    ['T1 relaxation time',       f'{p.t1_us:.0f}',  'µs'],
    ['Thermal occupation n_bar', f'{p.n_bar_th:.3f}', ''],
    ['T1 decay events (|1>)',    f'{data["n_relaxed"]}', '/ 1000 shots'],
    ['Thermal flips (|0>)',      f'{data["n_thermal"]}', '/ 1000 shots'],
    ['Single-shot SNR',          f'{snr:.2f}',   'dB'],
    ['Shots per state',          '1000',         ''],
    ['GMM fidelity',             f'{F_gmm*100:.3f}', '%'],
    ['LDA fidelity',             f'{F_lda*100:.3f}', '%'],
    ['LDA ROC AUC',              f'{auc_lda:.5f}', ''],
    ['P(1|0)  LDA',              f'{M_lda[1,0]*100:.3f}', '%'],
    ['P(0|1)  LDA',              f'{M_lda[0,1]*100:.3f}', '%'],
    ['Max fidelity (full int.)', f'{fidels.max()*100:.3f}', '%'],
]

csv_path = OUT / 'readout_summary.csv'
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(rows)

print(f"\n{'─'*60}")
print("Results summary:")
for row in rows[1:]:
    print(f"  {row[0]:<30} {row[1]:>8} {row[2]}")

print(f"\nAll outputs saved to  {OUT}")
print(SEP)
