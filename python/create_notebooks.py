#!/usr/bin/env python3
"""
create_notebooks.py — Generate Jupyter Notebooks for Project 2
===============================================================

Creates 4 .ipynb files in the notebooks/ directory using nbformat.
Run this once; then execute the notebooks with:

    jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace

or open them in VS Code / JupyterLab.

Usage
-----
    cd d:\\Resume_Projects\\Project2_Qubit_Readout
    python python/create_notebooks.py
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import nbformat as nbf
from pathlib import Path

NB_DIR = Path(ROOT) / 'notebooks'
NB_DIR.mkdir(exist_ok=True)


def _md(*lines):
    return nbf.v4.new_markdown_cell('\n'.join(lines))


def _code(*lines):
    return nbf.v4.new_code_cell('\n'.join(lines))


def _nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }
    nb.metadata['language_info'] = {'name': 'python', 'version': '3.12.0'}
    return nb


# ============================================================================
# Notebook 01 — Qubit Physics
# ============================================================================

nb01 = _nb([
    _md(
        '# Notebook 01 — Dispersive Readout: Qubit Physics',
        '',
        '**Project 2 — Superconducting Qubit Readout Signal Processing Pipeline**',
        '',
        '---',
        '',
        '## Theory',
        '',
        'In the **dispersive regime** ($g \\ll |\\omega_q - \\omega_r|$), the '
        'Jaynes-Cummings Hamiltonian reduces to:',
        '',
        '$$H_{\\rm eff}/\\hbar = (\\omega_r + \\chi \\sigma_z)\\,a^\\dagger a '
        '\\;+\\; \\frac{\\omega_q}{2}\\sigma_z$$',
        '',
        'The cavity resonance shifts by $\\pm\\chi$ depending on the qubit state — '
        'this is the **dispersive shift** that enables quantum non-demolition readout.',
        '',
        '| Qubit state | Cavity frequency | Drive detuning $\\delta$ |',
        '|:---:|:---:|:---:|',
        '| $|0\\rangle$ | $\\omega_r + \\chi$ | $\\omega_d - (\\omega_r + \\chi)$ |',
        '| $|1\\rangle$ | $\\omega_r - \\chi$ | $\\omega_d - (\\omega_r - \\chi)$ |',
        '',
        'In the rotating frame at $\\omega_d$, the classical cavity field obeys:',
        '',
        '$$\\frac{d\\alpha}{dt} = -\\left(\\frac{\\kappa}{2} + i\\delta_s\\right)\\alpha + \\varepsilon$$',
        '',
        'with steady state $\\alpha_{\\rm ss} = \\varepsilon\\,/\\,(\\kappa/2 + i\\delta_s)$.',
    ),

    _code(
        'import sys, os',
        'sys.path.insert(0, os.path.abspath(".."))',
        '',
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'from src.transmon import (',
        '    TransmonParams, DEFAULT_PARAMS, simulate_shots,',
        '    snr_db, steady_state_analytical, print_hamiltonian)',
        '',
        '%matplotlib inline',
        'plt.rcParams.update({"figure.dpi": 120, "font.size": 10})',
    ),

    _code(
        '# Display physical parameters and Hamiltonian',
        'p = DEFAULT_PARAMS',
        'print_hamiltonian(p)',
        'print()',
        'print(f"Drive frequency      : {p.omega_d/(2*np.pi)/1e9:.3f} GHz  (at bare cavity)")',
        'print(f"Noise sigma          : {p.noise_sigma}")',
        'print(f"Readout window       : {p.t_end*1e6:.1f} µs  ({p.n_time} samples)")',
    ),

    _md(
        '## 1.1 Noise-free cavity field buildup',
        '',
        'Solving $d\\alpha/dt = -(\\kappa/2 + i\\delta_s)\\alpha + \\varepsilon$ '
        'with $\\alpha(0)=0$.',
    ),

    _code(
        'from src.transmon import simulate_cavity',
        '',
        'alpha_0, t = simulate_cavity(0)',
        'alpha_1, _ = simulate_cavity(1)',
        '',
        'ss0, ss1 = steady_state_analytical()',
        'print(f"Steady-state |0>: {ss0:.4f}   (I={ss0.real:.3f}, Q={ss0.imag:.3f})")',
        'print(f"Steady-state |1>: {ss1:.4f}   (I={ss1.real:.3f}, Q={ss1.imag:.3f})")',
        'print(f"Separation |a0-a1| = {abs(ss0-ss1):.4f}")',
    ),

    _code(
        'fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)',
        'fig.suptitle("Dispersive Readout — Noise-free Cavity Field Buildup", fontsize=12, fontweight="bold")',
        't_us = t * 1e6',
        '',
        'config = [',
        '    ("Re(α)  |0⟩", alpha_0.real, "#2196F3"),',
        '    ("Im(α)  |0⟩", alpha_0.imag, "#2196F3"),',
        '    ("Re(α)  |1⟩", alpha_1.real, "#E91E63"),',
        '    ("Im(α)  |1⟩", alpha_1.imag, "#E91E63"),',
        ']',
        'for ax, (title, trace, col) in zip(axes.flat, config):',
        '    ax.plot(t_us, trace, color=col, linewidth=1.8)',
        '    ax.axhline(0, color="gray", linewidth=0.5)',
        '    ax.set_xlabel("Time (µs)"); ax.set_ylabel("Amplitude (a.u.)")',
        '    ax.set_title(title)',
        '    # Mark steady state',
        '    if "|0" in title and "Re" in title: ss_val = ss0.real',
        '    elif "|0" in title: ss_val = ss0.imag',
        '    elif "Re" in title: ss_val = ss1.real',
        '    else: ss_val = ss1.imag',
        '    ax.axhline(ss_val, color=col, linewidth=1, linestyle="--", alpha=0.6, label="Steady state")',
        '    ax.legend(fontsize=8)',
        'plt.show()',
    ),

    _md(
        '## 1.2 Monte-Carlo shot simulation (1000 shots × 2 states)',
        '',
        'AWGN is added to each shot independently, modelling amplifier noise and '
        'quantum vacuum fluctuations.',
    ),

    _code(
        'data    = simulate_shots(n_shots=1000, rng_seed=42)',
        'shots_0 = data["shots_0"]   # (1000, 512) complex',
        'shots_1 = data["shots_1"]',
        '',
        'snr = snr_db(alpha_0, alpha_1, p.noise_sigma)',
        'print(f"Single-shot SNR at steady state: {snr:.2f} dB")',
    ),

    _code(
        '# Plot: few individual shots + ensemble mean',
        'fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)',
        'fig.suptitle("Noisy Single Shots vs Ensemble Mean", fontsize=12, fontweight="bold")',
        '',
        'n_ex = 15',
        'for ax, (shots, alpha, col, label) in zip(axes, [',
        '    (shots_0, alpha_0, "#2196F3", "|0⟩"),',
        '    (shots_1, alpha_1, "#E91E63", "|1⟩"),',
        ']):',
        '    ax.plot(t_us, shots[:n_ex].real.T, color=col, alpha=0.2, linewidth=0.7)',
        '    ax.plot(t_us, alpha.real, color=col, linewidth=2.2, label=f"Mean {label}")',
        '    ax.set_xlabel("Time (µs)"); ax.set_ylabel("Re(α) (a.u.)")',
        '    ax.set_title(f"Re[α(t)] for state {label}  ({n_ex} shots shown)")',
        '    ax.legend()',
        'plt.show()',
    ),

    _code(
        '# IQ scatter at t = t_end (last sample, before integration)',
        'fig, ax = plt.subplots(figsize=(6, 5))',
        'ax.scatter(shots_0[:, -1].real, shots_0[:, -1].imag,',
        '           s=5, alpha=0.4, color="#2196F3", label="|0⟩", rasterized=True)',
        'ax.scatter(shots_1[:, -1].real, shots_1[:, -1].imag,',
        '           s=5, alpha=0.4, color="#E91E63", label="|1⟩", rasterized=True)',
        'ax.plot(ss0.real, ss0.imag, "b*", markersize=12, label=f"SS |0\u27e9 = {ss0:.3f}")',
        'ax.plot(ss1.real, ss1.imag, "r*", markersize=12, label=f"SS |1\u27e9 = {ss1:.3f}")',
        'ax.set_xlabel("I (a.u.)"); ax.set_ylabel("Q (a.u.)")',
        'ax.set_title(f"Raw IQ Scatter at t = {p.t_end*1e6:.1f} µs  (SNR = {snr:.1f} dB)")',
        'ax.legend(markerscale=2); ax.set_aspect("equal")',
        'plt.tight_layout(); plt.show()',
    ),
])


# ============================================================================
# Notebook 02 — DDC Signal Processing
# ============================================================================

nb02 = _nb([
    _md(
        '# Notebook 02 — Digital Down-Conversion & Signal Processing',
        '',
        '**Project 2 — Superconducting Qubit Readout Signal Processing Pipeline**',
        '',
        '---',
        '',
        '## Theory',
        '',
        'The heterodyne detection chain converts the cavity output to a digital '
        'baseband I/Q signal through five stages:',
        '',
        '| Stage | Operation | Purpose |',
        '|:---:|:---|:---|',
        '| 1 | IF upconversion | Simulate heterodyne: $s_{\\rm rf}(t) = \\mathrm{Re}[\\alpha(t)\\,e^{j\\omega_{\\rm IF}t}]$ |',
        '| 2 | DDC mixing | Multiply by $e^{-j\\omega_{\\rm IF}t}$ → complex baseband |',
        '| 3 | FIR LPF | Remove image at $-2f_{\\rm IF}$; 63-tap Hamming-windowed sinc |',
        '| 4 | Decimation | Downsample by 8 (after LPF to prevent aliasing) |',
        '| 5 | Integration | Box-car or matched filter → single I/Q point per shot |',
    ),

    _code(
        'import sys, os',
        'sys.path.insert(0, os.path.abspath(".."))',
        '',
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'from src.transmon import simulate_shots',
        'from src.readout_chain import (',
        '    process_shots_batch, process_single_shot,',
        '    DEFAULT_CHAIN, build_lpf)',
        '',
        '%matplotlib inline',
        'plt.rcParams.update({"figure.dpi": 120, "font.size": 10})',
        '',
        'data    = simulate_shots(n_shots=200, rng_seed=0)',
        't       = data["t"]',
        'shots_0 = data["shots_0"]',
        'shots_1 = data["shots_1"]',
        'alpha_0 = data["alpha_0"]',
        'alpha_1 = data["alpha_1"]',
        'template = alpha_0 - alpha_1',
    ),

    _md('## 2.1 FIR filter frequency response'),

    _code(
        'from scipy.signal import freqz',
        '',
        'h    = build_lpf(DEFAULT_CHAIN)',
        'w, H = freqz(h, worN=4096, fs=DEFAULT_CHAIN.fs)',
        '',
        'fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)',
        'fig.suptitle(f"FIR LPF — {DEFAULT_CHAIN.fir_taps}-tap Hamming-windowed sinc", fontweight="bold")',
        '',
        'axes[0].plot(w/1e6, 20*np.log10(np.abs(H)+1e-12), "#1565C0")',
        'axes[0].axvline(DEFAULT_CHAIN.fir_cutoff/1e6, color="red", linestyle="--",',
        '                label=f"Cutoff {DEFAULT_CHAIN.fir_cutoff/1e6:.0f} MHz")',
        'axes[0].set_xlim([0, 100]); axes[0].set_ylim([-80, 5])',
        'axes[0].set_xlabel("Frequency (MHz)"); axes[0].set_ylabel("|H(f)| (dB)")',
        'axes[0].set_title("Magnitude response"); axes[0].legend()',
        '',
        'axes[1].plot(w/1e6, np.unwrap(np.angle(H, deg=False))*180/np.pi, "#6A1B9A")',
        'axes[1].set_xlim([0, DEFAULT_CHAIN.fir_cutoff/1e6*3])',
        'axes[1].set_xlabel("Frequency (MHz)"); axes[1].set_ylabel("Phase (°)")',
        'axes[1].set_title("Phase response (passband)")',
        'plt.show()',
    ),

    _md('## 2.2 Processing pipeline — single shot waterfall'),

    _code(
        'import matplotlib.gridspec as gridspec',
        '',
        'shot_res = process_single_shot(shots_0[0], t)',
        't_us     = t * 1e6',
        'n_dec    = shot_res["after_dec"].shape[0]',
        't_dec_us = np.linspace(0, t[-1]*1e6, n_dec)',
        '',
        'fig = plt.figure(figsize=(11, 9), constrained_layout=True)',
        'fig.suptitle("Heterodyne Readout — Single Shot Pipeline",',
        '             fontsize=12, fontweight="bold")',
        'gs   = gridspec.GridSpec(5, 1, figure=fig)',
        'axs  = [fig.add_subplot(gs[i]) for i in range(5)]',
        '',
        'stages = [',
        '    ("1 — Raw IF  (f_IF=10 MHz)",     t_us,     shot_res["raw_rf"].real,    "#455A64"),',
        '    ("2 — After DDC mixing",           t_us,     shot_res["after_ddc"].real, "#1565C0"),',
        '    ("3 — After FIR LPF",              t_us,     shot_res["after_lpf"].real, "#6A1B9A"),',
        '    ("4 — After decimation (÷8)",      t_dec_us, shot_res["after_dec"].real, "#2E7D32"),',
        '    ("5 — Box-car integration window", t_dec_us, shot_res["after_dec"].real, "#BF360C"),',
        ']',
        'for i, (ax, (lbl, tax, sig, col)) in enumerate(zip(axs, stages)):',
        '    ax.plot(tax, sig, color=col, linewidth=0.9)',
        '    ax.set_title(lbl, fontsize=9, loc="left")',
        '    ax.set_ylabel("Amp.", fontsize=8)',
        '    if i < 4: ax.set_xticklabels([])',
        '    if i == 4:',
        '        n = len(sig); start = int(0.5*n)',
        '        ax.axvspan(t_dec_us[start], t_dec_us[-1], alpha=0.15, color=col,',
        '                   label="50% window")',
        '        ax.legend(fontsize=8)',
        'axs[-1].set_xlabel("Time (µs)")',
        'plt.show()',
    ),

    _md('## 2.3 Batch processing — integrated IQ clouds'),

    _code(
        'chain_0 = process_shots_batch(shots_0, t, template=template)',
        'chain_1 = process_shots_batch(shots_1, t, template=template)',
        '',
        'iq_bc_0 = chain_0["boxcar_iq"]',
        'iq_bc_1 = chain_1["boxcar_iq"]',
        'iq_mf_0 = chain_0["mf_iq"]',
        'iq_mf_1 = chain_1["mf_iq"]',
        '',
        'print(f"|0> box-car: mean=({iq_bc_0.real.mean():.4f}, {iq_bc_0.imag.mean():.4f})",',
        '      f"  std=({iq_bc_0.real.std():.4f}, {iq_bc_0.imag.std():.4f})") ',
        'print(f"|1> box-car: mean=({iq_bc_1.real.mean():.4f}, {iq_bc_1.imag.mean():.4f})",',
        '      f"  std=({iq_bc_1.real.std():.4f}, {iq_bc_1.imag.std():.4f})") ',
    ),

    _code(
        'fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)',
        'fig.suptitle("Integrated IQ Clouds", fontsize=12, fontweight="bold")',
        '',
        '# Box-car',
        'ax = axes[0]',
        'ax.scatter(iq_bc_0.real, iq_bc_0.imag, s=6, alpha=0.4,',
        '           color="#2196F3", label="|0⟩", rasterized=True)',
        'ax.scatter(iq_bc_1.real, iq_bc_1.imag, s=6, alpha=0.4,',
        '           color="#E91E63", label="|1⟩", rasterized=True)',
        'ax.set_xlabel("I"); ax.set_ylabel("Q")',
        'ax.set_title("Box-car integrator (50%)"); ax.legend(markerscale=3)',
        '',
        '# Matched filter',
        'ax = axes[1]',
        'ax.hist(iq_mf_0, bins=40, alpha=0.6, color="#2196F3",',
        '        label="|0⟩", density=True)',
        'ax.hist(iq_mf_1, bins=40, alpha=0.6, color="#E91E63",',
        '        label="|1⟩", density=True)',
        'ax.set_xlabel("Matched-filter output (a.u.)")',
        'ax.set_ylabel("Probability density")',
        'ax.set_title("Matched-filter integrator"); ax.legend()',
        'plt.show()',
    ),
])


# ============================================================================
# Notebook 03 — State Discrimination
# ============================================================================

nb03 = _nb([
    _md(
        '# Notebook 03 — State Discrimination',
        '',
        '**Project 2 — Superconducting Qubit Readout Signal Processing Pipeline**',
        '',
        '---',
        '',
        '## Theory',
        '',
        '### Readout Fidelity',
        '',
        'The **assignment matrix** $M$ encodes classification errors:',
        '',
        '$$M = \\begin{pmatrix} P(0|0) & P(0|1) \\\\ P(1|0) & P(1|1) \\end{pmatrix}$$',
        '',
        'Readout fidelity: $F = 1 - \\frac{P(0|1) + P(1|0)}{2}$',
        '',
        '### GMM Discriminator',
        '',
        'Fits a 2-component Gaussian Mixture Model on combined IQ data; '
        'label ambiguity resolved by proximity to known centroids.',
        '',
        '### LDA Discriminator',
        '',
        'Finds the optimal linear boundary $\\mathbf{w}^T \\mathbf{x} + b = 0$ '
        'that maximises the Fisher ratio. Equivalent to matched filter + threshold '
        'under Gaussian noise.',
    ),

    _code(
        'import sys, os',
        'sys.path.insert(0, os.path.abspath(".."))',
        '',
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'from matplotlib.patches import Ellipse',
        '',
        'from src.transmon      import simulate_shots',
        'from src.readout_chain import process_shots_batch',
        'from src.discriminator import (',
        '    GMMDiscriminator, LDADiscriminator,',
        '    assignment_matrix, readout_fidelity,',
        '    compute_roc, fidelity_vs_integration_time)',
        '',
        '%matplotlib inline',
        'plt.rcParams.update({"figure.dpi": 120, "font.size": 10})',
        '',
        'data     = simulate_shots(n_shots=1000, rng_seed=42)',
        't        = data["t"]',
        'shots_0  = data["shots_0"]',
        'shots_1  = data["shots_1"]',
        'template = data["alpha_0"] - data["alpha_1"]',
        '',
        'chain_0  = process_shots_batch(shots_0, t, template=template)',
        'chain_1  = process_shots_batch(shots_1, t, template=template)',
        'iq_0     = chain_0["boxcar_iq"]',
        'iq_1     = chain_1["boxcar_iq"]',
    ),

    _md('## 3.1 Fit discriminators'),

    _code(
        'gmm = GMMDiscriminator().fit(iq_0, iq_1)',
        'lda = LDADiscriminator().fit(iq_0, iq_1)',
        '',
        'M_gmm = assignment_matrix(gmm, iq_0, iq_1)',
        'M_lda = assignment_matrix(lda, iq_0, iq_1)',
        'F_gmm = readout_fidelity(M_gmm)',
        'F_lda = readout_fidelity(M_lda)',
        '',
        'print("=" * 50)',
        'print(f"GMM  fidelity : {F_gmm*100:.3f}%")',
        'print(f"     P(0|1)   : {M_gmm[0,1]*100:.3f}%")',
        'print(f"     P(1|0)   : {M_gmm[1,0]*100:.3f}%")',
        'print("─" * 50)',
        'print(f"LDA  fidelity : {F_lda*100:.3f}%")',
        'print(f"     P(0|1)   : {M_lda[0,1]*100:.3f}%")',
        'print(f"     P(1|0)   : {M_lda[1,0]*100:.3f}%")',
        'print("=" * 50)',
    ),

    _md('## 3.2 IQ scatter with decision boundaries'),

    _code(
        'def draw_ellipses(ax, gmm_disc, n_std=2.0):',
        '    for mean, cov, col in zip(gmm_disc.means, gmm_disc.covariances,',
        '                              ["#2196F3", "#E91E63"]):',
        '        eigvals, eigvecs = np.linalg.eigh(cov)',
        '        order   = eigvals.argsort()[::-1]',
        '        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]',
        '        angle   = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))',
        '        w, h    = 2*n_std*np.sqrt(eigvals)',
        '        ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,',
        '                             edgecolor=col, facecolor="none", lw=2,',
        '                             linestyle="--", zorder=5))',
        '',
        'fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)',
        'fig.suptitle("IQ Plane — State Discrimination", fontsize=12, fontweight="bold")',
        '',
        'for ax, (disc, M, F, title_sfx) in zip(axes, [',
        '    (gmm, M_gmm, F_gmm, "GMM"),',
        '    (lda, M_lda, F_lda, "LDA"),',
        ']):',
        '    ax.scatter(iq_0.real, iq_0.imag, s=5, alpha=0.35,',
        '               color="#2196F3", label="|0⟩", rasterized=True)',
        '    ax.scatter(iq_1.real, iq_1.imag, s=5, alpha=0.35,',
        '               color="#E91E63", label="|1⟩", rasterized=True)',
        '    if title_sfx == "GMM":',
        '        draw_ellipses(ax, gmm)',
        '    else:',
        '        xlim = (min(iq_0.real.min(), iq_1.real.min())-0.05,',
        '                max(iq_0.real.max(), iq_1.real.max())+0.05)',
        '        xs, ys = lda.decision_line_points(xlim)',
        '        ax.plot(xs, ys, "k--", lw=1.5, label="LDA boundary")',
        '        ax.set_xlim(xlim)',
        '    ax.set_xlabel("I (a.u.)"); ax.set_ylabel("Q (a.u.)")',
        '    ax.set_title(f"{title_sfx}  (F = {F*100:.2f}%)")',
        '    ax.legend(markerscale=3)',
        'plt.show()',
    ),

    _md('## 3.3 Confusion matrix'),

    _code(
        'fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)',
        'fig.suptitle("Assignment Matrices", fontsize=12, fontweight="bold")',
        '',
        'for ax, (M, F, label) in zip(axes, [',
        '    (M_gmm, F_gmm, "GMM"),',
        '    (M_lda, F_lda, "LDA"),',
        ']):',
        '    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)',
        '    ax.set_xticks([0,1]); ax.set_xticklabels(["|0⟩", "|1⟩"])',
        '    ax.set_yticks([0,1]); ax.set_yticklabels(["Readout 0", "Readout 1"])',
        '    ax.set_xlabel("Prepared state")',
        '    ax.set_title(f"{label}  F={F*100:.2f}%")',
        '    for i in range(2):',
        '        for j in range(2):',
        '            ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center",',
        '                    fontsize=12, fontweight="bold",',
        '                    color="white" if M[i,j]>0.5 else "black")',
        '    plt.colorbar(im, ax=ax, fraction=0.046)',
        'plt.show()',
    ),

    _md('## 3.4 ROC curve'),

    _code(
        'from sklearn.metrics import auc as sklearn_auc',
        '',
        'fpr_g, tpr_g, _, auc_g = compute_roc(gmm, iq_0, iq_1)',
        'fpr_l, tpr_l, _, auc_l = compute_roc(lda, iq_0, iq_1)',
        '',
        'fig, ax = plt.subplots(figsize=(5.5, 5))',
        'ax.plot(fpr_g, tpr_g, color="#9C27B0", lw=2, label=f"GMM  AUC={auc_g:.4f}")',
        'ax.plot(fpr_l, tpr_l, color="#1565C0", lw=2, label=f"LDA  AUC={auc_l:.4f}")',
        'ax.plot([0,1],[0,1], "k--", lw=1, label="Random")',
        'ax.set_xlabel("False Positive Rate  P(1|0)")',
        'ax.set_ylabel("True Positive Rate   P(1|1)")',
        'ax.set_title("ROC Curve"); ax.legend(); ax.set_aspect("equal")',
        'plt.tight_layout(); plt.show()',
        'print(f"LDA AUC = {auc_l:.5f}")',
    ),

    _md('## 3.5 Fidelity vs integration time'),

    _code(
        'fracs, fidels = fidelity_vs_integration_time(',
        '    chain_0["after_dec"], chain_1["after_dec"], n_fractions=30)',
        '',
        'fig, ax = plt.subplots(figsize=(6, 4))',
        'ax.plot(fracs*100, fidels*100, "o-", color="#2E7D32", markersize=4)',
        'ax.set_xlabel("Integration window (% of readout pulse)")',
        'ax.set_ylabel("Readout fidelity (%)")',
        'ax.set_title("Fidelity vs Integration Time")',
        'ax.axhline(fidels.max()*100, color="gray", linestyle="--", linewidth=1,',
        '           label=f"Max: {fidels.max()*100:.2f}%")',
        'ax.legend()',
        'plt.tight_layout(); plt.show()',
        'print(f"Maximum fidelity: {fidels.max()*100:.3f}% at {fracs[np.argmax(fidels)]*100:.0f}% integration")',
    ),
])


# ============================================================================
# Notebook 04 — Full Pipeline (end-to-end)
# ============================================================================

nb04 = _nb([
    _md(
        '# Notebook 04 — Full Readout Pipeline (End-to-End)',
        '',
        '**Project 2 — Superconducting Qubit Readout Signal Processing Pipeline**',
        '',
        '---',
        '',
        '## Overview',
        '',
        'This notebook runs the complete readout pipeline in a single '
        'sequential flow — from qubit physics simulation through digital '
        'signal processing to state discrimination — and produces the '
        'publication-quality output figures saved to `outputs/`.',
        '',
        '```',
        'TransmonParams                        Physical parameters',
        '    │',
        '    ▼',
        'simulate_shots()          ──────────► shots_0, shots_1   (1000 × 512 complex)',
        '    │',
        '    ▼',
        'process_shots_batch()     ──────────► boxcar_iq, mf_iq, after_dec',
        '    │',
        '    ▼',
        'LDADiscriminator.fit()    ──────────► decision boundary',
        '    │',
        '    ▼',
        'readout_fidelity()        ──────────► F = 1 − (P(0|1)+P(1|0))/2',
        '```',
    ),

    _code(
        'import sys, os',
        'sys.path.insert(0, os.path.abspath(".."))',
        '',
        'import numpy as np',
        'import matplotlib',
        'matplotlib.use("Agg")',
        'import matplotlib.pyplot as plt',
        'import matplotlib.gridspec as gridspec',
        'from matplotlib.patches import Ellipse',
        'from pathlib import Path',
        '',
        'from src.transmon      import simulate_shots, snr_db, steady_state_analytical',
        'from src.readout_chain import process_shots_batch',
        'from src.discriminator import (',
        '    GMMDiscriminator, LDADiscriminator,',
        '    assignment_matrix, readout_fidelity,',
        '    compute_roc, fidelity_vs_integration_time, snr_vs_detuning)',
        '',
        'OUT = Path("../outputs")',
        'OUT.mkdir(exist_ok=True)',
    ),

    _md('## Step 1 — Qubit physics'),

    _code(
        'data     = simulate_shots(n_shots=1000, rng_seed=42)',
        't        = data["t"]',
        'alpha_0  = data["alpha_0"]',
        'alpha_1  = data["alpha_1"]',
        'shots_0  = data["shots_0"]',
        'shots_1  = data["shots_1"]',
        'p        = data["params"]',
        'template = alpha_0 - alpha_1',
        '',
        'ss0, ss1 = steady_state_analytical(p)',
        'snr      = snr_db(alpha_0, alpha_1, p.noise_sigma)',
        'print(f"χ/2π = {p.chi/(2*np.pi)/1e6:.1f} MHz  κ/2π = {p.kappa/(2*np.pi)/1e6:.1f} MHz")',
        'print(f"Single-shot SNR = {snr:.2f} dB")',
    ),

    _md('## Step 2 — DDC processing'),

    _code(
        'chain_0 = process_shots_batch(shots_0, t, template=template)',
        'chain_1 = process_shots_batch(shots_1, t, template=template)',
        'iq_0    = chain_0["boxcar_iq"]',
        'iq_1    = chain_1["boxcar_iq"]',
        'print(f"|0> IQ centroid: {iq_0.mean():.4f}")',
        'print(f"|1> IQ centroid: {iq_1.mean():.4f}")',
    ),

    _md('## Step 3 — State discrimination + metrics'),

    _code(
        'gmm = GMMDiscriminator().fit(iq_0, iq_1)',
        'lda = LDADiscriminator().fit(iq_0, iq_1)',
        '',
        'M_lda = assignment_matrix(lda, iq_0, iq_1)',
        'F_lda = readout_fidelity(M_lda)',
        'fpr, tpr, _, auc_score = compute_roc(lda, iq_0, iq_1)',
        'fracs, fidels = fidelity_vs_integration_time(',
        '    chain_0["after_dec"], chain_1["after_dec"])',
        '',
        'print(f"LDA readout fidelity : {F_lda*100:.3f}%")',
        'print(f"LDA ROC AUC          : {auc_score:.5f}")',
        'print(f"Max fidelity (sweep) : {fidels.max()*100:.3f}%")',
    ),

    _md('## Step 4 — Publication figure (2×3 summary)'),

    _code(
        'fig = plt.figure(figsize=(14, 9), constrained_layout=True)',
        'fig.suptitle(',
        '    "Superconducting Qubit Dispersive Readout — Full Pipeline",',
        '    fontsize=14, fontweight="bold")',
        '',
        'gs   = gridspec.GridSpec(2, 3, figure=fig)',
        'ax11 = fig.add_subplot(gs[0, 0])  # Cavity buildup',
        'ax12 = fig.add_subplot(gs[0, 1])  # IQ scatter',
        'ax13 = fig.add_subplot(gs[0, 2])  # Confusion matrix',
        'ax21 = fig.add_subplot(gs[1, 0])  # Fidelity vs time',
        'ax22 = fig.add_subplot(gs[1, 1])  # ROC',
        'ax23 = fig.add_subplot(gs[1, 2])  # SNR vs detuning',
        '',
        't_us = t * 1e6',
        '',
        '# (A) Cavity field buildup',
        'ax11.plot(t_us, alpha_0.real, "#2196F3", lw=2, label="|0⟩")',
        'ax11.plot(t_us, alpha_1.real, "#E91E63", lw=2, label="|1⟩")',
        'ax11.set_xlabel("Time (µs)"); ax11.set_ylabel("Re(α)")',
        'ax11.set_title("(A) Cavity Field Buildup"); ax11.legend()',
        '',
        '# (B) IQ scatter',
        'ax12.scatter(iq_0.real, iq_0.imag, s=3, alpha=0.3,',
        '             color="#2196F3", rasterized=True, label="|0⟩")',
        'ax12.scatter(iq_1.real, iq_1.imag, s=3, alpha=0.3,',
        '             color="#E91E63", rasterized=True, label="|1⟩")',
        'xlim = (min(iq_0.real.min(),iq_1.real.min())-0.05,',
        '        max(iq_0.real.max(),iq_1.real.max())+0.05)',
        'xs, ys = lda.decision_line_points(xlim)',
        'ax12.plot(xs, ys, "k--", lw=1.5)',
        'ax12.set_xlabel("I"); ax12.set_ylabel("Q")',
        'ax12.set_title(f"(B) IQ Scatter  F={F_lda*100:.1f}%"); ax12.legend(markerscale=3)',
        '',
        '# (C) Assignment matrix',
        'im = ax13.imshow(M_lda, cmap="Blues", vmin=0, vmax=1)',
        'ax13.set_xticks([0,1]); ax13.set_xticklabels(["|0⟩","|1⟩"])',
        'ax13.set_yticks([0,1]); ax13.set_yticklabels(["R=0","R=1"])',
        'ax13.set_title("(C) Assignment Matrix (LDA)")',
        'for i in range(2):',
        '    for j in range(2):',
        '        ax13.text(j, i, f"{M_lda[i,j]:.3f}", ha="center", va="center",',
        '                  fontsize=11, fontweight="bold",',
        '                  color="white" if M_lda[i,j]>0.5 else "black")',
        '',
        '# (D) Fidelity vs time',
        'ax21.plot(fracs*100, fidels*100, "o-", color="#2E7D32", ms=4)',
        'ax21.set_xlabel("Integration window (%)"); ax21.set_ylabel("Fidelity (%)")',
        'ax21.set_title("(D) Fidelity vs Integration Time")',
        '',
        '# (E) ROC curve',
        'ax22.plot(fpr, tpr, color="#1565C0", lw=2, label=f"AUC={auc_score:.4f}")',
        'ax22.plot([0,1],[0,1],"k--",lw=1)',
        'ax22.set_xlabel("FPR"); ax22.set_ylabel("TPR")',
        'ax22.set_title("(E) ROC Curve"); ax22.legend(); ax22.set_aspect("equal")',
        '',
        '# (F) SNR vs detuning',
        'det_MHz, snr_norm = snr_vs_detuning()',
        'ax23.plot(det_MHz, snr_norm, color="#6A1B9A", lw=2)',
        'ax23.set_xlabel("Drive detuning (MHz)"); ax23.set_ylabel("Norm. SNR")',
        'ax23.set_title("(F) Analytical SNR vs Detuning")',
        '',
        'fig.savefig(OUT / "full_pipeline_summary.png", bbox_inches="tight", dpi=150)',
        'print(f"Saved → {OUT / \'full_pipeline_summary.png\'}")',
        'plt.show()',
    ),

    _md('## Results Summary'),

    _code(
        'print("=" * 52)',
        'print("  RESULTS SUMMARY")',
        'print("=" * 52)',
        'rows = [',
        '    ("χ/2π",             f"{p.chi/(2*np.pi)/1e6:.1f}",    "MHz"),',
        '    ("κ/2π",             f"{p.kappa/(2*np.pi)/1e6:.1f}",  "MHz"),',
        '    ("ε/2π",             f"{p.epsilon/(2*np.pi)/1e6:.1f}","MHz"),',
        '    ("Single-shot SNR",  f"{snr:.2f}",                    "dB"),',
        '    ("Shots per state",  "1000",                          ""),',
        '    ("LDA fidelity",     f"{F_lda*100:.3f}",              "%"),',
        '    ("ROC AUC (LDA)",    f"{auc_score:.5f}",              ""),',
        '    ("Max fidelity",     f"{fidels.max()*100:.3f}",       "%"),',
        '    ("P(1|0)",           f"{M_lda[1,0]*100:.3f}",         "%"),',
        '    ("P(0|1)",           f"{M_lda[0,1]*100:.3f}",         "%"),',
        ']',
        'for name, val, unit in rows:',
        '    print(f"  {name:<22} {val:>8} {unit}")',
        'print("=" * 52)',
    ),
])


# ============================================================================
# Write notebooks to disk
# ============================================================================

notebooks = [
    ('01_qubit_physics.ipynb',        nb01),
    ('02_ddc_signal_processing.ipynb', nb02),
    ('03_state_discrimination.ipynb',  nb03),
    ('04_full_pipeline.ipynb',         nb04),
]

for fname, nb in notebooks:
    path = NB_DIR / fname
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created → notebooks/{fname}")

print(f"\nAll notebooks written to {NB_DIR}")
print("To execute them:")
print("  jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace")
