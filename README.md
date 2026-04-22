# Superconducting Qubit Readout Signal Processing Pipeline

A full dispersive readout simulation and signal processing chain for superconducting
qubits — from quantum physics to state discrimination — implemented as three Python
source modules and four Jupyter notebooks.

## What This Demonstrates

| Skill area | Evidence |
|---|---|
| Quantum hardware knowledge | Jaynes-Cummings dispersive limit, dressed cavity frequencies, AWGN shot noise model |
| Digital signal processing | Heterodyne DDC, FIR LPF design, decimation, matched filtering, Wiener filter |
| Machine learning | GMM unsupervised clustering, LDA optimal linear classifier, ROC / AUC |
| Readout error mitigation | Calibration-matrix inversion, M3 sparse approximation, fidelity improvement |
| Multi-qubit systems | Frequency-multiplexed 2-tone readout, crosstalk analysis, latency modelling |
| Python / scientific stack | scipy ODE solver, scipy.signal, sklearn, matplotlib, Qiskit quantum_info |

---

## Hardware Context

In a real superconducting qubit experiment:

1. A microwave tone drives the readout cavity at ~6.5 GHz
2. The transmitted/reflected signal carries a qubit-state-dependent phase shift
3. A heterodyne receiver down-converts to ~10 MHz IF, then a DSP chain extracts the I/Q point
4. A classifier assigns the I/Q point to |0⟩ or |1⟩ — this is **dispersive readout**

This project simulates every step of that chain.

---

## Physics

In the dispersive limit ($g \ll |\omega_q - \omega_r|$), the Jaynes-Cummings Hamiltonian reduces to:

$$H_\mathrm{eff}/\hbar = (\omega_r + \chi\,\sigma_z)\,a^\dagger a \;+\; \frac{\omega_q}{2}\sigma_z$$

The cavity resonance shifts by $\pm\chi$ depending on qubit state.  Driving at $\omega_d = \omega_r$, the cavity field in rotating frame obeys:

$$\frac{d\alpha}{dt} = -\left(\frac{\kappa}{2} + i\delta_s\right)\alpha + \varepsilon$$

with steady state $\alpha_\mathrm{ss} = \varepsilon\,/\,(\kappa/2 + i\delta_s)$.

### Parameters

| Symbol | Value | Description |
|:---:|:---:|:---|
| $\omega_r/2\pi$ | 6.5 GHz | Bare cavity frequency |
| $\omega_q/2\pi$ | 5.0 GHz | Qubit frequency |
| $\chi/2\pi$ | 1.0 MHz | Dispersive shift |
| $\kappa/2\pi$ | 2.0 MHz | Cavity linewidth |
| $\varepsilon/2\pi$ | 1.0 MHz | Drive amplitude |
| $f_\mathrm{IF}$ | 10 MHz | Heterodyne IF frequency |

---

## Repository Structure

```
Project2_Qubit_Readout/
├── src/
│   ├── transmon.py              # Dispersive Hamiltonian + cavity ODE (scipy + Qiskit)
│   ├── readout_chain.py         # DDC + FIR LPF + decimation + integration
│   ├── discriminator.py         # GMM + LDA classifiers, fidelity, ROC
│   ├── error_mitigation.py      # Cal-matrix inversion + M3 mitigation ★
│   ├── wiener_filter.py         # Optimal Wiener filter (beats matched filter under coloured noise) ★
│   ├── crosstalk_readout.py     # 2-tone multiplexed readout, ZZ crosstalk, S21 spectrum ★
│   ├── latency_model.py         # End-to-end decision latency model + T1 penalty ★
│   └── readout_optimization.py  # ROC threshold / window 2D / QE sweep / active reset ★★
├── notebooks/
│   ├── 01_qubit_physics.ipynb
│   ├── 02_ddc_signal_processing.ipynb
│   ├── 03_state_discrimination.ipynb
│   └── 04_full_pipeline.ipynb
├── python/
│   ├── run_pipeline.py
│   ├── run_optimization.py      # ★★ Advanced optimisation demo (4 figures)
│   └── create_notebooks.py
├── outputs/
│   ├── 01_cavity_field_buildup.png
│   ├── 02_ddc_pipeline.png
│   ├── 03_iq_scatter_gmm.png
│   ├── 04_roc_confusion.png
│   ├── 05_fidelity_vs_time.png
│   ├── full_pipeline_summary.png
│   ├── error_mitigation.png           # ★  Bell state mitigation: fidelity 0.93 → 1.00
│   ├── wiener_filter.png              # ★  d' comparison: Wiener 3.7× > matched filter
│   ├── crosstalk_s21.png              # ★  S21 spectrum for all 4 qubit states
│   ├── crosstalk_iq.png               # ★  IQ clouds — 2-tone multiplexed
│   ├── crosstalk_fidelity_vs_df.png   # ★  Fidelity vs resonator spacing
│   ├── latency_breakdown.png          # ★  Latency pie charts: 3 architectures
│   ├── latency_fidelity.png           # ★  T1 error vs integration time
│   ├── 07_roc_threshold.png           # ★★ ROC + Youden threshold + confusion matrices
│   ├── 08_integration_window.png      # ★★ 2-D fidelity heatmap + ring-up overlay
│   ├── 09_quantum_efficiency.png      # ★★ Fidelity & SNR vs quantum efficiency η
│   └── 10_active_reset.png            # ★★ Residual |1⟩ vs reset rounds & latency
├── requirements.txt
└── README.md
```

★ New in v2  ★★ New in v3

---

## Quick Start

```powershell
cd Project2_Qubit_Readout
pip install -r requirements.txt

# Generate all output plots (no Jupyter needed)
python python/run_pipeline.py

# v3: Advanced readout optimisation (generates figures 07–10)
python python/run_optimization.py

# Create and open notebooks
python python/create_notebooks.py
jupyter notebook notebooks/
```

---

## Simulation Results

### Cavity Field Buildup

![Cavity field buildup](outputs/01_cavity_field_buildup.png)

*Noise-free cavity field evolution for |0⟩ (blue) and |1⟩ (red). The fields build up
to different steady-state amplitudes in the IQ plane due to the dispersive frequency shift ±χ.
Individual noisy shots overlaid (translucent).*

### Heterodyne Processing Chain

![DDC pipeline](outputs/02_ddc_pipeline.png)

*Stage-by-stage transformation of the raw IF signal: upconversion → DDC mixing →
FIR LPF (63-tap, cutoff 4 MHz) → decimation (÷8) → box-car integration window.*

### IQ Scatter + State Discrimination

![IQ scatter](outputs/03_iq_scatter_gmm.png)

*Left: IQ clouds with 2σ GMM ellipses. Right: LDA linear decision boundary.
The |0⟩ and |1⟩ states separate clearly in the IQ plane — the quadrature
component carries the qubit-state information.*

### ROC Curve + Assignment Matrix

![ROC and confusion](outputs/04_roc_confusion.png)

*Left: ROC curve (LDA AUC > 0.999). Right: assignment matrix — diagonal elements
close to 1.0 indicate high-fidelity readout.*

### Fidelity vs Integration Time

![Fidelity vs time](outputs/05_fidelity_vs_time.png)

*Left: readout fidelity improves as the integration window grows — converges once
the cavity reaches steady state. Right: analytical SNR vs drive frequency detuning,
showing the optimal operating point at $\Delta = 0$.*

---

## v3 Readout Optimisation

Demo scenario: χ/2π=1 MHz, κ/2π=2 MHz, **T1=3 µs**, σ=0.8 (HEMT-limited, η≈3.5%),
n̄_th=5%.  The 1 µs readout window causes 28% of |1⟩ shots to decay mid-window,
creating an asymmetric error matrix that all four techniques exploit.

### ROC-Based Threshold Optimisation (Youden's J)

![ROC threshold](outputs/07_roc_threshold.png)

Standard LDA uses P(|1⟩)=0.5 as its decision threshold.  With T1 relaxation
creating bimodal |1⟩ shot statistics, maximising Youden's J = TPR − FPR shifts
the optimal threshold to **0.047**, converting partially-decayed shots that scored
below the default 0.5 boundary into correct |1⟩ assignments.

| Metric | Default (0.50) | Youden (0.047) |
|:---|---:|---:|
| F0 (P correct \| |0⟩) | 94.7% | 94.4% |
| F1 (P correct \| |1⟩) | 82.6% | 91.4% |
| **Assignment fidelity** | **88.67%** | **92.85%** |
| **Gain** | — | **+4.17 pp** |

*Key insight*: Youden's J is optimal for balanced assignment fidelity; for active reset
the default threshold is better because it minimises the false-alarm rate ε₀₁.

### 2-D Integration Window Optimisation

![Integration window](outputs/08_integration_window.png)

Independently sweeping start (t_on) and end (t_off) of the box-car integration
reveals a trade-off surface: the ring-up transient (t < 2/κ ≈ 160 ns) carries
sub-optimal signal, while late-window samples accumulate T1-decayed |1⟩ shots.

| Window | Fidelity |
|:---|---:|
| Full [0, 1.0] µs | 88.67% |
| Optimal [0.04, 0.25] µs | **93.60%** |
| **Gain** | **+4.92 pp** |

The 2-D heatmap makes the ring-up and T1 constraints visible simultaneously.
The contour at F=88.67% marks the region where truncating the window first helps.

### Quantum Efficiency Sweep

![Quantum efficiency](outputs/09_quantum_efficiency.png)

The effective noise scales as σ_eff = σ_ql / √η.  Modelling a single-mode
heterodyne snapshot at steady state (bandwidth ≈ κ/2π), the fidelity sweeps across
the full dynamic range as η varies from 1% to 100%.

| η | Hardware | SNR | F_LDA |
|:---|:---|---:|---:|
| 1.00 | Quantum limit | 10.4 dB | 99.2% |
| 0.40 | JPA + HEMT | 6.4 dB | 97.3% |
| 0.10 | Good TWPA | 0.5 dB | 83.2% |
| 0.05 | HEMT only | −2.6 dB | 77.2% |
| 0.01 | Room-temp amp | −9.6 dB | 62.9% |

SNR follows SNR ∝ η (10 dB per decade) as expected for added-noise amplification.
GMM slightly outperforms LDA at low η due to the non-Gaussian T1 tail.

### Active Reset Convergence

![Active reset](outputs/10_active_reset.png)

Using clean-state F0/F1 values (n̄_th=0 training, intrinsic AWGN+T1 noise only):

| Threshold | Round 1 | Round 3 | Suppression |
|:---|---:|---:|---:|
| Default (0.50)  | 0.77% | **0.018%** | **278×** |
| Youden (0.006)  | 1.02% | 0.749% | 7× |

**Default threshold dominates active reset** because ε₀₁=0.0% (zero false alarms
with well-separated Gaussians), so the iterative map p₁ᵢ₊₁ = p₁ᵢ·ε₁₀ converges
purely on the T1-limited miss rate ε₁₀=15%.  The Youden threshold's ε₀₁=0.7%
creates a non-zero steady-state floor at p₁_ss = ε₀₁/(ε₀₁+F1) ≈ 0.75%.

This is the key design trade-off: optimise the threshold for **assignment fidelity**
(Youden) *or* for **active reset depth** (default/conservative), not both simultaneously.

---



| Metric | Value |
|:---|---:|
| χ/2π | 1.0 MHz |
| κ/2π | 2.0 MHz |
| Single-shot SNR (default params) | ~14 dB |
| GMM readout fidelity (ideal) | > 99.9% |
| LDA readout fidelity (ideal) | > 99.9% |
| LDA ROC AUC (ideal) | > 0.9999 |
| Youden threshold gain (T1=3 µs, σ=0.8) | **+4.17 pp** |
| Integration window optimisation gain | **+4.92 pp** |
| QE sweep: F at η=1 vs η=0.05 | 99.2% vs 77.2% |
| Active reset suppression (3 rounds) | **278×** (5.0% → 0.018%) |

---

## v2 Improvements

### Readout Error Mitigation

![Error mitigation](outputs/error_mitigation.png)

*Bell state |Φ+⟩ probability distribution before and after readout error mitigation.
With ε₀=3%, ε₁=4% assignment errors: raw fidelity 0.932 → mitigated 1.000 (exact)
/ 1.000 (M3). Mitigation overhead: 1.16× statistical noise inflation.*

### Optimal Wiener Filter

![Wiener filter](outputs/wiener_filter.png)

*Under coloured (1/f + white) noise, the Wiener filter outperforms the matched filter
by 3.7× in d'-metric (d'=0.097 vs 0.026). +39.9 dB SNR gain is achieved by
down-weighting frequency bins where the noise PSD dominates.*

### Multi-Qubit Frequency-Multiplexed Readout

![S21 spectrum](outputs/crosstalk_s21.png)

*Transmission S21(f) for all four two-qubit states. Two Lorentzian features at f_r0=5.00 GHz
and f_r1=5.15 GHz (Δf=150 MHz) disperse depending on qubit state. ZZ coupling (50 kHz)
causes a frequency shift visible as a splitting of the f_r1 feature.*

![IQ clouds multiplexed](outputs/crosstalk_iq.png)

*IQ clouds for each resonator after digital down-conversion to the respective tone.
Q0→Q1 crosstalk phase: 1.92° (ZZ-dominated). Q1→Q0 crosstalk: 0.10% (spectral leakage).*

### Real-Time Decision Latency Model

![Latency breakdown](outputs/latency_breakdown.png)

| Architecture | Total latency | Latency/T₁ | T₁ error penalty |
|---|---|---|---|
| State-of-art 2023 (JPA+fast FPGA) | 619 ns | 0.62% | 0.617% |
| Typical 2023 (HEMT+FPGA) | 1272 ns | 1.27% | 1.264% |
| Early 2018 (slow FIR) | 3593 ns | 3.59% | 3.529% |

*Latency budget for a typical superconducting qubit experiment (T₁=100 μs).
The dominant contributions are integration window and classical communication.
Surface code distance-3 tolerates ≈1% per-cycle error — current hardware is marginal.*

---

## References

1. Blais A. et al., *Circuit quantum electrodynamics*, Rev. Mod. Phys. **93**, 025005 (2021)
2. Wallraff A. et al., *Strong coupling of a single photon to a superconducting qubit*, Nature **431**, 162 (2004)
3. Krantz P. et al., *A quantum engineer's guide to superconducting qubits*, Appl. Phys. Rev. **6**, 021318 (2019)
4. Gambetta J. et al., *Qubit-photon interactions in a cavity*, PRA **74**, 042318 (2006)
5. Jeffrey E. et al., *Fast accurate state measurement with superconducting qubits*, PRL **112**, 190504 (2014)
6. Heinsoo J. et al., *Rapid high-fidelity multiplexed readout*, PRApplied **10**, 034040 (2018)
7. Reed M. et al., *High-fidelity readout in circuit quantum electrodynamics*, PRL **105**, 173601 (2010)
8. Johnson J. et al., *Heralded state preparation in a superconducting qubit*, PRL **109**, 050506 (2012)
