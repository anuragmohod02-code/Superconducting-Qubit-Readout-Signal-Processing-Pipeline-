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
│   ├── transmon.py           # Dispersive Hamiltonian + cavity ODE (scipy + Qiskit)
│   ├── readout_chain.py      # DDC + FIR LPF + decimation + integration
│   ├── discriminator.py      # GMM + LDA classifiers, fidelity, ROC
│   ├── error_mitigation.py   # Cal-matrix inversion + M3 mitigation ★
│   ├── wiener_filter.py      # Optimal Wiener filter (beats matched filter under coloured noise) ★
│   ├── crosstalk_readout.py  # 2-tone multiplexed readout, ZZ crosstalk, S21 spectrum ★
│   └── latency_model.py      # End-to-end decision latency model + T1 penalty ★
├── notebooks/
│   ├── 01_qubit_physics.ipynb
│   ├── 02_ddc_signal_processing.ipynb
│   ├── 03_state_discrimination.ipynb
│   └── 04_full_pipeline.ipynb
├── python/
│   ├── run_pipeline.py
│   └── create_notebooks.py
├── outputs/
│   ├── 01_cavity_field_buildup.png
│   ├── 02_ddc_pipeline.png
│   ├── 03_iq_scatter_gmm.png
│   ├── 04_roc_confusion.png
│   ├── 05_fidelity_vs_time.png
│   ├── full_pipeline_summary.png
│   ├── error_mitigation.png      # ★ Bell state mitigation: fidelity 0.93 → 1.00
│   ├── wiener_filter.png         # ★ d' comparison: Wiener 3.7× > matched filter
│   ├── crosstalk_s21.png         # ★ S21 spectrum for all 4 qubit states
│   ├── crosstalk_iq.png          # ★ IQ clouds — 2-tone multiplexed
│   ├── crosstalk_fidelity_vs_df.png  # ★ Fidelity vs resonator spacing
│   ├── latency_breakdown.png     # ★ Latency pie charts: 3 architectures
│   └── latency_fidelity.png      # ★ T1 error vs integration time
├── requirements.txt
└── README.md
```

★ New in v2

---

## Quick Start

```powershell
cd Project2_Qubit_Readout
pip install -r requirements.txt

# Generate all output plots (no Jupyter needed)
python python/run_pipeline.py

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

## Key Results

| Metric | Value |
|:---|---:|
| χ/2π | 1.0 MHz |
| κ/2π | 2.0 MHz |
| Single-shot SNR | ~14 dB |
| GMM readout fidelity | > 99.9% |
| LDA readout fidelity | > 99.9% |
| LDA ROC AUC | > 0.9999 |

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
