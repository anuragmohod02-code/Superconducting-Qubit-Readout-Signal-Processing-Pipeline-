"""
wiener_filter.py — Optimal Wiener Filter for Qubit Readout
===========================================================

Replaces the simple matched filter with a Wiener filter that is optimal
under coloured (non-white) noise — as is typically the case in superconducting
qubit readout chains where:
  - System noise (HEMT amplifier): adds ~20 K noise temperature
  - Quantum noise from the Josephson parametric amplifier (JPA): near-SQL noise
  - 1/f-type flux noise
  - Amplifier saturation: Johnson noise floor from 50 Ω load

Filter design (frequency domain):
    W(f) = H*(f) / (|H(f)|² + S_n(f) / S_s(f))

where:
    H(f)   = signal template spectrum (difference of |0⟩ and |1⟩ mean traces)
    S_n(f) = noise power spectral density (estimated from |0⟩ shots)
    S_s(f) = signal power spectral density (from template)

For white noise (S_n = const): reduces to standard matched filter.
For coloured noise: Wiener filter up-weights frequency bins where SNR is high.

The output is a single real number per shot (the filter output at t=T).

References
----------
Bultink C. et al., PRApplied 6, 034008 (2016)  — optimal Wiener filter for readout
Ryan C. et al., PRApplied 5, 014001 (2016)      — digital signal processing chain
Gambetta J. et al., PRA 76, 012325 (2007)       — quantum signal theory
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Wiener filter design ─────────────────────────────────────────────────────────

@dataclass
class WienerFilter:
    """
    Optimal Wiener filter for single-shot IQ readout discrimination.

    Fit on calibration traces, apply to single shots.
    Reduces to matched filter when noise is white (flat S_n).
    """
    n_taps:    int   = 0              # set after fit()
    fs:        float = 512e6          # sample rate (Hz)
    fir_coeff: np.ndarray = field(default_factory=lambda: np.array([]))
    snr_gain_db: float = 0.0          # integration gain vs simple box-car

    def fit(
        self,
        cal_0:      np.ndarray,   # (N_shots, N_time) complex baseband traces, state |0⟩
        cal_1:      np.ndarray,   # (N_shots, N_time) complex baseband traces, state |1⟩
        noise_from: str = "state0",
        regularise: float = 1e-4,  # Tikhonov regularisation: prevents div-by-zero
    ) -> 'WienerFilter':
        """
        Estimate Wiener filter from calibration data.

        Parameters
        ----------
        cal_0, cal_1 : calibration trace arrays (complex baseband)
        noise_from   : 'state0' | 'state1' — which state to estimate noise from
        regularise   : relative regularisation level (adds reg×max(S_n) to denominator)
        """
        n_time = cal_0.shape[1]
        self.n_taps = n_time

        # Signal template: difference of mean traces
        mean_0 = cal_0.mean(axis=0)   # (n_time,)
        mean_1 = cal_1.mean(axis=0)
        template = mean_1 - mean_0    # optimal direction in IQ space

        # Noise estimation: PSD from residuals of selected state
        noise_cal = cal_0 if noise_from == "state0" else cal_1
        noise_mean = noise_cal.mean(axis=0)
        residuals = noise_cal - noise_mean[np.newaxis, :]  # shape (N, n_time)

        # Average PSD (one-sided, real part)
        noise_psd = np.mean(np.abs(np.fft.rfft(residuals, axis=1))**2, axis=0)

        # Signal PSD
        template_fft = np.fft.rfft(template)
        signal_psd   = np.abs(template_fft)**2

        # Wiener filter in frequency domain (complex)
        reg_floor = regularise * noise_psd.max()
        W_f = template_fft.conj() / (signal_psd + noise_psd + reg_floor)

        # Normalise filter to unit energy (so output scale is comparable to MF)
        w_norm = np.sqrt(np.sum(np.abs(W_f)**2) + 1e-30)
        W_f_normed = W_f / w_norm

        # Store complex filter in time domain
        self._h_complex = np.fft.irfft(W_f_normed, n=n_time)
        # Convert back to time domain (real-valued FIR coefficients)
        w_t = self._h_complex.real
        self.fir_coeff = w_t

        # Estimate SNR gain vs box-car integration
        snr_wiener = float(np.sum(signal_psd / (noise_psd + reg_floor)))
        snr_boxcar = float(
            np.abs(np.sum(template))**2 / (np.mean(noise_psd) * n_time + 1e-15)
        )
        if snr_boxcar > 0:
            self.snr_gain_db = 10 * np.log10(max(snr_wiener / snr_boxcar, 1e-10))

        return self

    def apply(self, shots: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filter to shot traces using frequency-domain inner product.

        Parameters
        ----------
        shots : (N_shots, N_time) real or complex traces

        Returns
        -------
        y : (N_shots,) real filter output
        """
        if self.fir_coeff.size == 0:
            raise RuntimeError("Call fit() before apply().")
        single = shots.ndim == 1
        if single:
            shots = shots[np.newaxis, :]
        n = shots.shape[1]
        # Frequency-domain inner product: ∑_f W*(f) · X(f)  (Parseval-style correlation)
        X = np.fft.rfft(shots, axis=1)
        W_f = np.fft.rfft(self.fir_coeff, n=n)
        y = (W_f.conj()[np.newaxis, :] * X).sum(axis=1).real / n
        return float(y[0]) if single else y

    def separation(self, cal_0: np.ndarray, cal_1: np.ndarray) -> float:
        """
        Compute the normalised separation between |0⟩ and |1⟩ filtered outputs.

        Returns d' (d-prime), the signal-to-noise of the filter output distribution.
        """
        y0 = self.apply(cal_0)
        y1 = self.apply(cal_1)
        mu_diff  = abs(y1.mean() - y0.mean())
        sigma_avg = 0.5 * (y0.std() + y1.std()) + 1e-15
        return float(mu_diff / sigma_avg)


# ── Comparison: matched filter ───────────────────────────────────────────────────

def matched_filter(
    shots:  np.ndarray,
    cal_0:  np.ndarray,
    cal_1:  np.ndarray,
) -> np.ndarray:
    """
    Standard matched filter: project onto (mean_1 - mean_0)* template.

    Returns real-valued projection for each shot.
    """
    mean_0 = cal_0.mean(axis=0)
    mean_1 = cal_1.mean(axis=0)
    template = (mean_1 - mean_0).conj()   # matched filter kernel
    # Dot product (sum over time)
    if shots.ndim == 1:
        return float(np.sum(shots * template).real)
    return np.sum(shots * template[np.newaxis, :], axis=1).real


# ── Demo / CLI ────────────────────────────────────────────────────────────────────

def demo_wiener():
    """
    Demonstrate Wiener filter vs matched filter on simulated readout traces.

    Simulates a coloured noise environment (1/f + white) and shows the SNR
    improvement from using the Wiener filter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    print("=" * 60)
    print("  Optimal Wiener Filter vs Matched Filter — Readout Demo")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ── Simulation parameters
    N_SHOTS = 2000
    N_TIME  = 512     # time samples
    FS      = 512e6   # Hz
    T       = N_TIME / FS

    # Transient signal: exponential rise on qubit state (real-valued I channel)
    t    = np.linspace(0, T, N_TIME)
    t_us = t * 1e6

    # |1⟩ has larger IQ amplitude due to photon population
    env_0 = 0.10 * (1 - np.exp(-t / (0.05 * T)))   # |0⟩ mean (real)
    env_1 = 0.50 * (1 - np.exp(-t / (0.05 * T)))   # |1⟩ mean (real)

    # Coloured noise: white + 1/f (real-valued)
    def _coloured_noise(n_shots, n_time, sigma_white, sigma_1f):
        freqs = np.fft.rfftfreq(n_time, d=1.0 / FS) + 1.0
        psd_1f  = sigma_1f**2 / freqs
        psd_wht = (sigma_white**2) * np.ones_like(freqs)
        psd     = psd_1f + psd_wht
        amp     = np.sqrt(psd * FS / 2)
        phi = rng.uniform(0, 2 * np.pi, (n_shots, len(freqs)))
        spec = amp[np.newaxis, :] * np.exp(1j * phi)
        return np.fft.irfft(spec, n=n_time, axis=1).real   # real-valued noise

    sigma_w, sigma_1f = 0.08, 0.06
    noise_0 = _coloured_noise(N_SHOTS, N_TIME, sigma_w, sigma_1f)
    noise_1 = _coloured_noise(N_SHOTS, N_TIME, sigma_w, sigma_1f)

    # Real-valued traces (I channel only; standard for post-DDC data)
    traces_0 = env_0[np.newaxis, :] + noise_0   # (N_SHOTS, N_TIME)
    traces_1 = env_1[np.newaxis, :] + noise_1

    # ── Matched filter (using held-out calibration, same split as Wiener)
    half = N_SHOTS // 2
    y_mf_0 = matched_filter(traces_0[half:], traces_0[:half], traces_1[:half])
    y_mf_1 = matched_filter(traces_1[half:], traces_0[:half], traces_1[:half])
    sep_mf = abs(y_mf_1.mean() - y_mf_0.mean()) / (0.5*(y_mf_0.std() + y_mf_1.std()))

    # ── Wiener filter
    wf = WienerFilter(fs=FS)
    # Use first half as calibration, second half as test
    wf.fit(traces_0[:half], traces_1[:half])
    y_wf_0 = wf.apply(traces_0[half:])
    y_wf_1 = wf.apply(traces_1[half:])
    sep_wf = wf.separation(traces_0[half:], traces_1[half:])

    print(f"\nMatched filter  d' = {sep_mf:.3f}")
    print(f"Wiener filter   d' = {sep_wf:.3f}  (+{wf.snr_gain_db:.1f} dB)")

    # ── Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Optimal Wiener Filter vs Matched Filter — Qubit Readout (Coloured Noise)",
                 fontsize=12, fontweight="bold")

    # Panel 1: mean traces + noise PSD
    ax = axes[0]
    ax.plot(t_us[:64], env_0[:64], "b-",  linewidth=2, label="|0⟩ mean")
    ax.plot(t_us[:64], env_1[:64], "r-",  linewidth=2, label="|1⟩ mean")
    ax.plot(t_us[:64], traces_0[0, :64], "b--", alpha=0.4, linewidth=0.8, label="|0⟩ shot")
    ax.plot(t_us[:64], traces_1[0, :64], "r--", alpha=0.4, linewidth=0.8, label="|1⟩ shot")
    ax.set_xlabel("Time (μs)"); ax.set_ylabel("Amplitude (arb.)")
    ax.set_title("Readout Traces (first 64 samples)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Matched filter histograms
    ax = axes[1]
    lim = max(abs(y_mf_0).max(), abs(y_mf_1).max()) * 1.1
    bins = np.linspace(-lim, lim, 60)
    ax.hist(y_mf_0, bins=bins, color="#3498DB", alpha=0.65, label="|0⟩")
    ax.hist(y_mf_1, bins=bins, color="#E74C3C", alpha=0.65, label="|1⟩")
    ax.set_xlabel("Filter output"); ax.set_ylabel("Counts")
    ax.set_title(f"Matched Filter  d'={sep_mf:.2f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: Wiener filter histograms
    ax = axes[2]
    lim = max(abs(y_wf_0).max(), abs(y_wf_1).max()) * 1.1
    bins = np.linspace(-lim, lim, 60)
    ax.hist(y_wf_0, bins=bins, color="#3498DB", alpha=0.65, label="|0⟩")
    ax.hist(y_wf_1, bins=bins, color="#E74C3C", alpha=0.65, label="|1⟩")
    ax.set_xlabel("Filter output"); ax.set_ylabel("Counts")
    ax.set_title(f"Wiener Filter  d'={sep_wf:.2f}  (+{wf.snr_gain_db:.1f} dB)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wiener_filter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    demo_wiener()
