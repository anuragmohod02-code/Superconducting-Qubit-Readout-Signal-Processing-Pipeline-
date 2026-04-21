"""
readout_chain.py — Digital Readout Signal Processing Chain
==========================================================

Implements the full heterodyne readout signal processing pipeline:

  Stage 1 │ Upconvert   │ α(t) [complex BB] → real IF signal at f_IF
  Stage 2 │ DDC         │ mix with exp(−j·2π·f_IF·t) → complex BB
  Stage 3 │ FIR LPF     │ 63-tap windowed-sinc, cutoff 4 MHz (Hamming)
  Stage 4 │ Decimate    │ stride by factor 8 (after LPF avoids aliasing)
  Stage 5a│ Box-car     │ time-average over last `fraction` of trace
  Stage 5b│ Matched flt │ project onto (α0_mean − α1_mean)* template

All functions operate on batched (n_shots, n_time) arrays for efficiency.

References
----------
Gambetta et al., PRA 76, 012325 (2007)
Jeffrey et al., PRL 112, 190504 (2014)
Ryan et al., PRApplied 5, 014001 (2016)
"""

import numpy as np
from scipy.signal import firwin, lfilter
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Chain parameters
# ---------------------------------------------------------------------------

@dataclass
class ChainParams:
    """Heterodyne readout chain configuration."""
    f_if:         float = 10e6    # Intermediate frequency (Hz)
    fs:           float = 512e6   # ADC sample rate — must match n_time / t_end
    fir_taps:     int   = 63      # FIR filter length (odd for linear phase)
    fir_cutoff:   float = 4e6     # LPF cutoff frequency (Hz)
    decimate_by:  int   = 8       # Integer decimation factor
    int_fraction: float = 0.5     # Box-car window: last fraction of trace


DEFAULT_CHAIN = ChainParams()


# ---------------------------------------------------------------------------
# FIR low-pass filter design
# ---------------------------------------------------------------------------

def build_lpf(params: ChainParams = DEFAULT_CHAIN) -> np.ndarray:
    """
    Design a linear-phase FIR LPF using the windowed-sinc method (Hamming window).

    Cutoff is normalised to Nyquist: f_c / (f_s / 2).
    """
    cutoff_norm = params.fir_cutoff / (params.fs / 2.0)
    cutoff_norm = float(np.clip(cutoff_norm, 1e-4, 0.999))
    h = firwin(params.fir_taps, cutoff_norm, window='hamming')
    return h


def _lpf_complex(sig: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply real FIR filter independently to I and Q channels."""
    return lfilter(h, 1.0, sig.real, axis=-1) + 1j * lfilter(h, 1.0, sig.imag, axis=-1)


# ---------------------------------------------------------------------------
# Batch pipeline (vectorised over shots)
# ---------------------------------------------------------------------------

def process_shots_batch(
    shots:    np.ndarray,
    t:        np.ndarray,
    template: np.ndarray = None,
    params:   ChainParams = DEFAULT_CHAIN,
) -> dict:
    """
    Run the full heterodyne processing chain on a batch of shots.

    Parameters
    ----------
    shots    : (n_shots, n_time) complex — cavity field (rotating frame)
    t        : (n_time,)         float  — time axis in seconds
    template : (n_time,)         complex — difference template for matched filter;
                                  typically alpha_0 − alpha_1 (noise-free)
    params   : ChainParams

    Returns
    -------
    dict with snapshots at each pipeline stage:
        'raw_rf'    : (n_shots, n_time) float  — after IF upconversion
        'after_ddc' : (n_shots, n_time) complex — after DDC mixing
        'after_lpf' : (n_shots, n_time) complex — after LPF
        'after_dec' : (n_shots, n_dec)  complex — after decimation
        'boxcar_iq' : (n_shots,)        complex — box-car integrated IQ
        'mf_iq'     : (n_shots,)        float   — matched-filter output (if template given)
    """
    h = build_lpf(params)

    # ── Stage 1: Upconvert to IF ─────────────────────────────────────────────
    # raw_rf(t) = Re[α(t) · exp(j·2π·f_IF·t)]
    carrier = np.exp(1j * 2.0 * np.pi * params.f_if * t)        # (n_time,)
    raw_rf  = np.real(shots * carrier[np.newaxis, :])             # (n_shots, n_time)

    # ── Stage 2: DDC ─────────────────────────────────────────────────────────
    # Multiply by complex LO: exp(−j·2π·f_IF·t)
    # → recovers complex baseband + image at −2f_IF (removed by LPF)
    lo        = np.exp(-1j * 2.0 * np.pi * params.f_if * t)      # (n_time,)
    after_ddc = raw_rf * lo[np.newaxis, :]                        # (n_shots, n_time) complex

    # ── Stage 3: FIR LPF ─────────────────────────────────────────────────────
    after_lpf = _lpf_complex(after_ddc, h)                        # (n_shots, n_time)

    # ── Stage 4: Decimate ────────────────────────────────────────────────────
    after_dec = after_lpf[:, ::params.decimate_by]                # (n_shots, n_dec)

    # ── Stage 5a: Box-car integration ────────────────────────────────────────
    boxcar_iq = _boxcar(after_dec, params.int_fraction)           # (n_shots,) complex

    result = {
        'raw_rf':    raw_rf,
        'after_ddc': after_ddc,
        'after_lpf': after_lpf,
        'after_dec': after_dec,
        'boxcar_iq': boxcar_iq,
    }

    # ── Stage 5b: Matched filter ──────────────────────────────────────────────
    if template is not None:
        # Process the template through the same chain
        tmpl_rf  = np.real(template * carrier)
        tmpl_ddc = tmpl_rf * lo
        tmpl_lpf = _lpf_complex(tmpl_ddc, h)
        tmpl_dec = tmpl_lpf[::params.decimate_by]                 # (n_dec,)

        result['mf_iq']       = _matched_filter(after_dec, tmpl_dec)
        result['template_dec'] = tmpl_dec

    return result


# ---------------------------------------------------------------------------
# Single-shot pipeline (for visualisation of individual stages)
# ---------------------------------------------------------------------------

def process_single_shot(
    shot:   np.ndarray,
    t:      np.ndarray,
    params: ChainParams = DEFAULT_CHAIN,
) -> dict:
    """
    Same as process_shots_batch but for a single 1-D shot.
    Returns same keys; arrays have shape (n_time,) or (n_dec,).
    """
    result = process_shots_batch(shot[np.newaxis, :], t, params=params)
    return {k: v[0] if v.ndim == 2 else v for k, v in result.items()}


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def _boxcar(sig: np.ndarray, fraction: float) -> np.ndarray:
    """
    Box-car: average over the last `fraction` of the time axis.
    sig : (..., n_time) complex
    Returns: (...,) complex
    """
    n     = sig.shape[-1]
    start = int((1.0 - fraction) * n)
    return np.mean(sig[..., start:], axis=-1)


def _matched_filter(sig: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Matched filter: project each shot onto the (normalised) template.
        w      = template* / ||template||²
        output = Re(sig @ w)   →  scalar per shot

    sig      : (n_shots, n_dec) complex
    template : (n_dec,)         complex — typically (α0 − α1) after decimation
    Returns  : (n_shots,)       float
    """
    norm_sq = np.dot(template.conj(), template).real
    if norm_sq < 1e-30:
        raise ValueError("Matched-filter template has near-zero norm.")
    weights = template.conj() / norm_sq
    return np.real(sig @ weights)


# Public aliases used by discriminator / notebooks
boxcar_integrate = _boxcar
matched_filter_integrate = _matched_filter
