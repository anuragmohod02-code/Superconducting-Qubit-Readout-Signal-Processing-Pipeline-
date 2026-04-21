"""
transmon.py — Dispersive Readout IQ Simulation
===============================================

Models a transmon qubit coupled to a readout cavity in the dispersive limit.

Physics
-------
In the dispersive regime (g << |ωq − ωr|) the Jaynes-Cummings Hamiltonian
reduces to:

    H_eff/ℏ = (ωr + χ·σz)·a†a  +  (ωq/2)·σz

The cavity resonance shifts by ±χ depending on qubit state:

    |0⟩  →  cavity at  ωr + χ
    |1⟩  →  cavity at  ωr − χ

In the rotating frame at drive frequency ωd, the cavity field obeys:

    dα/dt = −(κ/2 + i·δs)·α + ε

where:
    δ0 = ωd − (ωr + χ)   (drive detuning when qubit in |0⟩)
    δ1 = ωd − (ωr − χ)   (drive detuning when qubit in |1⟩)
    κ   = cavity linewidth (energy decay rate)
    ε   = drive amplitude

Steady-state (t → ∞):  α_ss = ε / (κ/2 + i·δs)

We use Qiskit SparsePauliOp to display the Hamiltonian symbolically,
and scipy.integrate.solve_ivp for time evolution.

References
----------
Blais et al., Rev. Mod. Phys. 93, 025005 (2021)
Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)
Wallraff et al., Nature 431, 162 (2004)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field

# Optional: use Qiskit to display the Hamiltonian in Pauli form
try:
    from qiskit.quantum_info import SparsePauliOp
    _QISKIT_OK = True
except ImportError:
    _QISKIT_OK = False


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

@dataclass
class TransmonParams:
    """
    Dispersive readout parameters.
    All angular frequencies in rad/s; all times in seconds.
    """
    # ── Qubit ────────────────────────────────────────────────────────────────
    omega_q:      float = 2 * np.pi * 5.0e9     # Qubit frequency
    alpha_q:      float = 2 * np.pi * (-200e6)  # Anharmonicity  (α ≈ −200 MHz)
    # ── Readout cavity ───────────────────────────────────────────────────────
    omega_r:      float = 2 * np.pi * 6.5e9     # Bare cavity frequency
    kappa:        float = 2 * np.pi * 2.0e6     # Cavity linewidth (κ/2π = 2 MHz)
    # ── Qubit-cavity coupling (dispersive limit: χ ≈ g²/|ωq−ωr|) ───────────
    chi:          float = 2 * np.pi * 1.0e6     # Dispersive shift (χ/2π = 1 MHz)
    # ── Drive ────────────────────────────────────────────────────────────────
    omega_d:      float = 2 * np.pi * 6.5e9     # Drive frequency (at bare cavity)
    epsilon:      float = 2 * np.pi * 1.0e6     # Drive amplitude (rad/s)
    # ── Noise ────────────────────────────────────────────────────────────────
    noise_sigma:  float = 0.15                   # AWGN σ on each I/Q sample
    # ── Simulation ───────────────────────────────────────────────────────────
    t_end:        float = 1.0e-6                 # Readout window (1 µs)
    n_time:       int   = 512                    # Time samples per trace


DEFAULT_PARAMS = TransmonParams()


# ---------------------------------------------------------------------------
# Qiskit Hamiltonian display (educational)
# ---------------------------------------------------------------------------

def print_hamiltonian(params: TransmonParams = DEFAULT_PARAMS) -> None:
    """
    Print the dispersive Hamiltonian in Pauli operator form using Qiskit.
    H = ωr·(I−Z)/2 ⊗ I  +  ωq/2·I ⊗ Z  +  χ·(I−Z)/2 ⊗ Z
    (qubit tensored with a 2-level truncation of the cavity)
    """
    if not _QISKIT_OK:
        print("Qiskit not available; skipping Hamiltonian display.")
        return

    chi   = params.chi   / (2 * np.pi) / 1e6
    omega_r = params.omega_r / (2 * np.pi) / 1e9
    omega_q = params.omega_q / (2 * np.pi) / 1e9
    kappa = params.kappa / (2 * np.pi) / 1e6

    print("\n  Dispersive Hamiltonian (rotating wave approximation):")
    print(f"  H/ℏ = (ωr + χ·σz)·a†a  +  (ωq/2)·σz")
    print(f"")
    print(f"  Parameters:")
    print(f"    ωr/2π  = {omega_r:.2f} GHz  (bare cavity)")
    print(f"    ωq/2π  = {omega_q:.2f} GHz  (qubit)")
    print(f"    χ/2π   = {chi:.2f} MHz  (dispersive shift)")
    print(f"    κ/2π   = {kappa:.2f} MHz  (cavity linewidth)")
    print(f"")
    print(f"  Dressed cavity frequencies:")
    print(f"    |0⟩  →  (ωr + χ)/2π = {(params.omega_r + params.chi)/(2*np.pi)/1e9:.4f} GHz")
    print(f"    |1⟩  →  (ωr − χ)/2π = {(params.omega_r - params.chi)/(2*np.pi)/1e9:.4f} GHz")


# ---------------------------------------------------------------------------
# Cavity field ODE
# ---------------------------------------------------------------------------

def _cavity_ode(t, alpha_vec, kappa, delta, epsilon):
    """
    ODE right-hand side for:  dα/dt = −(κ/2 + i·δ)·α + ε

    State vector: alpha_vec = [Re(α), Im(α)]
    """
    a_re, a_im = alpha_vec
    da_re = -(kappa / 2) * a_re + delta * a_im + epsilon
    da_im = -(kappa / 2) * a_im - delta * a_re
    return [da_re, da_im]


def simulate_cavity(qubit_state: int,
                    params: TransmonParams = DEFAULT_PARAMS) -> tuple:
    """
    Simulate the cavity field for a single qubit state (noise-free).

    Parameters
    ----------
    qubit_state : 0 or 1
    params      : TransmonParams

    Returns
    -------
    alpha_t : complex ndarray of shape (n_time,)
              Cavity field in rotating frame
    t_eval  : float ndarray of shape (n_time,)
              Time axis in seconds
    """
    # Cavity dressed frequency: ωr + χ for |0⟩, ωr − χ for |1⟩
    sign      = +1 if qubit_state == 0 else -1
    omega_cav = params.omega_r + sign * params.chi
    delta     = params.omega_d - omega_cav          # drive detuning (rad/s)

    t_eval = np.linspace(0.0, params.t_end, params.n_time)

    sol = solve_ivp(
        _cavity_ode,
        [0.0, params.t_end],
        [0.0, 0.0],                                  # α(0) = 0
        t_eval=t_eval,
        args=(params.kappa, delta, params.epsilon),
        method='RK45',
        rtol=1e-9,
        atol=1e-12,
        dense_output=False,
    )

    return sol.y[0] + 1j * sol.y[1], t_eval


# ---------------------------------------------------------------------------
# Monte-Carlo shot simulation
# ---------------------------------------------------------------------------

def simulate_shots(
    n_shots:  int  = 1000,
    params:   TransmonParams = DEFAULT_PARAMS,
    rng_seed: int  = 42,
) -> dict:
    """
    Simulate dispersive readout for n_shots of each qubit state.

    AWGN noise is added to each shot independently, modelling:
      - Amplifier noise referred to the cavity output
      - Quantum vacuum fluctuations

    Returns
    -------
    dict with keys:
        't'          : (n_time,) time axis in seconds
        'alpha_0'    : (n_time,) complex — noise-free |0⟩ cavity field
        'alpha_1'    : (n_time,) complex — noise-free |1⟩ cavity field
        'shots_0'    : (n_shots, n_time) complex — noisy |0⟩ traces
        'shots_1'    : (n_shots, n_time) complex — noisy |1⟩ traces
        'params'     : TransmonParams used
    """
    rng = np.random.default_rng(rng_seed)

    alpha_0, t = simulate_cavity(0, params)
    alpha_1, _ = simulate_cavity(1, params)

    sigma = params.noise_sigma

    # Independent AWGN on each shot and each time sample
    noise_0 = (rng.standard_normal((n_shots, params.n_time))
               + 1j * rng.standard_normal((n_shots, params.n_time))) * sigma
    noise_1 = (rng.standard_normal((n_shots, params.n_time))
               + 1j * rng.standard_normal((n_shots, params.n_time))) * sigma

    shots_0 = alpha_0[np.newaxis, :] + noise_0     # (n_shots, n_time)
    shots_1 = alpha_1[np.newaxis, :] + noise_1

    return {
        't':       t,
        'alpha_0': alpha_0,
        'alpha_1': alpha_1,
        'shots_0': shots_0,
        'shots_1': shots_1,
        'params':  params,
    }


# ---------------------------------------------------------------------------
# SNR utilities
# ---------------------------------------------------------------------------

def compute_snr_linear(alpha_0: np.ndarray,
                       alpha_1: np.ndarray,
                       noise_sigma: float) -> float:
    """
    Single-shot SNR at steady state (last 10% of trace):
        SNR = |α0_ss − α1_ss|² / (4·σ²)
    """
    n    = len(alpha_0)
    tail = slice(int(0.9 * n), n)
    sig  = np.mean(np.abs(alpha_0[tail] - alpha_1[tail]) ** 2)
    return sig / (4.0 * noise_sigma ** 2)


def snr_db(alpha_0: np.ndarray,
           alpha_1: np.ndarray,
           noise_sigma: float) -> float:
    """Return single-shot SNR in dB."""
    return 10 * np.log10(compute_snr_linear(alpha_0, alpha_1, noise_sigma))


def steady_state_analytical(params: TransmonParams = DEFAULT_PARAMS) -> tuple:
    """
    Compute analytical steady-state cavity field for |0⟩ and |1⟩.

    α_ss = ε / (κ/2 + i·δ)

    Returns (alpha_ss_0, alpha_ss_1) as complex scalars.
    """
    eps = params.epsilon
    kp2 = params.kappa / 2

    delta_0 = params.omega_d - (params.omega_r + params.chi)
    delta_1 = params.omega_d - (params.omega_r - params.chi)

    alpha_ss_0 = eps / (kp2 + 1j * delta_0)
    alpha_ss_1 = eps / (kp2 + 1j * delta_1)

    return alpha_ss_0, alpha_ss_1
