"""
discriminator.py — IQ State Discrimination
===========================================

Classifies single-shot readout IQ points as qubit state |0⟩ or |1⟩.

Two classifiers:
  GMMDiscriminator  — Gaussian Mixture Model (2 components), unsupervised fit
  LDADiscriminator  — Linear Discriminant Analysis, supervised, optimal linear

Metrics computed:
  assignment_matrix          : 2×2 matrix M[i,j] = P(readout=i | state=j)
  readout_fidelity           : F = 1 − (P(0|1) + P(1|0)) / 2
  compute_roc                : ROC curve + AUC via sklearn
  fidelity_vs_integration_time : fidelity as function of readout window length

References
----------
Magesan et al., PRA 91, 012325 (2015)     — assignment matrix / fidelity
Bultink et al., PRApplied 6, 034008 (2016) — optimal matched filter weights
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc


# ---------------------------------------------------------------------------
# Data formatting helpers
# ---------------------------------------------------------------------------

def iq_to_xy(iq: np.ndarray) -> np.ndarray:
    """
    Convert complex IQ array → (N, 2) real array [I_column, Q_column].
    If already real (e.g. matched-filter output), return (N, 1).
    """
    if np.iscomplexobj(iq):
        return np.column_stack([iq.real, iq.imag])
    return iq.reshape(-1, 1)


def _combine(iq_0: np.ndarray, iq_1: np.ndarray):
    """Stack both states into (X, y) for sklearn."""
    X = np.vstack([iq_to_xy(iq_0), iq_to_xy(iq_1)])
    y = np.concatenate([np.zeros(len(iq_0)), np.ones(len(iq_1))]).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# GMM Discriminator
# ---------------------------------------------------------------------------

class GMMDiscriminator:
    """
    Gaussian Mixture Model discriminator (2-component, full covariance).

    Fits GMM on combined |0⟩ + |1⟩ data, then resolves the label ambiguity
    by matching each Gaussian component to the known state with the nearest
    centroid (measured along the I axis, which is typically the dominant axis
    of separation for drive at bare cavity frequency).
    """

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            n_init=5,
            max_iter=200,
        )
        self._label_map: dict = {0: 0, 1: 1}

    def fit(self, iq_0: np.ndarray, iq_1: np.ndarray) -> 'GMMDiscriminator':
        """Fit GMM, then resolve |0⟩↔|1⟩ label assignment."""
        X, _ = _combine(iq_0, iq_1)
        self.gmm.fit(X)

        # Component with lower I-mean → |0⟩  (physically: less photon number)
        means_i         = self.gmm.means_[:, 0]
        comp_for_0      = int(np.argmin(means_i))
        comp_for_1      = 1 - comp_for_0
        self._label_map = {comp_for_0: 0, comp_for_1: 1}
        return self

    def predict(self, iq: np.ndarray) -> np.ndarray:
        raw = self.gmm.predict(iq_to_xy(iq))
        return np.array([self._label_map[int(c)] for c in raw])

    def predict_proba(self, iq: np.ndarray) -> np.ndarray:
        """Return P(state=1) for each shot."""
        proba  = self.gmm.predict_proba(iq_to_xy(iq))
        comp_1 = next(k for k, v in self._label_map.items() if v == 1)
        return proba[:, comp_1]

    @property
    def means(self) -> np.ndarray:
        """GMM component means, sorted by label: row 0 → |0⟩, row 1 → |1⟩."""
        order = [k for k, v in sorted(self._label_map.items(), key=lambda x: x[1])]
        return self.gmm.means_[order]

    @property
    def covariances(self) -> np.ndarray:
        """GMM covariance matrices, sorted by label."""
        order = [k for k, v in sorted(self._label_map.items(), key=lambda x: x[1])]
        return self.gmm.covariances_[order]


# ---------------------------------------------------------------------------
# LDA Discriminator
# ---------------------------------------------------------------------------

class LDADiscriminator:
    """
    Linear Discriminant Analysis discriminator.

    Finds the optimal linear decision boundary in the IQ plane.
    Equivalent to matched filter + threshold when noise is Gaussian.
    """

    def __init__(self):
        self.lda = LinearDiscriminantAnalysis(solver='svd')

    def fit(self, iq_0: np.ndarray, iq_1: np.ndarray) -> 'LDADiscriminator':
        X, y = _combine(iq_0, iq_1)
        self.lda.fit(X, y)
        return self

    def predict(self, iq: np.ndarray) -> np.ndarray:
        return self.lda.predict(iq_to_xy(iq))

    def predict_proba(self, iq: np.ndarray) -> np.ndarray:
        return self.lda.predict_proba(iq_to_xy(iq))[:, 1]

    @property
    def coef(self) -> np.ndarray:
        return self.lda.coef_[0]

    @property
    def intercept(self) -> float:
        return float(self.lda.intercept_[0])

    def decision_line_points(self, xlim: tuple, n: int = 200) -> tuple:
        """
        Return (x, y) coordinates of the LDA decision boundary for plotting.
        Boundary: coef[0]*x + coef[1]*y + intercept = 0
        """
        xs = np.linspace(xlim[0], xlim[1], n)
        if abs(self.coef[1]) < 1e-12:
            return np.full(n, -self.intercept / self.coef[0]), xs
        ys = -(self.coef[0] * xs + self.intercept) / self.coef[1]
        return xs, ys


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def assignment_matrix(predictor, iq_0: np.ndarray, iq_1: np.ndarray) -> np.ndarray:
    """
    Compute the 2×2 assignment matrix:

        M[i, j] = P(readout = i | qubit state = j)

        M = [[P(0|0),  P(0|1)],
             [P(1|0),  P(1|1)]]

    An ideal readout has M = I.
    """
    pred_0 = predictor.predict(iq_0)
    pred_1 = predictor.predict(iq_1)
    p00 = np.mean(pred_0 == 0)
    p10 = np.mean(pred_0 == 1)
    p01 = np.mean(pred_1 == 0)
    p11 = np.mean(pred_1 == 1)
    return np.array([[p00, p01],
                     [p10, p11]])


def readout_fidelity(M: np.ndarray) -> float:
    """
    Readout fidelity:  F = 1 − (P(0|1) + P(1|0)) / 2
                         = 1 − (M[0,1] + M[1,0]) / 2
    """
    return 1.0 - (M[0, 1] + M[1, 0]) / 2.0


def compute_roc(predictor,
                iq_0: np.ndarray,
                iq_1: np.ndarray) -> tuple:
    """
    ROC curve and AUC for the discriminator.

    Returns
    -------
    (fpr, tpr, thresholds, auc_score)
    """
    iq_all = np.concatenate([iq_0, iq_1])
    y_all  = np.concatenate([np.zeros(len(iq_0)),
                              np.ones(len(iq_1))]).astype(int)
    scores         = predictor.predict_proba(iq_all)
    fpr, tpr, thr  = roc_curve(y_all, scores)
    auc_score      = auc(fpr, tpr)
    return fpr, tpr, thr, auc_score


def fidelity_vs_integration_time(
    shots_0_dec: np.ndarray,
    shots_1_dec: np.ndarray,
    n_fractions: int = 25,
) -> tuple:
    """
    Sweep the box-car integration window from t=0 forward.

    Integrates samples 0 … int(frac·N) to show how fidelity builds up as the
    cavity field rings up from zero toward steady state.  This reflects the
    real experimental trade-off: longer readout → more signal, but also more
    exposure to T1 relaxation.

    Parameters
    ----------
    shots_0_dec : (n_shots, n_dec) complex — |0⟩ after LPF + decimate
    shots_1_dec : (n_shots, n_dec) complex — |1⟩ after LPF + decimate
    n_fractions : number of integration durations to test

    Returns
    -------
    fractions  : (n_fractions,) — fraction of readout window used (0 → 1)
    fidelities : (n_fractions,) — LDA readout fidelity at each duration
    """
    fractions  = np.linspace(0.04, 1.0, n_fractions)
    fidelities = []

    for frac in fractions:
        n   = shots_0_dec.shape[-1]
        end = max(int(frac * n), 1)
        iq_0 = np.mean(shots_0_dec[:, :end], axis=-1)   # integrate t=0 → frac·T
        iq_1 = np.mean(shots_1_dec[:, :end], axis=-1)

        disc = LDADiscriminator().fit(iq_0, iq_1)
        M    = assignment_matrix(disc, iq_0, iq_1)
        fidelities.append(readout_fidelity(M))

    return fractions, np.array(fidelities)


def snr_vs_detuning(
    n_detunings: int = 30,
    chi_target:  float = 2 * np.pi * 1e6,
    base_kappa:  float = 2 * np.pi * 2e6,
    epsilon:     float = 2 * np.pi * 1e6,
) -> tuple:
    """
    Compute analytical single-shot SNR as the drive frequency is swept
    from (ωr − 3χ) to (ωr + 3χ) relative to the bare cavity.

    SNR = |α_ss(0) − α_ss(1)|² / (4·σ²)   [σ absorbed into ε/κ scale]

    Returns
    -------
    detunings_MHz : (n_detunings,) in MHz
    snr_linear    : (n_detunings,) linear SNR (normalised to peak)
    """
    detunings = np.linspace(-3 * chi_target, 3 * chi_target, n_detunings)

    snrs = []
    for det in detunings:
        # drive at omega_r + det
        delta_0 = det - chi_target    # delta for |0⟩ state
        delta_1 = det + chi_target    # delta for |1⟩ state
        kp2     = base_kappa / 2.0

        alpha_ss_0 = epsilon / (kp2 + 1j * delta_0)
        alpha_ss_1 = epsilon / (kp2 + 1j * delta_1)

        separation = abs(alpha_ss_0 - alpha_ss_1) ** 2
        snrs.append(separation)

    snrs = np.array(snrs)
    return detunings / (2 * np.pi * 1e6), snrs / snrs.max()
