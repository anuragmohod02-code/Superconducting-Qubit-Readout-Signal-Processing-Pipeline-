"""
latency_model.py — Real-Time Readout Decision Latency Model
=============================================================

Models the end-to-end latency of a real-time qubit state discrimination
pipeline and quantifies its impact on quantum error correction (QEC).

Latency sources modelled
------------------------
1. **Readout pulse duration** T_int: typically 200–1000 ns (dispersive regime)
2. **ADC sampling** T_adc: 1–2 ns depending on board
3. **DDC + FIR filter group delay**: proportional to filter length (taps/2 / fs)
4. **Decimation delay**: additional latency from averaging
5. **Threshold discriminator**: sub-clock-cycle in FPGA, modelled as 1 cycle
6. **Classical communication** T_comm: PCIe / network latency (1–10 μs)
7. **Classical processing** T_proc: threshold comparison, syndrome decoding
8. **Qubit T1 / T2**: decoherence during the latency window

Key result: **total_latency vs qubit T1** determines whether real-time
feedback is viable for a given architecture.

Typical numbers (superconducting qubit, 2023):
  T_int = 500 ns, T_adc = 1 ns, T_fir = 62 ns, T_dec = 15 ns,
  T_comm = 500 ns, T_proc = 200 ns → total ≈ 1.3 μs
  T1 ≈ 100 μs  → latency/T1 ≈ 1.3% (viable for surface code distance 3)

References
----------
Fowler A. et al., PRA 86, 032324 (2012)      — surface code latency requirements
Andersen C. et al., npj Quantum Inf 6, 1 (2020) — real-time feedback latency
Bultink C. et al., PRApplied 6, 034008 (2016) — hardware latency breakdown
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass


# ── Latency model ─────────────────────────────────────────────────────────────────

@dataclass
class ReadoutLatency:
    """
    End-to-end readout decision latency model.

    All times in nanoseconds.
    """
    T_int:     float = 500.0    # Integration window (ns)
    T_adc:     float = 2.0      # ADC conversion latency (ns)
    T_fir_taps:int   = 63       # FIR filter length (taps)
    fs_mhz:    float = 512.0    # ADC sample rate (MHz)
    decimate:  int   = 8        # Decimation factor
    T_disc:    float = 2.0      # Threshold discriminator FPGA latency (ns)
    T_comm:    float = 500.0    # Classical communication latency (ns)
    T_proc:    float = 200.0    # Classical processing / syndrome decoding (ns)
    T_reset:   float = 0.0      # Active reset latency if used (ns)

    @property
    def T_fir_ns(self) -> float:
        """Group delay of FIR filter = (N_taps - 1) / (2 * fs)"""
        return (self.T_fir_taps - 1) / (2.0 * self.fs_mhz) * 1e3   # ns

    @property
    def T_decimation_ns(self) -> float:
        """Extra latency from decimation stage (half a decimation interval)"""
        return (self.decimate / 2.0) / self.fs_mhz * 1e3  # ns

    @property
    def total_ns(self) -> float:
        return (self.T_int + self.T_adc + self.T_fir_ns
                + self.T_decimation_ns + self.T_disc
                + self.T_comm + self.T_proc + self.T_reset)

    def breakdown(self) -> dict:
        return {
            "Integration window (T_int)":     self.T_int,
            "ADC conversion":                  self.T_adc,
            "FIR group delay":                 self.T_fir_ns,
            "Decimation delay":                self.T_decimation_ns,
            "Threshold discriminator":         self.T_disc,
            "Classical communication (T_comm)":self.T_comm,
            "Classical processing (T_proc)":   self.T_proc,
            "Active reset (T_reset)":          self.T_reset,
        }

    def fidelity_penalty(self, T1_ns: float, T2_ns: float) -> dict:
        """
        Estimate the fidelity penalty from decoherence during latency.

        For T1 decay: ε_T1 = 1 - exp(-t_lat / T1)
        For T2 dephasing: ε_T2 = 1 - exp(-t_lat / T2)
        (These bound the error; actual penalty depends on protocol.)
        """
        t = self.total_ns
        eps_T1 = 1.0 - np.exp(-t / T1_ns)
        eps_T2 = 1.0 - np.exp(-t / T2_ns)
        return {
            "latency_ns":    t,
            "T1_ns":         T1_ns,
            "T2_ns":         T2_ns,
            "eps_T1":        eps_T1,
            "eps_T2":        eps_T2,
            "latency_over_T1": t / T1_ns,
        }


# ── Reference architectures ───────────────────────────────────────────────────────

ARCHITECTURES = {
    "State-of-art 2023\n(fast FPGA + JPA)": ReadoutLatency(
        T_int=200, T_adc=1, T_fir_taps=31, fs_mhz=1000,
        decimate=4, T_disc=1, T_comm=300, T_proc=100),
    "Typical 2023\n(FPGA + HEMT)": ReadoutLatency(
        T_int=500, T_adc=2, T_fir_taps=63, fs_mhz=512,
        decimate=8, T_disc=2, T_comm=500, T_proc=200),
    "Early systems\n(2018, slow FIR)": ReadoutLatency(
        T_int=800, T_adc=4, T_fir_taps=127, fs_mhz=250,
        decimate=16, T_disc=5, T_comm=2000, T_proc=500),
}


# ── Demo / CLI ────────────────────────────────────────────────────────────────────

def demo_latency():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Real-Time Readout Decision Latency Model")
    print("=" * 60)

    T1_ns  = 100_000.0   # 100 μs — typical superconducting qubit
    T2_ns  = 60_000.0    # 60 μs

    for arch_name, lat in ARCHITECTURES.items():
        pen = lat.fidelity_penalty(T1_ns, T2_ns)
        print(f"\n{'─'*50}")
        print(f"Architecture: {arch_name.replace(chr(10), ' ')}")
        print(f"  Total latency:      {lat.total_ns:.0f} ns")
        print(f"  Latency / T1:       {pen['latency_over_T1']*100:.2f}%")
        print(f"  T1 error penalty:   {pen['eps_T1']*100:.3f}%")
        print(f"  T2 dephasing:       {pen['eps_T2']*100:.3f}%")

    # ── Figure 1: Latency breakdown bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Readout Decision Latency — Architecture Comparison",
                 fontsize=12, fontweight="bold")

    cmap = plt.get_cmap("tab10")
    for ax, (arch_name, lat) in zip(axes, ARCHITECTURES.items()):
        bd = lat.breakdown()
        labels = [k for k, v in bd.items() if v > 0]
        values = [v for v in bd.values() if v > 0]
        colors = [cmap(i) for i in range(len(labels))]
        wedges, texts, autotexts = ax.pie(
            values, labels=None, colors=colors,
            autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
            pctdistance=0.7, startangle=90)
        ax.set_title(f"{arch_name}\nTotal: {lat.total_ns:.0f} ns", fontsize=10)

    # Legend from first chart
    handles = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(8)]
    bd_ref = list(ARCHITECTURES.values())[0].breakdown()
    fig.legend(handles, list(bd_ref.keys()), loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(out_dir, "latency_breakdown.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved → {os.path.join(out_dir, 'latency_breakdown.png')}")

    # ── Figure 2: T1 error penalty vs integration time sweep
    T_int_range = np.linspace(100, 2000, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Latency Impact on Qubit Fidelity",
                 fontsize=12, fontweight="bold")

    for arch_name, lat_base in ARCHITECTURES.items():
        eps_arr = []
        for T_int in T_int_range:
            lat = ReadoutLatency(
                T_int=T_int,
                T_adc=lat_base.T_adc,
                T_fir_taps=lat_base.T_fir_taps,
                fs_mhz=lat_base.fs_mhz,
                decimate=lat_base.decimate,
                T_disc=lat_base.T_disc,
                T_comm=lat_base.T_comm,
                T_proc=lat_base.T_proc,
            )
            pen = lat.fidelity_penalty(T1_ns, T2_ns)
            eps_arr.append(pen["eps_T1"] * 100)
        label = arch_name.replace("\n", " ")
        ax1.plot(T_int_range / 1e3, eps_arr, linewidth=2, label=label)

    ax1.set_xlabel("Integration time T_int (μs)")
    ax1.set_ylabel("T1 error during latency (%)")
    ax1.set_title(f"Error vs Integration Window  (T1={T1_ns/1e3:.0f} μs)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.35)

    # Panel 2: total latency vs T1/T2 for current design
    lat_cur = ARCHITECTURES["Typical 2023\n(FPGA + HEMT)"]
    T1_range = np.logspace(3, 6, 200)  # 1 μs to 1 ms
    eps_T1_arr = [lat_cur.fidelity_penalty(T1, T1 * 0.6)["eps_T1"] * 100 for T1 in T1_range]
    ax2.semilogx(T1_range / 1e3, eps_T1_arr, color="#E74C3C", linewidth=2)
    ax2.axhline(0.1, color="#888", linestyle="--", linewidth=1, label="0.1% threshold")
    ax2.axvline(lat_cur.total_ns / 1e3, color="#3498DB", linestyle="--",
                linewidth=1, label=f"T_lat = {lat_cur.total_ns:.0f} ns")
    ax2.set_xlabel("Qubit T1 (μs)")
    ax2.set_ylabel("T1 error during latency (%)")
    ax2.set_title(f"Typical 2023 design  (T_lat={lat_cur.total_ns:.0f} ns)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.35)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_fidelity.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(out_dir, 'latency_fidelity.png')}")


if __name__ == "__main__":
    demo_latency()
