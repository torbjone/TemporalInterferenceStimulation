"""Parameter scan over carrier frequency and beat frequency (the analysis
frequency) for the single-cell model in single_cell_simulations.py.

This is a variant of single_cell_param_scan.py. There the stimulation strength
(target_stim_dVm) and noise level (noise_level_Vm) were scanned at a fixed
1000/1020 Hz carrier pair. Here it is the other way round: the stimulus is
still a two-frequency (temporal-interference) drive, but now the *carrier
frequency* and the *beat frequency* are scanned while the stimulation strength
and noise level are held constant (target_stim_dVm=0.3 mV, noise_level_Vm=4 mV).

For each (carrier_freq, beat_freq) grid point the two carriers are placed at
``[carrier_freq, carrier_freq + beat_freq]`` so the firing-rate envelope beats
at ``beat_freq``. That beat frequency is also the analysis frequency at which
the firing-rate power and SNR are read off - so, unlike the original scan, the
analysis frequency changes from grid point to grid point.

Three firing-rate quantities are stored as matrices:
    1. firing_rate : mean number of spikes per second
    2. fr_power    : power of the firing rate at the beat (analysis) frequency
    3. fr_SNR      : SNR of the firing rate at the beat (analysis) frequency
The three matrices are shown as filled contour plots, one panel each.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from single_cell_simulations import (
    run_single_cell_simulation,
    simplify_axes,
)
# analyze_firing_rate is identical to the one used by the strength/noise scan.
from single_cell_param_scan import analyze_firing_rate


def run_param_scan(carrier_freqs, beat_freqs, const_params,
                   firing_rate_bin_size=0.1,
                   save_dir="results/param_scan_freq"):
    """Run the 2D carrier x beat scan and return (firing_rate, fr_power, fr_SNR).

    Each matrix has shape (len(carrier_freqs), len(beat_freqs)) so that rows
    index the carrier frequency (x-axis) and columns the beat frequency
    (y-axis). For every grid point the carriers are placed at
    ``[carrier, carrier + beat]`` and the firing-rate power/SNR are evaluated at
    ``beat`` (the beat frequency is the analysis frequency). Individual runs are
    cached by run_single_cell_simulation via a per-grid-point sim_name."""
    n_carrier = len(carrier_freqs)
    n_beat = len(beat_freqs)

    firing_rate = np.full((n_carrier, n_beat), np.nan)
    fr_power = np.full((n_carrier, n_beat), np.nan)
    fr_SNR = np.full((n_carrier, n_beat), np.nan)

    for i, carrier in enumerate(carrier_freqs):
        for j, beat in enumerate(beat_freqs):
            sim_name = f"scan_carrier{carrier:.1f}_beat{beat:.4f}"
            print(f"\n[{i * n_beat + j + 1}/{n_carrier * n_beat}] "
                  f"carrier_freq={carrier:.1f} Hz, beat_freq={beat:.4f} Hz")

            sim_params = dict(const_params,
                              f_values=[carrier, carrier + beat],
                              sim_name=sim_name,
                              save_dir=save_dir,
                              save_Vm=False)

            results = run_single_cell_simulation(**sim_params)
            # The analysis frequency is this grid point's beat frequency.
            power, snr = analyze_firing_rate(
                results, const_params["sim_time"], beat,
                bin_size=firing_rate_bin_size)

            firing_rate[i, j] = results["firing_rate"]
            fr_power[i, j] = power
            fr_SNR[i, j] = snr

    return firing_rate, fr_power, fr_SNR


def plot_param_scan(firing_rate, fr_power, fr_SNR,
                    carrier_freqs, beat_freqs,
                    save_name="param_scan_freq.png",
                    snr_levels=None, n_levels=14):
    """Plot the three scan matrices as filled contour panels (contourf).

    Carrier frequency is placed on the x-axis and beat frequency on the y-axis.
    The matrices are stored as [carrier, beat], so each is transposed to
    [beat, carrier] before plotting.

    `snr_levels` sets the contour boundaries for the SNR panel (a logarithmic
    colour scale is matched to them, with labelled contour lines overlaid); it
    defaults to a log-spaced set if not given. The firing-rate and power panels
    use `n_levels` automatically-placed levels."""
    if snr_levels is None:
        snr_levels = [1, 2, 5, 10, 20, 50, 100, 200]
    snr_levels = np.asarray(snr_levels, dtype=float)

    # Grid of data coordinates: x = carrier frequency, y = beat frequency. Each
    # matrix is transposed from [carrier, beat] to [beat, carrier] to match
    # (Y, X).
    X, Y = np.meshgrid(carrier_freqs, beat_freqs)

    # (matrix, title, levels, log_scale) - the SNR panel uses manual levels on
    # a logarithmic colour scale; the others use automatic linear levels.
    panels = [
        (firing_rate, "Firing rate (Hz)", n_levels, False),
        (fr_power, "Firing-rate power at beat freq", n_levels, False),
        (fr_SNR, "Firing-rate SNR at beat freq", snr_levels, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, bottom=0.15, top=0.9)

    for ax, (matrix, title, levels, log_scale) in zip(axes, panels):
        Z = matrix.T
        if log_scale:
            # Match the colour normalisation to the manual level range; extend
            # both ends so out-of-range cells still get the end colours.
            norm = LogNorm(vmin=levels[0], vmax=levels[-1])
            cf = ax.contourf(X, Y, Z, levels=levels, norm=norm, cmap="hot",
                             extend="both")
            # Overlay labelled contour lines for readability.
            cl = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4)
            ax.clabel(cl, fmt="%g", fontsize=7)
        else:
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="hot")
        ax.set_title(title)
        ax.set_xlabel("carrier frequency (Hz)")
        ax.set_ylabel("beat frequency (Hz)")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    simplify_axes(list(axes))
    fig.savefig(save_name, dpi=150)
    print(f"\nSaved figure to '{save_name}'")
    return fig


if __name__ == "__main__":
    # Stimulation strength and noise level are held constant; the carrier and
    # beat frequencies are scanned instead.
    const_params = dict(
        sim_time=10000e3,
        target_stim_dVm=0.3,
        noise_level_Vm=4.0,
        seed=2,
        V_th=-50.,
        E_m=-60.,
        C=100,
        tau_m=10,
        resolution=0.1,
        force_rerun=False,
    )

    # Scanned axes.
    carrier_freqs = np.linspace(500, 3000, 11)   # Hz
    beat_freqs = np.linspace(10, 100, 10)         # Hz

    rerun_scan = True

    if rerun_scan:
        firing_rate, fr_power, fr_SNR = run_param_scan(
            carrier_freqs, beat_freqs, const_params)

        # Store the matrices (with their axes) for later reuse.
        os.makedirs("results/param_scan_freq", exist_ok=True)
        np.savez("results/param_scan_freq/param_scan_freq_matrices.npz",
                 carrier_freqs=carrier_freqs,
                 beat_freqs=beat_freqs,
                 firing_rate=firing_rate,
                 fr_power=fr_power,
                 fr_SNR=fr_SNR)
    else:
        # Load the matrices (and their axes) from the previously saved scan.
        matrix_path = "results/param_scan_freq/param_scan_freq_matrices.npz"
        with np.load(matrix_path) as data:
            carrier_freqs = data["carrier_freqs"]
            beat_freqs = data["beat_freqs"]
            firing_rate = data["firing_rate"]
            fr_power = data["fr_power"]
            fr_SNR = data["fr_SNR"]
        print(f"Loaded scan matrices from '{matrix_path}'")

    plot_param_scan(firing_rate, fr_power, fr_SNR,
                    carrier_freqs, beat_freqs,
                    save_name="param_scan_freq_log.png")
