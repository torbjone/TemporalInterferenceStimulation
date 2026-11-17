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
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from single_cell_simulations import (
    run_single_cell_simulation,
    plot_single_cell_results,
    simplify_axes,
    mark_subplots
)
# analyze_firing_rate is identical to the one used by the strength/noise scan.
from single_cell_param_scan import (analyze_firing_rate, _octave_levels,
                                    snr_ref_from_band_ratio)


def run_param_scan(carrier_freqs, beat_freqs, const_params,
                   firing_rate_bin_size=None, use_welch=False, welch_segments=4,
                   freq_resolution=0.005, band_halfwidth=0.05,
                   freq_window=5.0, ref_bandwidth=0.01,
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
    fr_z_score = np.full((n_carrier, n_beat), np.nan)

    if firing_rate_bin_size is None:
        firing_rate_bin_size = const_params["resolution"]

    for i, carrier in enumerate(carrier_freqs):
        for j, beat in enumerate(beat_freqs):
            sim_name = f"scan_carrier{carrier:.1f}_beat{beat:.4f}"
            print(f"\n[{i * n_beat + j + 1}/{n_carrier * n_beat}] "
                  f"carrier_freq={carrier:.1f} Hz, beat_freq={beat:.4f} Hz")

            sim_params = dict(const_params,
                              f_values=[carrier, carrier + beat],
                              sim_name=sim_name,
                              save_dir=save_dir,
                              save_Vm="cut")

            results = run_single_cell_simulation(**sim_params)
            plot_single_cell_results(results, sim_params, fig_folder="param_scan_freq_figs",
                                     use_welch=use_welch, welch_segments=welch_segments)
            # The analysis frequency is this grid point's beat frequency.
            power, snr, snr2 = analyze_firing_rate(
                results, const_params["sim_time"], beat,
                use_welch=use_welch, welch_segments=welch_segments,
                bin_size=firing_rate_bin_size,
                freq_resolution=freq_resolution,
                band_halfwidth=band_halfwidth,
                freq_window=freq_window,
                ref_bandwidth=ref_bandwidth)

            firing_rate[i, j] = results["firing_rate"]
            fr_power[i, j] = power
            fr_SNR[i, j] = snr2["snr_ref"]
            fr_z_score[i, j] = snr2["z_score"]
            print("comparison old, ref SNR: ", snr, snr2["snr_ref"])
            print("z_score: ", snr2["z_score"])

    return firing_rate, fr_power, fr_SNR, fr_z_score


def plot_param_scan(firing_rate, fr_power, fr_SNR, fr_z_score,
                    carrier_freqs, beat_freqs,
                    save_name="param_scan_freq.png",
                    snr_levels=None, fr_power_levels=None, n_levels=14,
                    ref_bandwidth=0.01):
    """Plot the three scan matrices as filled contour panels (contourf).

    Carrier frequency is placed on the x-axis and beat frequency on the y-axis.
    The matrices are stored as [carrier, beat], so each is transposed to
    [beat, carrier] before plotting.

    `snr_levels` sets the contour boundaries for the SNR panel (a logarithmic
    colour scale is matched to them, with labelled contour lines overlaid); it
    defaults to a log-spaced set if not given. The firing-rate and power panels
    use `n_levels` automatically-placed levels."""
    # Octave-spaced levels spanning the data: the power and SNR panels now show
    # resolution-independent band-integrated measures (see
    # analyze_firing_rate), whose absolute scale depends on the firing rate, so
    # the levels are derived from the data rather than hard-coded.
    if snr_levels is None:
        # snr_ref = 0 means "no line", so (unlike the band ratio) there is no
        # natural floor at 1; the level range is capped by _octave_levels.
        snr_levels = _octave_levels(fr_SNR)
    if fr_power_levels is None:
        fr_power_levels = _octave_levels(fr_power)
    fr_levels = np.linspace(np.nanmin(firing_rate), np.nanmax(firing_rate), 7)
    fr_psd_levels = np.asarray(fr_power_levels, dtype=float)
    snr_levels = np.asarray(snr_levels, dtype=float)

    z_score_levels = np.linspace(0, 150, 11)

    # Grid of data coordinates: x = carrier frequency, y = beat frequency. Each
    # matrix is transposed from [carrier, beat] to [beat, carrier] to match
    # (Y, X).
    X, Y = np.meshgrid(carrier_freqs, beat_freqs)

    # (matrix, title, levels, log_scale) - the SNR panel uses manual levels on
    # a logarithmic colour scale; the others use automatic linear levels.
    panels = [
        (firing_rate, "firing rate (Hz)", fr_levels, False, "Hz"),
        (fr_power, "firing-rate power at beat frequency", fr_psd_levels, True, "(spikes/s)²"),
        (fr_SNR, "firing-rate SNR at beat frequency\n"
                 f"(B$_\\mathrm{{ref}}$ = {ref_bandwidth:g} Hz)",
         snr_levels, True, ''),
        # (fr_z_score, "z-score for peak at beat frequency", z_score_levels, False, ''),
    ]


    print(fr_SNR)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(wspace=0.4, left=0.04, right=0.95, bottom=0.13, top=0.93)

    for ax, (matrix, title, levels, log_scale, label) in zip(axes, panels):
        Z = matrix.T
        if log_scale:
            # Match the colour normalisation to the manual level range; extend
            # both ends so out-of-range cells still get the end colours.
            norm = LogNorm(vmin=levels[0], vmax=levels[-1])
            cf = ax.contourf(X, Y, np.abs(Z), levels=levels, norm=norm, cmap="PuBuGn_r",
                             extend="both")
            # Overlay labelled contour lines for readability.
            cl = ax.contour(X, Y, np.abs(Z), levels=levels, colors="k",
                            linewidths=0.4, extend="both")
            ax.clabel(cl, fmt="%g", fontsize=10)
        else:
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="PuBuGn_r")

            cl = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4)
            ax.clabel(cl, fmt="%g", fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("carrier frequency (Hz)")
        ax.set_ylabel("beat frequency (Hz)", labelpad=-5)
        #fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        if log_scale:
            cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(levels)
            cbar.set_ticklabels([f"{l:g}" for l in levels])

        else:
            cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="")
        cbar.set_label(label)

        ax.plot(2000, 20, '*', c='orange', ms=8)

    simplify_axes(list(axes))
    mark_subplots(list(axes), "DEF", ypos=1.05)
    fig.savefig(join("figures", save_name), dpi=150)
    print(f"\nSaved figure to '{save_name}'")
    return fig


if __name__ == "__main__":
    # Stimulation strength and noise level are held constant; the carrier and
    # beat frequencies are scanned instead.
    dt = 0.025
    noise_type = "ou"
    use_welch = True
    welch_segments = 4
    noise_level_Vm = 5.0
    const_params = dict(
        sim_time=5000e3,
        target_stim_dVm=0.3,
        noise_level_Vm=noise_level_Vm,
        seed=2,
        V_th=-50.,
        E_m=-60.,
        C=100,
        tau_m=10,
        resolution=dt,
        force_rerun=False,
        noise_type=noise_type,
        description="param scan OU noise",
    )

    # Scanned axes.
    carrier_freqs = np.linspace(500, 3000, 11)   # Hz
    beat_freqs = np.linspace(10, 100, 10)         # Hz

    rerun_scan = False
    # Fixed frequency resolution (df) + band-integrated SNR/z-score, which is
    # independent of df (see calculate_band_snr_and_zscore), as in Fig. 1.
    freq_resolution = 0.005
    band_halfwidth = 0.05
    # Reference noise bandwidth for the SNR panel (must be stated in the
    # caption); the line power is compared with the noise power in B_ref.
    ref_bandwidth = 0.01
    if freq_resolution is not None:
        welch_addon = f"_df{freq_resolution}_band{band_halfwidth}"
    else:
        welch_addon = f"_welch_{welch_segments}" if use_welch else ""
    if rerun_scan:
        firing_rate, fr_power, fr_SNR, fr_z_score = run_param_scan(
            carrier_freqs, beat_freqs, const_params, use_welch=use_welch,
            welch_segments=welch_segments,
            freq_resolution=freq_resolution,
            band_halfwidth=band_halfwidth,
            ref_bandwidth=ref_bandwidth)


        # Store the matrices (with their axes) for later reuse.
        os.makedirs("results/param_scan_freq", exist_ok=True)
        np.savez(f"results/param_scan_freq/param_scan_freq_matrices_dt:{dt}{welch_addon}.npz",
                 carrier_freqs=carrier_freqs,
                 beat_freqs=beat_freqs,
                 firing_rate=firing_rate,
                 fr_power=fr_power,
                 fr_SNR=fr_SNR,
                 ref_bandwidth=ref_bandwidth,
                 fr_z_score=fr_z_score,)
    else:
        # Load the matrices (and their axes) from the previously saved scan.
        matrix_path = f"results/param_scan_freq/param_scan_freq_matrices_dt:{dt}{welch_addon}.npz"
        with np.load(matrix_path) as data:
            carrier_freqs = data["carrier_freqs"]
            beat_freqs = data["beat_freqs"]
            firing_rate = data["firing_rate"]
            fr_power = data["fr_power"]
            fr_SNR = data["fr_SNR"]
            fr_z_score = data["fr_z_score"]
            stored_ref_bandwidth = (float(data["ref_bandwidth"])
                                    if "ref_bandwidth" in data else None)
        print(f"Loaded scan matrices from '{matrix_path}'")
        if stored_ref_bandwidth is None and freq_resolution is not None:
            # Older files stored snr_band_ratio; convert to the fixed-reference
            # bandwidth SNR instead of re-running the scan.
            fr_SNR = snr_ref_from_band_ratio(
                fr_SNR, band_halfwidth=band_halfwidth,
                freq_resolution=freq_resolution,
                ref_bandwidth=ref_bandwidth)
            print(f"Converted stored snr_band_ratio to snr_ref "
                  f"(B_ref = {ref_bandwidth} Hz)")

    plot_param_scan(firing_rate, fr_power, fr_SNR, fr_z_score,
                    carrier_freqs, beat_freqs,
                    n_levels=6,
                    ref_bandwidth=ref_bandwidth,
                    save_name=f"param_scan_freq_dt:{dt}{welch_addon}.pdf")
