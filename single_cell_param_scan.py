"""Parameter scan over stimulation strength (target_stim_dVm) and noise level
(noise_level_Vm) for the single-cell model in single_cell_simulations.py.

The stimulus is a two-frequency (temporal-interference) drive with carriers at
1000/1020 Hz, so the envelope beats at 20 Hz. All other parameters are held
constant. For each (target_stim_dVm, noise_level_Vm) grid point the simulation
is run (and cached) and three firing-rate quantities are stored as matrices:
    1. firing_rate : mean number of spikes per second
    2. fr_power    : power of the firing rate at the analysis frequency (20 Hz)
    3. fr_SNR      : SNR of the firing rate at the analysis frequency (20 Hz)
The three matrices are shown as color plots (imshow), one panel each.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import welch

from single_cell_simulations import (
    run_single_cell_simulation,
    find_spike_rate,
    return_freq_and_psd,
    compute_SNR,
    calculate_spectral_snr_and_zscore,
    simplify_axes,
    plot_single_cell_results, mark_subplots,
)

def analyze_firing_rate(results, sim_time, analysis_freq, bin_size,
                        use_welch=False, welch_segments=10,):
    """From a simulation `results` dict, return the firing-rate power and SNR
    at `analysis_freq` (Hz), computed the same way as plot_single_cell_results:
    bin the spikes into a rate, take its PSD, and read off the value / SNR at
    the frequency bin nearest `analysis_freq`."""
    spike_rate, t_bins = find_spike_rate(results["spike_times"], bin_size, sim_time)

    if use_welch:
        # Sampling rate (measurements per sec)
        fs_fr = 1000 / bin_size
        freqs, psd = welch(spike_rate, fs=fs_fr, nperseg=len(spike_rate) // welch_segments,
                                  noverlap=(len(spike_rate) // welch_segments) // 2)
    else:
        freqs, psd = return_freq_and_psd(t_bins, spike_rate)
        psd = psd[0]

    arg_f = np.argmin(np.abs(freqs - analysis_freq))
    power = psd[arg_f]
    snr = compute_SNR(freqs, psd, analysis_freq)
    snr2 = calculate_spectral_snr_and_zscore(freqs, psd, analysis_freq, freq_window=5)

    return power, snr, snr2


def run_param_scan(target_stim_dVms, noise_level_Vms, const_params,
                   analysis_freq=20.0, firing_rate_bin_size=None,
                   save_dir="results/param_scan", use_welch=False, welch_segments=8,):
    """Run the 2D scan and return (firing_rate, fr_power, fr_SNR) matrices.

    Each matrix has shape (len(noise_level_Vms), len(target_stim_dVms)) so that
    rows index the noise level (y-axis) and columns the stimulation strength
    (x-axis). Individual runs are cached by run_single_cell_simulation via a
    per-grid-point sim_name."""
    n_noise = len(noise_level_Vms)
    n_dVm = len(target_stim_dVms)

    if firing_rate_bin_size is None:
        firing_rate_bin_size = const_params["resolution"]


    firing_rate = np.full((n_noise, n_dVm), np.nan)
    fr_power = np.full((n_noise, n_dVm), np.nan)
    fr_SNR = np.full((n_noise, n_dVm), np.nan)
    fr_z_score = np.full((n_noise, n_dVm), np.nan)

    for i, noise in enumerate(noise_level_Vms):
        for j, dVm in enumerate(target_stim_dVms):
            sim_name = f"scan_dVm{dVm:.4f}_noise{noise:.4f}"
            print(f"\n[{i * n_dVm + j + 1}/{n_noise * n_dVm}] "
                  f"target_stim_dVm={dVm:.4f} mV, noise_level_Vm={noise:.4f} mV")

            sim_params = dict(const_params,
                              target_stim_dVm=dVm,
                              noise_level_Vm=noise,
                              sim_name=sim_name,
                              save_dir=save_dir,
                              save_Vm=True)

            results = run_single_cell_simulation(**sim_params)
            plot_single_cell_results(results, sim_params)
            power, snr, snr2 = analyze_firing_rate(
                results, const_params["sim_time"], analysis_freq,
                bin_size=firing_rate_bin_size,  use_welch=use_welch, welch_segments=welch_segments)

            firing_rate[i, j] = results["firing_rate"]
            fr_power[i, j] = power
            fr_SNR[i, j] = snr2["snr_peak_ratio"]
            fr_z_score[i, j] = snr2["z_score"]
            print("comparison old, new SNR: ", snr, snr2["snr_peak_ratio"])
            print("z_score: ", snr2["z_score"])

    return firing_rate, fr_power, fr_SNR, fr_z_score


def plot_param_scan(firing_rate, fr_power, fr_SNR, fr_z_score,
                    target_stim_dVms, noise_level_Vms,
                    analysis_freq=20.0, save_name="param_scan.png",
                    snr_levels=None, n_levels=14):
    """Plot the three scan matrices as filled contour panels (contourf).

    Noise level is placed on the x-axis and target stim dVm on the y-axis. The
    matrices are stored as [noise, dVm], so each is transposed to [dVm, noise]
    before plotting.

    `snr_levels` sets the contour boundaries for the SNR panel (a logarithmic
    colour scale is matched to them, with labelled contour lines overlaid); it
    defaults to a log-spaced set if not given. The firing-rate and power panels
    use `n_levels` automatically-placed levels."""
    if snr_levels is None:
        snr_levels = [2, 5, 10, 20, 50, 100, 200]

    fr_levels = [25, 50, 100, 200, 400, 800, 1600, 3200]
    snr_levels = np.asarray(snr_levels, dtype=float)
    z_score_levels = np.arange(0, 20)[::2]


    # Grid of data coordinates: x = noise level, y = target stim dVm. Each
    # matrix is transposed from [noise, dVm] to [dVm, noise] to match (Y, X).
    X, Y = np.meshgrid(noise_level_Vms, target_stim_dVms)

    # (matrix, title, levels, log_scale) - the SNR panel uses manual levels on
    # a logarithmic colour scale; the others use automatic linear levels.
    panels = [
        (firing_rate, "Firing rate (Hz)", n_levels, False, "Hz"),
        (fr_power, f"Firing-rate power at {analysis_freq:.0f} Hz", fr_levels, True, "|firing rate|²/Hz"),
        (fr_SNR, f"Firing-rate SNR at {analysis_freq:.0f} Hz", snr_levels, True, ''),
        (fr_z_score, "z-score for peak at beat frequency", z_score_levels, False, ''),

    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, bottom=0.15, top=0.9)

    for ax, (matrix, title, levels, log_scale, label) in zip(axes, panels):
        Z = matrix.T
        if log_scale:
            # Match the colour normalisation to the manual level range; extend
            # both ends so out-of-range cells still get the end colours.
            norm = LogNorm(vmin=levels[0], vmax=levels[-1])
            cf = ax.contourf(X, Y, Z, levels=levels, norm=norm, cmap="PuBuGn_r",
                             extend="both")
            # Overlay labelled contour lines for readability.
            cl = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4)
            ax.clabel(cl, fmt="%g", fontsize=10)
        else:
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="PuBuGn_r")

            cl = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4)
            ax.clabel(cl, fmt="%g", fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("noise level Vm (mV)")
        ax.set_ylabel("target stim dVm (mV)")
        if log_scale:
           cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
           cbar.set_ticks(levels)
           cbar.set_ticklabels([f"{int(l)}" for l in levels])

        else:
           cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="")
        cbar.set_label(label)
        ax.plot(4, 0.3, '*', c='orange', ms=8)

    simplify_axes(list(axes))
    mark_subplots(list(axes), ypos=1.05)
    fig.savefig(save_name, dpi=150)
    print(f"\nSaved figure to '{save_name}'")
    return fig


if __name__ == "__main__":
    # Two carriers 20 Hz apart -> the firing-rate envelope beats at 20 Hz,
    # which is the frequency at which power/SNR are evaluated.
    analysis_freq = 20.0
    carrier_freq = 1000.0
    dt = 0.05

    # Parameters held constant across the scan.
    const_params = dict(
        sim_time=10000e3,
        f_values=[carrier_freq, carrier_freq + analysis_freq],
        seed=2,
        V_th=-50.,
        E_m=-60.,
        C=100,
        tau_m=10,
        resolution=dt,
        force_rerun=True,
    )

    # Scanned axes:
    target_stim_dVms = np.linspace(0, 1, 11)   # mV
    noise_level_Vms = np.linspace(3, 9, 13)    # mV

    rerun_scan = True
    welch_segments = 10
    use_welch = True

    if rerun_scan:
        firing_rate, fr_power, fr_SNR, fr_z_score = run_param_scan(
            target_stim_dVms, noise_level_Vms, const_params,
            analysis_freq=analysis_freq, use_welch=use_welch, welch_segments=welch_segments,)

        # Store the matrices (with their axes) for later reuse.
        os.makedirs("results/param_scan", exist_ok=True)
        np.savez(f"results/param_scan/param_scan_matrices_welch:{use_welch}:{welch_segments}_dt{dt}.npz",
                 target_stim_dVms=target_stim_dVms,
                 noise_level_Vms=noise_level_Vms,
                 firing_rate=firing_rate,
                 fr_power=fr_power,
                 fr_SNR=fr_SNR,
                 analysis_freq=analysis_freq,
                 fr_z_score=fr_z_score,)
    else:
        # Load the matrices (and their axes) from the previously saved scan.
        matrix_path = f"results/param_scan/param_scan_matrices_welch:{use_welch}:{welch_segments}_dt{dt}.npz"
        with np.load(matrix_path) as data:
            target_stim_dVms = data["target_stim_dVms"]
            noise_level_Vms = data["noise_level_Vms"]
            firing_rate = data["firing_rate"]
            fr_power = data["fr_power"]
            fr_SNR = data["fr_SNR"]
            analysis_freq = float(data["analysis_freq"],)
            fr_z_score = data["fr_z_score"]
        print(f"Loaded scan matrices from '{matrix_path}'")

    plot_param_scan(firing_rate, fr_power, fr_SNR, fr_z_score,
                    target_stim_dVms, noise_level_Vms,
                    analysis_freq=analysis_freq,
                    save_name=f"param_scan_welch:{use_welch}:{welch_segments}_dt{dt}.pdf")
