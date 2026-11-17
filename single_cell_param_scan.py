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
from os.path import join

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
    welch_psd_fixed_resolution,
    calculate_band_snr_and_zscore,
    simplify_axes,
    plot_single_cell_results, mark_subplots,
)

def analyze_firing_rate(results, sim_time, analysis_freq, bin_size,
                        use_welch=False, welch_segments=10,
                        freq_resolution=0.005, band_halfwidth=0.05,
                        freq_window=5.0, ref_bandwidth=0.01):
    """From a simulation `results` dict, return the firing-rate power and SNR
    at `analysis_freq` (Hz): bin the spikes into a rate, take its PSD, and
    quantify the spectral line at `analysis_freq`.

    By default the PSD is a Welch estimate with a *fixed* frequency resolution
    `freq_resolution` (Hz), i.e. nperseg = fs / freq_resolution, so the bin
    width df is identical for every grid point and every simulation length
    (longer runs average more segments instead of giving a finer df). The line
    is then quantified by its power *integrated* over a band of half-width
    `band_halfwidth` (Hz) around `analysis_freq`, compared with equally wide
    control bands within +/- `freq_window` Hz
    (calculate_band_snr_and_zscore). Unlike the peak-bin measure, both the
    returned power and the SNR/z-score are then invariant with respect to df,
    so the numbers are directly comparable across the whole scan and across
    figures.

    Set `freq_resolution=None` to fall back to the old behaviour: a Welch PSD
    with `welch_segments` segments (`use_welch=True`) or a plain periodogram,
    and the peak-bin SNR of calculate_spectral_snr_and_zscore.

    The reported SNR ('snr_ref') is the band-integrated line power divided by
    the noise power in a *fixed* reference bandwidth `ref_bandwidth` (Hz),
    rather than by the noise in the integration band. Dividing by the noise in
    the integration band ('snr_band_ratio') bounds the ratio by
    1 + P_line/(N0*B_band) and, with B_band = 0.105 Hz, compresses everything
    into 1-2; using a fixed, explicitly stated B_ref keeps the measure
    df-invariant while preserving the dynamic range. B_ref must be given in
    the figure caption.

    Returns (power, snr_old, snr_res), where `power` is the band-integrated
    excess (line) power in (spikes/s)^2 in band mode and the raw PSD value at
    the nearest bin in the fallback mode."""
    spike_rate, t_bins = find_spike_rate(results["spike_times"], bin_size, sim_time)
    fs_fr = 1000 / bin_size

    if freq_resolution is not None:
        freqs, psd = welch_psd_fixed_resolution(
            spike_rate, fs_fr, freq_resolution=freq_resolution)
    elif use_welch:
        # Sampling rate (measurements per sec)
        freqs, psd = welch(spike_rate, fs=fs_fr, nperseg=len(spike_rate) // welch_segments,
                                  noverlap=(len(spike_rate) // welch_segments) // 2)
    else:
        freqs, psd = return_freq_and_psd(t_bins, spike_rate)
        psd = psd[0]

    snr = compute_SNR(freqs, psd, analysis_freq)

    if freq_resolution is not None:
        snr2 = calculate_band_snr_and_zscore(
            freqs, psd, analysis_freq, band_halfwidth=band_halfwidth,
            freq_window=freq_window, ref_bandwidth=ref_bandwidth)
        power = snr2["excess_power"]
    else:
        arg_f = np.argmin(np.abs(freqs - analysis_freq))
        power = psd[arg_f]
        snr2 = calculate_spectral_snr_and_zscore(freqs, psd, analysis_freq,
                                                 freq_window=freq_window)
        snr2["snr_band_ratio"] = snr2["snr_peak_ratio"]
        snr2["snr_ref"] = snr2["snr_peak_ratio"]

    return power, snr, snr2


def run_param_scan(target_stim_dVms, noise_level_Vms, const_params,
                   analysis_freq=20.0, firing_rate_bin_size=None,
                   save_dir="results/param_scan", use_welch=False, welch_segments=8,
                   freq_resolution=0.005, band_halfwidth=0.05,
                   freq_window=5.0, ref_bandwidth=0.01):
    """Run the 2D scan and return (firing_rate, fr_power, fr_SNR) matrices.

    `fr_SNR` holds the band-integrated line power divided by the noise power in
    the fixed reference bandwidth `ref_bandwidth` (Hz), i.e. 'snr_ref' of
    calculate_band_snr_and_zscore - see analyze_firing_rate for why the ratio
    against the integration band is unsuitable as a display quantity.

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
                              save_Vm="cut")

            results = run_single_cell_simulation(**sim_params)
            # plot_single_cell_results(results,
            #                          sim_params,
            #                          fig_folder="param_scan_figs",
            #                          use_welch=use_welch,
            #                          welch_segments=welch_segments)
            power, snr, snr2 = analyze_firing_rate(
                results, const_params["sim_time"], analysis_freq,
                bin_size=firing_rate_bin_size,  use_welch=use_welch,
                welch_segments=welch_segments,
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


def snr_ref_from_band_ratio(snr_band_ratio, band_halfwidth=0.05,
                            freq_resolution=0.005, ref_bandwidth=0.01):
    """Convert a stored `snr_band_ratio` matrix to the fixed-reference-bandwidth
    SNR ('snr_ref'), so old scan matrices can be re-plotted without re-running.

    With B_band = n_band_bins * df the identity is
        snr_ref = (snr_band_ratio - 1) * B_band / B_ref,
    since snr_band_ratio - 1 = P_line / (N0 * B_band)."""
    n_band = 2 * int(np.floor(band_halfwidth / freq_resolution)) + 1
    b_band = n_band * freq_resolution
    return (np.asarray(snr_band_ratio, dtype=float) - 1.0) * b_band / ref_bandwidth


def _octave_levels(matrix, lo_floor=None, n_min=5, n_max=9):
    """Octave-spaced (factor-2) contour levels spanning the finite, positive
    values of `matrix`, for use with a logarithmic colour scale.

    `lo_floor` optionally clamps the lowest level from below (e.g. 1.0 for an
    SNR ratio, for which 1 means "pure noise"). At most `n_max` octaves are
    kept, counted down from the top, so a few near-zero grid points do not
    stretch the colour scale over a dozen decades."""
    vals = np.asarray(matrix, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.array([1.0, 2.0])
    lo = np.min(vals)
    if lo_floor is not None:
        lo = max(lo, lo_floor)
    hi = max(np.max(vals), lo * 2)
    e_lo = int(np.floor(np.log2(lo)))
    e_hi = int(np.ceil(np.log2(hi)))
    while e_hi - e_lo + 1 < n_min:
        e_hi += 1
    if e_hi - e_lo + 1 > n_max:
        e_lo = e_hi - n_max + 1
    return 2.0**np.arange(e_lo, e_hi + 1)


def plot_param_scan(firing_rate, fr_power, fr_SNR, fr_z_score,
                    target_stim_dVms, noise_level_Vms,
                    analysis_freq=20.0, save_name="param_scan.png",
                    snr_levels=None, fr_power_levels=None, n_levels=14,
                    ref_bandwidth=0.01):
    """Plot the three scan matrices as filled contour panels (contourf).

    Noise level is placed on the x-axis and target stim dVm on the y-axis. The
    matrices are stored as [noise, dVm], so each is transposed to [dVm, noise]
    before plotting.

    `snr_levels` sets the contour boundaries for the SNR panel (a logarithmic
    colour scale is matched to them, with labelled contour lines overlaid); it
    defaults to a log-spaced set if not given. The firing-rate and power panels
    use `n_levels` automatically-placed levels."""
    # Octave-spaced levels spanning the finite, positive part of the data.
    # Both matrices are now resolution-independent band-integrated measures
    # (see analyze_firing_rate), but their absolute scale still depends on the
    # firing rate and the stimulation strength, so the levels are derived from
    # the data rather than hard-coded.
    if snr_levels is None:
        # snr_ref = 0 means "no line", so (unlike the band ratio) there is no
        # natural floor at 1; the level range is capped by _octave_levels.
        snr_levels = _octave_levels(fr_SNR)
    if fr_power_levels is None:
        fr_power_levels = _octave_levels(fr_power)
    fr_levels = np.asarray(fr_power_levels, dtype=float)
    snr_levels = np.asarray(snr_levels, dtype=float)
    z_score_levels = np.logspace(0., 3, 10)
    print(f"z_score_levels: {z_score_levels}")


    # Grid of data coordinates: x = noise level, y = target stim dVm. Each
    # matrix is transposed from [noise, dVm] to [dVm, noise] to match (Y, X).
    X, Y = np.meshgrid(noise_level_Vms, target_stim_dVms)

    print(fr_power)
    # (matrix, title, levels, log_scale) - the SNR panel uses manual levels on
    # a logarithmic colour scale; the others use automatic linear levels.
    panels = [
        (firing_rate, "Firing rate (Hz)", n_levels, False, "Hz"),
        (fr_power, f"Firing-rate power at {analysis_freq:.0f} Hz", fr_levels, True, "(spikes/s)²"),
        (fr_SNR, f"Firing-rate SNR at {analysis_freq:.0f} Hz\n"
                 f"(B$_\\mathrm{{ref}}$ = {ref_bandwidth:g} Hz)",
         snr_levels, True, ''),
        # (fr_z_score, "z-score for peak at beat frequency", z_score_levels,True, ''),
    ]

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
            cl = ax.contour(X, Y, np.abs(Z), levels=levels, colors="k", linewidths=0.4, extend="both")
            ax.clabel(cl, fontsize=10)
        else:
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="PuBuGn_r")

            cl = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4)
            ax.clabel(cl, fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("noise level Vm (mV)")
        ax.set_ylabel("target stim dVm (mV)", labelpad=-2)
        if log_scale:
           cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
           cbar.set_ticks(levels)
           cbar.set_ticklabels([f"{l:g}" for l in levels])

        else:
           cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="")
        cbar.set_label(label, labelpad=-1)
        ax.plot(5, 0.3, '*', c='orange', ms=8)

    simplify_axes(list(axes))
    mark_subplots(list(axes), ypos=1.05)
    fig.savefig(join("figures", save_name), dpi=150)
    print(f"\nSaved figure to '{save_name}'")
    return fig


if __name__ == "__main__":
    # Two carriers 20 Hz apart -> the firing-rate envelope beats at 20 Hz,
    # which is the frequency at which power/SNR are evaluated.
    analysis_freq = 20.0
    carrier_freq = 2000.0
    dt = 0.025
    noise_type = "ou"
    # Parameters held constant across the scan.
    const_params = dict(
        sim_time=5000e3,
        f_values=[carrier_freq, carrier_freq + analysis_freq],
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

    # Scanned axes:
    target_stim_dVms = np.linspace(0, 1, 11)   # mV
    noise_level_Vms = np.linspace(3, 15, 25)   # mV

    rerun_scan = False
    welch_segments = 4
    use_welch = True
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
            target_stim_dVms, noise_level_Vms, const_params,
            analysis_freq=analysis_freq, use_welch=use_welch,
            welch_segments=welch_segments,
            freq_resolution=freq_resolution,
            band_halfwidth=band_halfwidth,
            ref_bandwidth=ref_bandwidth,)

        # Store the matrices (with their axes) for later reuse.
        os.makedirs("results/param_scan", exist_ok=True)
        np.savez(f"results/param_scan/param_scan_matrices_dt{dt}{welch_addon}.npz",
                 target_stim_dVms=target_stim_dVms,
                 noise_level_Vms=noise_level_Vms,
                 firing_rate=firing_rate,
                 fr_power=fr_power,
                 fr_SNR=fr_SNR,
                 ref_bandwidth=ref_bandwidth,
                 analysis_freq=analysis_freq,
                 fr_z_score=fr_z_score,)
    else:
        # Load the matrices (and their axes) from the previously saved scan.
        matrix_path = f"results/param_scan/param_scan_matrices_dt{dt}{welch_addon}.npz"
        with np.load(matrix_path) as data:
            target_stim_dVms = data["target_stim_dVms"]
            noise_level_Vms = data["noise_level_Vms"]
            firing_rate = data["firing_rate"]
            fr_power = data["fr_power"]
            fr_SNR = data["fr_SNR"]
            analysis_freq = float(data["analysis_freq"],)
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
                    target_stim_dVms, noise_level_Vms,
                    analysis_freq=analysis_freq,
                    ref_bandwidth=ref_bandwidth,
                    save_name=f"param_scan_dt{dt}{welch_addon}.pdf")
