"""Feed-forward network with a balanced excitatory/inhibitory presynaptic
population, built on the single-cell model in single_cell_simulations.py.

This is a variant of network_simulations.py with an unconnected presynaptic
population. All neurons are driven by the *same* common
temporal-interference stimulus (the sine) and by individual noise (same
amplitude, independent realisation per neuron).

Recorded and returned:
    - membrane potential (Vm) and spike times of a subset of E and of I,
    - full population firing rates of E and of I,
    - membrane potential (Vm) and spike times of neuron B.

Results are cached to a pickle file exactly as in run_single_cell_simulation:
a run whose saved parameters match the request is loaded instead of rerun;
pass ``force_rerun=True`` to always rerun. Set ``save_Vm=False`` to keep only
spike times (Vm traces dominate the file size for long simulations)."""

import os
import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from single_cell_simulations import (
    find_I,
    find_noise_std,
    find_spike_rate,
    return_freq_and_psd,
    calculate_spectral_snr_and_zscore,
    compute_SNR,
    simplify_axes,
)


def _draw_delays(n, mean, std, resolution, rng):
    """Draw `n` synaptic delays (ms) from a normal distribution, clipped to be
    at least one time step and rounded onto the resolution grid (both required
    by NEST). With std <= 0 every delay equals `mean` (rounded to the grid)."""
    if std > 0:
        delays = rng.normal(mean, std, size=n)
    else:
        delays = np.full(n, float(mean))
    # Round to the resolution grid and enforce delay >= resolution.
    delays = np.round(delays / resolution) * resolution
    delays = np.maximum(delays, resolution)
    return delays


def run_network_simulation(pop_size=100,
                           sim_time=10e3,
                           f_values=[1000, 1020],
                           target_stim_dVm=0.3,
                           noise_level_Vm=4.0,
                           C=100,
                           V_th=-50.0,
                           E_m=-60.0,
                           tau_m=10,
                           n_record=5,
                           pop_rate_bin_size=0.1,
                           n_threads=1,
                           seed=2,
                           resolution=0.1,
                           sim_name="network_ei_test",
                           save_dir="results",
                           force_rerun=False,
                           save_Vm=True):
    """Simulate the (E, I) -> B feed-forward network and return a results dict.

    Parameters
    ----------
    pop_size : int
        Total number of presynaptic neurons. Split into an excitatory
        population E of ``round(exc_fraction * pop_size)`` neurons and an
        inhibitory population I holding the remainder.
    sim_time : float
        Simulation time (ms), excluding the initial `cutoff` transient.
    f_values : list
        Carrier frequencies (Hz) of the shared stimulus. Two values give a
        temporal-interference drive beating at their difference frequency.
    target_stim_dVm : float
        Subthreshold Vm-deviation amplitude (mV) each sine alone would produce;
        converted to an ac_generator current, shared by all E and I neurons.
    noise_level_Vm : float
        Vm-noise standard deviation (mV) for each presynaptic neuron. Same
        amplitude for all, but each neuron gets its own independent realisation.
    C, V_th, E_m, tau_m : float
        Neuron parameters, shared by populations E, I and neuron B.
    n_record, n_record_I : int
        Number of neurons, evenly spaced across their population,
        whose Vm and spikes are recorded.
    pop_rate_bin_size : float
        Bin width (ms) used to compute the full population firing rates of E
        and of I that drive neuron B.
    n_threads : int
        Number of local (shared-memory) threads NEST uses to parallelise the
        simulation across CPU cores. Because the per-thread RNG streams change
        with the thread count, the spike realisation depends on `n_threads`;
        it is therefore part of the cached parameter set.
    seed : int
        Master seed for the NEST kernel and for the delay draws.
    resolution : float
        Simulation time step (ms).
    sim_name, save_dir, force_rerun, save_Vm
        Caching / output controls, as in run_single_cell_simulation.
    """

    # All input parameters that define the simulation. Saved together with the
    # results so that a cached run can be verified against the current request.
    params = {
        "pop_size": pop_size,
        "sim_time": sim_time,
        "f_values": f_values,
        "target_stim_dVm": target_stim_dVm,
        "noise_level_Vm": noise_level_Vm,
        "C": C,
        "V_th": V_th,
        "E_m": E_m,
        "tau_m": tau_m,
        "n_record": n_record,
        "pop_rate_bin_size": pop_rate_bin_size,
        "n_threads": n_threads,
        "seed": seed,
        "resolution": resolution,
        "sim_name": sim_name,
        "save_Vm": save_Vm,
    }

    save_path = os.path.join(save_dir, f"{sim_name}.pkl")

    # Try to load a previous run instead of simulating again.
    if not force_rerun and os.path.isfile(save_path):
        try:
            with open(save_path, "rb") as f:
                saved = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            # A truncated/corrupt cache file (e.g. an interrupted write from a
            # previous run) - discard it and rerun the simulation.
            print(f"Could not load saved results from '{save_path}' ({e}) - "
                  f"rerunning simulation.")
            saved = None

        if saved is not None:
            if saved.get("params") == params:
                print(f"Loading saved results from '{save_path}'")
                return saved["results"]

            print(f"Saved parameters in '{save_path}' differ from the requested "
                  f"parameters - rerunning simulation.")

    import nest

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.INFO
    nest.SetKernelStatus({"print_time": True})
    # local_num_threads must be set before any nodes are created. Threading
    # (shared memory) parallelises the run across `n_threads` CPU cores within
    # this single process, so all recordings remain locally retrievable.
    nest.SetKernelStatus({"resolution": resolution, "local_num_threads": n_threads})
    nest.rng_seed = seed

    cutoff = 1000  # ms of initial transient discarded from the recordings

    # --- Presynaptic populations E and I: identical neurons, not connected ----
    neuron_params = {
        "V_th": V_th,
        "V_m": E_m,
        "C_m": C,
        "V_reset": E_m,
        "E_L": E_m,
        "I_e": 0.0,
        "tau_m": tau_m,
    }
    pop_E = nest.Create("iaf_psc_alpha", pop_size, params=neuron_params)

    # --- Independent noise, common amplitude, per presynaptic neuron ----------
    # A single noise_generator sends the SAME signal to all its targets, so
    # independent noise requires one device per neuron (each device draws its
    # own stream from the kernel RNG). Same std -> same amplitude, different
    # realisation per neuron.
    I_noise = find_noise_std(noise_level_Vm, tau_m=tau_m, C_m=C, resolution=resolution)
    noise_E = nest.Create(
        "noise_generator", pop_size,
        params={"mean": 0.0, "std": I_noise, "dt": resolution})


    # --- Shared temporal-interference stimulus --------------------------------
    # One ac_generator per carrier frequency, connected to every E and I
    # neuron, so both populations see the identical stimulus current.
    I_amp = [find_I(target_stim_dVm, f_, tau_m=tau_m, C_m=C) for f_ in f_values]
    sine = nest.Create(
        "ac_generator", len(f_values),
        params=[{"amplitude": a_, "frequency": f_} for a_, f_ in zip(I_amp, f_values)])


    # --- Recorders ------------------------------------------------------------
    # Record Vm + spikes for an evenly-spaced subset of E and of I, plus B.
    def _record_subset(pop, n_record):
        if pop is None:
            return pop, np.array([], dtype=int)
        n_pop = len(pop)
        n_record = int(min(n_record, n_pop))
        idx = np.unique(np.linspace(0, n_pop - 1, n_record, dtype=int))
        return pop[idx.tolist()], idx

    rec_E, _ = _record_subset(pop_E, n_record)

    mm_E = nest.Create("multimeter", params={
        "interval": resolution, "record_from": ["V_m"], "start": cutoff})
    sr_E = nest.Create("spike_recorder", params={"start": cutoff})

    # Spikes of the ENTIRE populations, to reconstruct the full population rates
    # that drive neuron B (all E and I neurons project to B, not just subsets).
    sr_E_all = nest.Create("spike_recorder", params={"start": cutoff})


    # --- Connections ----------------------------------------------------------
    # Shared stimulus and per-neuron noise onto both populations.

    nest.Connect(sine, pop_E, "all_to_all")
    nest.Connect(noise_E, pop_E, "one_to_one")

    # Recordings: subsets of E and I (Vm + spikes), all of E and I (spikes
    # only), and B.

    nest.Connect(mm_E, rec_E)
    nest.Connect(rec_E, sr_E)
    nest.Connect(pop_E, sr_E_all)

    # --- Simulate -------------------------------------------------------------
    nest.Simulate(sim_time + cutoff + 1)

    # --- Collect a presynaptic population -------------------------------------
    def _collect_pop(rec, sr, sr_all, mm):
        """Assemble the per-population results dict from its recorders."""
        if rec is None:
            return {
                "recorded_ids": np.array([], dtype=int),
                "spike_times": {},
                "population_rate": np.zeros(0),
                "population_rate_times": np.zeros(0),
                "population_rate_bin_size": pop_rate_bin_size,
                "mean_population_rate": 0.0,
            }
        rec_ids = np.atleast_1d(np.array(rec.global_id))
        sr_events = sr.get("events")
        spike_times = {}
        for gid in rec_ids:
            mask = sr_events["senders"] == gid
            spike_times[int(gid)] = np.sort(sr_events["times"][mask] - cutoff)

        all_times = sr_all.get("events")["times"] - cutoff
        pop_rate, t_bins = find_spike_rate(all_times, pop_rate_bin_size, sim_time)
        out = {
            "recorded_ids": rec_ids,
            "spike_times": spike_times,
            "population_rate": pop_rate,
            "population_rate_times": t_bins[:-1],
            "population_rate_bin_size": pop_rate_bin_size,
            # spikes/s summed over the whole population.
            "mean_population_rate": len(all_times) / sim_time * 1000,
        }
        if save_Vm:
            mm_events = mm.get("events")
            Vm = np.vstack([
                mm_events["V_m"][mm_events["senders"] == gid] for gid in rec_ids
            ]) if len(rec_ids) else np.empty((0, 0))
            out["Vm"] = Vm
        return out

    E_res = _collect_pop(rec_E, sr_E, sr_E_all, mm_E)

    results = {
        "pop_size": pop_size,
        "f_values": f_values,
        "target_stim_dVm": target_stim_dVm,
        "I_amp": I_amp,
        "noise_level_Vm": noise_level_Vm,
        "I_noise": I_noise,
        "E": E_res,
    }



    # --- Save (atomic) --------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"params": params, "results": results}, f)
    os.replace(tmp_path, save_path)
    print(f"Saved results to '{save_path}'")
    return results


def plot_network_results(results, sim_params,
                         tlim=[5, 6], max_f=2000,
                         firing_rate_bin_size=None, use_welch=True, welch_segments=10):
    """Plot the network output.

    Time-domain panels: the recorded subsets of E stacked Vm traces,
    and a spike raster of those subsets.

    Spectral panels (mirroring plot_single_cell_results, each with a full
    log-log spectrum and a zoom around the stimulus/beat frequency, annotated
    with the SNR there):
        - PSD of the NET population spikes
          firing rate, i.e. the signal B integrates),


    `tlim` is the time window (s) shown for the Vm traces; `max_f` the upper
    frequency (Hz) of the full spectra; `firing_rate_bin_size` the bin (ms)
    used for neuron B's firing-rate spectrum. The B-Vm spectrum requires the
    simulation to have been run with save_Vm=True."""
    V_th = sim_params["V_th"]
    sim_name = sim_params["sim_name"]
    sim_time = sim_params["sim_time"]

    if firing_rate_bin_size is None:
        firing_rate_bin_size = sim_params["resolution"]

    # Stimulus/beat frequency at which the SNR is evaluated: the difference
    # frequency for a two-carrier (TI) drive, or the single carrier otherwise.
    stim_freqs = sim_params["f_values"]
    if len(stim_freqs) == 2:
        stim_freq = np.abs(stim_freqs[1] - stim_freqs[0])
    elif len(stim_freqs) == 1:
        stim_freq = stim_freqs[0]
    else:
        raise ValueError("f_values must be a list of length 1 or 2")

    print(results)

    E = results["E"]
    rec_ids_E = np.atleast_1d(np.array(E["recorded_ids"]))
    n_E_rec = len(rec_ids_E)

    # --- Spectra --------------------------------------------------------------
    # Net presynaptic drive onto B: the weighted sum of the full E and I
    # population rates (same bins). This is the current-like signal B actually
    # integrates; with balanced input its mean ~ 0 while the beat-frequency
    # modulation survives.

    net_drive = E["population_rate"]
    net_times = E["population_rate_times"]


    if use_welch:
        fs_fr = 1000. / firing_rate_bin_size
        print(firing_rate_bin_size)
        freqs_net, psd_net = welch(net_drive, fs=fs_fr, nperseg=len(net_drive) // welch_segments,
                                  noverlap=(len(net_drive) // welch_segments) // 2)

    else:
        freqs_net, psd_net = return_freq_and_psd(net_times, net_drive)
        psd_net = psd_net[0]


    # snr_net_old = compute_SNR(freqs_net, psd_net, stim_freq)
    snr_net_old = compute_SNR(freqs_net, psd_net, stim_freq)

    snr_net_res = calculate_spectral_snr_and_zscore(freqs_net, psd_net, stim_freq, freq_window=5)

    print("Old SNR: ", snr_net_old)
    print("New calc: ", snr_net_res)

    snr_net_E = snr_net_res['snr_peak_ratio']

    plt.close("all")
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.7, wspace=0.35,
                          left=0.07, right=0.98, top=0.96, bottom=0.05)

    # --- Time-domain panels (span all three columns) --------------------------
    # Neuron B membrane potential with spike markers.

    # Recorded subsets of E (blue) and I (red), Vm traces offset vertically.
    ax_A = fig.add_subplot(
        gs[0, :], xlabel="time (s)", ylabel=r"$V_{\rm m}$ (mV), offset",
        title=f"Presynaptic E ({results['pop_size']}) neurons "
              f"- recorded subset")
    offset = 25.0
    k = 0
    if "Vm" in E and E["Vm"].size:
        for row in range(len(rec_ids_E)):
            ax_A.plot(E['population_rate_times']/ 1000, E["Vm"][row] + k * offset, 'C0', lw=0.6)
            k += 1

    ax_A.set_xlim(tlim)

    # Spike raster: E subset (blue), I subset (red), then B on top.
    ax_r = fig.add_subplot(gs[1, :], xlabel="time (s)", ylabel="neuron",
                           title="Spike raster (E subset, I subset, + B)")
    row = 0
    yticks, yticklabels = [], []
    for gid in rec_ids_E:
        st = E["spike_times"][int(gid)]
        ax_r.vlines(st / 1000, row + 0.6, row + 1.4, color="C0", lw=0.5)
        yticks.append(row + 1); yticklabels.append(f"E{gid}")
        row += 1

    ax_r.set_yticks(yticks)
    ax_r.set_yticklabels(yticklabels)
    ax_r.set_xlim(tlim)

    # --- Spectral panels: full log-log (row 3) + zoom (row 4) -----------------
    def _plot_psd(col, freqs, psd, snr, ylabel, title, zoom_factor):
        mask = freqs < max_f
        arg = np.argmin(np.abs(freqs - stim_freq))

        ax_full = fig.add_subplot(gs[2, :], xlim=[1, max_f],
                                  xlabel="frequency (Hz)", ylabel=ylabel,
                                  title=f"{title}\nSNR = {snr:.2f}")
        ax_full.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')

        ax_zoom = fig.add_subplot(gs[3, :], xlim=[stim_freq - 10, stim_freq + 10],
                                  xlabel="frequency (Hz)", ylabel=ylabel,
                                  title=f"zoom @ {stim_freq:.0f} Hz")
        ax_zoom.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')

        if len(psd.shape) == 1:
            ax_full.loglog(freqs[mask], psd[mask], 'k')
            ax_zoom.plot(freqs[mask], psd[mask], 'k')
        elif len(psd.shape) == 2:
            ax_full.loglog(freqs[mask], psd[0][mask], 'b')
            ax_zoom.plot(freqs[mask], psd[0][mask], 'b')

        else:
            raise RuntimeError("psd must be 1D or 2D")

        if len(psd.shape) == 1:
            top = psd[arg] * zoom_factor
            if np.isfinite(top) and top > 0:
                ax_zoom.set_ylim(0, top)
        elif len(psd.shape) == 2:
            top = psd[0][arg] * zoom_factor
            if np.isfinite(top) and top > 0:
                ax_zoom.set_ylim(0, top)
        else:
            raise RuntimeError("psd must be 1D or 2D")
        return ax_full, ax_zoom

    _plot_psd(0, freqs_net, np.array([psd_net]), snr_net_E, r"|firing rate|²/Hz",
              "Presynaptic drive", 1.2)


    simplify_axes(fig.axes)
    out_name = join("figures", f"{sim_name}_network_pop_size_{int(sim_params['sim_time'] / 1000)}s.png")
    plt.savefig(out_name, dpi=150)
    print(f"Saved figure to '{out_name}'")
    return fig


if __name__ == "__main__":

    weight_factor = 4.3
    target_stim_dVm = 0.03
    dt = 0.025
    carrier_f = 2000
    beat_f = 20

    sim_params = dict(
        pop_size=100000,
        sim_time=1000e3,
        f_values=[carrier_f, carrier_f + beat_f],
        target_stim_dVm=target_stim_dVm,
        noise_level_Vm=4.0,
        C=100,
        V_th=-50.,
        E_m=-60.,
        tau_m=10,
        n_record=5,
        pop_rate_bin_size=dt,
        seed=2,
        resolution=dt,
        n_threads=4,
        sim_name=f"network_pop_size_test_{target_stim_dVm}_{carrier_f}_{beat_f}",
        force_rerun=False,
        save_Vm=True,
    )

    results = run_network_simulation(**sim_params)
    plot_network_results(results, sim_params)
