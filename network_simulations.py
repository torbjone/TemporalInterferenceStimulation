"""Feed-forward network built on the single-cell model in
single_cell_simulations.py.

A population A of ``pop_size`` identical iaf_psc_alpha neurons (same parameters
as the single-cell example) is driven by a common temporal-interference
stimulus and by individual noise (same amplitude, independent realisation per
neuron). Population A is not internally connected. Every A neuron projects onto
a single postsynaptic neuron B through excitatory synapses whose delays are
drawn from a distribution and whose alpha-current time constant (tau_syn) is a
parameter. Neuron B has its own noise level.

Recorded and returned:
    - membrane potential (Vm) and spike times of a subset of population A,
    - membrane potential (Vm) and spike times of neuron B.

Results are cached to a pickle file exactly as in run_single_cell_simulation:
a run whose saved parameters match the request is loaded instead of rerun;
pass ``force_rerun=True`` to always rerun. Set ``save_Vm=False`` to keep only
spike times (Vm traces dominate the file size for long simulations)."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from single_cell_simulations import (
    find_I,
    find_noise_std,
    find_spike_rate,
    return_freq_and_psd,
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
                           noise_level_Vm_B=4.0,
                           syn_weight=20.0,
                           syn_delay=1.5,
                           syn_delay_std=0.0,
                           tau_syn=2.0,
                           C=100,
                           V_th=-50.0,
                           E_m=-60.0,
                           tau_m=10,
                           n_record_A=5,
                           pop_rate_bin_size=1.0,
                           n_threads=1,
                           seed=2,
                           resolution=0.1,
                           sim_name="network_test",
                           save_dir="results",
                           force_rerun=False,
                           save_Vm=True):
    """Simulate the A -> B feed-forward network and return a results dict.

    Parameters
    ----------
    pop_size : int
        Number of neurons N in population A.
    sim_time : float
        Simulation time (ms), excluding the initial `cutoff` transient.
    f_values : list
        Carrier frequencies (Hz) of the shared stimulus. Two values give a
        temporal-interference drive beating at their difference frequency.
    target_stim_dVm : float
        Subthreshold Vm-deviation amplitude (mV) each sine alone would produce;
        converted to an ac_generator current, shared by all A neurons.
    noise_level_Vm : float
        Vm-noise standard deviation (mV) for each A neuron. Same amplitude for
        all, but each neuron gets its own independent noise realisation.
    noise_level_Vm_B : float
        Vm-noise standard deviation (mV) for neuron B.
    syn_weight : float
        Weight (pA) of every A -> B synapse.
    syn_delay, syn_delay_std : float
        Mean and standard deviation (ms) of the A -> B synaptic delay
        distribution (normal, clipped to >= resolution and rounded to grid).
    tau_syn : float
        Alpha-current synaptic time constant (ms) of neuron B's incoming
        excitatory synapses (tau_syn_ex).
    C, V_th, E_m, tau_m : float
        Neuron parameters, shared by population A and neuron B.
    n_record_A : int
        Number of A neurons (evenly spaced across the population) whose Vm and
        spikes are recorded.
    pop_rate_bin_size : float
        Bin width (ms) used to compute the full population-A firing rate (the
        summed spike rate of all N neurons) that drives neuron B.
    n_threads : int
        Number of local (shared-memory) threads NEST uses to parallelise the
        simulation across CPU cores. Because the per-thread RNG streams change
        with the thread count, the spike realisation depends on `n_threads`;
        it is therefore part of the cached parameter set.
    seed : int
        Master seed for the NEST kernel and for the delay draw.
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
        "noise_level_Vm_B": noise_level_Vm_B,
        "syn_weight": syn_weight,
        "syn_delay": syn_delay,
        "syn_delay_std": syn_delay_std,
        "tau_syn": tau_syn,
        "C": C,
        "V_th": V_th,
        "E_m": E_m,
        "tau_m": tau_m,
        "n_record_A": n_record_A,
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

    # --- Population A: pop_size identical neurons, not interconnected ---------
    neuron_params = {
        "V_th": V_th,
        "V_m": E_m,
        "C_m": C,
        "V_reset": E_m,
        "E_L": E_m,
        "I_e": 0.0,
        "tau_m": tau_m,
    }
    pop_A = nest.Create("iaf_psc_alpha", pop_size, params=neuron_params)

    # --- Neuron B: same neuron parameters, with a synaptic time constant ------
    # tau_syn_ex sets the alpha-current time constant of B's incoming (A -> B)
    # excitatory synapses. A neurons receive only current inputs, so tau_syn
    # is irrelevant for them and is left at its default.
    neuron_B = nest.Create("iaf_psc_alpha",
                           params=dict(neuron_params, tau_syn_ex=tau_syn))

    # --- Independent noise, common amplitude, per A neuron --------------------
    # A single noise_generator sends the SAME signal to all its targets, so
    # independent noise requires one device per neuron (each device draws its
    # own stream from the kernel RNG). Same std -> same amplitude, different
    # realisation per neuron.
    I_noise_A = find_noise_std(noise_level_Vm, tau_m=tau_m, C_m=C, resolution=resolution)
    noise_A = nest.Create(
        "noise_generator", pop_size,
        params={"mean": 0.0, "std": I_noise_A, "dt": resolution})

    # --- Noise for neuron B ---------------------------------------------------
    I_noise_B = find_noise_std(noise_level_Vm_B, tau_m=tau_m, C_m=C, resolution=resolution)
    noise_B = nest.Create(
        "noise_generator",
        params={"mean": 0.0, "std": I_noise_B, "dt": resolution})

    # --- Shared temporal-interference stimulus --------------------------------
    # One ac_generator per carrier frequency, connected to every A neuron, so
    # the whole population sees the identical stimulus current.
    I_amp = [find_I(target_stim_dVm, f_, tau_m=tau_m, C_m=C) for f_ in f_values]
    sine = nest.Create(
        "ac_generator", len(f_values),
        params=[{"amplitude": a_, "frequency": f_} for a_, f_ in zip(I_amp, f_values)])

    # --- A -> B synaptic delays -----------------------------------------------
    rng = np.random.default_rng(seed)
    delays = _draw_delays(pop_size, syn_delay, syn_delay_std, resolution, rng)

    # --- Recorders ------------------------------------------------------------
    # Record Vm + spikes for an evenly-spaced subset of A, and for all of B.
    n_record_A = int(min(n_record_A, pop_size))
    record_idx = np.linspace(0, pop_size - 1, n_record_A, dtype=int)
    record_idx = np.unique(record_idx)
    rec_A = pop_A[record_idx.tolist()]

    mm_A = nest.Create("multimeter", params={
        "interval": resolution, "record_from": ["V_m"], "start": cutoff})
    sr_A = nest.Create("spike_recorder", params={"start": cutoff})

    # Spikes of the ENTIRE population A, to reconstruct the full population rate
    # that drives neuron B (all N neurons project to B, not just the subset).
    sr_A_all = nest.Create("spike_recorder", params={"start": cutoff})

    mm_B = nest.Create("multimeter", params={
        "interval": resolution, "record_from": ["V_m"], "start": cutoff})
    sr_B = nest.Create("spike_recorder", params={"start": cutoff})

    # --- Connections ----------------------------------------------------------
    # Shared stimulus and per-neuron noise onto population A.
    nest.Connect(sine, pop_A, "all_to_all")
    nest.Connect(noise_A, pop_A, "one_to_one")

    # Noise onto B.
    nest.Connect(noise_B, neuron_B)

    # A -> B, one connection per A neuron with its own delay.
    for src, d in zip(pop_A, delays):
        nest.Connect(src, neuron_B,
                     syn_spec={"weight": syn_weight, "delay": float(d)})

    # Recordings: subset of A (Vm + spikes), all of A (spikes only), and B.
    nest.Connect(mm_A, rec_A)
    nest.Connect(rec_A, sr_A)
    nest.Connect(pop_A, sr_A_all)
    nest.Connect(mm_B, neuron_B)
    nest.Connect(neuron_B, sr_B)

    # --- Simulate -------------------------------------------------------------
    nest.Simulate(sim_time + cutoff + 1)

    # --- Collect A subset -----------------------------------------------------
    rec_A_ids = rec_A.global_id
    rec_A_ids = np.atleast_1d(np.array(rec_A_ids))

    sr_A_events = sr_A.get("events")
    A_spike_times = {}
    for gid in rec_A_ids:
        mask = sr_A_events["senders"] == gid
        A_spike_times[int(gid)] = np.sort(sr_A_events["times"][mask] - cutoff)

    # Full population-A firing rate driving B: pool spikes from all N neurons
    # and bin. `population_rate` is the summed rate (spikes/s across the whole
    # population); divide by pop_size for the mean per-neuron rate.
    all_A_times = sr_A_all.get("events")["times"] - cutoff
    population_rate, t_bins = find_spike_rate(all_A_times, pop_rate_bin_size, sim_time)
    population_rate_times = t_bins[:-1]
    mean_population_rate = len(all_A_times) / sim_time * 1000  # spikes/s, whole pop

    results = {
        "pop_size": pop_size,
        "f_values": f_values,
        "target_stim_dVm": target_stim_dVm,
        "I_amp": I_amp,
        "noise_level_Vm": noise_level_Vm,
        "noise_level_Vm_B": noise_level_Vm_B,
        "I_noise_A": I_noise_A,
        "I_noise_B": I_noise_B,
        "syn_weight": syn_weight,
        "syn_delay": syn_delay,
        "syn_delay_std": syn_delay_std,
        "delays": delays,
        "tau_syn": tau_syn,
        "A": {
            "recorded_ids": rec_A_ids,
            "spike_times": A_spike_times,
            "population_rate": population_rate,
            "population_rate_times": population_rate_times,
            "population_rate_bin_size": pop_rate_bin_size,
            "mean_population_rate": mean_population_rate,
        },
        "B": {
            "spike_times": np.sort(sr_B.get("events")["times"] - cutoff),
        },
    }
    results["B"]["firing_rate"] = len(results["B"]["spike_times"]) / sim_time * 1000

    if save_Vm:
        # multimeter Vm arrays, one row per recorded A neuron (ordered as
        # rec_A_ids), and a single row for B, with a shared time axis.
        mm_A_events = mm_A.get("events")
        A_Vm = np.vstack([
            mm_A_events["V_m"][mm_A_events["senders"] == gid] for gid in rec_A_ids
        ]) if len(rec_A_ids) else np.empty((0, 0))
        B_Vm = mm_B.get("events")["V_m"]
        times = np.arange(A_Vm.shape[1] if A_Vm.size else len(B_Vm)) * resolution
        results["A"]["Vm"] = A_Vm
        results["A"]["times"] = times
        results["B"]["Vm"] = B_Vm
        results["B"]["times"] = times

    # --- Save (atomic) --------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"params": params, "results": results}, f)
    os.replace(tmp_path, save_path)
    print(f"Saved results to '{save_path}'")
    return results


def plot_network_results(results, sim_params, tlim=[5, 6], max_f=2000,
                         firing_rate_bin_size=0.1):
    """Plot the network output.

    Time-domain panels (as before): neuron B's Vm with spikes, the recorded
    subset of A as stacked Vm traces, and a spike raster of that subset plus B.

    Spectral panels (mirroring plot_single_cell_results, each with a full
    log-log spectrum and a zoom around the stimulus/beat frequency, annotated
    with the SNR there):
        - PSD of population A's TOTAL firing rate (all N cells, not just the
          plotted subset),
        - PSD of neuron B's membrane potential,
        - PSD of neuron B's firing rate.

    `tlim` is the time window (s) shown for the Vm traces; `max_f` the upper
    frequency (Hz) of the full spectra; `firing_rate_bin_size` the bin (ms)
    used for neuron B's firing-rate spectrum. The B-Vm spectrum requires the
    simulation to have been run with save_Vm=True."""
    V_th = sim_params["V_th"]
    sim_name = sim_params["sim_name"]
    sim_time = sim_params["sim_time"]

    # Stimulus/beat frequency at which the SNR is evaluated: the difference
    # frequency for a two-carrier (TI) drive, or the single carrier otherwise.
    stim_freqs = sim_params["f_values"]
    if len(stim_freqs) == 2:
        stim_freq = np.abs(stim_freqs[1] - stim_freqs[0])
    elif len(stim_freqs) == 1:
        stim_freq = stim_freqs[0]
    else:
        raise ValueError("f_values must be a list of length 1 or 2")

    A = results["A"]
    B = results["B"]
    rec_ids = np.atleast_1d(np.array(A["recorded_ids"]))
    n_A = len(rec_ids)

    # --- Spectra --------------------------------------------------------------
    # Population A total firing rate (computed over all N cells in the sim).
    freqs_pA, psd_pA = return_freq_and_psd(A["population_rate_times"], A["population_rate"])
    psd_pA = psd_pA[0]
    snr_pA = compute_SNR(freqs_pA, psd_pA, stim_freq)

    # Neuron B firing rate (bin the B spike train, then take its PSD).
    b_rate, b_bins = find_spike_rate(B["spike_times"], firing_rate_bin_size, sim_time)
    freqs_Bfr, psd_Bfr = return_freq_and_psd(b_bins, b_rate)
    psd_Bfr = psd_Bfr[0]
    snr_Bfr = compute_SNR(freqs_Bfr, psd_Bfr, stim_freq)

    # Neuron B membrane potential (only available when Vm was saved).
    have_B_vm = "Vm" in B
    if have_B_vm:
        freqs_Bvm, psd_Bvm = return_freq_and_psd(B["times"], B["Vm"])
        psd_Bvm = psd_Bvm[0]
        snr_Bvm = compute_SNR(freqs_Bvm, psd_Bvm, stim_freq)

    plt.close("all")
    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(5, 3, hspace=0.7, wspace=0.35,
                          left=0.07, right=0.98, top=0.96, bottom=0.05)

    # --- Time-domain panels (span all three columns) --------------------------
    # Neuron B membrane potential with spike markers.
    ax_B = fig.add_subplot(
        gs[0, :], xlabel="time (s)", ylabel=r"$V_{\rm m}$ (mV)",
        title=r"Neuron B - firing rate: {0:.2f} Hz".format(B["firing_rate"]))
    if have_B_vm:
        ax_B.plot(B["times"] / 1000, B["Vm"], 'k', lw=0.7)
        ax_B.set_xlim(tlim)
    for t in B["spike_times"]:
        ax_B.vlines(x=t / 1000, ymin=V_th, ymax=V_th + 5, color="r")

    # Recorded subset of population A, Vm traces offset vertically.
    ax_A = fig.add_subplot(
        gs[1, :], xlabel="time (s)", ylabel=r"$V_{\rm m}$ (mV), offset",
        title=f"Population A (subset of {n_A} of {results['pop_size']} neurons)")
    if "Vm" in A and A["Vm"].size:
        offset = 25.0
        for k, gid in enumerate(rec_ids):
            ax_A.plot(A["times"] / 1000, A["Vm"][k] + k * offset, lw=0.6)
        ax_A.set_xlim(tlim)

    # Spike raster for the recorded A subset plus B on top.
    ax_r = fig.add_subplot(gs[2, :], xlabel="time (s)", ylabel="neuron",
                           title="Spike raster (recorded A subset + B)")
    for k, gid in enumerate(rec_ids):
        st = A["spike_times"][int(gid)]
        ax_r.vlines(st / 1000, k + 0.6, k + 1.4, color="C0", lw=0.5)
    ax_r.vlines(B["spike_times"] / 1000, n_A + 0.6, n_A + 1.4, color="r", lw=0.5)
    ax_r.set_yticks(list(range(1, n_A + 1)) + [n_A + 1])
    ax_r.set_yticklabels([f"A{gid}" for gid in rec_ids] + ["B"])
    ax_r.set_xlim(tlim)

    # --- Spectral panels: full log-log (row 3) + zoom (row 4) -----------------
    def _plot_psd(col, freqs, psd, snr, ylabel, title, zoom_factor):
        mask = freqs < max_f
        arg = np.argmin(np.abs(freqs - stim_freq))

        ax_full = fig.add_subplot(gs[3, col], xlim=[1, max_f],
                                  xlabel="frequency (Hz)", ylabel=ylabel,
                                  title=f"{title}\nSNR = {snr:.2f}")
        ax_full.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')
        ax_full.loglog(freqs[mask], psd[mask], 'k')

        ax_zoom = fig.add_subplot(gs[4, col], xlim=[stim_freq - 10, stim_freq + 10],
                                  xlabel="frequency (Hz)", ylabel=ylabel,
                                  title=f"zoom @ {stim_freq:.0f} Hz")
        ax_zoom.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')
        ax_zoom.plot(freqs[mask], psd[mask], 'k')
        top = psd[arg] * zoom_factor
        if np.isfinite(top) and top > 0:
            ax_zoom.set_ylim(0, top)
        return ax_full, ax_zoom

    _plot_psd(0, freqs_pA, psd_pA, snr_pA, r"|rate$_A$|²/Hz",
              "Population A total firing rate", 1.2)
    if have_B_vm:
        _plot_psd(1, freqs_Bvm, psd_Bvm, snr_Bvm, r"|$V_{\rm m}$|²/Hz",
                  "Neuron B membrane potential", 5)
    else:
        ax = fig.add_subplot(gs[3, 1], title="Neuron B Vm\n(save_Vm=False)")
        ax.axis("off")
    _plot_psd(2, freqs_Bfr, psd_Bfr, snr_Bfr, "|firing rate|²/Hz",
              "Neuron B firing rate", 1.2)

    simplify_axes(fig.axes)
    out_name = f"{sim_name}_network_{int(sim_params['sim_time'] / 1000)}s.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved figure to '{out_name}'")
    return fig


if __name__ == "__main__":
    sim_params = dict(
        pop_size=10000,
        sim_time=5e3,
        f_values=[1000, 1020],
        target_stim_dVm=0.1,
        noise_level_Vm=4.0,
        noise_level_Vm_B=0.0,
        syn_weight=0.15,
        syn_delay=2,
        syn_delay_std=0.0,
        tau_syn=2.0,
        C=100,
        V_th=-50.,
        E_m=-60.,
        tau_m=10,
        n_record_A=5,
        pop_rate_bin_size=0.1,
        seed=2,
        resolution=0.1,
        n_threads=4,
        sim_name="network_test_weaker_0.1",
        force_rerun=False,
        save_Vm=True,
    )

    results = run_network_simulation(**sim_params)
    plot_network_results(results, sim_params)
