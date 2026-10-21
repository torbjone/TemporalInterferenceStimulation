"""Single-neuron simulation driven by spiking data from an unconnected population.

The spike trains produced by ``unconnected_pop_TI.py``
(``run_network_simulation``) are replayed into a single postsynaptic neuron B
through a balanced E/I synapse arrangement identical to
``network_simulations_ei.py``:

  - 80 % of the population neurons are treated as excitatory (weight +w),
  - 20 % as inhibitory (weight -inh_weight_factor * w).

With the default 80/20 split and ``inh_weight_factor=4`` the mean synaptic
drive onto B cancels, while any beat-frequency modulation in the population
activity survives.  Neuron B can optionally receive additional Gaussian noise.

Results are cached to a pickle file. Pass ``force_rerun=True`` to discard the
cache."""

import os
import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from matplotlib.colors import LogNorm

from single_cell_simulations import (
    find_I,
    find_noise_std,
    find_spike_rate,
    return_freq_and_psd,
    calculate_spectral_snr_and_zscore,
    compute_SNR,
    simplify_axes,
    mark_subplots,
    _apply_nb_style,
)
from unconnected_pop_TI import run_network_simulation as run_pop_simulation


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
                           noise_type="gaussian",
                           tau_ou=5.,
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
                           save_Vm=True,
                           save_all_spikes=False):
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
    noise_type : str
        "gaussian" (white noise, iaf_psc_alpha neurons) or "ou"
        (Ornstein-Uhlenbeck current with correlation time `tau_ou` ms). As in
        run_single_cell_simulation, the OU current is generated inside NEST
        by connecting the noise_generators to the filtered current port
        (receptor_type=1) of iaf_psc_exp neurons with tau_syn_ex = `tau_ou`,
        so it costs no extra memory.
    tau_ou : float
        OU correlation time (ms), only used for noise_type="ou".
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
    save_all_spikes : bool
        Also store the spike times and neuron indices (0 .. pop_size-1) of the
        whole population (results["E"]["all_spike_times"/"all_spike_ids"]),
        so that smaller sub-populations and shorter durations can be analysed
        afterwards (see analyze_pop_size_duration).
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
    if save_all_spikes:
        # Only added when used so existing cache files remain valid.
        params["save_all_spikes"] = True
    if noise_type == "ou":
        # Only added for OU noise so existing (gaussian) cache files remain valid.
        params.update({"noise_type": noise_type, "tau_ou": tau_ou,
                       "neuron_model": "iaf_psc_exp"})
    elif noise_type != "gaussian":
        raise ValueError(f"Unknown noise_type '{noise_type}'. "
                         f"Must be 'gaussian' or 'ou'.")

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
    if noise_type == "ou":
        # Identical to iaf_psc_alpha for the (unfiltered) sine input; the
        # exponential synaptic filter turns the noise into an OU current.
        neuron_params["tau_syn_ex"] = tau_ou
        pop_E = nest.Create("iaf_psc_exp", pop_size, params=neuron_params)
    else:
        pop_E = nest.Create("iaf_psc_alpha", pop_size, params=neuron_params)

    # --- Independent noise, common amplitude, per presynaptic neuron ----------
    # A single noise_generator sends the SAME signal to all its targets, so
    # independent noise requires one device per neuron (each device draws its
    # own stream from the kernel RNG). Same std -> same amplitude, different
    # realisation per neuron.
    if noise_type == "ou":
        I_noise, I_ou = find_noise_std(noise_level_Vm, tau_ou, tau_m=tau_m,
                                       C_m=C, resolution=resolution)
    else:
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
    sr_E_all = nest.Create("spike_recorder", params={"start": cutoff})


    # --- Connections ----------------------------------------------------------
    # Shared stimulus and per-neuron noise onto both populations.

    nest.Connect(sine, pop_E, "all_to_all")
    noise_syn_spec = {"receptor_type": 1} if noise_type == "ou" else None
    nest.Connect(noise_E, pop_E, "one_to_one", syn_spec=noise_syn_spec)

    # Recordings: subsets of E
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

        all_events = sr_all.get("events")
        all_times = all_events["times"] - cutoff
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
        if save_all_spikes:
            first_gid = int(np.min(np.atleast_1d(pop_E.global_id)))
            out["all_spike_times"] = all_times
            out["all_spike_ids"] = (all_events["senders"] - first_gid).astype(np.int32)
        return out

    E_res = _collect_pop(rec_E, sr_E, sr_E_all, mm_E)

    results = {
        "pop_size": pop_size,
        "f_values": f_values,
        "target_stim_dVm": target_stim_dVm,
        "I_amp": I_amp,
        "noise_level_Vm": noise_level_Vm,
        "I_noise": I_noise,
        "noise_type": noise_type,
        "E": E_res,
    }
    if noise_type == "ou":
        results.update({"tau_ou": tau_ou, "I_ou": I_ou})

    # --- Save (atomic) --------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"params": params, "results": results}, f)
    os.replace(tmp_path, save_path)
    print(f"Saved results to '{save_path}'")
    return results


def run_single_neuron_from_pop(
    # --- population (source) parameters ---
    pop_sim_params: dict,
    # --- neuron B parameters ---
    exc_fraction: float = 0.8,
    inh_weight_factor: float = 4.0,
    syn_weight: float = 20.0,
    syn_delay: float = 1.5,
    syn_delay_std: float = 0.0,
    tau_syn: float = 2.0,
    noise_level_Vm_B: float = 0.0,
    C: float = 100,
    V_th: float = -50.0,
    E_m: float = -60.0,
    tau_m: float = 10,
    # --- simulation / caching ---
    seed: int = 2,
    resolution: float = 0.1,
    sim_name: str = "single_neuron_from_pop",
    save_dir: str = "results",
    force_rerun: bool = False,
    save_Vm: bool = True,
):
    """Simulate a single neuron B driven by spike trains from an unconnected
    population and return a results dict.

    Parameters
    ----------
    pop_sim_params : dict
        Keyword arguments forwarded to ``unconnected_pop_TI.run_network_simulation``
        (with ``save_all_spikes=True`` added automatically). The population is
        run (or loaded from cache) first; its spike times are then replayed
        into neuron B.
    exc_fraction : float
        Fraction of the population treated as excitatory (default 0.8).
    inh_weight_factor : float
        Inhibitory weight multiplier: ``w_I = -inh_weight_factor * syn_weight``.
        With 80/20 split and factor 4 the mean input to B is balanced.
    syn_weight : float
        Excitatory synaptic weight onto B (pA, alpha-PSC peak).
    syn_delay, syn_delay_std : float
        Mean and std (ms) of the synaptic delay distribution.
        With std <= 0 all delays equal the mean.
    tau_syn : float
        Alpha-current synaptic time constant (ms) for both E and I inputs.
    noise_level_Vm_B : float
        Additional Gaussian noise on B expressed as Vm std (mV). 0 = no noise.
    C, V_th, E_m, tau_m : float
        Neuron B parameters.
    seed : int
        RNG seed for delay draws and neuron B's noise.
    resolution : float
        Simulation time step (ms).  Should match the population simulation.
    sim_name, save_dir, force_rerun, save_Vm : misc
        Caching / output controls.

    Returns
    -------
    dict with keys:
        ``B``      – spike_times, Vm (optional), times (optional), firing_rate
        ``pop``    – the full population results dict from unconnected_pop_TI
        ``params`` – all parameters that define this run
    """

    # ------------------------------------------------------------------
    # 1. Run (or load) the population simulation
    # ------------------------------------------------------------------
    pop_sim_params = dict(pop_sim_params, save_all_spikes=True)

    pop_results = run_pop_simulation(**pop_sim_params)

    pop_size = pop_results["pop_size"]
    sim_time = pop_sim_params["sim_time"]

    n_E = int(round(exc_fraction * pop_size))
    n_E = max(0, min(n_E, pop_size))
    n_I = pop_size - n_E
    inh_weight = -inh_weight_factor * syn_weight

    # ------------------------------------------------------------------
    # 2. Build the parameter dict for caching
    # ------------------------------------------------------------------
    params = {
        "pop_sim_name": pop_sim_params.get("sim_name", ""),
        "pop_size": pop_size,
        "sim_time": sim_time,
        "exc_fraction": exc_fraction,
        "inh_weight_factor": inh_weight_factor,
        "syn_weight": syn_weight,
        "syn_delay": syn_delay,
        "syn_delay_std": syn_delay_std,
        "tau_syn": tau_syn,
        "noise_level_Vm_B": noise_level_Vm_B,
        "C": C,
        "V_th": V_th,
        "E_m": E_m,
        "tau_m": tau_m,
        "seed": seed,
        "resolution": resolution,
        "sim_name": sim_name,
        "save_Vm": save_Vm,
    }

    save_path = os.path.join(save_dir, f"{sim_name}.pkl")

    if not force_rerun and os.path.isfile(save_path):
        try:
            with open(save_path, "rb") as f:
                saved = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(f"Could not load '{save_path}' ({e}) – rerunning.")
            saved = None

        if saved is not None:
            if saved.get("params") == params:
                print(f"Loading saved results from '{save_path}'")
                return saved["results"]
            print(f"Saved parameters differ – rerunning.")

    # ------------------------------------------------------------------
    # 3. Extract spike trains from the population results
    # ------------------------------------------------------------------
    E_data = pop_results["E"]
    all_spike_times = np.asarray(E_data["all_spike_times"])   # ms, relative to cutoff
    all_spike_ids   = np.asarray(E_data["all_spike_ids"])     # 0-based neuron index

    # Split by E / I: first n_E neurons are excitatory, the rest inhibitory.
    mask_E = all_spike_ids < n_E
    mask_I = ~mask_E

    # ------------------------------------------------------------------
    # 4. Set up NEST
    # ------------------------------------------------------------------
    import nest

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.INFO
    nest.SetKernelStatus({"print_time": True, "resolution": resolution})
    nest.rng_seed = seed

    cutoff = 1000  # ms – must match the population simulation

    # ------------------------------------------------------------------
    # 5. Neuron B
    # ------------------------------------------------------------------
    neuron_B = nest.Create(
        "iaf_psc_alpha",
        params={
            "V_th": V_th,
            "V_m": E_m,
            "C_m": C,
            "V_reset": E_m,
            "E_L": E_m,
            "I_e": 0.0,
            "tau_m": tau_m,
            "tau_syn_ex": tau_syn,
            "tau_syn_in": tau_syn,
        },
    )

    # ------------------------------------------------------------------
    # 6. Optional noise on B
    # ------------------------------------------------------------------
    if noise_level_Vm_B > 0.0:
        I_noise_B = find_noise_std(noise_level_Vm_B, tau_m=tau_m, C_m=C,
                                   resolution=resolution)
        noise_B = nest.Create(
            "noise_generator",
            params={"mean": 0.0, "std": I_noise_B, "dt": resolution},
        )
        nest.Connect(noise_B, neuron_B)
    else:
        I_noise_B = 0.0

    # ------------------------------------------------------------------
    # 7. Replay population spikes via spike_generator devices
    # ------------------------------------------------------------------
    # NEST's spike_generator expects spike times > 0 in the *simulation*
    # time frame (ms). The population spike times are already relative to
    # the cutoff (0 … sim_time ms). We shift them by `cutoff` so that the
    # replay starts after the same warm-up period.
    rng = np.random.default_rng(seed)

    def _make_delays(n):
        if syn_delay_std > 0:
            d = rng.normal(syn_delay, syn_delay_std, size=n)
        else:
            d = np.full(n, float(syn_delay))
        d = np.round(d / resolution) * resolution
        d = np.maximum(d, resolution)
        return d

    def _connect_population(neuron_ids, weight):
        """Create one spike_generator per neuron and connect it to B."""
        for nid in neuron_ids:
            spk = all_spike_times[all_spike_ids == nid] + cutoff  # shift to sim frame
            spk = np.sort(spk[spk > 0])
            sg = nest.Create("spike_generator",
                             params={"spike_times": list(spk)})
            delay = float(_make_delays(1)[0])
            nest.Connect(sg, neuron_B,
                         syn_spec={"weight": weight, "delay": delay})

    _connect_population(np.arange(n_E), syn_weight)
    _connect_population(np.arange(n_E, pop_size), inh_weight)

    # ------------------------------------------------------------------
    # 8. Recorders
    # ------------------------------------------------------------------
    mm_B = nest.Create("multimeter", params={
        "interval": resolution, "record_from": ["V_m"], "start": float(cutoff)})
    sr_B = nest.Create("spike_recorder", params={"start": float(cutoff)})

    nest.Connect(mm_B, neuron_B)
    nest.Connect(neuron_B, sr_B)

    # ------------------------------------------------------------------
    # 9. Simulate
    # ------------------------------------------------------------------
    nest.Simulate(sim_time + cutoff + 1)

    # ------------------------------------------------------------------
    # 10. Collect results
    # ------------------------------------------------------------------
    B_spike_times = np.sort(sr_B.get("events")["times"] - cutoff)
    B_firing_rate = len(B_spike_times) / sim_time * 1000

    B_res = {
        "spike_times": B_spike_times,
        "firing_rate": B_firing_rate,
    }

    if save_Vm:
        B_Vm = mm_B.get("events")["V_m"]
        B_times = np.arange(len(B_Vm)) * resolution
        B_res["Vm"] = B_Vm
        B_res["times"] = B_times

    results = {
        "B": B_res,
        "pop": pop_results,
        "n_E": n_E,
        "n_I": n_I,
        "exc_fraction": exc_fraction,
        "inh_weight_factor": inh_weight_factor,
        "syn_weight": syn_weight,
        "inh_weight": inh_weight,
        "syn_delay": syn_delay,
        "tau_syn": tau_syn,
        "noise_level_Vm_B": noise_level_Vm_B,
        "I_noise_B": I_noise_B,
        "params": params,
    }

    # ------------------------------------------------------------------
    # 11. Save (atomic)
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"params": params, "results": results}, f)
    os.replace(tmp_path, save_path)
    print(f"Saved results to '{save_path}'")
    return results


def plot_results(results, sim_params, tlim=(5, 6), max_f=2200,
                 firing_rate_bin_size=None, use_welch=True, welch_segments=10):
    """Plot neuron B's Vm trace, spike raster and firing-rate PSD.

    Parameters
    ----------
    results : dict
        Return value of ``run_single_neuron_from_pop``.
    sim_params : dict
        The ``pop_sim_params`` dict used for the population run (provides
        ``f_values``, ``sim_time``, ``V_th``, ``sim_name``).
    """
    f_values  = sim_params["f_values"]
    sim_time  = sim_params["sim_time"]
    V_th      = results["params"]["V_th"]
    sim_name  = results["params"]["sim_name"]

    if len(f_values) == 2:
        stim_freq = abs(f_values[1] - f_values[0])
    else:
        stim_freq = f_values[0]

    if firing_rate_bin_size is None:
        firing_rate_bin_size = results["params"]["resolution"]

    B = results["B"]
    have_Vm = "Vm" in B

    # Firing-rate PSD
    b_rate, b_bins = find_spike_rate(B["spike_times"], firing_rate_bin_size, sim_time)
    if use_welch:
        fs = 1000.0 / firing_rate_bin_size
        freqs_fr, psd_fr = welch(b_rate, fs=fs,
                                 nperseg=len(b_rate) // welch_segments,
                                 noverlap=(len(b_rate) // welch_segments) // 2)
    else:
        freqs_fr, psd_fr = return_freq_and_psd(b_bins, b_rate)
        psd_fr = psd_fr[0]

    snr_res = calculate_spectral_snr_and_zscore(freqs_fr, psd_fr, stim_freq, freq_window=5)
    snr_fr  = snr_res["snr_peak_ratio"]
    print(f"Neuron B firing rate : {B['firing_rate']:.2f} Hz")
    print(f"Firing-rate SNR @ {stim_freq:.0f} Hz: {snr_fr:.2f}  (z={snr_res['z_score']:.1f})")

    # Also compute population drive PSD for reference
    pop_E   = results["pop"]["E"]
    net_drive  = pop_E["population_rate"]
    net_times  = pop_E["population_rate_times"]
    print(pop_E)

    if use_welch:
        fs_pop = 1000.0 / firing_rate_bin_size
        freqs_pop, psd_pop = welch(net_drive, fs=fs_pop,
                                   nperseg=len(net_drive) // welch_segments,
                                   noverlap=(len(net_drive) // welch_segments) // 2)
    else:
        freqs_pop, psd_pop = return_freq_and_psd(net_times, net_drive)
        psd_pop = psd_pop[0]
    snr_pop = calculate_spectral_snr_and_zscore(freqs_pop, psd_pop, stim_freq, freq_window=5)
    print(f"Population drive SNR @ {stim_freq:.0f} Hz: {snr_pop['snr_peak_ratio']:.2f}")

    # Vm PSD
    if have_Vm:
        fs_vm = 1000.0 / results["params"]["resolution"]
        if use_welch:
            freqs_vm, psd_vm = welch(B["Vm"], fs=fs_vm,
                                     nperseg=len(B["Vm"]) // welch_segments,
                                     noverlap=(len(B["Vm"]) // welch_segments) // 2)
        else:
            freqs_vm, psd_vm = return_freq_and_psd(B["times"], B["Vm"])
            psd_vm = psd_vm[0]
        snr_vm = calculate_spectral_snr_and_zscore(freqs_vm, psd_vm, stim_freq, freq_window=5)
        print(f"Vm SNR @ {stim_freq:.0f} Hz: {snr_vm['snr_peak_ratio']:.2f}")

    # --- Figure ---
    plt.close("all")
    n_rows = 3
    fig = plt.figure(figsize=[10, 4])
    fig.subplots_adjust(hspace=0.55, left=0.05, right=0.97, top=0.95, bottom=0.05)

    ax_vm = fig.add_subplot(n_rows, 1, 1, xlim=tlim,
                                   ylabel=r"$V_{\rm m}$ (mV)", frameon=False, xticks=[], yticks=[])

    vm_norm = 10

    t_ = pop_E["population_rate_times"] / 1000
    tmask = (t_ > tlim[0]) & (t_ < tlim[1])

    t_ = t_[tmask]

    pre_psds = []

    for idx, id in enumerate(pop_E['recorded_ids']):
        vm_ = pop_E["Vm"][idx][tmask]
        vm_ -= np.mean(vm_)
        ax_vm.plot(t_, vm_ / vm_norm + 2 * idx, "k", lw=0.6)

        spike_times = pop_E['spike_times'][id]
        fr, t_bins = find_spike_rate(spike_times, firing_rate_bin_size, sim_time)
        if use_welch:
            fs_pre = 1000.0 / firing_rate_bin_size
            freqs_pre, psd_pre = welch(fr, fs=fs_pre,
                                       nperseg=len(fr) // welch_segments,
                                       noverlap=(len(fr) // welch_segments) // 2)
        else:
            freqs_pre, psd_pre = return_freq_and_psd(fr, t_bins)
            psd_pre = psd_pre[0]
        pre_psds.append(psd_pre)

        snr_pre = calculate_spectral_snr_and_zscore(freqs_pre, psd_pre, stim_freq, freq_window=5)

        print(f"Single-cell presynaptic drive SNR @ {stim_freq:.0f} Hz: {snr_pre['snr_peak_ratio']:.2f}")

    pre_psds = np.mean(pre_psds, axis=0)

    b_ = B["Vm"][tmask]
    b_ -= np.mean(b_)
    ax_vm.plot(t_, b_ / vm_norm - 4, 'r', lw=0.6)

    ax_vm.plot([tlim[-1] + 0.0025, tlim[-1] + 0.0025], [0, -1], c='k', clip_on=False)
    ax_vm.text(tlim[-1]+ 0.0025, -0.5, f"{vm_norm}\nmV" , ha="left", va="center")


    # ax = fig.add_subplot(n_rows, 1, 2);
    # ax.set_title(f"Neuron B Vm  (firing rate {B['firing_rate']:.2f} Hz, "
    #              f"w_E={results['syn_weight']:.1f} pA, "
    #              f"w_I={results['inh_weight']:.1f} pA)")
    # ax.set_xlabel("time (s)")
    # ax.set_ylabel(r"$V_{\rm m}$ (mV)")
    # ax.plot(B["times"] / 1000, B["Vm"], "k", lw=0.6)
    # ax.set_xlim(tlim)
    # for t in B["spike_times"]:
    #     ax.vlines(t / 1000, V_th, V_th + 5, color="r", lw=0.8)
    #
    # # Spike raster (just B)
    # ax = axes[row]; row += 1
    # ax.set_title("Neuron B – spike raster")
    # ax.set_xlabel("time (s)")
    # mask_rast = (B["spike_times"] / 1000 >= tlim[0]) & (B["spike_times"] / 1000 <= tlim[1])
    # ax.vlines(B["spike_times"][mask_rast] / 1000, 0.6, 1.4, color="r", lw=0.8)
    # ax.set_xlim(tlim)
    # ax.set_yticks([1]); ax.set_yticklabels(["B"])
    #
    # # Firing-rate PSD (full log-log)
    ax_pre_PSD = fig.add_subplot(3, 2, 3, title="presynaptic", xlabel="frequency (Hz)", ylabel="|firing rate|²/Hz")
    mask_f = (freqs_fr < max_f) & (freqs_fr > 1)
    # ax_pre_PSD.loglog(freqs_fr[mask_f], psd_pop[mask_f], "k")
    # for pre_psd in pre_psds:
    ax_pre_PSD.loglog(freqs_fr[mask_f], pre_psds[mask_f], "k")

    ax_fr_PSD = fig.add_subplot(3, 2, 4, title="postsynaptic", xlabel="frequency (Hz)", ylabel="|firing rate|²/Hz")
    ax_fr_PSD.loglog(freqs_fr[mask_f], psd_fr[mask_f], "k")

    # ax.axvline(stim_freq, lw=0.7, ls="--", color="gray")

    # # Firing-rate PSD (zoom)
    # ax = axes[row]; row += 1
    # ax.set_title(f"Firing-rate PSD zoom @ {stim_freq:.0f} Hz")
    # ax.set_xlabel("frequency (Hz)")
    # ax.set_ylabel("|firing rate|²/Hz")
    # ax.plot(freqs_fr[mask_f], psd_fr[mask_f], "k")
    # ax.set_xlim(stim_freq - 10, stim_freq + 10)
    # peak_val = psd_fr[np.argmin(np.abs(freqs_fr - stim_freq))]
    # if np.isfinite(peak_val) and peak_val > 0:
    #     ax.set_ylim(0, peak_val * 1.5)
    # ax.axvline(stim_freq, lw=0.7, ls="--", color="gray")

    simplify_axes(fig.axes)
    os.makedirs("figures", exist_ok=True)
    out_name = join("figures", f"{sim_name}_{int(sim_time / 1000)}s.png")
    plt.savefig(out_name, dpi=150)

    print(f"Saved figure to '{out_name}'")
    return fig

# ---------------------------------------------------------------------------
if __name__ == "__main__":

    dt = 0.025
    carrier_f = 2000
    beat_f = 20
    noise_type = "ou"
    noise_level_Vm = 5.0
    target_stim_dVm = 0.1

    # Parameters for the unconnected population (source of spikes)
    pop_sim_params = dict(
        pop_size=10000,
        sim_time=1000e3,
        f_values=[carrier_f, carrier_f + beat_f],
        target_stim_dVm=target_stim_dVm,
        noise_level_Vm=noise_level_Vm,
        noise_type=noise_type,
        tau_ou=5.0,
        C=100,
        V_th=-50.0,
        E_m=-60.0,
        tau_m=10,
        n_record=5,
        pop_rate_bin_size=dt,
        seed=2,
        resolution=dt,
        n_threads=4,
        sim_name=f"pop_size_duration_scan_{noise_type}_{noise_level_Vm}_{target_stim_dVm}_{carrier_f}_{beat_f}_dt{dt}",
        force_rerun=False,
        save_Vm=True,
    )


    w = 4.5
    g = 4.0
    sim_name = (f"single_neuron_from_pop_{noise_type}_{noise_level_Vm}_{target_stim_dVm}_{carrier_f}_{beat_f}_dt{dt}"
                f"w{w}_g{g}")

    results = run_single_neuron_from_pop(
        pop_sim_params=pop_sim_params,
        exc_fraction=0.8,
        inh_weight_factor=g,
        syn_weight=w,
        syn_delay=1.5,
        syn_delay_std=0.0,
        tau_syn=2.0,
        noise_level_Vm_B=0.0,
        C=100,
        V_th=-50.0,
        E_m=-60.0,
        tau_m=10,
        seed=3,
        resolution=dt,
        sim_name=sim_name,
        save_dir="results",
        force_rerun=False,
        save_Vm=True,
    )

    plot_results(results, pop_sim_params, tlim=(3.8, 4.2), use_welch=True,
                 welch_segments=4)
    #plt.show()
