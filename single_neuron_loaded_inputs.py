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
import sys
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


def _single_neuron_params_match(saved_params, current_params):
    """Check if saved parameters match current parameters, with backwards compatibility."""
    if saved_params == current_params:
        return True
    for k, v in current_params.items():
        if k in ("pop_sim_name_E", "pop_sim_name_I"):
            saved_val = saved_params.get(k, saved_params.get("pop_sim_name"))
            if saved_val != v:
                return False
        elif k == "pop_sim_name":
            saved_val = saved_params.get("pop_sim_name", saved_params.get("pop_sim_name_E"))
            if saved_val != v:
                return False
        else:
            if saved_params.get(k) != v:
                return False
    return True


def run_single_neuron_from_pop(
    # --- population (source) parameters ---
    pop_sim_params: dict = None,
    pop_sim_params_E: dict = None,
    pop_sim_params_I: dict = None,
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
    pop_sim_params : dict, optional
        Keyword arguments forwarded to ``unconnected_pop_TI.run_network_simulation``
        (with ``save_all_spikes=True`` added automatically). Used for both E and I
        if ``pop_sim_params_E`` and ``pop_sim_params_I`` are not separately given.
    pop_sim_params_E, pop_sim_params_I : dict, optional
        Separate parameter dicts for the excitatory and inhibitory populations,
        allowing E and I to be loaded from different simulations (e.g. different
        target_stim_dVm values).
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
        ``pop``    – the excitatory population results dict (for backwards compatibility)
        ``pop_E``  – the full E population results dict
        ``pop_I``  – the full I population results dict
        ``params`` – all parameters that define this run
    """

    # ------------------------------------------------------------------
    # 1. Run (or load) the population simulation(s)
    # ------------------------------------------------------------------
    if pop_sim_params_E is None and pop_sim_params_I is None:
        if pop_sim_params is None:
            raise ValueError("Must provide either pop_sim_params or pop_sim_params_E / pop_sim_params_I.")
        pop_sim_params_E = pop_sim_params
        pop_sim_params_I = pop_sim_params
    elif pop_sim_params_E is None:
        pop_sim_params_E = pop_sim_params
    elif pop_sim_params_I is None:
        pop_sim_params_I = pop_sim_params

    pop_sim_params_E = dict(pop_sim_params_E, save_all_spikes=True)
    pop_sim_params_I = dict(pop_sim_params_I, save_all_spikes=True)

    if pop_sim_params_E == pop_sim_params_I:
        pop_results_E = run_pop_simulation(**pop_sim_params_E)
        pop_results_I = pop_results_E
    else:
        pop_results_E = run_pop_simulation(**pop_sim_params_E)
        pop_results_I = run_pop_simulation(**pop_sim_params_I)

    pop_size_E = pop_results_E["pop_size"]
    pop_size_I = pop_results_I["pop_size"]
    sim_time = pop_sim_params_E["sim_time"]

    n_E = int(round(exc_fraction * pop_size_E))
    n_E = max(0, min(n_E, pop_size_E))
    n_I = int(round((1.0 - exc_fraction) * pop_size_I))
    n_I = max(0, min(n_I, pop_size_I))
    inh_weight = -inh_weight_factor * syn_weight

    # ------------------------------------------------------------------
    # 2. Build the parameter dict for caching
    # ------------------------------------------------------------------
    params = {
        "pop_sim_name_E": pop_sim_params_E.get("sim_name", ""),
        "pop_sim_name_I": pop_sim_params_I.get("sim_name", ""),
        "pop_sim_name": pop_sim_params_E.get("sim_name", ""),
        "pop_size": pop_size_E,
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
            if _single_neuron_params_match(saved.get("params", {}), params):
                print(f"Loading saved results from '{save_path}'")
                return saved["results"]
            print(f"Saved parameters differ – rerunning.")

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
    rng = np.random.default_rng(seed)

    def _make_delays(n):
        if syn_delay_std > 0:
            d = rng.normal(syn_delay, syn_delay_std, size=n)
        else:
            d = np.full(n, float(syn_delay))
        d = np.round(d / resolution) * resolution
        d = np.maximum(d, resolution)
        return d

    def _connect_population(pop_data, neuron_ids, weight):
        """Create one spike_generator per neuron in neuron_ids and connect it to B."""
        all_spike_times = np.asarray(pop_data["all_spike_times"])
        all_spike_ids   = np.asarray(pop_data["all_spike_ids"])
        for nid in neuron_ids:
            spk = all_spike_times[all_spike_ids == nid] + cutoff  # shift to sim frame
            spk = np.sort(spk[spk > 0])
            sg = nest.Create("spike_generator",
                             params={"spike_times": list(spk)})
            delay = float(_make_delays(1)[0])
            nest.Connect(sg, neuron_B,
                         syn_spec={"weight": weight, "delay": delay})

    e_ids = np.arange(n_E)
    if pop_results_E is pop_results_I or pop_size_I >= n_E + n_I:
        i_ids = np.arange(n_E, n_E + n_I)
    else:
        i_ids = np.arange(n_I)

    _connect_population(pop_results_E["E"], e_ids, syn_weight)
    _connect_population(pop_results_I["E"], i_ids, inh_weight)

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

    def _strip_all_spikes(pop_res):
        """Return a shallow copy of pop_res with all_spike_times and all_spike_ids removed to avoid duplicating huge spike arrays."""
        if pop_res is None:
            return None
        res_copy = dict(pop_res)
        if "E" in res_copy and isinstance(res_copy["E"], dict):
            e_copy = dict(res_copy["E"])
            e_copy.pop("all_spike_times", None)
            e_copy.pop("all_spike_ids", None)
            res_copy["E"] = e_copy
        return res_copy

    pop_res_E_saved = _strip_all_spikes(pop_results_E)
    pop_res_I_saved = _strip_all_spikes(pop_results_I)

    results = {
        "B": B_res,
        "pop": pop_res_E_saved,
        "pop_E": pop_res_E_saved,
        "pop_I": pop_res_I_saved,
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
    fig = plt.figure(figsize=[10, 6])
    fig.subplots_adjust(hspace=0.65, left=0.08, right=0.97, top=0.95, bottom=0.1)

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


    # ax = fig.add_subplot(n_rows, 1, 2)
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
    # ax = fig.add_subplot(n_rows, 1, 3)
    # ax.set_title("Neuron B – spike raster")
    # ax.set_xlabel("time (s)")
    # mask_rast = (B["spike_times"] / 1000 >= tlim[0]) & (B["spike_times"] / 1000 <= tlim[1])
    # ax.vlines(B["spike_times"][mask_rast] / 1000, 0.6, 1.4, color="r", lw=0.8)
    # ax.set_xlim(tlim)
    # ax.set_yticks([1]); ax.set_yticklabels(["B"])

    # # Firing-rate PSD (full log-log)
    ax_pre_PSD = fig.add_subplot(3, 2, 3, title="presynaptic", xlabel="frequency (Hz)", ylabel="|firing rate|²/Hz")
    mask_f = (freqs_fr < max_f) & (freqs_fr > 1)
    # ax_pre_PSD.loglog(freqs_fr[mask_f], psd_pop[mask_f], "k")
    # for pre_psd in pre_psds:
    ax_pre_PSD.loglog(freqs_fr[mask_f], pre_psds[mask_f], "k")

    ax_fr_PSD = fig.add_subplot(3, 2, 4, title="postsynaptic", xlabel="frequency (Hz)", ylabel="|firing rate|²/Hz")
    ax_fr_PSD.loglog(freqs_fr[mask_f], psd_fr[mask_f], "k")


    # # # Vm PSD (full log-log)
    # ax_pre_Vm_PSD = fig.add_subplot(3, 2, 5, title="presynaptic", xlabel="frequency (Hz)", ylabel="|Vm|²/Hz")
    # mask_f = (freqs_fr < max_f) & (freqs_fr > 1)
    # # ax_pre_PSD.loglog(freqs_fr[mask_f], psd_pop[mask_f], "k")
    # # for pre_psd in pre_psds:
    # ax_pre_PSD.loglog(freqs_fr[mask_f], pre_psds[mask_f], "k")

    ax_Vm_PSD = fig.add_subplot(3, 2, 6, title="postsynaptic", xlabel="frequency (Hz)", ylabel="|Vm|²/Hz")
    mask_f = (freqs_vm < max_f) & (freqs_vm > 1)
    ax_Vm_PSD.loglog(freqs_vm[mask_f], psd_vm[mask_f], "k")



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


def _postsynaptic_snr_from_results(results, stim_freq, firing_rate_bin_size=None,
                                   use_welch=True, welch_segments=4,
                                   freq_window=5):
    """Return the postsynaptic (neuron B) SNR at `stim_freq`.

    The SNR is computed from the firing-rate PSD of B, exactly as in
    ``plot_results``.  If the membrane potential was saved, the Vm SNR is
    returned as well.
    """
    params = results["params"]
    sim_time = params["sim_time"]
    if firing_rate_bin_size is None:
        firing_rate_bin_size = params["resolution"]

    B = results["B"]

    b_rate, b_bins = find_spike_rate(B["spike_times"], firing_rate_bin_size, sim_time)
    if use_welch:
        fs = 1000.0 / firing_rate_bin_size
        freqs_fr, psd_fr = welch(b_rate, fs=fs,
                                 nperseg=len(b_rate) // welch_segments,
                                 noverlap=(len(b_rate) // welch_segments) // 2)
    else:
        freqs_fr, psd_fr = return_freq_and_psd(b_bins, b_rate)
        psd_fr = psd_fr[0]

    snr_fr = calculate_spectral_snr_and_zscore(freqs_fr, psd_fr, stim_freq,
                                               freq_window=freq_window)

    out = {
        "snr_fr": snr_fr["snr_peak_ratio"],
        "z_fr": snr_fr["z_score"],
        "firing_rate": B["firing_rate"],
        "pre_fr_E": np.nan,
        "pre_fr_I": np.nan,
        "snr_Vm": np.nan,
        "z_Vm": np.nan,
    }

    # Presynaptic E and I firing rates (average per cell in Hz)
    pop_E = results.get("pop_E")
    if pop_E is None:
        pop_E = results.get("pop")
    if pop_E is not None and isinstance(pop_E, dict):
        pop_size_E = pop_E.get("pop_size", results.get("params", {}).get("pop_size", 10000))
        e_pop_data = pop_E.get("E", pop_E)
        if isinstance(e_pop_data, dict) and "mean_population_rate" in e_pop_data:
            out["pre_fr_E"] = e_pop_data["mean_population_rate"] / pop_size_E if pop_size_E else np.nan

    pop_I = results.get("pop_I")
    if pop_I is None:
        pop_I = results.get("pop")
    if pop_I is not None and isinstance(pop_I, dict):
        pop_size_I = pop_I.get("pop_size", results.get("params", {}).get("pop_size", 10000))
        i_pop_data = pop_I.get("I")
        if i_pop_data is None:
            i_pop_data = pop_I.get("E", pop_I)
        if isinstance(i_pop_data, dict) and "mean_population_rate" in i_pop_data:
            out["pre_fr_I"] = i_pop_data["mean_population_rate"] / pop_size_I if pop_size_I else np.nan

    if "Vm" in B:
        fs_vm = 1000.0 / params["resolution"]
        if use_welch:
            freqs_vm, psd_vm = welch(B["Vm"], fs=fs_vm,
                                     nperseg=len(B["Vm"]) // welch_segments,
                                     noverlap=(len(B["Vm"]) // welch_segments) // 2)
        else:
            freqs_vm, psd_vm = return_freq_and_psd(B["times"], B["Vm"])
            psd_vm = psd_vm[0]
        snr_vm = calculate_spectral_snr_and_zscore(freqs_vm, psd_vm, stim_freq,
                                                   freq_window=freq_window)
        out["snr_Vm"] = snr_vm["snr_peak_ratio"]
        out["z_Vm"] = snr_vm["z_score"]

    return out


def build_pop_sim_params(
    target_stim_dVm=0.15,
    noise_type="ou",
    noise_level_Vm=5.0,
    carrier_f=2000,
    beat_f=20,
    dt=0.025,
    pop_size=10000,
    sim_time=1000e3,
    tau_ou=5.0,
    C=100,
    V_th=-50.0,
    E_m=-60.0,
    tau_m=10,
    n_record=5,
    seed=2,
    n_threads=4,
    force_rerun=False,
    save_Vm=True,
):
    """Build a standard pop_sim_params dictionary for unconnected_pop_TI."""
    return dict(
        pop_size=pop_size,
        sim_time=sim_time,
        f_values=[carrier_f, carrier_f + beat_f],
        target_stim_dVm=target_stim_dVm,
        noise_level_Vm=noise_level_Vm,
        noise_type=noise_type,
        tau_ou=tau_ou,
        C=C,
        V_th=V_th,
        E_m=E_m,
        tau_m=tau_m,
        n_record=n_record,
        pop_rate_bin_size=dt,
        seed=seed,
        resolution=dt,
        n_threads=n_threads,
        sim_name=f"pop_size_duration_scan_{noise_type}_{noise_level_Vm}_{target_stim_dVm}_{carrier_f}_{beat_f}_dt{dt}",
        force_rerun=force_rerun,
        save_Vm=save_Vm,
    )


def run_single_neuron_selective_stim(
    target_stim_dVm=0.15,
    stim_target="both",  # "both", "E", or "I"
    unstim_dVm=0.0,
    noise_type="ou",
    noise_level_Vm=5.0,
    carrier_f=2000,
    beat_f=20,
    dt=0.025,
    w=4.5,
    g=4.0,
    exc_fraction=0.8,
    syn_delay=1.5,
    syn_delay_std=0.0,
    tau_syn=2.0,
    noise_level_Vm_B=0.0,
    C=100,
    V_th=-50.0,
    E_m=-60.0,
    tau_m=10,
    seed=2,
    pop_size=10000,
    sim_time=1000e3,
    tau_ou=5.0,
    n_record=5,
    n_threads=4,
    force_rerun=False,
    save_Vm=True,
    save_dir="results",
):
    """Run a single-neuron simulation where either both E and I, only E,
    or only I receive the TI stimulation `target_stim_dVm`.

    The unstimulated subpopulation receives `unstim_dVm` (default 0.0).

    Parameters
    ----------
    target_stim_dVm : float
        Stimulation amplitude (target_stim_dVm) for stimulated population(s).
    stim_target : {"both", "E", "I"}
        Which population receives the stimulation:
        - "both": Both E and I cells receive target_stim_dVm.
        - "E" (or "E_only"): Only E cells receive target_stim_dVm, I cells receive unstim_dVm.
        - "I" (or "I_only"): Only I cells receive target_stim_dVm, E cells receive unstim_dVm.
    unstim_dVm : float
        Stimulation amplitude for the unstimulated population (default 0.0).
    """
    if stim_target in ("both", "all", "EI", None):
        dVm_E = target_stim_dVm
        dVm_I = target_stim_dVm
        sim_name = (f"single_neuron_from_pop_{noise_type}_{noise_level_Vm}_{target_stim_dVm}_"
                    f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}")
    elif stim_target in ("E", "E_only", "exc"):
        dVm_E = target_stim_dVm
        dVm_I = unstim_dVm
        sim_name = (f"single_neuron_from_pop_E_{noise_type}_{noise_level_Vm}_{target_stim_dVm}_"
                    f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}")
    elif stim_target in ("I", "I_only", "inh"):
        dVm_E = unstim_dVm
        dVm_I = target_stim_dVm
        sim_name = (f"single_neuron_from_pop_I_{noise_type}_{noise_level_Vm}_{target_stim_dVm}_"
                    f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}")
    else:
        raise ValueError(f"Unknown stim_target: {stim_target}. Must be 'both', 'E', or 'I'.")

    pop_sim_params_E = build_pop_sim_params(
        target_stim_dVm=dVm_E, noise_type=noise_type, noise_level_Vm=noise_level_Vm,
        carrier_f=carrier_f, beat_f=beat_f, dt=dt, pop_size=pop_size, sim_time=sim_time,
        tau_ou=tau_ou, C=C, V_th=V_th, E_m=E_m, tau_m=tau_m, n_record=n_record,
        seed=seed, n_threads=n_threads, force_rerun=force_rerun, save_Vm=save_Vm,
    )
    pop_sim_params_I = build_pop_sim_params(
        target_stim_dVm=dVm_I, noise_type=noise_type, noise_level_Vm=noise_level_Vm,
        carrier_f=carrier_f, beat_f=beat_f, dt=dt, pop_size=pop_size, sim_time=sim_time,
        tau_ou=tau_ou, C=C, V_th=V_th, E_m=E_m, tau_m=tau_m, n_record=n_record,
        seed=seed, n_threads=n_threads, force_rerun=force_rerun, save_Vm=save_Vm,
    )

    return run_single_neuron_from_pop(
        pop_sim_params_E=pop_sim_params_E,
        pop_sim_params_I=pop_sim_params_I,
        exc_fraction=exc_fraction,
        inh_weight_factor=g,
        syn_weight=w,
        syn_delay=syn_delay,
        syn_delay_std=syn_delay_std,
        tau_syn=tau_syn,
        noise_level_Vm_B=noise_level_Vm_B,
        C=C,
        V_th=V_th,
        E_m=E_m,
        tau_m=tau_m,
        seed=seed,
        resolution=dt,
        sim_name=sim_name,
        save_dir=save_dir,
        force_rerun=force_rerun,
        save_Vm=save_Vm,
    )


def find_simulated_target_stim_dVm(noise_type="ou", noise_level_Vm=5.0,
                                   carrier_f=2000, beat_f=20, dt=0.025,
                                   w=4.5, g=4.0, stim_target="both", save_dir="results"):
    """Scan `save_dir` for already simulated single-neuron runs and return a
    sorted list of ``(target_stim_dVm, file_path)`` tuples.

    Only runs matching the given stimulation / synapse parameters are
    returned.
    `stim_target` can be:
      - "both" (or "all", None): matches single_neuron_from_pop_{noise_type}_{noise_level_Vm}_{dVm}_{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}.pkl
      - "E" (or "E_only"): matches single_neuron_from_pop_E_{noise_type}_{noise_level_Vm}_{dVm}_{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}.pkl
      - "I" (or "I_only"): matches single_neuron_from_pop_I_{noise_type}_{noise_level_Vm}_{dVm}_{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}.pkl
    """
    import glob
    import re

    if stim_target in ("both", "all", "EI", None):
        pattern = (f"single_neuron_from_pop_{noise_type}_{noise_level_Vm}_*_"
                   f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}.pkl")
        regex = re.compile(
            rf"single_neuron_from_pop_{re.escape(str(noise_type))}_"
            rf"{re.escape(str(noise_level_Vm))}_([0-9.eE+-]+)_"
            rf"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}\.pkl$")
    elif stim_target in ("E", "E_only", "exc"):
        pattern = (f"single_neuron_from_pop_E_{noise_type}_{noise_level_Vm}_*_"
                   f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}.pkl")
        regex = re.compile(
            rf"single_neuron_from_pop_E_{re.escape(str(noise_type))}_"
            rf"{re.escape(str(noise_level_Vm))}_([0-9.eE+-]+)_"
            rf"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}\.pkl$")
    elif stim_target in ("I", "I_only", "inh"):
        pattern = (f"single_neuron_from_pop_I_{noise_type}_{noise_level_Vm}_*_"
                   f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}.pkl")
        regex = re.compile(
            rf"single_neuron_from_pop_I_{re.escape(str(noise_type))}_"
            rf"{re.escape(str(noise_level_Vm))}_([0-9.eE+-]+)_"
            rf"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}\.pkl$")
    else:
        raise ValueError(f"Unknown stim_target '{stim_target}'. Must be 'both', 'E', or 'I'.")

    found = []
    for path in glob.glob(os.path.join(save_dir, pattern)):
        m = regex.search(os.path.basename(path))
        if m is None:
            continue
        try:
            found.append((float(m.group(1)), path))
        except ValueError:
            continue
    return sorted(found)


def collect_snr_vs_target_stim_dVm(noise_type="ou", noise_level_Vm=5.0,
                                   carrier_f=2000, beat_f=20, dt=0.025,
                                   w=4.5, g=4.0, stim_target="both", save_dir="results",
                                   firing_rate_bin_size=None, use_welch=True,
                                   welch_segments=4, freq_window=5,
                                   cache_name=None, force_recompute=False):
    """Compute the postsynaptic SNR for every already simulated
    ``target_stim_dVm`` and return ``(dVm_values, snr_dict)``.

    The result files are several GB each, so they are loaded (and released)
    one at a time and the resulting SNR values are cached in a small pickle.
    """
    import gc

    target_tag = f"_{stim_target}" if stim_target not in ("both", "all", "EI", None) else ""
    if cache_name is None:
        cache_name = (f"snr_vs_target_stim_dVm{target_tag}_{noise_type}_{noise_level_Vm}_"
                      f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}"
                      f"_welch{use_welch}:{welch_segments}.pkl")
    cache_path = os.path.join(save_dir, cache_name)

    cache = {}
    if not force_recompute and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(f"Could not load SNR cache '{cache_path}' ({e}) – recomputing.")
            cache = {}

    stim_freq = beat_f
    runs = find_simulated_target_stim_dVm(noise_type=noise_type,
                                          noise_level_Vm=noise_level_Vm,
                                          carrier_f=carrier_f, beat_f=beat_f,
                                          dt=dt, w=w, g=g,
                                          stim_target=stim_target, save_dir=save_dir)
    if not runs:
        raise FileNotFoundError(
            f"No simulated single-neuron runs found in '{save_dir}' for stim_target={stim_target}, "
            f"noise_type={noise_type}, noise_level_Vm={noise_level_Vm}, "
            f"carrier_f={carrier_f}, beat_f={beat_f}, dt={dt}, w={w}, g={g}")

    print(f"[{stim_target}] Found {len(runs)} simulated target_stim_dVm values: "
          f"{[d for d, _ in runs]}")

    updated = False
    for dVm, path in runs:
        if dVm in cache and not force_recompute and "pre_fr_E" in cache[dVm]:
            continue
        print(f"Loading '{path}' (this may take a while) …")
        with open(path, "rb") as f:
            saved = pickle.load(f)
        cache[dVm] = _postsynaptic_snr_from_results(
            saved["results"], stim_freq,
            firing_rate_bin_size=firing_rate_bin_size, use_welch=use_welch,
            welch_segments=welch_segments, freq_window=freq_window)
        updated = True
        print(f"  target_stim_dVm={dVm}: postsynaptic SNR = "
              f"{cache[dVm]['snr_fr']:.2f} (z={cache[dVm]['z_fr']:.1f}), "
              f"post rate = {cache[dVm]['firing_rate']:.2f} Hz, "
              f"pre E rate = {cache[dVm]['pre_fr_E']:.2f} Hz, "
              f"pre I rate = {cache[dVm]['pre_fr_I']:.2f} Hz")
        del saved
        gc.collect()

    if updated:
        os.makedirs(save_dir, exist_ok=True)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp_path, cache_path)
        print(f"Saved SNR summary to '{cache_path}'")

    dVm_values = sorted(d for d, _ in runs)
    return dVm_values, cache


def plot_snr_vs_target_stim_dVm(noise_type="ou", noise_level_Vm=5.0,
                                carrier_f=2000, beat_f=20, dt=0.025,
                                w=4.5, g=4.0, stim_target="compare", save_dir="results",
                                firing_rate_bin_size=None, use_welch=True,
                                welch_segments=4, freq_window=5,
                                plot_Vm=False, force_recompute=False):
    """Plot the postsynaptic firing rate SNR, postsynaptic firing rate, and
    presynaptic E/I population firing rates at the beat frequency as a function
    of ``target_stim_dVm`` for all already simulated values.

    Parameters
    ----------
    stim_target : str or list of str
        "both": plot only runs where both E and I are stimulated.
        "E": plot only runs where E cells are stimulated.
        "I": plot only runs where I cells are stimulated.
        "compare" or ["both", "E", "I"]: plot all available conditions on the same figure.
    """
    if stim_target == "compare":
        targets = ["both", "E", "I"]
    elif isinstance(stim_target, (list, tuple)):
        targets = list(stim_target)
    else:
        targets = [stim_target]

    styles = {
        "both": dict(color="k", marker="o", linestyle="-", label="E + I stimulated"),
        "E": dict(color="tab:blue", marker="s", linestyle="--", label="Only E stimulated"),
        "I": dict(color="tab:red", marker="^", linestyle="-.", label="Only I stimulated"),
        "E_only": dict(color="tab:blue", marker="s", linestyle="--", label="Only E stimulated"),
        "I_only": dict(color="tab:red", marker="^", linestyle="-.", label="Only I stimulated"),
    }

    target_data = {}
    for target in targets:
        try:
            dVm_values, snr = collect_snr_vs_target_stim_dVm(
                noise_type=noise_type, noise_level_Vm=noise_level_Vm,
                carrier_f=carrier_f, beat_f=beat_f, dt=dt, w=w, g=g,
                stim_target=target, save_dir=save_dir,
                firing_rate_bin_size=firing_rate_bin_size,
                use_welch=use_welch, welch_segments=welch_segments,
                freq_window=freq_window, force_recompute=force_recompute)
            target_data[target] = (dVm_values, snr)
        except FileNotFoundError as err:
            print(f"Skipping {target}: {err}")
            continue

    if not target_data:
        raise FileNotFoundError(f"No simulations found for targets {targets} in '{save_dir}'.")

    valid_targets = list(target_data.keys())
    n_rows = len(valid_targets)
    n_cols = 4

    plt.close("all")
    fig = plt.figure(figsize=[3.2 * n_cols, 3.0 * n_rows])
    fig.subplots_adjust(left=0.07, bottom=0.09, right=0.98, top=0.93,
                        wspace=0.4, hspace=0.5)

    # One shared y-axis per column (first row defines the scale).
    col_axes = {1: None, 2: None, 3: None, 4: None}

    for row, target in enumerate(valid_targets):
        dVm_values, snr = target_data[target]
        style = styles.get(target, dict(marker="o", label=target))
        target_label = style.get("label", target)

        # Column 1: presynaptic rates
        ax_pre = fig.add_subplot(n_rows, n_cols, row * n_cols + 1,
                                 xlabel=r"target stim d$V_{\rm m}$ (mV)",
                                 ylabel="presynaptic rate (Hz)",
                                 sharey=col_axes[1])
        col_axes[1] = col_axes[1] or ax_pre
        ax_pre.set_title(target_label, fontsize=9)
        pre_fr_E = [snr[d].get("pre_fr_E", np.nan) for d in dVm_values]
        pre_fr_I = [snr[d].get("pre_fr_I", np.nan) for d in dVm_values]
        ax_pre.plot(dVm_values, pre_fr_E, color="tab:blue", marker="s", linestyle="-", label="Presynaptic E")
        ax_pre.plot(dVm_values, pre_fr_I, color="tab:red", marker="^", linestyle="--", label="Presynaptic I")
        ax_pre.legend(frameon=False, fontsize=8)

        # Column 2: postsynaptic rate
        ax_fr = fig.add_subplot(n_rows, n_cols, row * n_cols + 2,
                                xlabel=r"target stim d$V_{\rm m}$ (mV)",
                                ylabel="post-synaptic rate (Hz)",
                                sharey=col_axes[2])
        col_axes[2] = col_axes[2] or ax_fr
        ax_fr.set_title(target_label, fontsize=9)
        fr = [snr[d]["firing_rate"] for d in dVm_values]
        ax_fr.plot(dVm_values, fr, color="k", marker="o", linestyle="-")

        # Column 3: SNR peak ratio
        ax_snr = fig.add_subplot(n_rows, n_cols, row * n_cols + 3,
                                 xlabel=r"target stim d$V_{\rm m}$ (mV)",
                                 ylabel="SNR (peak ratio)",
                                 yscale="log",
                                 sharey=col_axes[3])
        col_axes[3] = col_axes[3] or ax_snr
        ax_snr.set_title(target_label, fontsize=9)
        snr_fr = [snr[d]["snr_fr"] for d in dVm_values]
        ax_snr.plot(dVm_values, snr_fr, color="k", marker="o", linestyle="-")

        # Column 4: z-score
        ax_z = fig.add_subplot(n_rows, n_cols, row * n_cols + 4,
                               xlabel=r"target stim d$V_{\rm m}$ (mV)",
                               ylabel="z-score",
                               yscale="log",
                               sharey=col_axes[4])
        col_axes[4] = col_axes[4] or ax_z
        ax_z.set_title(target_label, fontsize=9)
        z_fr = [snr[d]["z_fr"] for d in dVm_values]
        ax_z.plot(dVm_values, z_fr, color="k", marker="o", linestyle="-")

        # Keep the y tick labels in every row, even though the y-axes are shared.
        for ax_ in (ax_pre, ax_fr, ax_snr, ax_z):
            ax_.tick_params(labelleft=True)

    simplify_axes(fig.axes)
    mark_subplots(fig.axes, ypos=1.05)
    os.makedirs("figures", exist_ok=True)
    target_tag = "_".join(valid_targets) if len(valid_targets) > 1 else valid_targets[0]
    out_name = join("figures",
                    f"snr_vs_target_stim_dVm_{target_tag}_{noise_type}_{noise_level_Vm}_"
                    f"{carrier_f}_{beat_f}_dt{dt}w{w}_g{g}.pdf")
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

    w = 4.5

    # Set to True to only plot the postsynaptic SNR as a function of
    # target_stim_dVm for all already simulated values (no new simulation).
    snr_vs_dVm_only = True

    if snr_vs_dVm_only:
        for g in [4.0]:
            for stim_target in ["compare"]:
                try:
                    plot_snr_vs_target_stim_dVm(
                        noise_type=noise_type,
                        noise_level_Vm=noise_level_Vm,
                        carrier_f=carrier_f,
                        beat_f=beat_f,
                        dt=dt,
                        w=w,
                        g=g,
                        stim_target=stim_target,
                        save_dir="results",
                        use_welch=True,
                        welch_segments=4,
                    )
                except FileNotFoundError as err:
                    print(f"Skipping plot for g={g}, stim_target={stim_target}: {err}")
        raise SystemExit


    for g in [4.0]:
        for stim_target in ["E", "both"]:
            for target_stim_dVm in [0.0]:

                # Run single-neuron simulation with selective stimulation:
                results = run_single_neuron_selective_stim(
                    target_stim_dVm=target_stim_dVm,
                    stim_target=stim_target,
                    noise_type=noise_type,
                    noise_level_Vm=noise_level_Vm,
                    carrier_f=carrier_f,
                    beat_f=beat_f,
                    dt=dt,
                    w=w,
                    g=g,
                    exc_fraction=0.8,
                    syn_delay=1.5,
                    syn_delay_std=0.0,
                    tau_syn=2.0,
                    noise_level_Vm_B=0.0,
                    C=100,
                    V_th=-50.0,
                    E_m=-60.0,
                    tau_m=10,
                    seed=2,
                    pop_size=10000,
                    sim_time=1000e3,
                    force_rerun=False,
                    save_Vm=True,
                    save_dir="results",
                )

                pop_sim_params = build_pop_sim_params(
                    target_stim_dVm=target_stim_dVm,
                    noise_type=noise_type,
                    noise_level_Vm=noise_level_Vm,
                    carrier_f=carrier_f,
                    beat_f=beat_f,
                    dt=dt,
                )

                plot_results(results, pop_sim_params, tlim=(3.8, 4.2), use_welch=True,
                             welch_segments=4)


    for g in [4.0]:
        for stim_target in ["compare"]:
            try:
                plot_snr_vs_target_stim_dVm(
                    noise_type=noise_type,
                    noise_level_Vm=noise_level_Vm,
                    carrier_f=carrier_f,
                    beat_f=beat_f,
                    dt=dt,
                    w=w,
                    g=g,
                    stim_target=stim_target,
                    save_dir="results",
                    use_welch=True,
                    welch_segments=4,
                )
            except FileNotFoundError as err:
                print(f"Skipping plot for g={g}, stim_target={stim_target}: {err}")
