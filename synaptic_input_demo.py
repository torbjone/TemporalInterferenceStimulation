"""Toy demo: a single iaf_psc_alpha neuron driven by synaptic input whose
rate is locked to two frequencies (by default 20 Hz and 2000 Hz).

Each input frequency f has its own sinusoidal_poisson_generator, i.e. an
inhomogeneous Poisson process with instantaneous rate

    r(t) = input_rate + input_ampl * sin(2*pi * f * t),

relayed by `n_inputs` parrot_neurons (one independent realisation each).

The input spike trains are generated ONCE (generate_input_spike_trains) and
replayed through spike_generators in every simulation, so the synaptic input
times are identical across all cases. The cases are:

    reference : very fast synapses, single fixed delay, no Vm noise, no spikes
    tau_syn   : realistic synaptic time constant
    delay_std : 1 ms std on the synaptic delays
    Vm noise  : 1 mV std noise on the membrane potential
    spikes    : spike threshold lowered to -50 mV (occasional spikes)

each differing from the reference by exactly one modification.

Plotted: the Vm traces of all cases in one panel (top row), and for each case
the PSD of the synaptic input current (left) and of the membrane potential
(right).

Results are cached to pickle files exactly as in run_single_cell_simulation.
"""

import os
import pickle
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from single_cell_simulations import (
    find_noise_std,
    calculate_spectral_snr_and_zscore,
    simplify_axes,
)

CUTOFF = 1000  # ms of initial transient discarded from the recordings


def _load_cache(save_path, params, force_rerun):
    """Return cached results from `save_path` if they were produced with
    `params`, otherwise None."""
    if force_rerun or not os.path.isfile(save_path):
        return None
    try:
        with open(save_path, "rb") as f:
            saved = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        # A truncated/corrupt cache file (e.g. an interrupted write from a
        # previous run) - discard it and rerun.
        print(f"Could not load saved results from '{save_path}' ({e}) - rerunning.")
        return None
    if saved.get("params") == params:
        print(f"Loading saved results from '{save_path}'")
        return saved["results"]
    print(f"Saved parameters in '{save_path}' differ from the requested "
          f"parameters - rerunning.")
    return None


def _save_cache(save_path, params, results):
    """Atomically pickle `params` and `results` to `save_path`."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"params": params, "results": results}, f)
    os.replace(tmp_path, save_path)
    print(f"Saved results to '{save_path}'")


def generate_input_spike_trains(sim_time=10e3,
                                input_freqs=(20.0, 2000.0),
                                input_rate=10.0,
                                input_ampl=1.0,
                                n_inputs=50,
                                seed=2,
                                resolution=0.1,
                                sim_name="synaptic_demo_inputs",
                                save_dir="results",
                                force_rerun=False):
    """Generate the frequency-locked presynaptic spike trains once.

    For every frequency in `input_freqs` a sinusoidal_poisson_generator drives
    `n_inputs` parrot_neurons, whose spikes are recorded over the full
    simulation (including the initial `CUTOFF` transient). Returns a dict
    with the list of spike-time arrays (ms, one per presynaptic input) and the
    input frequency each input is locked to. These trains are replayed in
    run_synaptic_input_simulation, so all simulations receive identical input.
    """
    params = {
        "sim_time": sim_time,
        "input_freqs": tuple(input_freqs),
        "input_rate": input_rate,
        "input_ampl": input_ampl,
        "n_inputs": n_inputs,
        "seed": seed,
        "resolution": resolution,
        "cutoff": CUTOFF,
    }
    save_path = os.path.join(save_dir, f"{sim_name}.pkl")
    cached = _load_cache(save_path, params, force_rerun)
    if cached is not None:
        return cached

    import nest

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.WARNING
    nest.SetKernelStatus({"resolution": resolution})
    nest.rng_seed = seed

    recorders = []
    for f in input_freqs:
        spg = nest.Create("sinusoidal_poisson_generator", params={
            "rate": input_rate,
            "amplitude": input_ampl,
            "frequency": f,
            "phase": 0.0,
        })
        parrots = nest.Create("parrot_neuron", n_inputs)
        nest.Connect(spg, parrots)  # all_to_all, independent trains per parrot
        rec = nest.Create("spike_recorder")
        nest.Connect(parrots, rec)
        recorders.append((parrots, rec))

    nest.Simulate(sim_time + CUTOFF + 1)

    spike_trains, train_freqs = [], []
    for f, (parrots, rec) in zip(input_freqs, recorders):
        ev = rec.get("events")
        for gid in parrots.tolist():
            spike_trains.append(np.sort(ev["times"][ev["senders"] == gid]))
            train_freqs.append(f)

    results = {"spike_trains": spike_trains, "train_freqs": np.array(train_freqs)}
    _save_cache(save_path, params, results)
    return results


def run_synaptic_input_simulation(inputs,
                                  input_params,
                                  sim_time=10e3,
                                  syn_weight=12.0,
                                  syn_delay=3.0,
                                  syn_delay_std=0.0,
                                  tau_syn=2.0,
                                  tau_syn_ref=2.0,
                                  conserve_charge=True,
                                  noise_level_Vm=0.0,
                                  C=100,
                                  V_th=0.0,
                                  E_m=-60.0,
                                  tau_m=10,
                                  seed=2,
                                  resolution=0.1,
                                  sim_name="synaptic_demo",
                                  save_dir="results",
                                  force_rerun=False):
    """Simulate a single neuron driven by the pre-generated input trains.

    Parameters
    ----------
    inputs : dict
        Output of generate_input_spike_trains. Each spike train is replayed by
        its own spike_generator, so input times are identical across runs.
    input_params : dict
        Parameters passed to generate_input_spike_trains (only used to verify
        cached results).
    sim_time : float
        Simulation time (ms), excluding the initial `CUTOFF` transient. Must
        match the duration the input trains were generated for.
    syn_weight : float
        Weight (pA) of every input -> neuron synapse, interpreted at the
        reference time constant `tau_syn_ref` (see `conserve_charge`).
    syn_delay : float
        Mean synaptic delay (ms) of the input -> neuron connections.
    syn_delay_std : float
        Standard deviation (ms) of the synaptic delays. 0 gives a single
        delay shared by all synapses. Delays are drawn (numpy, seeded by
        `seed`) from a normal distribution, clipped at `resolution` and
        rounded to the simulation grid.
    tau_syn : float
        Alpha-current synaptic time constant (ms) (tau_syn_ex).
    tau_syn_ref : float
        Reference synaptic time constant (ms) at which `syn_weight` is defined.
        Only used when `conserve_charge=True`.
    conserve_charge : bool
        If True, use w_eff = syn_weight * tau_syn_ref / tau_syn so the charge
        per synaptic event (proportional to weight * tau_syn in
        iaf_psc_alpha) - and hence the mean drive - is independent of
        `tau_syn`, isolating the synaptic low-pass filtering effect.
    noise_level_Vm : float
        Vm-noise standard deviation (mV) added via a noise_generator
        (0 disables it).
    C, V_th, E_m, tau_m : float
        Neuron parameters, as in run_single_cell_simulation.
    seed : int
        Seed for the NEST kernel (Vm noise) and for the delay distribution.
    resolution : float
        Simulation time step (ms).
    sim_name, save_dir, force_rerun
        Caching / output controls, as in run_single_cell_simulation.
    """
    params = {
        "input_params": input_params,
        "sim_time": sim_time,
        "syn_weight": syn_weight,
        "syn_delay": syn_delay,
        "syn_delay_std": syn_delay_std,
        "tau_syn": tau_syn,
        "tau_syn_ref": tau_syn_ref,
        "conserve_charge": conserve_charge,
        "noise_level_Vm": noise_level_Vm,
        "C": C,
        "V_th": V_th,
        "E_m": E_m,
        "tau_m": tau_m,
        "seed": seed,
        "resolution": resolution,
        "sim_name": sim_name,
    }
    save_path = os.path.join(save_dir, f"{sim_name}.pkl")
    cached = _load_cache(save_path, params, force_rerun)
    if cached is not None:
        return cached

    import nest

    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.WARNING
    nest.SetKernelStatus({"resolution": resolution})
    nest.rng_seed = seed

    # --- Postsynaptic neuron --------------------------------------------------
    neuron = nest.Create("iaf_psc_alpha", params={
        "V_th": V_th,
        "V_m": E_m,
        "C_m": C,
        "V_reset": E_m,
        "E_L": E_m,
        "I_e": 0.0,
        "tau_m": tau_m,
        "tau_syn_ex": tau_syn,
    })

    # --- Replayed synaptic input ----------------------------------------------
    spike_trains = inputs["spike_trains"]
    n_syn = len(spike_trains)
    sgs = nest.Create("spike_generator", n_syn)
    sgs.set([{"spike_times": st} for st in spike_trains])

    if syn_delay_std > 0:
        rng = np.random.default_rng(seed)
        delays = rng.normal(syn_delay, syn_delay_std, n_syn)
    else:
        delays = np.full(n_syn, float(syn_delay))
    delays = np.round(np.maximum(delays, resolution) / resolution) * resolution

    syn_weight_eff = syn_weight * tau_syn_ref / tau_syn if conserve_charge else syn_weight
    nest.Connect(sgs, neuron, "all_to_all",
                 syn_spec={"weight": syn_weight_eff,
                           "delay": delays.reshape(1, n_syn)})

    # --- Optional membrane noise ----------------------------------------------
    if noise_level_Vm > 0:
        I_noise = find_noise_std(noise_level_Vm, tau_m=tau_m, C_m=C, resolution=resolution)
        noise = nest.Create("noise_generator",
                            params={"mean": 0.0, "std": I_noise, "dt": resolution})
        nest.Connect(noise, neuron)
    else:
        I_noise = 0.0

    # --- Recorders ------------------------------------------------------------
    multimeter = nest.Create("multimeter", params={
        "interval": resolution, "record_from": ["V_m", "I_syn_ex"], "start": CUTOFF})
    spike_recorder = nest.Create("spike_recorder", params={"start": CUTOFF})
    nest.Connect(multimeter, neuron)
    nest.Connect(neuron, spike_recorder)

    # --- Simulate -------------------------------------------------------------
    nest.Simulate(sim_time + CUTOFF + 1)

    ts = np.sort(spike_recorder.get("events")["times"] - CUTOFF)
    events = multimeter.get("events")
    V_m = events["V_m"]

    results = {
        "tau_syn": tau_syn,
        "syn_weight_eff": syn_weight_eff,
        "delays": delays,
        "spike_times": ts,
        "firing_rate": len(ts) / sim_time * 1000,
        "noise_level_Vm": noise_level_Vm,
        "I_noise": I_noise,
        "Vm": V_m,
        "I_syn": events["I_syn_ex"],
        "times": np.arange(len(V_m)) * resolution,
    }
    _save_cache(save_path, params, results)
    return results


def plot_synaptic_input_results(case_results, input_freqs, resolution,
                                fig_name, tlim=(1.0, 2.0), max_f=5000,
                                welch_segments=8, snr_window=5):
    """Plot the Vm traces of all cases in the top row, and below one row per
    case with the PSD of the synaptic input current (left) and the membrane
    potential (right). The reference case (first entry) is drawn in light
    gray behind every other case for comparison.

    `case_results` is a list of (label, results) tuples; `tlim` the time
    window (s) of the Vm traces; `max_f` the upper frequency (Hz) of the PSDs,
    which are computed with Welch's method using `welch_segments` segments
    (50% overlap). Each PSD panel is annotated with the SNR (peak power over
    the local median noise floor within +-`snr_window` Hz) at every input
    frequency."""
    colors = ["k", "tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown"]

    def _psd(sig):
        nperseg = len(sig) // welch_segments
        freqs, psd = welch(sig, fs=1000. / resolution,
                           nperseg=nperseg, noverlap=nperseg // 2)
        mask = (freqs > 0) & (freqs < max_f)
        return freqs[mask], psd[mask]

    psds = [(_psd(res["I_syn"]), _psd(res["Vm"])) for _, res in case_results]

    n_cases = len(case_results)
    plt.close("all")
    fig = plt.figure(figsize=(11, 2.0 * (n_cases + 1)))
    gs = fig.add_gridspec(n_cases + 1, 2, hspace=0.7, wspace=0.25,
                          left=0.08, right=0.98, top=0.95, bottom=0.05)

    # --- Top row: Vm of all cases ---------------------------------------------
    ax_vm = fig.add_subplot(gs[0, :], xlabel="time (s)", ylabel=r"$V_{\rm m}$ (mV)",
                            xlim=tlim,
                            title="Membrane potential, identical synaptic input "
                                  "(locked to " +
                                  " and ".join(f"{f:.0f} Hz" for f in input_freqs) + ")")
    for i, (label, res) in enumerate(case_results):
        t = res["times"] / 1000
        m = (t >= tlim[0]) & (t <= tlim[1])
        ax_vm.plot(t[m], res["Vm"][m], color=colors[i], lw=0.7,
                   label=f"{label} ({res['firing_rate']:.2f} Hz)")
        sp = res["spike_times"] / 1000
        sp = sp[(sp >= tlim[0]) & (sp <= tlim[1])]
        if len(sp):
            ymax = ax_vm.get_ylim()[1]
            ax_vm.vlines(sp, ymax - 2, ymax, color=colors[i], lw=1.5)
    ax_vm.legend(fontsize=7, frameon=False, ncol=n_cases, loc="lower center",
                 bbox_to_anchor=(0.5, 1.12))

    # --- PSD rows -------------------------------------------------------------
    for i, (label, _) in enumerate(case_results):
        for col, (name, unit) in enumerate([("Synaptic input current", r"pA$^2$/Hz"),
                                            ("Membrane potential", r"mV$^2$/Hz")]):
            ax = fig.add_subplot(gs[i + 1, col], xlim=[1, max_f],
                                 ylabel=unit, title=f"{name}: {label}")
            if i == n_cases - 1:
                ax.set_xlabel("frequency (Hz)")
            for f in input_freqs:
                ax.axvline(f, lw=0.5, ls="--", color="gray")
            if i > 0:
                f_ref, p_ref = psds[0][col]
                ax.loglog(f_ref, p_ref, color="0.75", lw=1.0, label=case_results[0][0])
            freqs, psd = psds[i][col]
            ax.loglog(freqs, psd, color=colors[i], lw=0.8, label=label)
            if i > 0:
                ax.legend(fontsize=7, frameon=False, loc="lower left")

            snr_text = []
            for f in input_freqs:
                snr = calculate_spectral_snr_and_zscore(freqs, psd, f, freq_window=snr_window)
                snr_text.append(f"SNR @ {f:.0f} Hz: {snr['snr_peak_ratio']:.1f}")
                print(f"{label} | {name} | SNR @ {f:.0f} Hz: {snr['snr_peak_ratio']:.2f} "
                      f"(z-score {snr['z_score']:.1f})")
            ax.text(0.99, 0.97, "\n".join(snr_text), transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))

    simplify_axes(fig.axes)
    os.makedirs("figures", exist_ok=True)
    out_name = join("figures", f"{fig_name}.pdf")
    plt.savefig(out_name, dpi=150)
    print(f"Saved figure to '{out_name}'")
    return fig


if __name__ == "__main__":

    dt = 0.025
    sim_time = 20e3

    input_params = dict(
        sim_time=sim_time,
        input_freqs=(20.0, 2000.0),
        input_rate=10.0,
        input_ampl=5.0,
        n_inputs=50,
        seed=2,
        resolution=dt,
    )
    inputs = generate_input_spike_trains(**input_params)

    # Reference: very fast synapses, one shared delay, no noise, no spikes.
    reference = dict(
        sim_time=sim_time,
        syn_weight=10.0,
        syn_delay=2.0,
        syn_delay_std=0.0,
        tau_syn=2,
        tau_syn_ref=2,
        noise_level_Vm=0.0,
        C=100,
        V_th=0.,
        E_m=-60.,
        tau_m=10,
        seed=2,
        resolution=dt,
    )
    # Each case is the reference with one individual modification.
    cases = [
        ("reference", {}),
        #(r"$\tau_{\rm syn}$ = 2 ms", {"tau_syn": 2.0}),
        ("delay std = 0.25 ms", {"syn_delay_std": 0.25}),
        (r"$V_{\rm m}$ noise = 0.05 mV", {"noise_level_Vm": 0.05}),
        (r"$V_{\rm th}$ = -50 mV", {"V_th": -50.}),
    ]

    case_results = []
    for i, (label, mod) in enumerate(cases):
        sim_params = {**reference, **mod, "sim_name": f"synaptic_demo_case{i}"}
        case_results.append((label, run_synaptic_input_simulation(
            inputs, input_params, **sim_params)))
        print(f"{label}: firing rate {case_results[-1][1]['firing_rate']:.2f} Hz")

    plot_synaptic_input_results(case_results, input_params["input_freqs"], dt,
                                fig_name=f"synaptic_demo_cases_{int(sim_time / 1000)}s", tlim=(9.5, 10))
