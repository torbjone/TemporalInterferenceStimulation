"""Toy demo: a single iaf_psc_alpha neuron driven by synaptic input whose
rate is locked to a given frequency.

The presynaptic input is a sinusoidal_poisson_generator, i.e. an inhomogeneous
Poisson process with instantaneous rate

    r(t) = input_rate + input_ampl * sin(2*pi * input_freq * t),

so the *rate* of the synaptic input is grid-/phase-locked to `input_freq`. The
generator drives `n_inputs` parrot_neurons (each relaying one independent
realisation of that modulated process), and every parrot projects onto the
single postsynaptic neuron. Recording the parrots therefore gives exactly the
synaptic input spike train the neuron receives.

Plotted (mirroring plot_single_cell_results):
    - the membrane potential trace (with spikes),
    - the PSD of the membrane potential,
    - the PSD of the neuron's firing rate,
    - the PSD of the synaptic input rate,
each spectrum shown full-band and zoomed around `input_freq`, annotated with
its SNR there.

Results are cached to a pickle file exactly as in run_single_cell_simulation.
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from single_cell_simulations import (
    find_noise_std,
    find_spike_rate,
    return_freq_and_psd,
    compute_SNR,
    simplify_axes,
)


def run_synaptic_input_simulation(sim_time=10e3,
                                  input_freq=20.0,
                                  input_rate=15.0,
                                  input_ampl=10.0,
                                  n_inputs=100,
                                  syn_weight=15.0,
                                  syn_delay=1.5,
                                  tau_syn=2.0,
                                  tau_syn_ref=2.0,
                                  conserve_charge=True,
                                  noise_level_Vm=0.0,
                                  C=100,
                                  V_th=-50.0,
                                  E_m=-60.0,
                                  tau_m=10,
                                  seed=2,
                                  resolution=0.1,
                                  sim_name="synaptic_demo",
                                  save_dir="results",
                                  force_rerun=False,
                                  save_Vm=True):
    """Simulate a single neuron driven by frequency-locked synaptic input.

    Parameters
    ----------
    sim_time : float
        Simulation time (ms), excluding the initial `cutoff` transient.
    input_freq : float
        Frequency (Hz) the synaptic input rate is locked to.
    input_rate : float
        Mean (DC) rate (spikes/s) of each presynaptic Poisson process.
    input_ampl : float
        Modulation amplitude (spikes/s) of the presynaptic rate at input_freq.
        Kept <= input_rate so the instantaneous rate stays non-negative.
    n_inputs : int
        Number of presynaptic parrot_neurons (independent realisations of the
        common modulated rate) projecting onto the neuron.
    syn_weight : float
        Weight (pA) of every input -> neuron synapse, interpreted at the
        reference time constant `tau_syn_ref` (see `conserve_charge`).
    syn_delay : float
        Synaptic delay (ms) of the input -> neuron connections. A single value
        shared by all synapses (no delay distribution).
    tau_syn : float
        Alpha-current synaptic time constant (ms) of the neuron's incoming
        excitatory synapses (tau_syn_ex).
    tau_syn_ref : float
        Reference synaptic time constant (ms) at which `syn_weight` is defined.
        Only used when `conserve_charge=True`.
    conserve_charge : bool
        If True, compensate the weight so the total charge per synaptic event
        (proportional to weight * tau_syn) is independent of `tau_syn`:
        w_eff = syn_weight * tau_syn_ref / tau_syn. In NEST's iaf_psc_alpha the
        weight sets the PSC *peak* while the PSC *integral* scales with
        tau_syn, so this keeps the mean current into the cell (hence mean Vm
        and firing rate) unchanged as `tau_syn` is varied, isolating the
        synaptic low-pass filtering effect. If False, `syn_weight` is used as
        given (constant PSC peak, tau-dependent mean drive).
    noise_level_Vm : float
        Optional Vm-noise standard deviation (mV) added to the neuron via a
        noise_generator (0 disables it).
    C, V_th, E_m, tau_m : float
        Neuron parameters, as in run_single_cell_simulation.
    seed : int
        Master seed for the NEST kernel.
    resolution : float
        Simulation time step (ms).
    sim_name, save_dir, force_rerun, save_Vm
        Caching / output controls, as in run_single_cell_simulation.
    """

    # All input parameters that define the simulation. Saved together with the
    # results so that a cached run can be verified against the current request.
    params = {
        "sim_time": sim_time,
        "input_freq": input_freq,
        "input_rate": input_rate,
        "input_ampl": input_ampl,
        "n_inputs": n_inputs,
        "syn_weight": syn_weight,
        "syn_delay": syn_delay,
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
    nest.verbosity = nest.VerbosityLevel.WARNING
    nest.SetKernelStatus({"resolution": resolution})
    nest.rng_seed = seed

    cutoff = 1000  # ms of initial transient discarded from the recordings

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

    # --- Frequency-locked synaptic input --------------------------------------
    # sinusoidal_poisson_generator: instantaneous rate = rate + amplitude*sin,
    # so the input rate is locked to `input_freq`. With individual_spike_trains
    # (the default) each target parrot relays its own realisation of that rate.
    spg = nest.Create("sinusoidal_poisson_generator", params={
        "rate": input_rate,
        "amplitude": input_ampl,
        "frequency": input_freq,
        "phase": 0.0,
    })
    parrots = nest.Create("parrot_neuron", n_inputs)
    nest.Connect(spg, parrots)  # all_to_all, independent trains per parrot

    # Charge-conserving weight: the alpha-PSC integral scales with tau_syn (its
    # peak equals the weight), so scaling the weight by tau_syn_ref/tau_syn
    # keeps the per-event charge - and thus the mean drive - fixed as tau_syn
    # is varied, isolating the synaptic low-pass filtering effect.
    syn_weight_eff = syn_weight * tau_syn_ref / tau_syn if conserve_charge else syn_weight
    nest.Connect(parrots, neuron,
                 syn_spec={"weight": syn_weight_eff, "delay": syn_delay})

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
        "interval": resolution, "record_from": ["V_m"], "start": cutoff})
    spike_recorder = nest.Create("spike_recorder", params={"start": cutoff})
    input_recorder = nest.Create("spike_recorder", params={"start": cutoff})

    nest.Connect(multimeter, neuron)
    nest.Connect(neuron, spike_recorder)
    nest.Connect(parrots, input_recorder)  # the synaptic input spike train

    # --- Simulate -------------------------------------------------------------
    nest.Simulate(sim_time + cutoff + 1)

    ts = np.sort(spike_recorder.get("events")["times"] - cutoff)
    input_ts = np.sort(input_recorder.get("events")["times"] - cutoff)

    results = {
        "input_freq": input_freq,
        "input_rate": input_rate,
        "input_ampl": input_ampl,
        "n_inputs": n_inputs,
        "tau_syn": tau_syn,
        "syn_weight_eff": syn_weight_eff,
        "spike_times": ts,
        "firing_rate": len(ts) / sim_time * 1000,
        "input_spike_times": input_ts,
        "noise_level_Vm": noise_level_Vm,
        "I_noise": I_noise,
    }
    if save_Vm:
        V_m = multimeter.get("events")["V_m"]
        results["Vm"] = V_m
        results["times"] = np.arange(len(V_m)) * resolution

    # --- Save (atomic) --------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"params": params, "results": results}, f)
    os.replace(tmp_path, save_path)
    print(f"Saved results to '{save_path}'")
    return results


def plot_synaptic_input_results(results, sim_params, tlim=[0.5, 1.0], max_f=2000,
                                firing_rate_bin_size=0.1, input_bin_size=0.1):
    """Plot the membrane-potential trace and the PSDs of the membrane
    potential, the firing rate, and the synaptic input rate.

    Each spectrum is shown full-band (log-log) and zoomed around `input_freq`,
    annotated with its SNR there. `tlim` is the time window (s) shown for the
    Vm trace; `max_f` the upper frequency (Hz) of the full spectra. The Vm
    panels require the simulation to have been run with save_Vm=True."""
    V_th = sim_params["V_th"]
    sim_name = sim_params["sim_name"]
    sim_time = sim_params["sim_time"]
    resolution = sim_params["resolution"]
    input_freq = sim_params["input_freq"]

    have_vm = "Vm" in results

    # --- Spectra --------------------------------------------------------------
    # Neuron firing rate.
    spike_rate, t_bins = find_spike_rate(results["spike_times"], firing_rate_bin_size, sim_time)
    freqs_fr, psd_fr = return_freq_and_psd(t_bins, spike_rate)
    psd_fr = psd_fr[0]
    snr_fr = compute_SNR(freqs_fr, psd_fr, input_freq)

    # Synaptic input rate (pooled parrot spikes -> rate -> PSD).
    in_rate, in_bins = find_spike_rate(results["input_spike_times"], input_bin_size, sim_time)
    freqs_in, psd_in = return_freq_and_psd(in_bins, in_rate)
    psd_in = psd_in[0]
    snr_in = compute_SNR(freqs_in, psd_in, input_freq)

    # Membrane potential.
    if have_vm:
        freqs_vm, psd_vm = return_freq_and_psd(results["times"], results["Vm"])
        psd_vm = psd_vm[0]
        snr_vm = compute_SNR(freqs_vm, psd_vm, input_freq)

    plt.close("all")
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.35,
                          left=0.07, right=0.98, top=0.94, bottom=0.07)

    # --- Membrane-potential trace (spans all columns) -------------------------
    ax_vm = fig.add_subplot(
        gs[0, :], xlabel="time (s)", ylabel=r"$V_{\rm m}$ (mV)",
        title=(r"Single neuron, synaptic input rate locked to {0:.0f} Hz - "
               r"firing rate: {1:.2f} Hz".format(input_freq, results["firing_rate"])))
    if have_vm:
        ax_vm.plot(results["times"] / 1000, results["Vm"], 'k', lw=0.7)
        ax_vm.set_xlim(tlim)
    for t in results["spike_times"]:
        ax_vm.vlines(x=t / 1000, ymin=V_th, ymax=V_th + 5, color="r")

    # --- PSD panels: full log-log (row 1) + zoom (row 2) ----------------------
    def _plot_psd(col, freqs, psd, snr, ylabel, title, zoom_factor):
        mask = freqs < max_f
        arg = np.argmin(np.abs(freqs - input_freq))

        ax_full = fig.add_subplot(gs[1, col], xlim=[1, max_f],
                                  xlabel="frequency (Hz)", ylabel=ylabel,
                                  title=f"{title}\nSNR = {snr:.2f}")
        ax_full.axvline(x=input_freq, lw=0.5, ls='--', color='gray')
        ax_full.loglog(freqs[mask], psd[mask], 'k')

        ax_zoom = fig.add_subplot(gs[2, col], xlim=[input_freq - 10, input_freq + 10],
                                  xlabel="frequency (Hz)", ylabel=ylabel,
                                  title=f"zoom @ {input_freq:.0f} Hz")
        ax_zoom.axvline(x=input_freq, lw=0.5, ls='--', color='gray')
        ax_zoom.plot(freqs[mask], psd[mask], 'k')
        top = psd[arg] * zoom_factor
        if np.isfinite(top) and top > 0:
            ax_zoom.set_ylim(0, top)

    if have_vm:
        _plot_psd(0, freqs_vm, psd_vm, snr_vm, r"|$V_{\rm m}$|²/Hz",
                  "Membrane potential", 5)
    else:
        ax = fig.add_subplot(gs[1, 0], title="Membrane potential\n(save_Vm=False)")
        ax.axis("off")
    _plot_psd(1, freqs_fr, psd_fr, snr_fr, "|firing rate|²/Hz",
              "Firing rate", 1.2)
    _plot_psd(2, freqs_in, psd_in, snr_in, "|input rate|²/Hz",
              "Synaptic input rate", 1.2)

    simplify_axes(fig.axes)
    out_name = f"{sim_name}_{int(sim_time / 1000)}s.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved figure to '{out_name}'")
    return fig


if __name__ == "__main__":
    sim_params = dict(
        sim_time=100e3,
        input_freq=1000.0,
        input_rate=10.0,
        input_ampl=10.0,
        n_inputs=100,
        syn_weight=15.0,
        syn_delay=1.5,
        tau_syn=2,
        noise_level_Vm=0.5,
        C=100,
        V_th=-10.,
        E_m=-60.,
        tau_m=10,
        seed=2,
        resolution=0.1,
        sim_name="synaptic_demo_no_spikes_noise2",
        force_rerun=False,
        save_Vm=True,
    )

    results = run_synaptic_input_simulation(**sim_params)
    plot_synaptic_input_results(results, sim_params)
