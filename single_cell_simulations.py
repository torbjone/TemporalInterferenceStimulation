import os
import pickle
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch


def find_I(target_dVm, f, tau_m=10., C_m=100.):
    """Current amplitude (pA) for a sinusoidal input at frequency `f` (Hz) that
    produces a subthreshold membrane-potential deviation of amplitude
    `target_dVm` (mV) in an iaf_psc_alpha neuron, in the absence of noise.

    The neuron's subthreshold RC response gives
        |dVm| = I_amp * R / sqrt(1 + (2*pi*f*tau_m)^2),
    with membrane resistance R = tau_m / C_m (matching NEST's tau_m = R*C_m).
    Rearranged for I_amp and returned in pA, the unit NEST's ac_generator
    expects. `tau_m` is in ms and `C_m` in pF to match the simulation
    parameters."""
    tau_si = tau_m * 1e-3            # ms -> s
    C_si = C_m * 1e-12              # pF -> F
    R_si = tau_si / C_si            # Ohm; R = tau_m / C_m
    dV_si = target_dVm * 1e-3       # mV -> V
    I_si = dV_si * math.sqrt(1 + (2*math.pi*f*tau_si)**2) / R_si
    return I_si / 1e-12            # A -> pA


def find_noise_std(noise_level_Vm, tau_m=10., C_m=100., resolution=0.1):
    """Noise_generator current std (pA) that produces a membrane-potential
    standard deviation of `noise_level_Vm` mV in the subthreshold
    iaf_psc_alpha neuron, in the absence of any other input.

    The noise_generator injects a piecewise-constant Gaussian current, redrawn
    every `resolution` ms. Exact integration of the RC membrane over one step
    gives the AR(1) recursion x_{k+1} = a*x_k + R*(1-a)*I_k, with
    a = exp(-dt/tau_m) and R = tau_m/C_m, whose stationary std is
        sigma_V = R * sigma_I * sqrt((1 - a) / (1 + a)).
    Inverting for sigma_I and returning pA (NEST's unit). `tau_m`, `resolution`
    in ms and `C_m` in pF, to match the simulation parameters."""
    tau_si = tau_m * 1e-3           # ms -> s
    C_si = C_m * 1e-12             # pF -> F
    R_si = tau_si / C_si           # Ohm; R = tau_m / C_m
    dt_si = resolution * 1e-3      # ms -> s
    a = math.exp(-dt_si / tau_si)
    sigma_V = noise_level_Vm * 1e-3  # mV -> V
    sigma_I = sigma_V / (R_si * math.sqrt((1 - a) / (1 + a)))
    return sigma_I / 1e-12         # A -> pA


def return_freq_and_psd(tvec, sig):
    """ Returns the amplitude and frequency of the input signal"""
    import scipy.fftpack as ff
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec

    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig, axis=1)[:, pidxs[0]]

    amplitude = np.abs(Y)**2/Y.shape[1]
    return freqs, amplitude


def simplify_axes(axes):

    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


def find_spike_rate(spike_times, bin_size, sim_time):
    # creating bins with width = bin_size (ms)
    t_bins = np.arange(0, sim_time + bin_size, bin_size)

    # putting spike train into the bins
    spike_counts, _ = np.histogram(spike_times, bins=t_bins)

    # plt.plot(t_bins[1:], spike_counts)
    # for t in results["spike_times"]:
    #     plt.vlines(x=t, ymin=-1, ymax=0, color="black")
    #
    # plt.show()
    #

    # Spike rate
    spike_rate = spike_counts / (bin_size / 1000.0)

    return spike_rate, t_bins


def psd_welch(spike_times, sim_time, bin_size=0.25, segments=16,):
    """Computes the PSD of spiketrain given:
    spike_times : spike train,
    sim_time : duration of simulation
    bin_size : frequency resolution
    segments : number of segments the signal is to be divided into
    It returns PSD-values of the spike train, and corresponding frequencies"""
    spike_rate, spike_counts = find_spike_rate(spike_times, bin_size, sim_time)
    # Sampling rate (measurements per sec)
    fs = 1000 / bin_size

    # using scipys Welch function to find frequencies and psd values. overlap fraction is 0.5
    # This function splits the signal into overlapping segments, computes PSD for each segment and averages
    freqs, psd_values = welch(spike_rate, fs=fs, nperseg=len(spike_counts) // segments,
                              noverlap=(len(spike_counts) // segments) // 2)

    # if normalize:
    #     psd_values = psd_values / np.max(psd_values)

    # Cut off high-frequency tail to avoid artifacts near Nyquist
    # mask = freqs < f + 800
    # freqs = freqs[mask]
    # psd = psd_values[mask]

    return freqs, psd_values


def compute_SNR(freqs, psd_values, f):
    """Computes and returns SNR given PSD_values and corresponding frequencies."""

    window = 5  # 5 Hz-window around f

    # Frequency closest to f
    arg_f = np.argmin(np.abs(freqs - f))
    print(f"SNR frequency: {freqs[arg_f]} Hz")
    # Finding the indexes of frequencies around the input frequency
    noise_mask = (freqs >= freqs[arg_f] - window) & (freqs <= freqs[arg_f] + window) & (freqs != freqs[arg_f])

    # Local noise average
    noise_avg = np.mean(psd_values[noise_mask])

    # calculate SNR
    if noise_avg > 0:
        SNR = ((psd_values[arg_f]) / noise_avg)

    # Omit values that does not produce spikes
    else:
        SNR = np.nan

    return SNR


def run_single_cell_simulation(sim_time=10e3,
                               f_values=[10.0], target_stim_dVm=0.3, noise_level_Vm=4.0, C=100,
                               V_th=-50.0, E_m=-60.0, tau_m=10,
                               seed=2, resolution=0.1, sim_name="test",
                               save_dir="results", force_rerun=False,
                               save_Vm=True):
    """This function creates:
    -Single iaf_psc_alpha neuron (LIF w/alpha-shaped postsynaptic currents)
    -noise using a noise_generator to get noise from Gaussian distribution
    -spike_recorder to get the spike times
    -multimeter to plot the membrane potential manually
    -deterministic sinusoidal inputs using AC_generators

    One or two sine currents are connected to the neuron, as well as noise,
    to stimulate the neuron.
    The simulation is run and the function returns
    the dataframe "results" containing:
    -spike_times : times when an action potential occurs
    -V_m : membrane potential
    -V_th : threshold potential
    -sim_time: Simulation time
    -times : times when Vm was recorded

    The results are saved to a pickle file (together with all input
    parameters) inside ``save_dir``. On subsequent calls the function will
    load the saved results instead of rerunning the simulation, provided the
    saved parameters exactly match the current input parameters. Set
    ``force_rerun=True`` to always rerun and overwrite the saved file."""

    # All input parameters that define the simulation. Saved together with the
    # results so that a cached run can be verified against the current request.
    params = {
        "sim_time": sim_time,
        "f_values": f_values,
        "target_stim_dVm": target_stim_dVm,
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

    cutoff = 1000

    neuron = nest.Create("iaf_psc_alpha")

    neuron.set(V_th=V_th)
    neuron.set(V_m=E_m)
    neuron.set(C_m=C)
    neuron.set(V_reset=E_m)
    neuron.set(E_L=E_m)
    neuron.set(I_e=0)
    neuron.set(tau_m=tau_m)

    # Convert the requested Vm noise level (mV std) into the noise_generator
    # current std (pA) via the neuron's subthreshold RC response, so the noise
    # alone drives a `noise_level_Vm` mV standard deviation of the membrane.
    I_noise = find_noise_std(noise_level_Vm, tau_m=tau_m, C_m=C, resolution=resolution)

    noise = nest.Create(
        "noise_generator",
        1,
        params=[
            {"mean": 0.0, "std": I_noise, "dt": resolution},
        ]
    )

    multimeter = nest.Create(
        "multimeter",
        params={
            "interval": resolution,
            "record_from": ["V_m"],
            "start": cutoff,
        }
    )

    # Convert the requested Vm-deviation target into a per-frequency current
    # amplitude (pA) using the neuron's subthreshold RC response, so that each
    # sine alone would drive a `target_stim_dVm` mV deviation at its frequency.
    I_amp = [find_I(target_stim_dVm, f_, tau_m=tau_m, C_m=C) for f_ in f_values]

    sine = nest.Create(
        "ac_generator",
        len(f_values),
        params=[{"amplitude": a_, "frequency": f_} for a_, f_ in zip(I_amp, f_values)]
    )

    spike_recorder = nest.Create("spike_recorder",
                                 params={"start": cutoff})

    # Connections
    nest.Connect(multimeter, neuron)
    nest.Connect(neuron, spike_recorder)
    nest.Connect(sine, neuron)
    nest.Connect(noise[0], neuron)

    # Simulating and recording results
    nest.Simulate(sim_time + cutoff + 1)

    mm_data = multimeter.get("events")
    events = spike_recorder.get("events")
    ts = events["times"] - cutoff
    V_m = mm_data["V_m"]
    times = np.arange(len(V_m)) * resolution #mm_data["times"] - cutoff

    # Return results
    results = {
        "spike_times": ts,
        "firing_rate": len(ts) / sim_time * 1000,
        "target_stim_dVm": target_stim_dVm,
        "I_amp": I_amp,
        "noise_level_Vm": noise_level_Vm,
        "I_noise": I_noise,
    }
    if save_Vm:
        results.update({ "Vm": V_m, "times": times ,})


    # Save the results together with all input parameters so the run can be
    # loaded (and verified) instead of rerun next time.
    os.makedirs(save_dir, exist_ok=True)
    # Write to a temporary file first and atomically replace the target, so an
    # interrupted write can never leave a truncated cache file behind.
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"params": params, "results": results}, f)
    os.replace(tmp_path, save_path)
    print(f"Saved results to '{save_path}'")
    return results


def plot_single_cell_results(results, sim_params,
                             welch_segments=1, max_f=2000, tlim=[5, 6],
                             firing_rate_bin_size=0.1, use_welch=False):
    """Analyses and plots the output of run_single_cell_simulation.

    Computes the Vm and firing-rate spectra (with their SNR at stim_freq) and
    produces the membrane-potential trace and (zoomed) PSD panels, saving the
    figure to a PNG named after the simulation. Simulation-defining values
    that are not part of `results` (sim_time, V_th, resolution, sim_name) are
    read from `sim_params`."""

    print("Plotting single cell results...")
    sim_time = sim_params["sim_time"]
    V_th = sim_params["V_th"]
    resolution = sim_params["resolution"]
    sim_name = sim_params["sim_name"]
    stim_freqs = sim_params["f_values"]
    if len(stim_freqs) == 2:
        stim_freq = np.abs(stim_freqs[1] - stim_freqs[0])
    elif len(stim_freqs) == 1:
        stim_freq = stim_freqs[0]
    else:
        raise ValueError("stim_freqs must be a list of length 1 or 2")

    print(f"Max Vm deviation: ", np.max(np.abs(results["Vm"] - np.mean(results["Vm"]))))

    spike_rate, t_bins = find_spike_rate(results["spike_times"], firing_rate_bin_size, sim_time)

    print("Calculating PSDs...")
    if use_welch:
        # Sampling rate (measurements per sec)
        fs_fr = 1000 / firing_rate_bin_size
        freqs_fr, fr_psd = welch(spike_rate, fs=fs_fr, nperseg=len(spike_rate) // welch_segments,
                                  noverlap=(len(spike_rate) // welch_segments) // 2)

        fs = 1000 / resolution
        freqs_vm, vm_psd = welch(results["Vm"], fs=fs, nperseg=len(results["Vm"]) // welch_segments,
                                 noverlap=(len(results["Vm"]) // welch_segments) // 2)

    else:
        freqs_fr, fr_psd = return_freq_and_psd(t_bins, spike_rate)
        fr_psd = fr_psd[0]

        freqs_vm, vm_psd = return_freq_and_psd(results["times"], results["Vm"])
        vm_psd = vm_psd[0]

    fr_SNR = compute_SNR(freqs_fr, fr_psd, stim_freq)
    vm_SNR = compute_SNR(freqs_vm, vm_psd, stim_freq)

    print("Making plot...")

    plt.close("all")
    fig = plt.figure(figsize=(10, 9))
    fig.subplots_adjust(wspace=0.5, right=0.98, top=0.95, hspace=0.6)
    ax_vm = fig.add_subplot(311, ylim=[-75, -44], xlabel="time (s)", ylabel=r"$V_{\rm m}$ (mV)",
                            title=r"Firing rate: {0:.2f} Hz; STD($V_m$): {1:.2f} mV".format(
                                results["firing_rate"], np.std(results["Vm"])))
    ax_vm_psd = fig.add_subplot(323, title=f"SNR = {vm_SNR:.2f}", xlim=[1, max_f],
                                xlabel="frequency (Hz)", ylabel=r"|$V_{\rm m}$|²/Hz")
    ax_fr_psd = fig.add_subplot(324, title=f"SNR = {fr_SNR:.2f}", xlim=[1, max_f],
                                xlabel="frequency (Hz)", ylabel="|firing rate|²/Hz")

    ax_vm_psd_zoom = fig.add_subplot(325, title=f"SNR = {vm_SNR:.2f}",
                                     # xlim=[np.min(stim_freqs) - 10, np.max(stim_freqs) + 10],
                                     xlim=[stim_freq - 10, stim_freq + 10],
                                     ylim=[0, vm_psd[np.argmin(np.abs(freqs_vm - stim_freq))] * 5],
                                     xlabel="frequency (Hz)", ylabel=r"|$V_{\rm m}$|²/Hz")
    ax_fr_psd_zoom = fig.add_subplot(326, title=f"SNR = {fr_SNR:.2f}",
                                     # xlim=[np.min(stim_freqs) - 10, np.max(stim_freqs) + 10],
                                     xlim=[stim_freq - 10, stim_freq + 10],
                                     ylim=[0, fr_psd[np.argmin(np.abs(freqs_fr - stim_freq))] * 1.2],
                                     xlabel="frequency (Hz)", ylabel="|firing rate|²/Hz")

    ax_vm.plot(results["times"] / 1000, results["Vm"], 'k')
    ax_vm.set_xlim(tlim)
    for t in results["spike_times"]:
        ax_vm.vlines(x=t / 1000, ymin=V_th, ymax=V_th + 5, color="r")

    ax_vm_psd.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')
    ax_fr_psd.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')
    ax_vm_psd_zoom.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')
    ax_fr_psd_zoom.axvline(x=stim_freq, lw=0.5, ls='--', color='gray')

    ax_vm_psd.loglog(freqs_vm[freqs_vm < max_f], vm_psd[freqs_vm < max_f], 'k')
    ax_fr_psd.loglog(freqs_fr[freqs_fr < max_f], fr_psd[freqs_fr < max_f], 'k')

    ax_vm_psd_zoom.plot(freqs_vm[freqs_vm < max_f], vm_psd[freqs_vm < max_f], 'k')
    ax_fr_psd_zoom.plot(freqs_fr[freqs_fr < max_f], fr_psd[freqs_fr < max_f], 'k')

    simplify_axes(fig.axes)

    plt.savefig(f"{sim_name}_{welch_segments}_{int(sim_time / 1000)}s.png")

    return fig




if __name__ == "__main__":
    sim_params_1ABC = dict(
        sim_time=10000e3,
        target_stim_dVm=0.0,
        f_values=[20],
        noise_level_Vm=4.,
        seed=2,
        V_th=-50.,
        resolution=0.1,
        sim_name="Fig1A-C",
        force_rerun=False,
    )

    sim_params_S1 = dict(
        sim_time=10e3,
        target_stim_dVm=0.3,
        f_values=[20],
        noise_level_Vm=0.,
        seed=2,
        V_th=-50.,
        resolution=0.1,
        sim_name="FigS1",
        force_rerun=False,
    )
    sim_params_1DEF = dict(
        sim_time=10000e3,
        target_stim_dVm=0.3,
        f_values=[20],
        noise_level_Vm=4.,
        seed=2,
        V_th=-50.,
        resolution=0.1,
        sim_name="Fig1D-F",
        force_rerun=False,
    )
    sim_params_S2 = dict(
        sim_time=10e3,
        target_stim_dVm=0.3,
        f_values=[1000],
        noise_level_Vm=0.,
        seed=2,
        V_th=-50.,
        resolution=0.1,
        sim_name="FigS2",
        force_rerun=False,
    )

    sim_params_1GHI = dict(
        sim_time=10000e3,
        target_stim_dVm=0.3,
        f_values=[1000],
        noise_level_Vm=4.,
        seed=2,
        V_th=-50.,
        resolution=0.1,
        sim_name="Fig1G-I",
        force_rerun=False,
    )

    sim_params_S3 = dict(
        sim_time=10e3,
        target_stim_dVm=0.3,
        f_values=[1000, 1020],
        noise_level_Vm=0.,
        seed=2,
        V_th=-50.,
        resolution=0.1,
        sim_name="FigS3",
        force_rerun=False,
    )

    sim_params_JKL = dict(
        sim_time=10000e3,
        target_stim_dVm=0.3,
        f_values=[1000, 1020],
        noise_level_Vm=4.,
        seed=2,
        V_th=-50.,
        resolution=0.1,
        sim_name="Fig1J-L",
        force_rerun=False,
    )



    sim_params_list = [sim_params_JKL]

    for sim_params in sim_params_list:
        results = run_single_cell_simulation(**sim_params)
        plot_single_cell_results(results, sim_params)
