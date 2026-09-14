"""Filters-only demonstration: does a frequency-locked peak in the synaptic
input rate survive the synaptic + membrane low-pass filtering?

No NEST here - everything is done with explicit signal processing so the
cause of the effect is transparent:

    1. Build the synaptic input as the summed spike count of `n_inputs`
       independent Poisson processes sharing a common rate that is modulated
       (grid-locked) at `input_freq`. Its PSD has a sharp peak at input_freq
       sitting on a flat Poisson shot-noise floor.
    2. Convolve with an alpha synaptic kernel (tau_syn)  -> synaptic current.
    3. Low-pass with the membrane RC (tau_m)             -> membrane potential.
    4. Repeat step 3 but add independent membrane current noise.

The point: pure linear filtering (steps 2-3) multiplies the peak AND its local
neighbourhood by the same |H(f)|^2, so the LOCAL peak-to-floor ratio (SNR) is
preserved - the peak stays visible, just at a much lower absolute level. It is
only step 4, adding noise with a shallower spectrum, that raises the local
floor at input_freq enough to bury the peak.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, lfilter

from single_cell_simulations import find_noise_std, compute_SNR, simplify_axes


def alpha_kernel(tau_syn, dt, n_tau=8):
    """Alpha PSC kernel sampled at step `dt` (ms), normalised to unit peak so
    that convolving with a unit spike gives a PSC of peak 1 (scaled by weight
    afterwards). tau_syn in ms."""
    t = np.arange(0, n_tau * tau_syn, dt)
    return (np.e / tau_syn) * t * np.exp(-t / tau_syn)


def membrane_filter(I_pA, tau_m, C, dt):
    """Subthreshold RC membrane response to an input current (pA), returned as
    a Vm deviation in mV. Exact exponential (AR(1)) integration:
        V[k] = a V[k-1] + R (1-a) I[k-1],  a = exp(-dt/tau_m),  R = tau_m/C.
    tau_m, dt in ms; C in pF."""
    tau_s = tau_m * 1e-3
    a = np.exp(-(dt * 1e-3) / tau_s)
    R = tau_s / (C * 1e-12)                 # Ohm
    # IIR: V[k] = a V[k-1] + R(1-a) I[k-1]; current pA -> A, volts -> mV.
    b = [0.0, R * (1 - a) * 1e-12 * 1e3]
    return lfilter(b, [1.0, -a], I_pA)


def transfer_functions(freqs, tau_syn, tau_m):
    """Analytic magnitude responses (unit DC gain) of the alpha synapse and the
    RC membrane, evaluated at `freqs` (Hz). tau in ms."""
    w_syn = 2 * np.pi * freqs * tau_syn * 1e-3
    w_m = 2 * np.pi * freqs * tau_m * 1e-3
    H_syn = 1.0 / (1.0 + w_syn ** 2)             # alpha (double pole)
    H_m = 1.0 / np.sqrt(1.0 + w_m ** 2)          # single RC
    return H_syn, H_m


def run_demo(input_freq=1000.0, input_rate=15.0, input_ampl=10.0, n_inputs=100,
             weight=15.0, tau_syn=2.0, tau_m=10.0, C=100.0, noise_level_Vm=1.0,
             T=20e3, dt=0.1, seed=2, nperseg=16384, save_name=None):
    rng = np.random.default_rng(seed)
    fs = 1000.0 / dt                       # sampling rate (Hz)

    # --- 1. Synaptic input: summed Poisson counts with rate locked to input_freq
    t = np.arange(0, T, dt)                                  # ms
    rate_t = input_rate + input_ampl * np.sin(2 * np.pi * input_freq * t / 1000.0)
    lam = n_inputs * rate_t * (dt / 1000.0)                  # expected counts / bin
    counts = rng.poisson(np.clip(lam, 0, None)).astype(float)
    input_rate_signal = counts / (dt / 1000.0)               # summed spikes/s

    # --- 2. Synaptic current: counts convolved with the alpha kernel (peak=weight)
    k = alpha_kernel(tau_syn, dt)
    I_syn = weight * np.convolve(counts, k)[:len(counts)]     # pA

    # --- 3. Membrane potential (pure filtering, no extra noise) ---------------
    Vm = membrane_filter(I_syn, tau_m, C, dt)                 # mV

    # --- 4. Membrane potential with independent current noise -----------------
    sigma_I = find_noise_std(noise_level_Vm, tau_m=tau_m, C_m=C, resolution=dt)
    I_noise = rng.normal(0.0, sigma_I, size=len(counts))
    Vm_noisy = membrane_filter(I_syn + I_noise, tau_m, C, dt)

    # --- PSDs (Welch, so the shot-noise floor is smooth) ----------------------
    def psd(x):
        f, p = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        return f, p

    stages = [
        ("1. Synaptic input rate", input_rate_signal, "|rate|²/Hz"),
        ("2. Synaptic current", I_syn, "|I$_{syn}$|²/Hz"),
        ("3. Vm (filters only)", Vm, "|V$_m$|²/Hz"),
        (f"4. Vm + {noise_level_Vm:g} mV noise", Vm_noisy, "|V$_m$|²/Hz"),
    ]

    fig, axes = plt.subplots(2, len(stages), figsize=(17, 7.5))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.09,
                        wspace=0.32, hspace=0.42)
    fig.suptitle(f"Frequency-locked input at {input_freq:.0f} Hz through the "
                 f"synaptic (tau={tau_syn:g} ms) and membrane (tau={tau_m:g} ms) "
                 f"filters", fontsize=12)

    print(f"\nPeak-to-local-floor SNR at {input_freq:.0f} Hz:")
    for col, (title, sig, ylabel) in enumerate(stages):
        f, p = psd(sig)
        snr = compute_SNR(f, p, input_freq)
        print(f"  {title:32s}: SNR = {snr:.3g}")

        m = f < min(2 * input_freq, fs / 2)
        ax = axes[0, col]
        ax.loglog(f[m], p[m], 'k', lw=0.8)
        ax.axvline(input_freq, lw=0.6, ls='--', color='gray')
        ax.set(title=f"{title}\nSNR@{input_freq:.0f}Hz = {snr:.2g}",
               xlabel="frequency (Hz)", ylabel=ylabel)

        # Zoom around input_freq (linear) to show the local peak vs local floor.
        zoom = (f > input_freq - 40) & (f < input_freq + 40)
        axz = axes[1, col]
        axz.plot(f[zoom], p[zoom], 'k', lw=0.9)
        axz.axvline(input_freq, lw=0.6, ls='--', color='gray')
        axz.set(title=f"zoom @ {input_freq:.0f} Hz",
                xlabel="frequency (Hz)", ylabel=ylabel)

    simplify_axes(list(axes.ravel()))
    if save_name is None:
        save_name = f"demo_filtering_{int(input_freq)}Hz.png"
    fig.savefig(save_name, dpi=150)
    print(f"\nSaved figure to '{save_name}'")
    return fig


if __name__ == "__main__":
    run_demo()
