
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import LFPy
from lfpykit import CurrentDipoleMoment
from lfpykit.eegmegcalc import NYHeadModel
#from brainsignals import neural_simulations as ns
#from brainsignals.plotting_convention import simplify_axes, mark_subplots

import ssl
import json
import shutil
import zipfile
import subprocess
from os.path import join
from urllib.request import urlopen

import neuron

np.random.seed(1234)

load_headmodel = False

if load_headmodel:
    _nyhead_candidates = ["sa_nyhead.mat",
                          os.path.expanduser("~/work/NY_head/sa_nyhead.mat")]
    _nyhead_file = next((pth for pth in _nyhead_candidates if os.path.isfile(pth)), None)
    nyhead = NYHeadModel(nyhead_file=_nyhead_file)

    cortex = nyhead.cortex
    elecs = np.array(nyhead.elecs)
    head_tri = np.array(nyhead.head_data["head"]["tri"]).T - 1
    head_vc = np.array(nyhead.head_data["head"]["vc"])
    cortex_tri = np.array(nyhead.head_data["cortex75K"]["tri"]).T - 1
    x_ctx, y_ctx, z_ctx = cortex
    x_h, y_h, z_h = head_vc[0, :], head_vc[1, :], head_vc[2, :]

    #eeg_elec_idx = 140
    #vertex_idx = np.argmax(np.abs(nyhead.lead_field_normal[:, eeg_elec_idx]))

    # Keep the left-hemisphere electrodes (as before)
    upper_idxs = np.where(elecs[1, :] > 0)[0]
    #elecs = elecs[:, upper_idxs]
    num_elecs = elecs.shape[1]


    #eeg_elec_idx = 111
    eeg_elec_idx = 222
    vertex_idx = np.argmax(np.abs(nyhead.lead_field_normal[:, eeg_elec_idx]))
    # dipole_loc = np.array([-80.9, -22.5, -9.1])  # left middle temporal gyrus (MNI mm)
    # vertex_idx = np.argmin(np.sqrt(np.sum((dipole_loc[:, None] - cortex)**2, axis=0)))
    # Closest electrode to the source
    eeg_elec_idx = np.argmin(np.sqrt(np.sum((cortex[:, vertex_idx, None] -
                                             elecs[:3, :]) ** 2, axis=0)))

    print(eeg_elec_idx)

    dipole_loc = cortex[:, vertex_idx]            # snap to nearest cortex vertex
    print("MTG source vertex:", np.round(dipole_loc, 1), "mm")
    nyhead.set_dipole_pos(dipole_loc)
    print("Vertex normal direction:", np.round(nyhead.cortex_normal_vec, 3))

    print("Closest EEG electrode location:", elecs[:, eeg_elec_idx], eeg_elec_idx)

    print("Max lead field: ", np.max(np.abs(nyhead.lead_field)))

    tES_field = nyhead.lead_field_normal[vertex_idx, eeg_elec_idx]

    print("tES lead field (V/m): ",  tES_field)
else:
    tES_field = 0.44291753583824356

dt = 2**-4
tstop = 1000
cutoff = 1000

n_tsteps_ = int((tstop + cutoff) / dt ) + 1
tvec = np.arange(n_tsteps_) * dt

stim_freqs = np.fft.fftfreq(int(tstop / dt), dt / 1000)

pidxs = stim_freqs > 0.0
stim_freqs = stim_freqs[pidxs]

stim_current = np.zeros(len(tvec))

for freq in stim_freqs:
    stim_current += np.sin(2 * np.pi * freq * tvec / 1000. + np.random.uniform(0, 2 * np.pi))

input_idx = 0
input_amp = 1e6  # nA

make_passive = True

# Human pyramidal (middle temporal gyrus), perisomatic.

model_id = 626170736

cell_name = "hay"
cell_name = f"allen_{model_id}"

allen_folder = "allen_cell_models"           # .../brainsignals/cell_models/allen
os.makedirs(allen_folder, exist_ok=True)
_loaded_mod_folders = set()


def _nrnivmodl_bin():
    """Locate nrnivmodl. Prefer the console-script wrapper next to the running
    python; the copy under neuron/.data/bin has a build path baked in and fails."""
    cand = join(os.path.dirname(sys.executable), "nrnivmodl")
    if os.path.isfile(cand):
        return cand
    exe = shutil.which("nrnivmodl")
    if exe and os.path.isfile(exe):
        return exe
    raise RuntimeError("nrnivmodl not found")


def download_allen_model(model_id):
    """Download and unzip an Allen neuronal model into allen_folder."""
    model_id = str(model_id)
    zip_path = join(allen_folder, f"neuronal_model_{model_id}.zip")
    out_dir = join(allen_folder, f"neuronal_model_{model_id}")
    url = f"https://api.brain-map.org/neuronal_model/download/{model_id}"
    print("Downloading Allen model", model_id, "from", url)
    u = urlopen(url, context=ssl._create_unverified_context())
    with open(zip_path, "wb") as f:
        f.write(u.read())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    os.remove(zip_path)
    return out_dir


def return_allen_cell_model(model_id, dt, tstop, cutoff, make_passive=False):
    """Build an LFPy.Cell from an Allen Cell Types biophysical model.
    Tested previously against the AllenSDK/BMTK versions of the same models."""
    model_id = str(model_id)
    model_folder = join(allen_folder, f"neuronal_model_{model_id}")
    if not os.path.isdir(model_folder):
        download_allen_model(model_id)

    mod_folder = join(model_folder, "modfiles")
    if not os.path.isdir(join(mod_folder, "x86_64")):
        print("Compiling mechanisms ...")
        cwd = os.getcwd()
        os.chdir(mod_folder)
        try:
            subprocess.run([_nrnivmodl_bin()], check=True)
        finally:
            os.chdir(cwd)

    if mod_folder not in _loaded_mod_folders:
        neuron.load_mechanisms(mod_folder)
        _loaded_mod_folders.add(mod_folder)

    params = json.load(open(join(model_folder, "fit_parameters.json")))
    manifest = json.load(open(join(model_folder, "manifest.json")))
    metadata = json.load(open(join(model_folder, "model_metadata.json")))
    morph_file = join(model_folder, "reconstruction.swc")
    model_type = manifest["biophys"][0]["model_type"]

    Ra = params["passive"][0]["ra"]
    if model_type == "Biophysical - perisomatic":
        e_pas = params["passive"][0]["e_pas"]
        cms = params["passive"][0]["cm"]

    neuron.h.celsius = params["conditions"][0]["celsius"]
    reversal_potentials = params["conditions"][0]["erev"]
    active_mechs = params["genome"]

    cell_parameters = {
        'morphology': morph_file,
        'v_init': -85,
        'passive': False,
        'nsegs_method': 'fixed_length',
        'max_nsegs_length': 10.,
        'dt': dt,
        'tstart': 0,          # let the cell settle; recorders start at t=0
        'tstop': tstop + cutoff,
        'pt3d': True,
        'extracellular': True,   # needed for imem
        'custom_code': [join(allen_folder, 'remove_axon.hoc')],
    }
    cell = LFPy.Cell(**cell_parameters)
    cell.metadata = metadata
    cell.manifest = manifest

    if make_passive and model_type != "Biophysical - perisomatic":
        raise RuntimeError("make_passive only implemented for perisomatic models")

    for sec in neuron.h.allsec():
        sec.insert("pas")
        sectype = sec.name().split("[")[0]
        if model_type == "Biophysical - perisomatic":
            sec.e_pas = e_pas
            for cm_dict in cms:
                if cm_dict["section"] == sectype:
                    exec("sec.cm = {}".format(cm_dict["cm"]))
        sec.Ra = Ra

        for sec_dict in active_mechs:
            if sec_dict["section"] == sectype:

                if sec_dict["mechanism"] == "":
                    # This is the passive mechanism
                    if sec_dict["name"] != "g_pas":
                        raise RuntimeError("Something wrong with model building function!")
                    exec("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))
                else:
                    if not make_passive:
                        if not sec.has_membrane(sec_dict["mechanism"]):
                            sec.insert(sec_dict["mechanism"])
                        exec("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))
        if not make_passive:
            for sec_dict in reversal_potentials:
                if sec_dict["section"] == sectype:
                    for key in sec_dict.keys():
                        if not key == "section":
                            exec("sec.{} = {}".format(key, sec_dict[key]))

    neuron.h.secondorder = 0
    return cell


def return_hay_cell(tstop, dt, cutoff, make_passive=False):
    if not os.path.isfile(join(hay_folder, 'morphologies', 'cell1.asc')):
        ns.download_hay_model()
    #
    # if make_passive:
    #     cell_params = {
    #         'morphology': join(hay_folder, 'morphologies', 'cell1.asc'),
    #         'passive': True,
    #         'passive_parameters': {"g_pas": 1 / 15000,
    #                                "e_pas": -70.},
    #         'nsegs_method': "lambda_f",
    #         "Ra": 100,
    #         "cm": 1.0,
    #         "lambda_f": 100,
    #         'dt': dt,
    #         'tstart': 0,
    #         'tstop': tstop + cutoff,
    #         'v_init': -70,
    #         'pt3d': True,
    #         'extracellular': True,
    #     }
    #
    #     cell = LFPy.Cell(**cell_params)
    #     cell.set_rotation(x=4.729, y=-3.166)
    #
    #     return cell
    # else:
    if not hasattr(neuron.h, "CaDynamics_E2"):
        neuron.load_mechanisms(join(hay_folder, 'mod'))
    cell_params = {
        'morphology': join(hay_folder, "morphologies", "cell1.asc"),
        'templatefile': [join(hay_folder, 'models', 'L5PCbiophys3.hoc'),
                         join(hay_folder, 'models', 'L5PCtemplate.hoc')],
        'templatename': 'L5PCtemplate',
        'templateargs': join(hay_folder, 'morphologies', 'cell1.asc'),
        'passive': False,
        'nsegs_method': None,
        'dt': dt,
        'tstart': 0,
        'tstop': tstop + cutoff,
        'v_init': -75,
        'celsius': 34,
        'pt3d': True,
        'extracellular': True,
    }

    cell = LFPy.TemplateCell(**cell_params)

    if make_passive:
        remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1", "SK_E2", "K_Tst", "K_Pst", "KdShu2007",
                       "Im", "Ih", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA", 'StochKv']
        ns.remove_active_mechanisms(remove_list, cell)

    cell.set_rotation(x=4.729, y=-3.166)
    return cell


def return_cell_model(tstop, dt, cutoff, make_passive, cell_name, rotation=None):
    neuron.h("forall delete_section()")

    if cell_name == "hay":
        cell = return_hay_cell(tstop, dt, cutoff,
                               make_passive=make_passive)
    elif "allen" in cell_name:
        cell_id = cell_name.split("_")[1]
        cell = return_allen_cell_model(cell_id, dt, tstop, cutoff,
                                       make_passive=make_passive)

    else:
        raise ValueError(f"Did not recognize cell name: {cell_name}")
    if rotation is not None:
        cell.set_rotation(x=rotation[0], y=rotation[1], z=rotation[2], rotation_order='xyz')

    return cell


def make_sinusoidal_current_stimuli(cell, input_idx, freqs, tvec, input_scaling=0.001):

    I = np.zeros(len(tvec))
    for freq in freqs:
        I += np.cos(2 * np.pi * freq * tvec/1000.)
    input_array = input_scaling * I
    noise_vec = neuron.h.Vector(input_array)

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, noise_vec


def return_freq_and_amplitude(tvec, sig):
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

    amplitude = np.abs(Y)/Y.shape[1]
    return freqs, amplitude

def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=-0.12, ypos=1.15):

    if not type(axes) is list:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.text(xpos, ypos, letters[idx].capitalize(),
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='demibold',
                fontsize=10,
                transform=ax.transAxes)

def simplify_axes(axes):

    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


def run_RT_based_simulation(cell_name, dt, tstop, cutoff, input_idx,
                            tES_current, lead_field):

    scale_factor = 1e0
    cell = return_cell_model(cell_name=cell_name, dt=dt, tstop=tstop,
                             cutoff=cutoff, make_passive=True,
                             rotation=[-np.pi/2, -np.pi/9, 0])

    noise_vec = neuron.h.Vector(tES_current * scale_factor)

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)

    cell.simulate(rec_vmem=True, rec_imem=True)

    keep_idxs = cell.tvec > cutoff
    cell.imem = cell.imem[:, keep_idxs]
    cell.vmem = cell.vmem[:, keep_idxs]
    cell.somav = cell.somav[keep_idxs]
    noise_vec = np.array(noise_vec)[keep_idxs]
    cell.tvec = cell.tvec[keep_idxs]
    cell.tvec -= cell.tvec[0]

    p = CurrentDipoleMoment(cell).get_transformation_matrix() @ cell.imem
    #p = p[2, :]

    eeg = lead_field * p[2, :] * 1E-9

    print("p: ", p)
    print("EEG: ", eeg)
    print(np.std(p))
    print(lead_field)

    #eeg -= np.mean(eeg)

    xf = np.fft.fftfreq(len(cell.tvec), cell.dt / 1000)
    pidxs = xf > 0.0
    xf = xf[pidxs]

    print(f"Extracted freqs: {xf}")
    print(f"Expected  freqs: {stim_freqs}")

    freqs, yf1 = return_freq_and_amplitude(cell.tvec, noise_vec)
    freqs, yf2 = return_freq_and_amplitude(cell.tvec, p)
    freqs, yf3 = return_freq_and_amplitude(cell.tvec, eeg)

    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(wspace=0.5, right=0.98, top=0.95, hspace=0.6)

    ax_neur = fig.add_subplot([0.0, 0., 0.3, 0.99], aspect=1, frameon=False,
                              xticks=[], yticks=[])

    ax1 = fig.add_subplot(332, xlabel="time (ms)", ylabel="input (nA)")
    ax2 = fig.add_subplot(335, xlabel="time (ms)", ylabel="$p_z$ (nAµm)")
    ax3 = fig.add_subplot(338, xlabel="time (ms)", ylabel="EEG µV")

    ax1_psd = fig.add_subplot(333, xlabel="frequency (Hz)", ylabel="input (nA)", ylim=[1e-1, 1e1])
    ax2_psd = fig.add_subplot(336, xlabel="frequency (Hz)", ylabel="$p_z$ (nAµm)")
    ax3_psd = fig.add_subplot(339, xlabel="frequency (Hz)", ylabel="EEG (µV)")

    ax_neur.plot(cell.x.T, cell.z.T, c='k', lw=0.5, zorder=1)

    ax1.plot(cell.tvec, noise_vec, c='k')
    ax2.plot(cell.tvec, p[2, :], label="$P_z$")
    ax2.plot(cell.tvec, p[1, :], label="$P_y$")
    ax2.plot(cell.tvec, p[0, :], label="$P_x$")

    ax3.plot(cell.tvec, eeg * 1e3, c='k')

    ax1_psd.loglog(freqs, yf1[0], c='k')
    ax2_psd.loglog(freqs, yf2[0], label="$P_x$")
    ax2_psd.loglog(freqs, yf2[1], label="$P_y$")
    ax2_psd.loglog(freqs, yf2[2], label="$P_z$")
    ax3_psd.loglog(freqs, yf3[0] * 1e3, c='k')

    ax2_psd.legend(frameon=False, ncol=1, loc="upper right")
    # ax2.legend(frameon=False, ncol=1, loc="upper right")

    simplify_axes(fig.axes)
    fig.savefig(f"control_RT_sim_{cell_name}.png")
    fig.savefig(f"control_RT_sim_{cell_name}.pdf")
    plt.close(fig)

    return eeg


eeg = run_RT_based_simulation(cell_name, dt, tstop, cutoff, input_idx,
                                           stim_current, tES_field)
