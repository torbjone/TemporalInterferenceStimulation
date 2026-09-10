import numpy as np
from numpy import random
import nest 


def build_simulation(sim_time=1000, second_sine=False, noisy=True, 
                  f1=10.0, a=100.0, SD=250.0,
                  delay_mean=1, delay_sd=1, C=100,
                  V_thresh=-50.0, E_m=-60.0, tau_m=10,
                  seed=np.random.randint(1,1e+6), beat=20, resolution=0.25):
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
    -times : times when Vm was recorded"""
    
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": resolution}) 
    nest.rng_seed = seed
  
    neuron = nest.Create("iaf_psc_alpha")
    
    neuron.set(V_th = V_thresh)
    neuron.set(V_m = E_m)
    neuron.set(C_m = C)
    neuron.set(V_reset = E_m)
    neuron.set(E_L = E_m)
    neuron.set(I_e = 0)
    neuron.set(tau_m = tau_m)
    

    noise = nest.Create(
        "noise_generator",
        1,
        params=[
            {"mean": 0.0, "std": SD, "dt": 0.5},

        ]
    )

    multimeter = nest.Create(
        "multimeter",
        params={
            "interval": resolution,
            "record_from": ["V_m"],
            "start" : 1000,
        }
    )

    f2=f1+beat
    sine = nest.Create(
        "ac_generator",
        2,
        params=[
            {"amplitude": a, "frequency": f1},
            {"amplitude": a, "frequency": f2},
        ]
    )


   
    spike_recorder = nest.Create("spike_recorder",
                                params = {
                                "start" : 1000})
    
    #Connections
    nest.Connect(multimeter, neuron)
    nest.Connect(neuron, spike_recorder)

    nest.rng_seed = seed

    if second_sine:
        nest.Connect(sine[0], neuron)
        nest.Connect(sine[1], neuron)
    else:
        nest.Connect(sine[0], neuron)

    if noisy:

        nest.Connect(
            noise[0],
            neuron
        )
        
    # Simulating and recording results 
    nest.set_verbosity("M_WARNING")
    nest.Simulate(sim_time + 1000)
    
    mm_data = multimeter.get("events")
    events = spike_recorder.get("events")
    senders = events["senders"]
    ts = events["times"]
    V_m =  mm_data["V_m"]
    times = mm_data["times"]

    #Return results
    results = {
        "spike_times": ts,
        "Vm": V_m,
        "V_th": V_thresh,
        "sim_time" : sim_time,
        "times" : times,
        "firing_rate" : len(ts)/sim_time*1000
    }
    
    return results


