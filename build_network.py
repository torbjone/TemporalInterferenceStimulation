import numpy as np
from numpy import random
import random
import math
import matplotlib.pyplot as plt
import itertools
import nest 

def positive_normal(mu, std, size, resolution):
    """This function draws random delays drawn from a Gaussian Distribution
   ensuring that the delays are bigger than the temporal resolution of the simulations.
   mu: mean value 
   std: standard deviation
   size: number of delays (i.e. #connections)"""
    
    delays = np.random.normal(mu, std, size)
    
    while np.any(delays < resolution):
        delays[delays < resolution] = np.random.normal(mu, std, np.sum(delays < resolution))
    return delays

def build_network(Cm_B=100, Cm_A=100,sim_time=1000, second_sine=False, noisy=True, common_noise=False,
                  f1=10.0, a=100.0, SD=250.0, delay_mean=1, delay_sd=0.25,
                  V_thresh=-50.0, E_m=-60.0, tau_m=10, weight=1, tau_syn_ex=2, 
                  seed=np.random.randint(0,1e6), beat=20, resolution=0.25,
                  measure_from_A=False, num_A=1000, neuron_type="iaf_psc_alpha", num_raster=100):
    
    """This function builds the network as described in the Methods-section. 
    It creates neurons and measuring devices, connections and runs the simulation.
    ------------------------------------------------------------------------------
    Arguments:
    sim_time: duration of simulation (ms)
    second_sine: Whether or not to connect the second AC-generator
    noisy: Whether or not noise generators are connected 
    f1: input frequency of AC-generator 
    a: current amplitude of AC-generator 
    SD: noise level (standard deviation of noise generator)
    delay_mean: mean delay (ms)
    delay_sd: standard deviation of delay (ms)
    V_thresh: Mem.pot. threshold for firing action potential (mV)
    E_m: Resting/reset potential (mV)
    weight: synaptic weight (pA)
    seed: random number to initialize the simulation 
    beat: offset frequency (f2-f1) (Hz)
    resolution : temporal resolution (ms)
    measure_from_A : Used in investigation of population A
    num_A : number of neurons in population A
    ----------------------------------------------------------------------
    Returns: dictionary containing recordings and data from the simulation
    """
    
    f2=f1+beat
    
    nest.ResetKernel()
    nest.SetKernelStatus({
    "resolution": resolution,
    "rng_seed": seed
    })
    nest.set_verbosity("M_WARNING")
    
    params_A = {
        "C_m": Cm_A,
        "tau_m": tau_m,
        "E_L": -60.0,
        "V_reset": E_m,
        "V_th": V_thresh,
        "V_m": E_m,
        "I_e" : 0,
        "t_ref" : 2,
        "tau_syn_ex" : tau_syn_ex
    }

    params_B = {
            "C_m": Cm_B,
            "tau_m": tau_m,
            "E_L": -60.0,
            "V_reset": E_m,
            "V_th": V_thresh,
            "V_m": E_m,
            "I_e" : 0,
            "t_ref" : 2,
            "tau_syn_ex" : tau_syn_ex
        }
    
    if neuron_type == "iaf_psc_delta":
        params_A.pop("tau_syn_ex")
        params_B.pop("tau_syn_ex")
    
    if neuron_type == "iaf_cond_exp":
        params_A.pop("tau_m")
        params_B.pop("tau_m")
    
    # Creating neurons, current generators and recording devices
    nodes_A = nest.Create(neuron_type, num_A, params=params_A)
    node_B = nest.Create(neuron_type, 1, params=params_B)



    sinus_1 = nest.Create(
            "ac_generator",
            params={"amplitude": a, "frequency": f1})
    
    sinus_2 = nest.Create(
            "ac_generator",
            params={"amplitude": a, "frequency": f2})
    
    
    noise = nest.Create("noise_generator", 
                        params={"mean" : 0, "std" : SD, "dt" : 0.5})
    
    
    if neuron_type == "iaf_cond_exp" or neuron_type == "iaf_psc_delta":
          multimeter = nest.Create(
            "multimeter", 2,
            params={
                "interval": resolution,
                "record_from": ["V_m"],
    
            })
    else:
    
        multimeter = nest.Create(
                "multimeter", 2,
                params={
                    "interval": resolution,
                    "record_from": ["V_m", "I_syn_ex"],
        
                })
    
    # spike_recorders: 
    # 1 for the B-neuron, 1 connected to all A-neurons, 1 for a single A-neuron
    spike_recorders = nest.Create("spike_recorder", 3) 
    spike_recorder_raster = nest.Create("spike_recorder", 1)
    
    # Spike recorder connections
    if len(nodes_A) >= num_raster:
        nest.Connect(nodes_A[:num_raster], spike_recorder_raster)
    nest.Connect(node_B, spike_recorders[0])
    nest.Connect(nodes_A, spike_recorders[1])
    nest.Connect(nodes_A[0], spike_recorders[2])
    
  # Connect B cell to population A
    if delay_sd != 0:
        random_delays = positive_normal(delay_mean, delay_sd, len(nodes_A), resolution)
    
        for idx, node in enumerate(nodes_A):
            nest.Connect(node, node_B, syn_spec={"weight": weight, 
                                                 "delay": random_delays[idx]
                                                })
    
    else:
        nest.Connect(nodes_A, node_B, syn_spec={"weight": weight, 
                                                "delay" : delay_mean
                                               })

  
        
    # Multimeter connections
    if measure_from_A:
        nest.Connect(multimeter[0], nodes_A[0])  
        nest.Connect(multimeter[1], nodes_A[1])
      
    if measure_from_A == False:
        nest.Connect(multimeter[0], node_B)
    
    # AC-generator connections
    nest.Connect(sinus_1, nodes_A)
    
    if second_sine:
        nest.Connect(sinus_2, nodes_A)

    # Noise generator connections
    if noisy and common_noise==False:
        for i, node in enumerate(nodes_A):
            nest.SetKernelStatus({"rng_seed": seed+i})
            noise_A = nest.Create("noise_generator", params={"mean": 0.0, "std": SD, "dt" : 0.5})
            nest.Connect(noise_A, node)
        
        nest.Connect(noise, node_B)
        
    if noisy and common_noise:
        nest.Connect(noise, nodes_A)
        nest.Connect(noise, node_B)
    
    # Simulate and extract results
    nest.Simulate(sim_time)
    mm_data = multimeter.get("events")[0]
    V_m = mm_data["V_m"]
    times = mm_data["times"]
    if neuron_type != "iaf_cond_exp" and neuron_type != "iaf_psc_delta":
        I_syn_ex = mm_data["I_syn_ex"]
    else:
        I_syn_ex = 0
    
    if measure_from_A == False:
        events = spike_recorders.get("events")[0]
        senders = events["senders"]
        ts = np.array(events["times"])
        
    else:
        # For comparison with single neuron plots
        events = spike_recorders.get("events")[2]
        senders = np.array(events["senders"])
        ts = np.array(events["times"])
      
        # To compare noise
        mm_data2 = multimeter.get("events")[1]
        V_m2 = mm_data2["V_m"]
        times2 = mm_data2["times"]
   
    
    tot_events = spike_recorders[1].get("events") 
    tot_senders = np.array(tot_events["senders"])
    tot_ts = np.array(tot_events["times"])
     
        
    # Dictionary containing results 
    results = {
        "spike_times": ts,
        "Vm": V_m,
        "V_th": V_thresh,
        "sim_time" : sim_time,
        "times" : times,
        "Number_of_spikes" : len(ts),
        "Spike_rate" : len(ts)/sim_time*1000, #spikes/sec
        "tot_spike_rate" : len(tot_ts)/sim_time*1000,
        "tot_ts" : tot_ts,
        "I_syn_ex" : I_syn_ex

    }

    if measure_from_A:
        results["Vm2"] = V_m2
        results["times2"] = times2
        
    if len(nodes_A) >= num_raster:
        results["spike_recorder_raster"] = spike_recorder_raster
        
                      
    return results


