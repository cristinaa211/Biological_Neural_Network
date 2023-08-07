import numpy as np 
import matplotlib.pyplot as plt
import math

def neuron_type(neuron_type):
    """Provides parameters for each type of neuron, according to the Izhikevich Neuron Model"""
    match neuron_type :
        case "FS":
            a = 0.1
            b = 0.2
            c = -65
            d = 2
        case "LTS":
            a = 0.02
            b = 0.25
            c = -65
            d = 2
        case "IB":
            a = 0.02
            b = 0.2
            c = -55
            d = 4
        case "CH":
            a = 0.02
            b = 0.2
            c = -50
            d = 2
        case "RS":
            a = 0.02
            b = 0.2
            c = -65
            d = 8
    return a , b , c , d 


def continous_impulse_encoding(neurontype, title, input_current = 7):
    a, b, c, d = neuron_type(neurontype)
    initial_time, total_time, dt = 100, 1000, 0.5
    T = math.ceil(total_time / dt)
    T = int(T)
    membrane_potential = np.zeros((T,1))
    recovery_variable = np.zeros_like(membrane_potential)
    current = np.zeros_like(membrane_potential)
    current_axis = np.zeros_like(current)
    membrane_potential[0] = -70
    recovery_variable[0] = -14
    for t in range(T-3):
        I_applied = [input_current if t*dt > initial_time else 0][0]
        current[t] = I_applied
        if I_applied > 10: I_applied = 10
        current_axis[t] = I_applied - 90
        I_applied = current[t]
        if membrane_potential[t] < 35:
            dv = (0.04 * membrane_potential[t] + 5)*membrane_potential[t] + 140 - recovery_variable[t]
            membrane_potential[t+1] = membrane_potential[t] + (dv + I_applied)*dt 
            du = a*(b*membrane_potential[t] - recovery_variable[t])
            recovery_variable[t+1] = recovery_variable[t] + dt*du
        else: 
            membrane_potential[t] = 35
            membrane_potential[t+1] = c
            recovery_variable[t+1] = recovery_variable[t] + d
    time_vector = dt * np.arange(-1, T-1)
    plt.figure()
    plt.plot(time_vector, membrane_potential)
    plt.plot(time_vector, current_axis, 'r')
    plt.xlabel('Time [ms]')
    plt.xlim([0, 950])
    plt.ylabel('Potential V [mV]')
    plt.legend('Membrane potential', 'Excitation current', loc = 'best')
    plt.title('Neuron Type: {}'.format(title))
    plt.show()


if __name__ == "__main__":
    neuron_types = {'CH' :'Chattering CH', 'LTS' :  'Low-threshold spikes LTS', 'RS' : 'Regular Spiking RS', 'FS' :  'Fast Spiking FS' }
    for neu_type in neuron_types.keys() :
        continous_impulse_encoding(neu_type, neuron_types[neu_type], 5)
