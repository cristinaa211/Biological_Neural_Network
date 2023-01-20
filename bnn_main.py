
import numpy as np 
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.special as sps  
import seaborn as sns 
from scipy.sparse import csr_matrix
import math
import pandas as pd 
import os
import pathlib
from scipy.signal import correlate
from scipy.stats import pearsonr
class biological_neural_network():
    def __init__(self, inhibitory_neuron_type, exhibitory_neuron_type, no_neurons, 
        no_synapses, inhibitory_prob, current, total_time, time_init, time_final, display):
        self.inh_neu_type = inhibitory_neuron_type
        self.exhib_neu_type = exhibitory_neuron_type
        self.no_neurons = no_neurons
        self.no_synapses = no_synapses
        self.inh_prob = inhibitory_prob
        self.current = current
        self.total_time = total_time
        self.time_init = time_init
        self.time_final = time_final
        self.display = display
        
    def izhikevich_parameters_initialization(self):
        a_exc, b_exc, c_exc, d_exc = self.neuron_type(self.exhib_neu_type)
        a_inh, b_inh, c_inh, d_inh = self.neuron_type(self.inh_neu_type)
        inh_value = self.inh_prob + 1
        exc_value = self.inh_prob + 1
        inh_value = np.random.rand(self.no_neurons, 1) < self.inh_prob
        # exc_value = [not x for x in inh_value]
        exc_value = ~inh_value
        inh = np.array([1 if x == True else 0 for x in inh_value])
        exc = np.array([1 if x == True else 0 for x in exc_value])
        a = a_exc * exc + a_inh * inh
        b = b_exc * exc + b_inh * inh
        c = c_exc * exc + c_inh * inh
        d = d_exc * exc + d_inh * inh
        self.inh = inh.reshape(len(inh),1)
        self.exc = exc.reshape(len(exc),1)
        return a, b, c, d

    def initialize_time_variables(self):
        self.dt = 0.5
        # total number of time steps
        self.T = math.ceil(self.total_time / self.dt)
        self.T = int(self.T)
        # Poisson firing rate
        self.frequency = 2*1e-3
         # time constant for synaptic conductance
        self.tau_g = 10 

    def initialize_synapse_matrix(self):
        # conductance of network's synapses
        self.g_conex = np.zeros((self.no_synapses,1))
        # reversal potential of network's synapses
        self.E_conex = np.zeros((self.no_synapses,1))
        # weights of network's synapses
        w_conex = 0.07 * np.ones((self.no_neurons, self.no_synapses))
        w_conex[np.random.rand(self.no_neurons, self.no_synapses) > 0.1] = 0
        self.w_conex = w_conex
        return True
        
    def initialize_membrane_potential_matrix_recovery_variable(self):
        self.membrane_potential = np.zeros((self.no_neurons, self.T))
        self.recovery_variable = np.zeros((self.no_neurons, self.T))
        self.crt_interm = np.zeros((self.no_neurons, self.T))
        self.membrane_potential[:,0] = -70
        self.recovery_variable[:,0] = -14

    def initialize_neuron_matrix(self):
        # initialize neuron matrix
        # # synaptic conductances
        self.g = np.zeros((self.no_neurons, 1))
        # synaptic reversal potentials
        self.E = np.zeros((self.no_neurons,1))
        self.E[self.inh] = -85
        # existence weights of inhibitory neuron
        weights = np.zeros((self.no_neurons, self.no_neurons))
        weights = weights.flatten().tolist()
        matrix_indices = np.random.random((self.no_neurons, self.no_neurons))
        flatten_matrix = matrix_indices.flatten()
        # choose indices for the weights of inhibitory neurons which will receive values from a Gamma distribution 
        indices = list(np.where(flatten_matrix < self.inh_prob)[0])
        gamma_distribution = np.random.gamma(shape = 2, scale = 0.003, size = (len(indices), 1))
        gamma_distribution = gamma_distribution.tolist()
        random_gamma_distr_values = random.choices(gamma_distribution, k=len(indices))
        i = 0
        for idx in indices:
            weights[idx] =  random_gamma_distr_values[i][0]
            i += 1
        distribution = [weights[idx] for idx in indices]
        sns.histplot(distribution)
        plt.xlabel('Weights values')
        # plt.show()
        weights = np.reshape(weights, (self.no_neurons , self.no_neurons))
        # the connections between the inhibitory neurons and the excitatory neurons are two times stronger
        weights[self.exc, self.inh] *= 2
        weights = csr_matrix(weights)
        self.weights = weights

    def generate_network_current(self):
        '''
        The method is trying to simulate the 
        membrane potential of the network by updating 
        the conductance of the network's synapses,
         calculating the applied current, updating the 
        conductance of the network, calculating the recurrent current, 
        and updating the membrane potential of the
         neurons using the Izhikevich model.
        Initializes two arrays (sum_0, sum_1) to store the sum of certain values, and a variable "fired" to store the number of neurons that have fired.
        Applies a current to the neurons if the current time step is the first time step.
        Determines whether a random subset of the synapses are active based on a probability determined by the time step and a predefined time window.
        Calculates the total current applied to the neurons based on the active synapses and the membrane potential of the neurons.
        Updates the membrane potential and recovery variable of the neurons using the Izhikevich model equations.
        Checks if any of the neurons have fired and updates their membrane potential accordingly.
        Stores certain values in different arrays for later use.
        '''
        self.izhikevich_parameters_initialization()
        self.initialize_time_variables()
        self.initialize_synapse_matrix()
        self.initialize_neuron_matrix()
        self.initialize_membrane_potential_matrix_recovery_variable()
        sum_0 = np.zeros((1, self.T))
        sum_1 = np.zeros((1, self.T))
        # store the number of neurons that have fired
        fired = 0
        for t in range(self.T-1) : 
            I_applied_0 = [self.current if t == 0 else np.zeros((self.no_neurons, 1))]
            variable_dt = t*self.dt
            p = [np.random.rand(self.no_synapses, 1) < self.frequency * self.dt if variable_dt > self.time_init and variable_dt < self.time_final else 0][0]
            if type(p) == int:
                self.g_conex += p
            else:
                self.g_conex = self.g_conex.ravel() +  p.flatten() 
                self.g_conex = np.expand_dims(self.g_conex, axis = 1)
            result_matrix = np.dot(self.w_conex , self.g_conex)
            I_applied = I_applied_0 + np.dot(self.w_conex, self.g_conex*self.E_conex)
            membrane_potential_column = self.membrane_potential[:,t]
            I_applied = I_applied - np.dot( result_matrix.ravel(), membrane_potential_column.ravel())
            self.g_conex = (1 - self.dt / self.tau_g) * self.g_conex
            try:
                self.g = np.add(self.g.ravel(), np.array(fired).ravel())
                self.g = np.expand_dims(self.g, axis = 1)
            except: 
                self.g += fired
            I_recurrent = np.dot(self.weights.todense(), self.g * self.E)
            self.g = (1 - self.dt / self.tau_g) * self.g
            try:
                I_total = np.add(I_applied, I_recurrent)
            except:
                I_applied = np.squeeze(I_applied, axis = 0)
                I_total = np.add(I_applied, I_recurrent)
            self.crt_interm[:,t] = I_total.ravel() 
            # Izhikevich neuron model:
            #-----------------------------
            first_part = 0.04 * self.membrane_potential[:, t] + 5* np.ones_like(self.recovery_variable[:, t])
            first_part =  np.expand_dims(first_part, axis=1)
            matrix_help = 140 * np.ones_like(self.recovery_variable[:, t])
            second_part = matrix_help- self.recovery_variable[:, t]
            second_part =  np.expand_dims(second_part, axis=1)
            second_part_1 =  second_part + I_total
            mem_pot = np.expand_dims( self.membrane_potential[:, t], axis = 1)
            third_part = first_part * mem_pot
            dv = third_part + second_part_1
            #------------------------------
            self.membrane_potential[:, t+1] = self.membrane_potential[:, t] + dv.ravel() * self.dt
            a, b, c, d = self.izhikevich_parameters_initialization()
            du = a * (b * self.membrane_potential[:, t] - self.recovery_variable[:, t])
            self.recovery_variable[:, t+1] = self.recovery_variable[:, t] + self.dt * du
            fired = self.membrane_potential[:, t] >= 35
            # fired = np.array([0 if x == False else 1 for x in fired])
            self.membrane_potential[fired, t] = 35
            self.membrane_potential[fired, t+1] = c[fired]
            self.recovery_variable[fired, t+1] = self.recovery_variable[fired, t] + d[fired]
            for j in range(0, self.no_neurons):
                sum_0[0,t] = sum_0[0,t] + self.membrane_potential[j,t]
                sum_1[0,t] = sum_1[0,t] + self.crt_interm[j,t]
        sum_0 = np.nan_to_num(sum_0)
        sum_1 = np.nan_to_num(sum_1)
        return sum_0, sum_1

    def plot_neuron_activation_map(self):
        spikes = (self.membrane_potential == 35)
        dim1, dim2 = spikes.shape
        time_step = np.arange(0, self.T) * self.dt
        spikes = [0 if x == False else 1 for x in spikes.flatten()]
        spikes = np.reshape(spikes, (dim1, dim2))
        spikes[self.inh, :] *= 2
        plt.figure()
        x, y = np.meshgrid(time_step, np.arange(0, self.no_neurons))
        col = ['k', 'r']
        for k in range(2):
            index = np.where(spikes == k) 
            plt.scatter(x[index], y[index], c = col[k], marker = '*')
        plt.xlim([0, self.T * self.dt])
        plt.ylim([0, self.no_neurons])
        plt.xlabel('Time [ms]')
        plt.ylabel('Neuron index')
        plt.title('Neurons activation map')
        plt.show()


    def plot_action_potentials(self):
        idx2, idx3 = self.select_random_excitatory_inhibitory_neurons()
        time_step = np.arange(0, self.T) * self.dt
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(time_step.ravel(), self.membrane_potential[0,:].ravel())
        plt.ylim([-100,100])
        plt.title('MP first neuron')
        plt.xlabel('Time step')
        plt.subplot(2, 2, 2)
        plt.plot(time_step.ravel(), self.membrane_potential[idx2,:].ravel())
        plt.title('MP random excitatory neuron')
        plt.xlabel('Time step')
        plt.ylim([-100,100])
        plt.subplot(2, 2, 3)
        plt.plot(time_step.ravel(), self.membrane_potential[idx3,:].ravel())
        plt.title('MP random inhibitory neuron')
        plt.xlabel('Time step')
        plt.ylim([-100,100])
        plt.subplot(2, 2, 4)
        plt.plot(time_step.ravel(), self.membrane_potential[-1,:].ravel())
        plt.title('MP last neuron')
        plt.ylim([-100,100])
        plt.xlabel('Time step')
        plt.show()

    def generate_plots(self, network_signal_value, network_current ):
        time_step = np.arange(0, self.T) * self.dt
        plt.figure()
        plt.plot(time_step.ravel(), network_signal_value[0].ravel() / 1000, 'b')
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [uV]')
        plt.title('Network signal')
        plt.show()

        plt.figure()
        plt.plot(time_step.ravel(), network_current[0].ravel(), 'b')
        plt.xlabel('Time [ms]')
        plt.ylabel('Current intensity [pA]')
        plt.title('Network current')
        plt.show()

        plt.figure()
        for p in range(self.no_neurons):
            plt.plot(time_step.ravel(), self.membrane_potential[p,:].ravel())
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.title('Action potentials at each moment of time')
        plt.show()

    def neuron_type(self, use_case):
        match use_case :
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

    def select_random_excitatory_inhibitory_neurons(self):
        exc_neurons = np.where(~self.inh)[0]
        inh_neurons = np.where(self.inh)[0]
        try:
            idx2 = np.random.choice(exc_neurons, 1)
            idx3 = np.random.choice(inh_neurons, 1)
        except:
            idx3 = 0
        return  int(idx2), int(idx3)

    def forward(self):
        network_signal_value, network_current = self.generate_network_current()
        if self.display == True:
            self.plot_neuron_activation_map()
            self.plot_action_potentials()
            self.generate_plots(network_signal_value, network_current )
        else: pass
        return network_signal_value, network_current


def generate_histogram(weights_matrix):
    count, bisn, ignored = plt.hist(weights_matrix, 50, density=True)
    plt.plot()
    
def generate_gamma_distribution(shape, scale, gamma_distribution): 
    count, bins, ignored = plt.hist(gamma_distribution, 50, density=True)
    y = bins**(shape-1)*(np.exp(-bins/scale) /  
                        (sps.gamma(shape)*scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')  
    plt.title('Gamma distribution for the values of inhibitory neurons')
    plt.show()
    
def perfom_fft_of_signal(signal, title):
    signal = np.int16((signal / signal.max()))
    fourier_transform = np.fft.rfft(signal- np.mean(signal), norm='ortho')
    frequency = np.fft.rfftfreq(len(signal))
    spectrum_magnitude = np.abs(fourier_transform)[0:] 
    plt.plot(frequency, spectrum_magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('{}'.format(title))
    plt.show()

def perfom_correlation_coefficient(signal1, signal2, display = False):
    signal1 = signal1.ravel() / max(signal2)
    signal2 = signal2.ravel() / max(signal2)
    correlation = np.correlate(signal1, signal2, mode='full')
    correlation = correlation / (np.linalg.norm(signal1) * np.linalg.norm(signal2))
    if display == True:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(0, len(signal1)), signal1 )
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [norm]')
        plt.title('EEG signal')
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(0, len(signal2)), signal2 )
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [norm]')
        plt.title('BNN signal')
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(0, len(correlation)), correlation)
        plt.xlabel('Index')
        plt.ylabel('Correlation coefficient [norm]')
        plt.title('Cross-correlation of EEG signal and BNN signal')
        plt.show()
    if len(signal1) > len(signal2):
        signal22 = np.resize(signal2, len(signal1))
        signal11 = signal1
    else:
        signal11 = np.resize(signal1, len(signal2))
        signal22 = signal2
    correlation_coef = np.corrcoef(signal11, signal22)[0, 1]
    return correlation_coef

def continous_impulse_encoding(use_case, input_current = 7):
    match use_case :
        case "FS":
            a = 0.1
            b = 0.2
            c = -65
            d = 2
            title = 'FS'
        case "LTS":
            a = 0.02
            b = 0.25
            c = -65
            d = 2
            title = 'LTS'
        case "IB":
            a = 0.02
            b = 0.2
            c = -55
            d = 4
            title = 'IB'
        case "CH":
            a = 0.02
            b = 0.2
            c = -50
            d = 2
            title = 'CH'
        case "RS":
            a = 0.02
            b = 0.2
            c = -65
            d = 8
            title = 'RS'
    initial_time, final_time, total_time, dt = 100, 700, 1000, 0.5
    T = math.ceil(total_time / dt)
    T = int(T)
    membrane_potential = np.zeros((T,1))
    recovery_variable = np.zeros_like(membrane_potential)
    current = np.zeros_like(membrane_potential)
    current_axis = np.zeros_like(current)
    membrane_potential[0] = -70
    recovery_variable[0] = -14
    for t in range(T-1):
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
    time_vector = dt * np.arange(0, T)
    plt.figure()
    plt.plot(time_vector, membrane_potential)
    plt.plot(time_vector, current_axis, 'r')
    plt.xlabel('Time [ms]')
    plt.ylabel('Potential V [mV]')
    plt.legend('Membrane potential', 'Excitation current', loc = 'best')
    plt.title('{}'.format(title))
    plt.show()





if __name__ == '__main__':  
    path_file = os.getcwd()
    path_file = pathlib.Path(path_file)
    filename = 'eeg_file_mathematics_subject.csv'
    eeg_file = pd.read_csv(path_file / filename)
    eeg_dataframe = pd.DataFrame(eeg_file)
    correlation_list = []
    neuron_types = ['CH', 'LTS', 'RS', 'FS']
    # for n in [10,100,1000]:
    #     inh_neuron_type = np.random.choice(neuron_types)
    #     exc_neuron_type = np.random.choice(neuron_types)
    #     bnn = biological_neural_network(inhibitory_neuron_type=inh_neuron_type, exhibitory_neuron_type=inh_neuron_type,
    #                                 no_neurons= n, no_synapses= 10000, inhibitory_prob= 5,
    #                                  current=5, total_time=2000, time_init=200, time_final=1700, display = False)
    #     network_signal_value, network_current = bnn.forward()
    #     for i in range(eeg_dataframe.shape[1]):
    #         if i == 0: pass
    #         eeg_signal = eeg_dataframe.iloc[:,i]
    #         correlation_value = perfom_correlation_coefficient(eeg_signal, network_signal_value[0])
    #         correlation_list.append((n, i, correlation_value,inh_neuron_type, exc_neuron_type ))
    # z,x,y, inh, exc = max(correlation_list, key = lambda x : x[2])
    # print('The electrode number {} has the higher similarity with the BNN signal, having the correlation coefficient = {}, for {} neurons.'.format(x, y, z))
    # print('The neurons type are : inhibitory = {} , excitatory = {}'.format(inh, exc))
    print('-----------------------------------------------')
    for neu_type in neuron_types :
        continous_impulse_encoding(neu_type, 7)


