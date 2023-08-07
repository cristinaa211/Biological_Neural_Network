import numpy as np 
import pandas as pd 
from bnn import BiologicalNeuralNetwork
from operations import correlation_coefficient
import logging
import matplotlib.pyplot as plt 

def compare_bnn_eeg_signals(eeg_csv_file, range_no_neurons, display_plots = False):
    """Computes the coefficient coefficient between the BNN signal and the signals from 22 electrodes.
    INPUT:
        eeg_csv_file     : csv format file 
        range_no_neurons : a list of number of neurons used for the BNN 
        display_plots    : if True then plots different plots 
    RETURN:
        correlation_list : a list of (number_of_neurons, electrode_number, correlation_coefficient_value, inh_neuron_type, exc_neuron_type) pairs for each simulation
        """
    eeg_file = pd.read_csv(eeg_csv_file)
    eeg_dataframe = pd.DataFrame(eeg_file)
    exc_neuron_types = ['CH', 'RS']
    inh_neuron_types = ['LTS', 'FS']
    correlation_list = []
    logging.basicConfig(filename="log.txt", level=logging.INFO)
    logging.info('NEW SIMULATION')
    colors = ['b', 'g', 'r']
    inh_neuron_type = np.random.choice(inh_neuron_types)
    exc_neuron_type = np.random.choice(exc_neuron_types)
    eeg_signal = eeg_dataframe.iloc[:,0]
    print('INHIBITORY NEURON: {}, EXHIBITORY NEURON: {}'.format(inh_neuron_type, exc_neuron_type))
    idx = 0
    plt.figure()
    for n in range_no_neurons:
        for i in range(0,2):
            bnn = BiologicalNeuralNetwork(inhibitory_neuron_type=inh_neuron_type, exhibitory_neuron_type=inh_neuron_type,
                                    no_neurons = n, no_synapses= 10*n, inhibitory_prob = 0.2,
                                    current = 5, total_time= 2200)
            network_signal_value = bnn.forward(display = display_plots)
            correlation_value, cross_corr = correlation_coefficient(eeg_signal, network_signal_value, i, display = False)
            correlation_list.append((n, i, correlation_value, inh_neuron_type, exc_neuron_type ))
            plt.plot(np.arange(-len(cross_corr)/2, len(cross_corr)/2), cross_corr, colors[idx], label = '{} neurons'.format(n))
        idx += 1
    plt.xlabel('Index')
    plt.legend(loc = 'best')
    plt.ylabel('Correlation coefficient')
    plt.title('Cross-correlation of EEG signal and BNN signal')
    plt.suptitle('Inhibitory neuron: {}, Exhibitory neuron: {}'.format(inh_neuron_type, exc_neuron_type))
    plt.show()
    nr_neurons, electrode, corr_coeff, inh, exc = max(correlation_list, key = lambda x : x[2])
    logging.info('The  signal from electrode number {} has the higher similarity with the BNN signal, having the correlation coefficient = {}. BNN has {} neurons.'.format(electrode, corr_coeff, nr_neurons))
    logging.info('The neurons type are : inhibitory = {} , excitatory = {}'.format(inh, exc))
    logging.info('-----------------------------------------------')
    return correlation_list


if __name__ == '__main__':  
    filename = './eeg_math_subj.csv'
    range_no_neurons = [10, 100, 1000]
    correlation_list = compare_bnn_eeg_signals(filename, range_no_neurons, display_plots= False)
    df_corr_list = pd.DataFrame(data = correlation_list, columns= ['no_neurons', 'electrode', 'correlation_coeff', 'inh_type', 'exc_type'])
    df_corr_list.to_csv('./correlation_list_2.csv')
