import numpy as np 
import pandas as pd 
from bnn import BiologicalNeuralNetwork
from operations import correlation_coefficient
import logging


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
    neuron_types = ['CH', 'LTS', 'RS', 'FS']
    correlation_list = []
    logging.basicConfig(filename="log.txt", level=logging.INFO)
    logging.info('NEW SIMULATION')
    for _ in range(9):
        inh_neuron_type = np.random.choice(neuron_types)
        exc_neuron_type = np.random.choice(neuron_types)
        for n in range_no_neurons:
            bnn = BiologicalNeuralNetwork(inhibitory_neuron_type=inh_neuron_type, exhibitory_neuron_type=inh_neuron_type,
                                        no_neurons= n, no_synapses= n * 10, inhibitory_prob = 0.5,
                                        current = 7, total_time= 2500)
            network_signal_value = bnn.forward(display = display_plots)
            for i in range(eeg_dataframe.shape[1]):
                if i == 0: pass
                eeg_signal = eeg_dataframe.iloc[:,i]
                correlation_value = correlation_coefficient(eeg_signal, network_signal_value[0],i, display = display_plots)
                correlation_list.append((n, i, correlation_value, inh_neuron_type, exc_neuron_type ))
    nr_neurons, electrode, corr_coeff, inh, exc = max(correlation_list, key = lambda x : x[2])
    for line in correlation_list:
        logging.info('The signal from electrode number {} has the correlation coefficient = {} with the BNN signal, BNN having {} neurons.'.format(line[1], line[2], line[0]))
    logging.info('The  signal from electrode number {} has the higher similarity with the BNN signal, having the correlation coefficient = {}. BNN has {} neurons.'.format(electrode, corr_coeff, nr_neurons))
    logging.info('The neurons type are : inhibitory = {} , excitatory = {}'.format(inh, exc))
    logging.info('-----------------------------------------------')
    return correlation_list


if __name__ == '__main__':  
    filename = './eeg_math_subj.csv'
    range_no_neurons = [10, 100, 1000]
    correlation_list = compare_bnn_eeg_signals(filename, range_no_neurons, display_plots= False)
    df_corr_list = pd.DataFrame(data = correlation_list, columns= ['no_neurons', 'electrode', 'correlation_coeff', 'inh_type', 'exc_type'])
    df_corr_list.to_csv('./correlation_list.csv')
