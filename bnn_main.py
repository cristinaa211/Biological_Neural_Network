
import numpy as np 

import random

class biological_neural_network():
    def __init__(self, inhibitory_neuron_type, exhibitory_neuron_type, no_neurons, 
    no_synapses, inhibitory_prob, current, total_time, time_init, time_final):
        self.inh_neu_type = inhibitory_neuron_type
        self.exhib_neu_type = exhibitory_neuron_type
        self.no_neurons = no_neurons
        self.no_synapses = no_synapses
        self.inh_prob = inhibitory_prob
        self.current = current
        self.total_time = total_time
        self.time_init = time_init
        self.time_final = time_final

    def initialize_time_variables(self):
        self.dt = 0.5
        self.T = self.total_time / self.dt
        self.frequency = 2*1e-3
        self.tau_g = 10 
        
    def izhikevich_parameters_initialization(self):
        a_exc, b_exc, c_exc, d_exc = self.neuron_type(self.exhib_neu_type)
        a_inh, b_inh, c_inh, d_inh = self.neuron_type(self.inh_neu_type)
        inh_value = self.inh_prob + 1
        exc_value = self.inh_prob + 1
        inh_value  = get_random_number(self.inh_prob, self.no_neurons)
        exc_value = [1 if number == 0 else 1 for number in inh_value]
        inh_value = np.array(inh_value)
        exc_value = np.array(exc_value)
        a = a_exc * exc_value + a_inh * inh_value
        b = b_exc * exc_value + b_inh * inh_value
        c = c_exc * exc_value + c_inh * inh_value
        d = d_exc * exc_value + d_inh * inh_value
        self.inh = inh_value
        self.exc = exc_value
        return a, b, c, d

    def initialize_synapse_matrix(self):
        self.g_conex = np.zeros([self.no_synapses,1])
        self.E_conex = np.zeros([self.no_synapses, 1])
        w_conex = 0.07 * np.ones([self.no_neurons, self.no_synapses])
        self.w_conex = choose_no_conections(w_conex)
        

    def initialize_membrane_potential_matrix_recovery_variable(self):
        self.membrane_potential = np.zeros([self.no_neurons, int(self.T)])
        self.recovery_variable = np.zeros([self.no_neurons, int(self.T)])
        self.membrane_potential[:,1] = -70 
        self.recovery_variable[:,1] = -14

    def initialize_neuron_matrix(self):
        # initialize neuron matrix
        self.g = np.zeros([self.no_neurons, 1])
        self.E = np.zeros([self.no_neurons, 1])
        self.E[self.inh] = -85
        self.weights = np.zeros([self.no_neurons, self.no_neurons])
        matrix_indices = random.randint(self.no_neurons, self.no_neurons)
        # indices = 

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



    def forward(self):
        a,b,c,d = self.izhikevich_parameters_initialization()
        self.initialize_time_variables()
        self.initialize_synapse_matrix()
        self.initialize_neuron_matrix()
        self.initialize_membrane_potential_matrix_recovery_variable()
        return a,b,c,d




def get_random_number(threshold, interval):
    random_nr = []
    for _ in range(interval):
        a = random.randint(0, 1)
        while a > threshold:
            a = random.randint(0, 1)
        random_nr.append(a)
    return random_nr

def choose_no_conections(w_conex):
    dim1, dim2 = w_conex.shape
    w_conex = w_conex.flatten()
    connections_zeros = np.random.choice(range(len(w_conex)), int(0.1 * len(w_conex)))
    for idx, item in enumerate(w_conex):
        if idx not in connections_zeros :
            w_conex[idx] = 0
    w_conex = np.reshape(w_conex, (dim1, dim2))
    return w_conex



bnn = biological_neural_network(inhibitory_neuron_type='FS', exhibitory_neuron_type='RS',
                                no_neurons= 1000, no_synapses= 100, inhibitory_prob= 0.1, current=5, total_time=1000, time_init=200, time_final=700)
a,b,c,d = bnn.forward()
print(a,b,c,d)

