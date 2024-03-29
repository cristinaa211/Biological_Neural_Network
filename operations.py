import matplotlib.pyplot as plt 
import numpy as np
import scipy.special as sps  
from scipy.fft import rfft, rfftfreq

def generate_histogram(weights_matrix):
    count, bisn = plt.hist(weights_matrix, 50, density=True)
    plt.plot(bisn, count)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of weights matrix')
    plt.show()

    
def generate_gamma_distribution(shape, scale, gamma_distribution): 
    count, bins = plt.hist(gamma_distribution, 50, density=True)
    y = bins**(shape-1)*(np.exp(-bins/scale) /  
                        (sps.gamma(shape)*scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')  
    plt.title('Gamma distribution for the values of inhibitory neurons')
    plt.show()

def fft_signal(signal, sample_rate, title):
    signal = signal.ravel() - np.mean(signal)
    magnitude = np.abs(rfft(signal))
    frequency = rfftfreq(len(signal), 1 / sample_rate)
    plt.plot(frequency, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('{}'.format(title))
    plt.show()


def correlation_coefficient(signal1, signal2, idx, display = True):
    signal1 = signal1.ravel() 
    signal2 = signal2.ravel()
    correlation = np.correlate(signal1, signal2, mode='full')
    correlation = correlation / np.linalg.norm(correlation)
    if display == True:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(0, len(signal1)), signal1 )
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [uV]')
        plt.title('EEG signal Electrode {}'.format(idx))
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(0, len(signal2)), signal2 )
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [uV]')
        plt.title('BNN signal')
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(-len(correlation)/2, len(correlation)/2), correlation)
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
    return correlation_coef, correlation
