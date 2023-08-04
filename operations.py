import matplotlib.pyplot as plt 
import numpy as np
import scipy.special as sps  



def generate_histogram(weights_matrix):
    count, bisn, ignored = plt.hist(weights_matrix, 50, density=True)
    plt.plot(bisn, count)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of weights matrix')
    plt.show()

    
def generate_gamma_distribution(shape, scale, gamma_distribution): 
    count, bins, ignored = plt.hist(gamma_distribution, 50, density=True)
    y = bins**(shape-1)*(np.exp(-bins/scale) /  
                        (sps.gamma(shape)*scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')  
    plt.title('Gamma distribution for the values of inhibitory neurons')
    # plt.show()
    
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

def correlation_coefficient(signal1, signal2, idx, display = True):
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
        plt.title('EEG signal Electrode {}'.format(idx))
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
