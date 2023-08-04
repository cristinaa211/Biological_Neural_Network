import mne 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import correlate

def xcorr(sig1, sig2): 
    "Plot cross-correlation (full) between two signals."
    N = max(len(sig1), len(sig2)) 
    n = min(len(sig1), len(sig2)) 
    if N == len(sig2): 
        lags = np.arange(-N + 1, n) 
    else: 
        lags = np.arange(-n + 1, N) 
    sig1 = sig1 / np.std(sig1)
    sig2 = sig2 / np.std(sig2)
    cross_corr = correlate(sig1, sig2 , 'same') / n
    if any(cross_corr) > 1 :
        print("something is wrong")
    return lags, cross_corr



def plot_eeg_signals(df):
    ch_names = list(df.columns.values)
    df_eeg = df.copy()
    df_eeg = df_eeg.transpose()
    info = mne.create_info(ch_names = ch_names, sfreq = 2048)
    raw = mne.io.RawArray(data = df_eeg, info = info)
    raw.plot()
    plt.show()
    print(raw)
    print(raw.info)

def compute_freq_bands(data, fs):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                'Theta': (4, 8),
                'Alpha': (8, 12),
                'Beta': (12, 30),
                'Gamma': (30, 45)}
    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                        (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    # Plot the data (using pandas here cause it's easy)
    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = eeg_bands.keys()
    df['val'] = [eeg_band_fft[band] for band in eeg_bands]
    return df

if __name__ == "__main__":
    filename = './eeg_math_subj.csv'
    df_eeg = pd.read_csv(filename)
    df_eeg = df_eeg.drop(columns=["'EDF Annotations'"]).reset_index(drop=True)
    print(df_eeg.head(5))
    plot_eeg_signals(df_eeg)
    columns = list(df_eeg.columns.values)
    idx = 0
    fig, axes = plt.subplots(nrows = 7, ncols = 3)
    params = {
    'axes.titlesize': 8,
    'axes.labelsize': 5}
    plt.rcParams.update(params)
    for col in columns:
        df_pr = compute_freq_bands(df_eeg[col], 2048)
        df_pr.plot(kind = 'bar', x='band', y='val', legend=False, ax = axes[idx % 7, idx % 3] )
        axes[idx % 7, idx % 3].set_title('{}'.format(col))
        idx += 1
    plt.show()

