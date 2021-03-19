import os
import glob
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq
from scipy.ndimage.filters import uniform_filter1d

SAMPLING_FREQ = 25.6 * 1000
CUTOFF_FREQ = 20000

def parse_labels(signal_files, fs=SAMPLING_FREQ, cutoff=CUTOFF_FREQ):

    cycle_labels = []
    for signal_file in tqdm(signal_files):
    
        df = pd.read_csv(signal_file, header=None)
        if len(df.columns) == 1:
                df = pd.read_csv(signal_file, sep=';', header=None)
                
        hsig = df.iloc[:, -2].values
        vsig = df.iloc[:, -1].values
        
        L = []
        for signal in [hsig, vsig]:
            X = fft(signal)
            freqs = fftfreq(len(signal)) * SAMPLING_FREQ
            
            X = np.abs(X[1:len(signal)//2])
            freqs = freqs[1:len(signal)//2]
            
            # moving average filter
            Xf = uniform_filter1d(X, size=5)
            
            L += [np.mean(Xf[ freqs <= cutoff ])]
        cycle_labels.append(np.max(L))
    cycle_labels = (np.array(cycle_labels) - np.min(cycle_labels)) / (np.max(cycle_labels) - np.min(cycle_labels))
    return cycle_labels
    
if __name__ == "__main__":

    data_dir = "data/Learning_set"
    output_dir = "data/Label"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cycle_names = os.listdir(data_dir)
    
    for cycle_name in cycle_names:
        
        print("Processing: {}".format(cycle_name))
        
        signal_files = glob.glob(os.path.join(data_dir, cycle_name, "acc_*.csv"))
        
        cycle_labels = parse_labels(signal_files, fs=SAMPLING_FREQ)
        """
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.plot(cycle_labels, c="red", label="label")
        plt.legend()
        plt.show()
        """
        #break
        np.save(os.path.join(output_dir, "{}.npy".format(cycle_name)), cycle_labels)
        