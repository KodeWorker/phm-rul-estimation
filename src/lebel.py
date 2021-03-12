import os
import glob
import math
import pandas as pd
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm
from datetime import datetime
from PyEMD import EMD

# specification from https://arxiv.org/abs/1812.03315
ETA=13
L_BALL=3.5 * 1e-3 # mm -> m
L_PITCH=25.6 * 1e-3 # mm -> m
THETA=0
ROTATION_FREQ_1=1800 / 60 # r/m -> r/sec
ROTATION_FREQ_2=1600 / 60 # r/m -> r/sec
DYNAMIC_LOAD_1=4000
DYNAMIC_LOAD_2=4200

def inner_race_frequency(n_balls, rotation_freq, contact_angle_radians, ball_diameter, pitch_diameter):
    return (n_balls/2) * rotation_freq * ( 1 + ball_diameter/pitch_diameter*np.cos(contact_angle_radians))

def outer_race_frequency(n_balls, rotation_freq, contact_angle_radians, ball_diameter, pitch_diameter):
    return (n_balls/2) * rotation_freq * ( 1 - ball_diameter/pitch_diameter*np.cos(contact_angle_radians))

# according to https://www.craftbearing.com/pdf/Frequency-Data.pdf
# function in Table I is incorrect!
def ball_frequency(n_balls, rotation_freq, contact_angle_radians, ball_diameter, pitch_diameter):
    return pitch_diameter/ball_diameter* rotation_freq * ( 1 - ((ball_diameter**2)/(pitch_diameter**2))*(np.cos(contact_angle_radians)**2))

# https://dsp.stackexchange.com/questions/48657/plotting-hilbert-and-marginal-spectra-in-python
# https://github.com/liangliannie/hht-spectrum/blob/master/source/hht.py

def hilb(s, unwrap=False):
    """
    Performs Hilbert transformation on signal s.
    Returns amplitude and phase of signal.
    Depending on unwrap value phase can be either
    in range [-pi, pi) (unwrap=False) or
    continuous (unwrap=True).
    """    
    H = hilbert(s)
    amp = np.abs(H)
    phase = np.arctan2(H.imag, H.real)
    if unwrap: phase = np.unwrap(phase)
    return amp, phase

def HilbertHaungTransform(imfs):
    """
    Performs Hilbert transformation on imfs.
    Returns frequency and amplitude of signal.
    """
    n_imfs = imfs.shape[0]
    f = []
    a = []
    for i in range(n_imfs - 1):
        # upper, lower = pyhht.utils.get_envelops(imfs[i, :])
        inst_imf = imfs[i, :]  # /upper
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        inst_freq = (2 * math.pi) / np.diff(phase)  #

        inst_freq = np.insert(inst_freq, len(inst_freq), inst_freq[-1])
        inst_amp = np.insert(inst_amp, len(inst_amp), inst_amp[-1])

        f.append(inst_freq)
        a.append(inst_amp)
    return np.asarray(f).T, np.asarray(a).T

def MarginalHilbertSpectrum(freq, amp):
    ftemp = []
    Atemp = []
    for f in np.unique(freq):
        idx = np.where(freq==f)[0]
        ftemp.append(f)
        Atemp.append(np.sum(amp[idx]))
        
    sort_idx = np.argsort(ftemp)
    ftemp = np.array(ftemp)[sort_idx]
    Atemp = np.array(Atemp)[sort_idx]
    return ftemp, Atemp

def get_label(freq, amp, f):
    idx = (np.abs(freq - f)).argmin()
    return amp[idx]

if __name__ == "__main__":

    data_dir = "data/Learning_set"
    output_dir = "data/Label"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    emd = EMD()
    cycle_names = os.listdir(data_dir)
    
    for cycle_name in cycle_names:
        
        print("Processing: {}".format(cycle_name))
        
        if "Bearing1" in cycle_name:
            ROTATION_FREQ = ROTATION_FREQ_1
            DYNAMIC_LOAD = DYNAMIC_LOAD_1
        elif "Bearing2" in cycle_name:
            ROTATION_FREQ = ROTATION_FREQ_2
            DYNAMIC_LOAD = DYNAMIC_LOAD_2
        elif "Bearing3" in cycle_name:
            print("no bearing spec")
            break
        
        f_inner = inner_race_frequency(ETA, ROTATION_FREQ, THETA , L_BALL, L_PITCH)
        f_outer = outer_race_frequency(ETA, ROTATION_FREQ, THETA, L_BALL, L_PITCH)
        f_ball = ball_frequency(ETA, ROTATION_FREQ, THETA, L_BALL, L_PITCH)
        
        print("f_outer: {:.2f} Hz, f_inner: {:.2f} Hz, f_ball: {:.2f} Hz".format(f_outer, f_inner, f_ball))
        
        signal_files = glob.glob(os.path.join(data_dir, cycle_name, "acc_*.csv"))
        cycle_labels = []
        for signal_file in tqdm(signal_files):
        
            df = pd.read_csv(signal_file, header=None)
            
            hsig = df.iloc[:, -2].values
            vsig = df.iloc[:, -1].values
            date = [datetime(2021, 1, 1, int(df.iloc[i, 0]), int(df.iloc[i, 1]), int(df.iloc[i, 2]), int(df.iloc[i, 3])) for i in range(len(df))]
            t = [(d-date[0]).total_seconds for d in date]
            
            L = []
            for data in [hsig, vsig]:
                
                imfs = emd(data, t)
                freq, amp = HilbertHaungTransform(imfs)
                freq, amp = MarginalHilbertSpectrum(freq, amp)
                
                #import matplotlib.pyplot as plt
                #plt.figure()
                #plt.plot(freq, amp)
                #plt.show()
                
                L += [get_label(freq, amp, f_inner), get_label(freq, amp, f_outer), get_label(freq, amp, f_ball)]
            cycle_labels.append(np.max(L))
            
            #print(freq)
            #break
        np.save(os.path.join(output_dir, "{}.npy".format(cycle_name)), np.array(cycle_labels))
        