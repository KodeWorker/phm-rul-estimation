from model import CNN
import torch
import joblib
import os
import glob
from label import parse_labels, SAMPLING_FREQ, CUTOFF_FREQ
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import uniform_filter1d

def rolling_estimation(regr, outputs, threshold, window_size, sliding_size):
    svr_outputs = []
    rolling_outputs = [x for x in outputs]
    overthreshold_cnt = 0
    cnt = 0
    while overthreshold_cnt < 3:
        
        mean = np.mean(rolling_outputs[-window_size:])
        var = np.var(rolling_outputs[-window_size:])
        #x = rolling_outputs[-window_size:]
        #in_feature = np.array([x + [mean, var]])
        in_feature = np.array([[mean, var]])
        #in_feature = np.array([x])
        output = regr.predict(in_feature)
        #print(in_feature, output)
        if output[0] >= threshold:
            overthreshold_cnt += 1
        
        cnt += 1
        rolling_outputs += [output[0]]
        svr_outputs += [output[0]]*sliding_size
        
        if cnt > 1000:
            break
        
    return svr_outputs

if __name__ == "__main__":
    
    threshold = 0.95
    window_size, sliding_size = 50, 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnn = CNN((1, 2, 2560)).to(device)
    cnn.load_state_dict(torch.load("weights/cnn.pt", device))
        
    regr = joblib.load('weights/regr.pkl')
    
    cycle_names = os.listdir("data/Test_set")
    
    cycle_names
    
    for cycle_name in cycle_names:
    
        # Pred
        test_files = glob.glob(os.path.join("data/Test_set", cycle_name, "acc_*.csv"))
        outputs = []
        for test_file in tqdm(test_files):
            df = pd.read_csv(test_file, header=None)
            if len(df.columns) == 1:
                df = pd.read_csv(test_file, sep=';', header=None)
            
            sig = df.iloc[:, -2:].values
            sig = np.transpose(sig, (1,0))
            sig = np.expand_dims(sig, axis=0)
            feature = torch.from_numpy(sig).float().to(device)
            output = cnn(feature)
            outputs.append(output.detach().cpu().numpy().flatten()[0])
        #print(outputs)
        svr_outputs = rolling_estimation(regr, outputs, threshold, window_size, sliding_size)
        
        # True
        total_files = glob.glob(os.path.join("data/Full_Test_Set", cycle_name, "acc_*.csv"))
        #cycle_labels = parse_labels(total_files, fs=SAMPLING_FREQ, cutoff=CUTOFF_FREQ)
        
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.title(cycle_name)
        #plt.plot(cycle_labels, c="red", label="true")
        plt.plot(outputs, c="blue", label="CNN-pred")
        plt.plot(range(len(outputs), len(outputs)+len(svr_outputs)),svr_outputs, linestyle="--", c="green", label="SVR-pred")
        plt.legend()
        plt.show()
        
        break