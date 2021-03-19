import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from scipy.signal import find_peaks

class CNNDataset(data.Dataset):
    
    def __init__(self, data_dir, label_dir):
        
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.filenames = []
        self.labels = []
        self.initialize()
    
    def initialize(self):
        lable_files = glob.glob(os.path.join(self.label_dir, "*.npy"))
        cycle_names = [os.path.basename(f).replace(".npy", "") for f in lable_files]
        for cycle_name, lable_file  in zip(cycle_names, lable_files):
            signal_files = glob.glob(os.path.join(self.data_dir, cycle_name, "acc_*.csv"))
            signal_labels = np.load(lable_file)
            
            self.filenames += signal_files
            self.labels += signal_labels.tolist()
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        
        df = pd.read_csv(filename, header=None)
        sig = df.iloc[:, -2:].values
        #sig = (sig - np.mean(sig))/np.std(sig)
        sig = np.transpose(sig, (1,0))
        data = torch.from_numpy(sig)
        label = torch.from_numpy(np.asarray([label]))
        
        return data, label
    
    def __len__(self):
        return len(self.filenames)

class CycleDataset(data.Dataset):
    
    def __init__(self, data_dir, label_dir, cycle_name):
        
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.cycle_name = cycle_name
        self.filenames = []
        self.labels = []
        self.initialize()
    
    def initialize(self):
    
        cycle_name = self.cycle_name
        lable_file = os.path.join(self.label_dir, "{}.npy".format(self.cycle_name))
        
        signal_files = glob.glob(os.path.join(self.data_dir, cycle_name, "acc_*.csv"))
        signal_labels = np.load(lable_file)
        
        self.filenames += signal_files
        self.labels += signal_labels.tolist()
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        
        df = pd.read_csv(filename, header=None)
        sig = df.iloc[:, -2:].values
        #sig = (sig - np.mean(sig))/np.std(sig)
        sig = np.transpose(sig, (1,0))
        data = torch.from_numpy(sig)
        label = torch.from_numpy(np.asarray([label]))
        
        return data, label
    
    def __len__(self):
        return len(self.filenames)
        
if __name__ == "__main__":
    
    data_dir = "data/Learning_set"
    label_dir = "data/Label"
    
    dataset = CycleDataset(data_dir, label_dir, "Bearing1_1")
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    import matplotlib.pyplot as plt
    
    for x,y in loader:
        
        print(x.shape)
        plt.figure()
        plt.plot(y)
        plt.show()
        
    """
    
    window_size = 50
    sliding_size = 1
    #! it should be estimated EDI
    dataset = SVRDataset(data_dir, label_dir, window_size, sliding_size)
    
    for x,y in dataset:
        print(x.shape, y.shape)
        break
    
    """