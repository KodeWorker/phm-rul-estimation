import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

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
            # normalize label
            signal_labels = (signal_labels - min(signal_labels)) / (max(signal_labels) - min(signal_labels))
            #print(min(signal_labels), max(signal_labels))
            
            self.filenames += signal_files
            self.labels += signal_labels.tolist()
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        
        df = pd.read_csv(filename, header=None)
        sig = df.iloc[:, -2:].values
        sig = np.amax(sig, axis=-1)
        sig = np.expand_dims(sig, axis=0)
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
        # normalize label
        signal_labels = (signal_labels - min(signal_labels)) / (max(signal_labels) - min(signal_labels))
        #print(min(signal_labels), max(signal_labels))
        
        self.filenames += signal_files
        self.labels += signal_labels.tolist()
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        
        df = pd.read_csv(filename, header=None)
        sig = df.iloc[:, -2:].values
        sig = np.amax(sig, axis=-1)
        sig = np.expand_dims(sig, axis=0)
        data = torch.from_numpy(sig)
        label = torch.from_numpy(np.asarray([label]))
        
        return data, label
    
    def __len__(self):
        return len(self.filenames)

"""
class SVRDataset(data.Dataset):
    
    def __init__(self, data_dir, label_dir, window_size, sliding_size):
        
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.window_size = window_size
        self.sliding_size = sliding_size
        self.pairs = []
        self.initialize()
    
    def initialize(self):
        lable_files = glob.glob(os.path.join(self.label_dir, "*.npy"))
        cycle_names = [os.path.basename(f).replace(".npy", "") for f in lable_files]
        for cycle_name, lable_file  in zip(cycle_names, lable_files):
            
            features = []
            labels = []
            
            signal_files = glob.glob(os.path.join(self.data_dir, cycle_name, "acc_*.csv"))
            
            signal_labels = np.load(lable_file)
            signal_labels = (signal_labels - min(signal_labels)) / (max(signal_labels) - min(signal_labels))
            
            for signal_file, signal_label in zip(signal_files, signal_labels):
                df = pd.read_csv(signal_file, header=None)
                sig = df.iloc[:, -2:].values
                sig = np.amax(sig, axis=-1)
                mean, std = np.mean(sig), np.std(sig)
                
                features += [(mean, std, signal_label)]
                labels += [signal_label]
            
            # create (features, target) pairs with window_size and sliding_size
            for i in range(self.window_size, len(features)-1, self.sliding_size):
                
                pair_features = []
                for j in range(self.window_size):
                    pair_features += [features[i-j]]
                pair_targets = labels[i+1]
                
                self.pairs += [(np.array(pair_features).flatten(), np.array([pair_targets]))]
            
    def __getitem__(self, index):
        feature, target = self.pairs[index]
        return feature, target
    
    def __len__(self):
        return len(self.pairs)
"""
if __name__ == "__main__":
    
    data_dir = "data/Learning_set"
    label_dir = "data/Label"
    
    
    dataset = CNNDataset(data_dir, label_dir)
    
    for x,y in dataset:
        print(x.shape, y.shape)
        break
    """
    
    window_size = 50
    sliding_size = 1
    #! it should be estimated EDI
    dataset = SVRDataset(data_dir, label_dir, window_size, sliding_size)
    
    for x,y in dataset:
        print(x.shape, y.shape)
        break
    
    """