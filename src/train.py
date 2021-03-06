from dataset import CNNDataset, CycleDataset
from model import CNN
from sklearn.svm import SVR
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import torch
import random
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_svr_features(outputs, window_size, sliding_size):
    features = []
    targets = []
    
    series = np.array(outputs).flatten()
    #print(series)
    #print("***", np.argwhere(series==np.max(series)))
    series = series[:np.argwhere(series==np.max(series))[0, 0] + 1]
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(series, c="red")
    plt.show()
    """
    #series = torch.flatten(outputs).detach().cpu().numpy()
    for i in range(window_size, len(series), sliding_size):
        
        index = len(series)-1 - i + window_size
        mean = np.mean(series[index-window_size:index])
        var = np.var(series[index-window_size:index])
        #x = series[index-window_size:index].flatten().tolist()
        #features += [x + [mean, var]]
        #features += [x]
        features += [[mean, var]]        
        targets += [series[index]]
    
    features, targets = np.asarray(features), np.asarray(targets)
    
    return features, targets

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random_state = 777
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    
    doTrainCNN = False
    doTrainSVR = True
    
    if doTrainCNN:
        
        cnn = CNN((1, 2, 2560)).to(device)
        #cnn.load_state_dict(torch.load("weights/cnn.pt", device))
        
        train_batch_size = 512
        train_dataset = CNNDataset("data/Learning_set", "data/Label")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        total_step = len(train_loader)
        
        criterion = SmoothL1Loss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-5)
        
        num_epochs = 50
        for epoch in range(num_epochs):
            for i, (signals, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                signals=signals.float().to(device)
                labels=labels.float().to(device)
                
                #print(signals.dtype)
                outputs = cnn(signals)
                #print(outputs.dtype, )
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                if (epoch+1) % 5 == 0 or epoch==0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        save_model_path = "weights/cnn.pt"
        torch.save(cnn.state_dict(), save_model_path) 
    
    if doTrainSVR:
    
        cnn = CNN((1, 2, 2560)).to(device)
        cnn.load_state_dict(torch.load("weights/cnn.pt", device))
        
        window_size, sliding_size = 50, 1
        regr = make_pipeline(StandardScaler(), SVR(C=5.09, epsilon=0.2))
        
        #cycle_name = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
        cycle_name = ["Bearing1_1", "Bearing1_2"]
        
        total_X = np.array([])
        total_Y = np.array([])
        for cycle_name in cycle_name:
        
            train_dataset = CycleDataset("data/Learning_set", "data/Label", cycle_name)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
            
            outputs = []
            true = []
            for i, (features, targets) in tqdm(enumerate(train_loader)):
                
                features = features.float().to(device)
                target = targets.float().to(device)
                    
                output = cnn(features)
            
                outputs.append(output.detach().cpu().numpy()[0])
                true.append(target.detach().cpu().numpy()[0])
            """
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(true, c="red", label="true")
            plt.plot(outputs, c="blue", label="pred")
            plt.legend()
            plt.show()
            """
            features, targets = get_svr_features(outputs, window_size, sliding_size)
            print(min(targets), max(targets), len(targets))
            
            if len(total_X) == 0 and len(total_Y) == 0:
                total_X = features
                total_Y = targets
            else:
                total_X = np.vstack((total_X, features))
                total_Y = np.append(total_Y, targets)
        
        #print(total_X.shape, total_Y.shape)
        regr.fit(total_X, total_Y)
        joblib.dump(regr, "weights/regr.pkl")