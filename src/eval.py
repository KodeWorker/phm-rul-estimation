from model import CNN
import torch
import joblib

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cnn = CNN((1, 1, 2560)).to(device)
    cnn.load_state_dict(torch.load("weights/cnn.pt", device))
        
    svr = joblib.load('weights/svr.pkl')
    result = loaded_model.score(X_test, Y_test)
    print(result)
