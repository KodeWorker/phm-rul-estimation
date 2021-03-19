import torch
from torch.nn import Module, Conv1d, MaxPool1d, Linear
from torch.nn.functional import relu
"""
class PrintLayer(Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
                    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
"""

class CNN(Module):
    def __init__(self, in_size):
        super().__init__()        
        #self.conv1 = Conv1d(in_size[1], 64, kernel_size=100, stride=50)
        self.conv1 = Conv1d(in_size[1], 64, kernel_size=4, stride=2)
        self.mp1 = MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = Conv1d(64, 64, kernel_size=2, stride=1)
        self.mp2 = MaxPool1d(kernel_size=2, stride=2)
        self.get_linear_size(torch.rand(in_size))
        self.fc1= Linear(self.linear_in_size,100)
        self.fc2 = Linear(100,1)
        #self.print_layer = PrintLayer()
        
    def get_linear_size(self, x):
        x = self.conv1(x)
        x = relu(self.mp1(x))
        x = self.conv2(x)
        x = relu(self.mp2(x))
        self.linear_in_size = x.numel()
        
    def forward(self, x):
    
        #x = self.print_layer(x)
        x = self.conv1(x)
        #x = self.print_layer(x)
        x = relu(self.mp1(x))
        #x = self.print_layer(x)
        x = self.conv2(x)
        #x = self.print_layer(x)
        x = relu(self.mp2(x))
        #x = self.print_layer(x)
        x = x.view(-1, self.linear_in_size)
        #x = self.print_layer(x)
        x = self.fc1(x)
        #x = self.print_layer(x)
        x = torch.sigmoid(self.fc2(x))
        #x = self.print_layer(x)
        return x
        
if __name__ == "__main__":
    in_tensor = torch.rand((20, 2, 2560))
    model = CNN((1, 2, 2560))
    out_tensor = model(in_tensor)
    print(out_tensor.shape)
    """
    torch.Size([1, 1, 2560])
    torch.Size([1, 64, 51])
    torch.Size([1, 64, 25])
    torch.Size([1, 64, 24])
    torch.Size([1, 64, 12])
    torch.Size([1, 768])
    torch.Size([1, 100])
    torch.Size([1, 1])
    """