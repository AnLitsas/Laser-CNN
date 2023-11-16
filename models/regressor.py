import torch 
import torch.nn as nn 


class Regressor(nn.Module):
    def __init__(self, activation_function = 'relu', dropout_strength=0.2):
        super(Regressor, self).__init__()
        self.dropout_strength = dropout_strength
        self.activation_functions = {'selu':  torch.selu, 'relu': torch.relu, 'elu': nn.ELU()}
        self.activation_function = self.activation_functions[activation_function]
        
        
        # 3 convolutional layers. Image size is 512x512
        self.conv1 = nn.Conv2d(1, 32, 3).to('cpu')
        self.bn1 = nn.BatchNorm2d(32).to('cpu')
        self.dropout1 = nn.Dropout2d(dropout_strength).to('cpu')
        
        self.conv2 = nn.Conv2d(32, 64, 3).to('cpu')
        self.bn2 = nn.BatchNorm2d(64).to('cpu')
        self.dropout2 = nn.Dropout2d(dropout_strength).to('cpu')
        
        self.conv3 = nn.Conv2d(64, 128, 3).to('cpu')
        self.bn3 = nn.BatchNorm2d(128).to('cpu')
        self.dropout3 = nn.Dropout2d(dropout_strength).to('cpu')
        
        # 2 fully connected layers
        self.fc1 = nn.Linear(128*8*8, 64).to('cpu')
        self.fc2 = nn.Linear(64, 32).to('cpu')
        self.fc3 = nn.Linear(32, 1).to('cpu')
        
        # Max pooling layer
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=3)
        
    def forward(self, x):
        x = self.max_pool2d(self.activation_function(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.max_pool2d(self.activation_function(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.max_pool2d(self.activation_function(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = x.view(-1, 128*8*8)
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        
        x = self.fc3(x)
       
        return x