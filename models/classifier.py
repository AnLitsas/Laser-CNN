import torch.nn as nn 
import torch 

class Classifier(nn.Module):
    def __init__(self, activation_function = 'relu', dropout_strength=0.2):
        super(Classifier, self).__init__()
        self.dropout_strength = dropout_strength
        self.activation_functions = {'selu':  torch.selu, 'relu': torch.relu, 'elu': nn.ELU()}
        self.activation_function = self.activation_functions[activation_function]
        

        # 3 convolutional layers. Image size is 256x256
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(dropout_strength)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(dropout_strength)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(dropout_strength)
        
        # 2 fully connected layers
        self.fc1 = nn.Linear(128*8*8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        
        # Max pooling layer
        self.max_pool2d = nn.MaxPool2d(3, 3)
        
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
        
        x = torch.softmax(self.fc3(x), dim=1)
        
        return x