import torch
import torch.nn as nn
import torch.nn.functional as F


# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    # Input layer (4 features of the flower) --> Hidden Layer1 (number of neurons)
    #  --> Hidden Layer2 (n) --> output (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, output_features=3):
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
