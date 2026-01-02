import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd



# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    # Input layer (4 features of the flower) --> Hidden Layer1 (number of neurons)
    #  --> Hidden Layer2 (n) --> output (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, output_features=3):
        super().__init__() # instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# Pick a manual seed for randomization
torch.manual_seed(41)


# Create an instance of model
model = Model()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

print(my_df.head())

# Change last column from strings to integers
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)

# Train Test Split! Sex X, y
X = my_df.drop('species', axis=1)
y = my_df['species']

# Convert these to numpy arrays
X = X.values
y = y.values

from sklearn.model_selection import train_test_split

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# Convert y labels to tensos long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criteria of the model to measure the error, how far the predictions are from
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

