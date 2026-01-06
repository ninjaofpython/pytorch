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

# Train our model
# Epochs (One run through all the training data in our neural network)
epochs = 200
losses = []

for i in range(epochs):
    # Go forward and get a prediction, forward pass maybe?
    y_pred = model.forward(X_train) # Get predicted results

    # Measure the loss/error, it's going to be high at first
    loss = criterion(y_pred, y_train) # Predicted values vs the y_train

    # Keep track of our losses
    losses.append(loss.detach().numpy())

    # Print every 10 epoch
    if i % 10 == 0:
        print(f"Epoch: {i} and loss: {loss}")

    # Do some back propagation: take the error rate of forward propagation and feed it back
    # Through the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Graph it out!
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("Epoch")
print(plt.show())


# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad(): # Basically turn off back propogation
    y_eval = model.forward(X_test) # X_test are features from our test set, y_eval will be predictions
    loss = criterion(y_eval, y_test) # Find the loss or error
    print("Here's the loss")
    print(loss)


correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if y_test[i] == 0:
            x = "setosa"
        elif y_test[i] == 1:
            x = 'versicolor'
        else:
            x = 'virginica'

        print(f"{i+1}.) {str(y_val)} \t {y_test[i]}")

        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct +=1

    print(f"We got correct {correct} xxxxxxx")


new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])

with torch.no_grad():
    print("New iris xxxxxxx")
    print(model(new_iris))


    

    
