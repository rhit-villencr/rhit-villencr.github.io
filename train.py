import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# All changeable parameters
csv = 'Concrete_train.csv'

inputColumns = [0,1,2,3,4,5,6,7]
outputColumns = [8]

batchSize = 100
learningRate = 0.01
num_epochs = 1000

hiddenSize = 32
num_hidden_layers = 1 # > 0

# Create PyTorch dataset
class CreateDataset(Dataset):
    def __init__(self,features,targets):
        self.features = torch.tensor(features,dtype=torch.float32)
        self.targets  = torch.tensor(targets,dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,idx):
        return self.features[idx], self.targets[idx]
    

# Define the neural network model
class NeuralNetRegressor(nn.Module):
    def __init__(self, input_size=len(inputColumns), hidden_size=hiddenSize, output_size=len(outputColumns)):
        super(NeuralNetRegressor, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        for i in range(1, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
    
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Pass through the hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        
        # Pass through the output layer
        x = self.output_layer(x)
        return x
    
    def predict(self, input_data):
        # Set the model to evaluation mode
        self.eval()
        # Convert the input data to a tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        # Perform a forward pass
        with torch.no_grad():  # Disable gradient calculation
            predictions = self(input_tensor)
        # Return the predictions as a numpy array
        return predictions.numpy()
 
def load_model(weights_path='neural_net_weights.pth'):
    model = NeuralNetRegressor()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model 
    
# Compute RMSE
def RMSE(y,yh):
    mse = ((y-yh)*(y-yh)).mean(axis=0)
    return np.sqrt(mse)

# Load .csv file 
df = pd.read_csv(csv)

# Set features to be the first 8 columns of the dataset
features = df.iloc[:, inputColumns]

# Set targets to be just the last column of the dataset
targets = df.iloc[:, outputColumns]

# Create PyTorch dataset
X = features.to_numpy()
Y = targets.to_numpy()
dataset = CreateDataset(X,Y)
data_loader = DataLoader(dataset,batch_size=batchSize,shuffle=True)

# Create the model instance
model = load_model()

# Define the loss function and the optimizer
cost_function = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=learningRate)

# Example of how to train the model (assuming you have your data loaders)
for epoch in range(1, num_epochs+1):
    for X,Y in data_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        Yh = model(X)
        
        # Compute the loss
        loss = cost_function(Yh,Y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0 or epoch == 1:
        percent = epoch/num_epochs
        print(f'{epoch}/{num_epochs}: [{"|"*int(percent * 50)}{"-"*int(50-(percent * 50))}]')
        
# Export the model to ONNX format
dummy_input = torch.randn(1, len(inputColumns))  # Use the appropriate input size
onnx_path = 'neural_net_model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, 
                  opset_version=11, do_constant_folding=True, 
                  input_names=['input'], output_names=['output'])      
        
# neural network RMSE
X = torch.tensor(features.to_numpy(),dtype=torch.float32)
Yh = model(X)
Yh = Yh.detach().numpy()
Y = targets.to_numpy()

print(f'RMSE of trained regressor: {RMSE(Y,Yh)[0]}')
