import torch
import torch.nn as nn
import torch.nn.functional as F

class DiabetesNN(nn.Module):
    """
    A customizable neural network for diabetes prediction with 4 fully connected layers.
    
    Parameters:
    - input_dim: Number of input features
    - hidden_dims: List of hidden layer dimensions (length 3)
    - dropout_rate: Dropout probability for regularization
    - activation: Activation function to use ('relu', 'leaky_relu', 'elu', or 'tanh')
    """
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.2, activation='relu'):
        super(DiabetesNN, self).__init__()
        
        # Validate inputs
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must be a list of length 3")
        
        # Define activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError("activation must be 'relu', 'leaky_relu', 'elu', or 'tanh'")
        
        # Define network layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dims[2], 1)
    
    def forward(self, x):
        # First hidden layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        
        # Output layer (no activation, as we'll use BCEWithLogitsLoss)
        x = self.fc4(x)
        
        return x

def get_model_summary(model, input_size):
    """
    Prints a summary of the model architecture.
    
    Parameters:
    - model: The PyTorch model
    - input_size: Tuple of input dimensions (batch_size, input_features)
    """
    from torchsummary import summary
    return summary(model, input_size=input_size[1:], batch_size=input_size[0]) 