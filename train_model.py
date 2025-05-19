#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.pytorch
from model import DiabetesNN
import joblib
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network model for diabetes prediction')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='/mnt/data/diabetes_project',
                        help='Directory containing the processed data files')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=str, default='64,32,16',
                        help='Comma-separated list of hidden layer dimensions (default: 64,32,16)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'elu', 'tanh'],
                        help='Activation function (default: relu)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization) (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer to use (default: adam)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (default: cpu)')
    parser.add_argument('--run_name', type=str, default='diabetes_nn',
                        help='Name for the MLflow run (default: diabetes_nn)')
    
    return parser.parse_args()

def load_data(data_dir):
    """Load and prepare the training and testing data."""
    print(f"Loading data from {data_dir}...")
    
    # Check if files exist first
    train_file = os.path.join(data_dir, 'diabetes_train.csv')
    test_file = os.path.join(data_dir, 'diabetes_test.csv')
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Train file exists: {os.path.exists(train_file)}")
        print(f"Test file exists: {os.path.exists(test_file)}")
        print("Listing directory contents:")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                print(f"- {os.path.join(root, file)}")
        raise FileNotFoundError(f"Could not find required data files in {data_dir}")
    
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Get column dtypes before processing
    print("Train data column types:")
    for col in train_data.columns:
        print(f"- {col}: {train_data[col].dtype}")
    
    # Make sure all data is numeric before converting to tensors
    # First, handle any categorical columns by checking if they need conversion
    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            print(f"Converting column {col} from object type to numeric")
            # For categorical columns, convert to category codes
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
            test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    
    # Fill any NaN values that might have been created
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    
    # Split into features and target
    X_train = train_data.drop('is_diabetic', axis=1).astype(np.float32).values
    y_train = train_data['is_diabetic'].astype(np.float32).values
    X_test = test_data.drop('is_diabetic', axis=1).astype(np.float32).values
    y_test = test_data['is_diabetic'].astype(np.float32).values
    
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset, X_train.shape[1]

def get_optimizer(optimizer_name, model_parameters, lr, weight_decay):
    """Get the optimizer based on the name."""
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience, device):
    """Train the model with early stopping."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # MLflow logging
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(best_model_state)
    
    # Plot and save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            # Convert outputs to probabilities and then to binary predictions
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    # For ROC-AUC, we need probabilities
    with torch.no_grad():
        all_probs = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs).flatten()
    roc_auc = roc_auc_score(all_targets, all_probs)
    
    return {
        'test_loss': test_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set the device
    device = torch.device(args.device)
    
    # Load data
    print("Loading data...")
    train_dataset, test_dataset, input_dim = load_data(args.data_dir)
    
    # Create data loaders
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout': args.dropout,
            'activation': args.activation,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'optimizer': args.optimizer,
            'patience': args.patience,
            'seed': args.seed
        })
        
        # Initialize model
        print("Initializing model...")
        model = DiabetesNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=args.dropout,
            activation=args.activation
        ).to(device)
        
        # Get optimizer
        optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate, args.weight_decay)
        
        # Define loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Train model
        print("Training model...")
        model, train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=args.epochs,
            patience=args.patience,
            device=device
        )
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log loss curve
        mlflow.log_artifact('loss_curve.png')
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Print results
        print("\nFinal Test Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Print MLflow tracking info
        print("\nMLflow run information:")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")
        
        # Save trained model to disk
        output_dir = args.data_dir
        torch.save(model.state_dict(), os.path.join(output_dir, 'diabetes_nn_model.pt'))
        print(f"Model saved to {os.path.join(output_dir, 'diabetes_nn_model.pt')}")

if __name__ == '__main__':
    main() 