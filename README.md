# Diabetes Prediction with PyTorch

This project contains a PyTorch-based neural network for predicting diabetes using the engineered features from the diabetes dataset. The model architecture is customizable, allowing for different hidden dimensions, activation functions, and training parameters.

## Project Structure

- `model.py`: Contains the neural network architecture definition
- `train_model.py`: Executable script for training the model with different configurations

## Model Architecture

The model is a 4-layer feed-forward neural network with:
- Customizable hidden dimensions for each layer
- Batch normalization after each hidden layer
- Dropout for regularization
- Multiple activation function options

## Running Training Jobs on Domino

Set-up your domino project
- Clone this repo and use your own github and create a Domino project
- Add download this csv datafile, add to your project: https://drive.google.com/file/d/1NUyd6LGyMVnSic_DbGHOEKuwZx1WXm0G/view?usp=sharing


To train a model on Domino with default parameters:

```bash
python train_model.py
```

For custom training configurations:

```bash
python train_model.py --hidden_dims 128,64,32 --dropout 0.3 --activation leaky_relu --batch_size 128 --epochs 200 --learning_rate 0.0005 --optimizer adam
```

## Available Parameters

### Model Parameters:
- `--hidden_dims`: Comma-separated list of hidden layer dimensions (default: '64,32,16')
- `--dropout`: Dropout rate (default: 0.2)
- `--activation`: Activation function (choices: relu, leaky_relu, elu, tanh; default: relu)

### Training Parameters:
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs to train (default: 100)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (L2 regularization) (default: 1e-5)
- `--patience`: Patience for early stopping (default: 10)
- `--optimizer`: Optimizer to use (choices: adam, sgd, rmsprop; default: adam)
- `--device`: Device to use for training (default: cpu)

### Experiment Tracking:
- `--run_name`: Name for the MLflow run (default: diabetes_nn)

## Features and Target

The model is trained on engineered features including:
- Weight, calories_wk, hrs_exercise_wk, exercise_intensity, annual_income
- Derived features: weight_category, exercise_level, income_category
- Target variable: is_diabetic (1 for diabetic, 0 for non-diabetic)

## MLflow Integration

The training script automatically logs:
- Model parameters
- Training and validation loss curves
- Test metrics (accuracy, precision, recall, F1 score, ROC AUC)
- The trained model itself

The trained model will be saved in the MLflow model registry and also as a PyTorch state dict. 
