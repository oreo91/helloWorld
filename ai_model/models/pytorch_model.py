import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time

class PyTorchNeuralNetwork(nn.Module):
    """
    A flexible neural network implementation using PyTorch.
    Supports various architectures, optimizers, and training features.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 activation: str = 'relu', dropout_rate: float = 0.0):
        """
        Initialize the PyTorch neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout rate for regularization
        """
        super(PyTorchNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation and dropout for hidden layers
            if i < len(layer_sizes) - 2:  # Not the last layer
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f"Using device: {self.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                   optimizer: str = 'adam', weight_decay: float = 0.0, 
                   verbose: bool = True) -> None:
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            weight_decay: L2 regularization parameter
            verbose: Whether to print training progress
        """
        # Convert to tensors and move to device
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        if optimizer.lower() == 'adam':
            opt = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            opt = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == 'rmsprop':
            opt = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Loss function
        if self.output_size == 1:
            criterion = nn.BCEWithLogitsLoss()  # Binary classification
        else:
            criterion = nn.CrossEntropyLoss()  # Multi-class classification
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, 
                                                        patience=10, verbose=False)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.train()  # Set to training mode
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                opt.zero_grad()
                
                outputs = self.forward(batch_X)
                
                if self.output_size == 1:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    train_correct += (predicted.squeeze() == batch_y.float()).sum().item()
                else:
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == batch_y).sum().item()
                
                loss.backward()
                opt.step()
                
                train_loss += loss.item()
                train_total += batch_y.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                self.eval()  # Set to evaluation mode
                with torch.no_grad():
                    val_outputs = self.forward(X_val_tensor)
                    
                    if self.output_size == 1:
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor.float())
                        val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()
                        val_correct = (val_predicted.squeeze() == y_val_tensor.float()).sum().item()
                    else:
                        val_loss = criterion(val_outputs, y_val_tensor)
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_correct = (val_predicted == y_val_tensor).sum().item()
                    
                    val_accuracy = val_correct / y_val_tensor.size(0)
                    
                    self.val_losses.append(val_loss.item())
                    self.val_accuracies.append(val_accuracy)
                    
                    # Learning rate scheduling
                    scheduler.step(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                if X_val is not None:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - "
                          f"Val Loss: {val_loss.item():.4f} - Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        self.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            
            if self.output_size == 1:
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                return predictions.cpu().numpy().astype(int)
            else:
                _, predictions = torch.max(outputs, 1)
                return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        self.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            
            if self.output_size == 1:
                probabilities = torch.sigmoid(outputs)
            else:
                probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on test data."""
        self.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            
            if self.output_size == 1:
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs.squeeze(), y_tensor.float())
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                accuracy = (predicted.squeeze() == y_tensor.float()).float().mean().item()
            else:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, y_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_tensor).float().mean().item()
            
            return loss.item(), accuracy
    
    def plot_training_history(self) -> None:
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.train_accuracies, label='Training Accuracy', color='blue')
        if self.val_accuracies:
            axes[1].plot(self.val_accuracies, label='Validation Accuracy', color='red')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Model loaded from {filepath}")


class ConvolutionalNeuralNetwork(nn.Module):
    """
    A Convolutional Neural Network for image classification tasks.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        """
        Initialize the CNN.
        
        Args:
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
            num_classes: Number of output classes
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the size of flattened features
        # This is a placeholder - actual size depends on input image size
        self.feature_size = 128 * 4 * 4  # Assuming 32x32 input images
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def _calculate_feature_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate the size of features after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.view(1, -1).size(1)