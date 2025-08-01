import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pickle

class NeuralNetwork:
    """
    A simple feedforward neural network implemented from scratch using NumPy.
    This is for educational purposes to understand how neural networks work internally.
    """
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01, activation: str = 'relu'):
        """
        Initialize the neural network.
        
        Args:
            layers: List of integers representing the number of neurons in each layer
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            # Xavier initialization
            weight = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Store training history
        self.loss_history = []
        self.accuracy_history = []
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = self._activation_function(x)
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_features)
            
        Returns:
            output: Network output
            activations: List of activations for each layer
            z_values: List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        current_activation = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i < len(self.weights) - 1:  # Hidden layers
                current_activation = self._activation_function(z)
            else:  # Output layer (use sigmoid for binary classification, softmax for multi-class)
                if self.layers[-1] == 1:  # Binary classification
                    current_activation = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                else:  # Multi-class classification
                    # Softmax
                    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                    current_activation = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            activations.append(current_activation)
        
        return current_activation, activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                z_values: List[np.ndarray]) -> None:
        """
        Backward propagation to compute gradients and update weights.
        
        Args:
            X: Input data
            y: True labels
            activations: Activations from forward pass
            z_values: Pre-activation values from forward pass
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        if self.layers[-1] == 1:  # Binary classification
            dz = activations[-1] - y.reshape(-1, 1)
        else:  # Multi-class classification
            dz = activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            dW[i] = np.dot(activations[i].T, dz) / m
            db[i] = np.mean(dz, axis=0, keepdims=True)
            
            if i > 0:  # Not the first layer
                da = np.dot(dz, self.weights[i].T)
                dz = da * self._activation_derivative(z_values[i-1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        m = y_true.shape[0]
        
        if self.layers[-1] == 1:  # Binary classification
            y_true = y_true.reshape(-1, 1)
            loss = -np.mean(y_true * np.log(y_pred + 1e-8) + 
                           (1 - y_true) * np.log(1 - y_pred + 1e-8))
        else:  # Multi-class classification
            loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              batch_size: Optional[int] = None, verbose: bool = True) -> None:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (samples, features)
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
        """
        if batch_size is None:
            batch_size = X.shape[0]
        
        # Convert labels to one-hot encoding for multi-class classification
        if self.layers[-1] > 1 and len(y.shape) == 1:
            num_classes = self.layers[-1]
            y_onehot = np.zeros((y.shape[0], num_classes))
            y_onehot[np.arange(y.shape[0]), y] = 1
            y = y_onehot
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass
                output, activations, z_values = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_batch, output)
                epoch_loss += loss
                
                # Compute accuracy
                if self.layers[-1] == 1:  # Binary classification
                    predictions = (output > 0.5).astype(int)
                    accuracy = np.mean(predictions.flatten() == y_batch.flatten())
                else:  # Multi-class classification
                    predictions = np.argmax(output, axis=1)
                    true_labels = np.argmax(y_batch, axis=1)
                    accuracy = np.mean(predictions == true_labels)
                
                epoch_accuracy += accuracy
                num_batches += 1
                
                # Backward pass
                self.backward(X_batch, y_batch, activations, z_values)
            
            # Average loss and accuracy for this epoch
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)
            
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        output, _, _ = self.forward(X)
        
        if self.layers[-1] == 1:  # Binary classification
            return (output > 0.5).astype(int)
        else:  # Multi-class classification
            return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        output, _, _ = self.forward(X)
        return output
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on test data."""
        predictions = self.predict(X)
        if self.layers[-1] == 1:  # Binary classification
            accuracy = np.mean(predictions.flatten() == y.flatten())
        else:  # Multi-class classification
            accuracy = np.mean(predictions == y)
        
        # Compute loss
        output, _, _ = self.forward(X)
        if self.layers[-1] > 1 and len(y.shape) == 1:
            num_classes = self.layers[-1]
            y_onehot = np.zeros((y.shape[0], num_classes))
            y_onehot[np.arange(y.shape[0]), y] = 1
            y = y_onehot
        
        loss = self.compute_loss(y, output)
        
        return loss, accuracy
    
    def plot_training_history(self) -> None:
        """Plot training loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layers': self.layers,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layers = model_data['layers']
        self.learning_rate = model_data['learning_rate']
        self.activation = model_data['activation']
        self.loss_history = model_data.get('loss_history', [])
        self.accuracy_history = model_data.get('accuracy_history', [])
        
        print(f"Model loaded from {filepath}")