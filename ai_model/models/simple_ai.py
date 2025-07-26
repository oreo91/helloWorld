#!/usr/bin/env python3
"""
Simple AI - Easy-to-use interface for creating AI models

This module provides a simplified interface for creating and training AI models
without needing to understand the underlying complexity.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import NeuralNetwork
from utils.data_utils import DataProcessor

class SimpleAI:
    """
    A simplified interface for creating AI models.
    Perfect for beginners who want to get started quickly.
    """
    
    def __init__(self, model_type='classifier', hidden_layers=[64, 32]):
        """
        Initialize a simple AI model.
        
        Args:
            model_type: Either 'classifier' or 'regressor'
            hidden_layers: List of hidden layer sizes
        """
        self.model_type = model_type
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_sample_data(self, n_samples=1000, n_features=20, difficulty='medium'):
        """
        Create sample data for testing your model.
        
        Args:
            n_samples: Number of data points
            n_features: Number of input features
            difficulty: 'easy', 'medium', or 'hard'
        """
        if difficulty == 'easy':
            noise = 0.1
            n_redundant = 2
        elif difficulty == 'medium':
            noise = 0.15
            n_redundant = 5
        else:  # hard
            noise = 0.25
            n_redundant = 10
            
        if self.model_type == 'classifier':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_redundant=n_redundant,
                n_informative=n_features - n_redundant,
                n_clusters_per_class=1,
                flip_y=noise,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise * 100,
                random_state=42
            )
            
        return X, y
    
    def train(self, X, y, test_size=0.2, epochs=100, learning_rate=0.01, verbose=True):
        """
        Train the AI model on your data.
        
        Args:
            X: Input features (2D array)
            y: Target values (1D array)
            test_size: Fraction of data to use for testing
            epochs: Number of training iterations
            learning_rate: How fast the model learns
            verbose: Whether to print progress
        """
        if verbose:
            print(f"🧠 Training {self.model_type} with {X.shape[0]} samples...")
            
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Create the neural network
        input_size = X_train_scaled.shape[1]
        output_size = 1
        layers = [input_size] + self.hidden_layers + [output_size]
        
        self.model = NeuralNetwork(
            layers=layers,
            learning_rate=learning_rate,
            activation='relu' if self.model_type == 'classifier' else 'linear'
        )
        
        # Train the model
        if self.model_type == 'classifier':
            y_train = y_train.reshape(-1, 1)
            
        self.model.train(X_train_scaled, y_train.reshape(-1, 1), epochs=epochs, verbose=verbose)
        self.is_trained = True
        
        # Evaluate the model
        if verbose:
            self.evaluate(verbose=True)
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
            
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'classifier':
            probabilities = self.model.predict_proba(X_scaled)
            return (probabilities > 0.5).astype(int).flatten()
        else:
            return self.model.predict_proba(X_scaled).flatten()
    
    def evaluate(self, verbose=True):
        """
        Evaluate the model performance on test data.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
            
        predictions = self.predict(self.scaler.inverse_transform(self.X_test))
        
        if self.model_type == 'classifier':
            accuracy = accuracy_score(self.y_test, predictions)
            
            if verbose:
                print(f"\n📊 Model Performance:")
                print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"   Test samples: {len(self.y_test)}")
                
            return {'accuracy': accuracy}
        else:
            mse = np.mean((self.y_test - predictions) ** 2)
            rmse = np.sqrt(mse)
            
            if verbose:
                print(f"\n📊 Model Performance:")
                print(f"   RMSE: {rmse:.3f}")
                print(f"   MSE: {mse:.3f}")
                print(f"   Test samples: {len(self.y_test)}")
                
            return {'rmse': rmse, 'mse': mse}
    
    def plot_training_history(self):
        """
        Plot the training history to see how the model learned.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")
            
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.model.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot accuracy (for classifiers)
        if self.model_type == 'classifier' and hasattr(self.model, 'accuracy_history'):
            plt.subplot(1, 2, 2)
            plt.plot(self.model.accuracy_history)
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Args:
            filename: Name of the file to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
            
        import pickle
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'hidden_layers': self.hidden_layers
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"✅ Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a trained model from a file.
        
        Args:
            filename: Name of the file to load the model from
        """
        import pickle
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.hidden_layers = model_data['hidden_layers']
        self.is_trained = True
        
        print(f"✅ Model loaded from {filename}")

# Example usage functions
def quick_classifier_example():
    """
    Quick example of creating and training a classifier.
    """
    print("🚀 Quick Classifier Example")
    print("=" * 30)
    
    # Create a simple AI classifier
    ai = SimpleAI(model_type='classifier')
    
    # Generate sample data
    X, y = ai.create_sample_data(n_samples=1000, difficulty='medium')
    
    # Train the model
    ai.train(X, y, epochs=50)
    
    # Plot training history
    ai.plot_training_history()
    
    return ai

def quick_regressor_example():
    """
    Quick example of creating and training a regressor.
    """
    print("🚀 Quick Regressor Example")
    print("=" * 30)
    
    # Create a simple AI regressor
    ai = SimpleAI(model_type='regressor')
    
    # Generate sample data
    X, y = ai.create_sample_data(n_samples=1000, difficulty='medium')
    
    # Train the model
    ai.train(X, y, epochs=50)
    
    # Plot training history
    ai.plot_training_history()
    
    return ai

if __name__ == "__main__":
    # Run both examples
    classifier = quick_classifier_example()
    regressor = quick_regressor_example()