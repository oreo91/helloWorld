#!/usr/bin/env python3
"""
Simple test script to verify the AI model implementation works correctly.
This creates a basic neural network and trains it on a simple dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.neural_network import NeuralNetwork
from utils.data_utils import DataProcessor

def test_basic_model():
    print("🧠 Testing Basic AI Model Implementation")
    print("=" * 50)
    
    # Generate a simple dataset
    print("📊 Generating synthetic dataset...")
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, 
                             n_informative=3, n_redundant=1, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Create and train the neural network
    print("\n🏗️  Creating neural network...")
    model = NeuralNetwork([4, 8, 4, 1], learning_rate=0.1, activation='relu')
    
    print("🚀 Training the model...")
    model.train(X_train_scaled, y_train.reshape(-1, 1), epochs=100, verbose=False)
    
    # Create history dictionary from stored history
    history = {
        'loss': model.loss_history,
        'accuracy': model.accuracy_history
    }
    
    # Test the model
    print("📈 Evaluating model performance...")
    train_predictions = model.predict_proba(X_train_scaled)
    test_predictions = model.predict_proba(X_test_scaled)
    
    # Calculate accuracy
    train_accuracy = np.mean((train_predictions > 0.5) == y_train.reshape(-1, 1))
    test_accuracy = np.mean((test_predictions > 0.5) == y_test.reshape(-1, 1))
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("📊 Training plots saved as 'training_results.png'")
    
    print("\n✅ Model test completed successfully!")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.4f}")
    
    return model, test_accuracy

def test_data_utils():
    print("\n🔧 Testing Data Utilities...")
    print("=" * 50)
    
    # Test data processor
    processor = DataProcessor()
    
    # Load a sample dataset
    X, y, feature_names = processor.load_sample_dataset('iris')
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    print(f"Feature names: {feature_names}")
    
    # Test preprocessing
    X_processed, y_processed = processor.preprocess_data(X, y)
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Data range: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
    
    print("✅ Data utilities test completed!")
    
    return processor

if __name__ == "__main__":
    try:
        # Test the basic model
        model, accuracy = test_basic_model()
        
        # Test data utilities
        processor = test_data_utils()
        
        print("\n🎉 All tests passed!")
        print(f"Your AI model achieved {accuracy:.1%} accuracy on the test set.")
        print("\nNext steps:")
        print("1. Run 'python examples/basic_classification.py' for a comprehensive demo")
        print("2. Explore the models/ directory to see different implementations")
        print("3. Check out examples/ for various use cases")
        print("4. Read README.md for detailed documentation")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Make sure all required packages are installed.")
        print("Run: pip install -r requirements.txt")