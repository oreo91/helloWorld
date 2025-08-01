#!/usr/bin/env python3
"""
Basic Classification Example using Custom Neural Network

This script demonstrates how to use the custom neural network implementation
for classification tasks. It includes data loading, preprocessing, training,
evaluation, and visualization.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import NeuralNetwork
from utils.data_utils import DataProcessor, plot_decision_boundary, evaluate_model_performance, plot_confusion_matrix, create_spiral_dataset

def main():
    print("=== Custom Neural Network Classification Demo ===\n")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Example 1: Iris Dataset (Multi-class classification)
    print("1. Training on Iris Dataset (Multi-class classification)")
    print("-" * 60)
    
    # Load iris dataset
    X, y, feature_names = data_processor.load_sample_dataset('iris')
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Features: {feature_names[:4]}...")  # Show first 4 features
    
    # Preprocess data
    X_processed, y_processed = data_processor.preprocess_data(X, y)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y_processed, test_size=0.2, val_size=0.2
    )
    
    # Create and train neural network
    # Network architecture: input_size -> 64 -> 32 -> num_classes
    nn_iris = NeuralNetwork(
        layers=[X_train.shape[1], 64, 32, len(np.unique(y))],
        learning_rate=0.01,
        activation='relu'
    )
    
    print(f"Training neural network with architecture: {nn_iris.layers}")
    nn_iris.train(X_train, y_train, epochs=500, batch_size=32, verbose=True)
    
    # Evaluate on test set
    test_loss, test_accuracy = nn_iris.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    predictions = nn_iris.predict(X_test)
    
    # Detailed evaluation
    metrics = evaluate_model_performance(y_test, predictions, 'classification')
    print(f"\nDetailed Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Plot training history
    nn_iris.plot_training_history()
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, predictions, ['Setosa', 'Versicolor', 'Virginica'])
    
    # Save the model
    nn_iris.save_model('ai_model/models/iris_model.pkl')
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Spiral Dataset (Binary classification with decision boundary)
    print("2. Training on Spiral Dataset (Binary classification)")
    print("-" * 60)
    
    # Create spiral dataset
    X_spiral, y_spiral = create_spiral_dataset(n_samples=300)
    print(f"Spiral dataset shape: {X_spiral.shape}")
    
    # Visualize original data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spiral, cmap=plt.cm.RdYlBu, s=50)
    plt.title('Original Spiral Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()
    
    # Preprocess spiral data
    data_processor_spiral = DataProcessor()  # New processor for spiral data
    X_spiral_processed, y_spiral_processed = data_processor_spiral.preprocess_data(
        X_spiral, y_spiral
    )
    
    # Split spiral data
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = data_processor_spiral.split_data(
        X_spiral_processed, y_spiral_processed, test_size=0.2, val_size=0.2
    )
    
    # Create neural network for spiral data (binary classification)
    nn_spiral = NeuralNetwork(
        layers=[2, 64, 64, 32, 1],  # Deeper network for complex pattern
        learning_rate=0.1,
        activation='relu'
    )
    
    print(f"Training neural network with architecture: {nn_spiral.layers}")
    nn_spiral.train(X_train_s, y_train_s, epochs=1000, batch_size=16, verbose=True)
    
    # Evaluate spiral model
    test_loss_s, test_accuracy_s = nn_spiral.evaluate(X_test_s, y_test_s)
    print(f"\nSpiral Test Results:")
    print(f"Loss: {test_loss_s:.4f}")
    print(f"Accuracy: {test_accuracy_s:.4f}")
    
    # Plot training history for spiral
    nn_spiral.plot_training_history()
    
    # Plot decision boundary
    plot_decision_boundary(nn_spiral, X_spiral_processed, y_spiral_processed, 
                         "Neural Network Decision Boundary - Spiral Data")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Synthetic Dataset with Feature Engineering
    print("3. Training on Synthetic Dataset with Feature Engineering")
    print("-" * 60)
    
    # Load synthetic dataset
    X_synth, y_synth, feature_names_synth = data_processor.load_sample_dataset('synthetic_classification')
    print(f"Synthetic dataset shape: {X_synth.shape}")
    print(f"Number of classes: {len(np.unique(y_synth))}")
    
    # Preprocess synthetic data
    X_synth_processed, y_synth_processed = data_processor.preprocess_data(X_synth, y_synth)
    
    # Create polynomial features
    X_synth_poly = data_processor.create_polynomial_features(X_synth_processed, degree=2)
    print(f"After polynomial features: {X_synth_poly.shape}")
    
    # Split synthetic data
    X_train_sy, X_val_sy, X_test_sy, y_train_sy, y_val_sy, y_test_sy = data_processor.split_data(
        X_synth_poly, y_synth_processed, test_size=0.2, val_size=0.2
    )
    
    # Create neural network for synthetic data
    nn_synth = NeuralNetwork(
        layers=[X_train_sy.shape[1], 128, 64, 32, len(np.unique(y_synth))],
        learning_rate=0.005,  # Lower learning rate for polynomial features
        activation='relu'
    )
    
    print(f"Training neural network with architecture: {nn_synth.layers}")
    nn_synth.train(X_train_sy, y_train_sy, epochs=300, batch_size=64, verbose=True)
    
    # Evaluate synthetic model
    test_loss_sy, test_accuracy_sy = nn_synth.evaluate(X_test_sy, y_test_sy)
    print(f"\nSynthetic Test Results:")
    print(f"Loss: {test_loss_sy:.4f}")
    print(f"Accuracy: {test_accuracy_sy:.4f}")
    
    # Plot training history for synthetic
    nn_synth.plot_training_history()
    
    # Comparison with different architectures
    print("\n" + "="*60 + "\n")
    print("4. Architecture Comparison")
    print("-" * 60)
    
    architectures = [
        [X_train.shape[1], 32, 3],  # Shallow network
        [X_train.shape[1], 64, 32, 3],  # Medium network
        [X_train.shape[1], 128, 64, 32, 3],  # Deep network
    ]
    
    results = []
    
    for i, arch in enumerate(architectures):
        print(f"\nTesting architecture {i+1}: {arch}")
        
        # Create and train model
        model = NeuralNetwork(layers=arch, learning_rate=0.01, activation='relu')
        model.train(X_train, y_train, epochs=200, batch_size=32, verbose=False)
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        results.append((arch, test_loss, test_accuracy))
        
        print(f"Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    
    # Plot architecture comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    losses = [r[1] for r in results]
    plt.bar(range(len(losses)), losses)
    plt.title('Test Loss by Architecture')
    plt.xlabel('Architecture')
    plt.ylabel('Loss')
    plt.xticks(range(len(losses)), [f'Arch {i+1}' for i in range(len(losses))])
    
    plt.subplot(1, 2, 2)
    accuracies = [r[2] for r in results]
    plt.bar(range(len(accuracies)), accuracies)
    plt.title('Test Accuracy by Architecture')
    plt.xlabel('Architecture')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(accuracies)), [f'Arch {i+1}' for i in range(len(accuracies))])
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Demo Complete ===")
    print("\nKey takeaways:")
    print("1. The custom neural network can handle both binary and multi-class classification")
    print("2. Deeper networks can learn more complex patterns (like the spiral)")
    print("3. Feature engineering (polynomial features) can improve performance")
    print("4. Different architectures have different trade-offs between complexity and performance")


if __name__ == "__main__":
    main()