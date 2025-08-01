#!/usr/bin/env python3
"""
Advanced PyTorch Neural Network Example

This script demonstrates advanced features of the PyTorch neural network
implementation including different optimizers, regularization techniques,
and comparison with the custom implementation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pytorch_model import PyTorchNeuralNetwork, ConvolutionalNeuralNetwork
from models.neural_network import NeuralNetwork  # Custom implementation for comparison
from utils.data_utils import DataProcessor, evaluate_model_performance, plot_confusion_matrix

def compare_optimizers():
    """Compare different optimizers on the same dataset."""
    print("=== Optimizer Comparison ===\n")
    
    # Load and prepare data
    data_processor = DataProcessor()
    X, y, feature_names = data_processor.load_sample_dataset('wine')
    X_processed, y_processed = data_processor.preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y_processed, test_size=0.2, val_size=0.2
    )
    
    print(f"Dataset: Wine classification")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    
    optimizers = ['adam', 'sgd', 'rmsprop']
    results = {}
    
    for optimizer in optimizers:
        print(f"\nTraining with {optimizer.upper()} optimizer...")
        
        # Create model
        model = PyTorchNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=[128, 64, 32],
            output_size=len(np.unique(y)),
            activation='relu',
            dropout_rate=0.3
        )
        
        # Train model
        model.train_model(
            X_train, y_train, X_val, y_val,
            epochs=200, batch_size=32, learning_rate=0.001,
            optimizer=optimizer, weight_decay=0.01,
            verbose=False
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        results[optimizer] = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'train_losses': model.train_losses,
            'val_losses': model.val_losses,
            'train_accuracies': model.train_accuracies,
            'val_accuracies': model.val_accuracies
        }
        
        print(f"{optimizer.upper()} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training loss comparison
    axes[0, 0].set_title('Training Loss Comparison')
    for optimizer in optimizers:
        axes[0, 0].plot(results[optimizer]['train_losses'], label=optimizer.upper())
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation loss comparison
    axes[0, 1].set_title('Validation Loss Comparison')
    for optimizer in optimizers:
        axes[0, 1].plot(results[optimizer]['val_losses'], label=optimizer.upper())
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training accuracy comparison
    axes[1, 0].set_title('Training Accuracy Comparison')
    for optimizer in optimizers:
        axes[1, 0].plot(results[optimizer]['train_accuracies'], label=optimizer.upper())
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation accuracy comparison
    axes[1, 1].set_title('Validation Accuracy Comparison')
    for optimizer in optimizers:
        axes[1, 1].plot(results[optimizer]['val_accuracies'], label=optimizer.upper())
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Final comparison
    print(f"\nFinal Test Results:")
    print("-" * 40)
    for optimizer in optimizers:
        print(f"{optimizer.upper():>10}: Acc={results[optimizer]['test_accuracy']:.4f}, Loss={results[optimizer]['test_loss']:.4f}")
    
    return results


def regularization_study():
    """Study the effect of different regularization techniques."""
    print("\n=== Regularization Study ===\n")
    
    # Load and prepare data
    data_processor = DataProcessor()
    X, y, _ = data_processor.load_sample_dataset('synthetic_classification')
    X_processed, y_processed = data_processor.preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y_processed, test_size=0.2, val_size=0.2
    )
    
    # Test different regularization configurations
    configs = [
        {'dropout': 0.0, 'weight_decay': 0.0, 'name': 'No Regularization'},
        {'dropout': 0.3, 'weight_decay': 0.0, 'name': 'Dropout Only'},
        {'dropout': 0.0, 'weight_decay': 0.01, 'name': 'L2 Regularization Only'},
        {'dropout': 0.3, 'weight_decay': 0.01, 'name': 'Dropout + L2'},
        {'dropout': 0.5, 'weight_decay': 0.05, 'name': 'Heavy Regularization'},
    ]
    
    results = {}
    
    for config in configs:
        print(f"Testing: {config['name']}")
        
        model = PyTorchNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=[256, 128, 64],  # Larger network to show overfitting
            output_size=len(np.unique(y)),
            activation='relu',
            dropout_rate=config['dropout']
        )
        
        model.train_model(
            X_train, y_train, X_val, y_val,
            epochs=300, batch_size=32, learning_rate=0.001,
            optimizer='adam', weight_decay=config['weight_decay'],
            verbose=False
        )
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        
        results[config['name']] = {
            'config': config,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'train_losses': model.train_losses,
            'val_losses': model.val_losses,
            'train_accuracies': model.train_accuracies,
            'val_accuracies': model.val_accuracies
        }
        
        print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Plot regularization comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    axes[0].set_title('Training vs Validation Loss')
    for name, result in results.items():
        axes[0].plot(result['train_losses'], '--', alpha=0.7, label=f'{name} (Train)')
        axes[0].plot(result['val_losses'], '-', label=f'{name} (Val)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True)
    
    # Final test accuracy
    names = list(results.keys())
    test_accs = [results[name]['test_accuracy'] for name in names]
    
    axes[1].bar(range(len(names)), test_accs)
    axes[1].set_title('Test Accuracy by Regularization')
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([name.replace(' ', '\\n') for name in names], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results


def framework_comparison():
    """Compare custom NumPy implementation with PyTorch implementation."""
    print("\n=== Framework Comparison (NumPy vs PyTorch) ===\n")
    
    # Load and prepare data
    data_processor = DataProcessor()
    X, y, _ = data_processor.load_sample_dataset('breast_cancer')
    X_processed, y_processed = data_processor.preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y_processed, test_size=0.2, val_size=0.2
    )
    
    print(f"Dataset: Breast Cancer (binary classification)")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Custom NumPy implementation
    print("\nTraining Custom NumPy Neural Network...")
    custom_model = NeuralNetwork(
        layers=[X_train.shape[1], 64, 32, 1],
        learning_rate=0.01,
        activation='relu'
    )
    
    import time
    start_time = time.time()
    custom_model.train(X_train, y_train, epochs=500, batch_size=32, verbose=False)
    custom_time = time.time() - start_time
    
    custom_test_loss, custom_test_acc = custom_model.evaluate(X_test, y_test)
    
    # PyTorch implementation
    print("Training PyTorch Neural Network...")
    pytorch_model = PyTorchNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[64, 32],
        output_size=1,
        activation='relu',
        dropout_rate=0.0
    )
    
    start_time = time.time()
    pytorch_model.train_model(
        X_train, y_train, X_val, y_val,
        epochs=500, batch_size=32, learning_rate=0.01,
        optimizer='adam', verbose=False
    )
    pytorch_time = time.time() - start_time
    
    pytorch_test_loss, pytorch_test_acc = pytorch_model.evaluate(X_test, y_test)
    
    # Results comparison
    print(f"\nResults Comparison:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Custom NumPy':<15} {'PyTorch':<15}")
    print("-" * 50)
    print(f"{'Test Accuracy':<20} {custom_test_acc:<15.4f} {pytorch_test_acc:<15.4f}")
    print(f"{'Test Loss':<20} {custom_test_loss:<15.4f} {pytorch_test_loss:<15.4f}")
    print(f"{'Training Time (s)':<20} {custom_time:<15.2f} {pytorch_time:<15.2f}")
    print(f"{'Device':<20} {'CPU':<15} {str(pytorch_model.device):<15}")
    
    # Plot training curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot(custom_model.loss_history, label='Custom NumPy')
    axes[0].plot(pytorch_model.train_losses, label='PyTorch')
    axes[0].set_title('Training Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(custom_model.accuracy_history, label='Custom NumPy')
    axes[1].plot(pytorch_model.train_accuracies, label='PyTorch')
    axes[1].set_title('Training Accuracy Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'custom': {'model': custom_model, 'accuracy': custom_test_acc, 'time': custom_time},
        'pytorch': {'model': pytorch_model, 'accuracy': pytorch_test_acc, 'time': pytorch_time}
    }


def learning_rate_analysis():
    """Analyze the effect of different learning rates."""
    print("\n=== Learning Rate Analysis ===\n")
    
    # Load data
    data_processor = DataProcessor()
    X, y, _ = data_processor.load_sample_dataset('iris')
    X_processed, y_processed = data_processor.preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y_processed, test_size=0.2, val_size=0.2
    )
    
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    results = {}
    
    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        
        model = PyTorchNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=[64, 32],
            output_size=len(np.unique(y)),
            activation='relu'
        )
        
        model.train_model(
            X_train, y_train, X_val, y_val,
            epochs=200, batch_size=16, learning_rate=lr,
            optimizer='sgd', verbose=False
        )
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        results[lr] = {
            'test_accuracy': test_accuracy,
            'train_losses': model.train_losses,
            'val_losses': model.val_losses
        }
        
        print(f"  Final test accuracy: {test_accuracy:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    axes[0].set_title('Training Loss by Learning Rate')
    for lr, result in results.items():
        axes[0].plot(result['train_losses'], label=f'LR={lr}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log')
    
    # Final accuracies
    lrs = list(results.keys())
    accs = [results[lr]['test_accuracy'] for lr in lrs]
    
    axes[1].semilogx(lrs, accs, 'o-')
    axes[1].set_title('Test Accuracy vs Learning Rate')
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    best_lr = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    print(f"\nBest learning rate: {best_lr} (Accuracy: {results[best_lr]['test_accuracy']:.4f})")
    
    return results


def main():
    """Run all experiments."""
    print("🚀 Advanced PyTorch Neural Network Experiments")
    print("=" * 60)
    
    # Run experiments
    optimizer_results = compare_optimizers()
    regularization_results = regularization_study()
    framework_results = framework_comparison()
    lr_results = learning_rate_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    # Best optimizer
    best_optimizer = max(optimizer_results.keys(), 
                        key=lambda x: optimizer_results[x]['test_accuracy'])
    print(f"Best Optimizer: {best_optimizer.upper()} "
          f"(Accuracy: {optimizer_results[best_optimizer]['test_accuracy']:.4f})")
    
    # Framework comparison
    custom_acc = framework_results['custom']['accuracy']
    pytorch_acc = framework_results['pytorch']['accuracy']
    faster_framework = 'Custom NumPy' if framework_results['custom']['time'] < framework_results['pytorch']['time'] else 'PyTorch'
    print(f"Framework Performance: PyTorch ({pytorch_acc:.4f}) vs Custom NumPy ({custom_acc:.4f})")
    print(f"Faster Framework: {faster_framework}")
    
    # Best learning rate
    best_lr = max(lr_results.keys(), key=lambda x: lr_results[x]['test_accuracy'])
    print(f"Best Learning Rate: {best_lr} (Accuracy: {lr_results[best_lr]['test_accuracy']:.4f})")
    
    print("\n💡 Key Insights:")
    print("- Different optimizers perform differently on various datasets")
    print("- Regularization helps prevent overfitting in complex models")
    print("- PyTorch provides GPU acceleration and automatic differentiation")
    print("- Learning rate is crucial for convergence and final performance")
    print("- Custom implementations help understand the underlying mechanics")


if __name__ == "__main__":
    main()