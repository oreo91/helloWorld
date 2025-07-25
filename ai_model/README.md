# 🧠 Custom AI Model Project

A comprehensive machine learning project featuring custom neural network implementations, both from scratch using NumPy and advanced implementations using PyTorch. This project is designed to help you understand the fundamentals of AI/ML while providing practical, production-ready code.

## 🚀 Project Overview

This project includes:

- **Custom Neural Network** - Built from scratch using only NumPy for educational purposes
- **PyTorch Implementation** - Advanced neural networks with modern features
- **Convolutional Neural Networks** - For image classification tasks
- **Data Processing Utilities** - Comprehensive data preprocessing and visualization tools
- **Example Scripts** - Ready-to-run examples demonstrating various ML concepts
- **Performance Analysis** - Tools to compare different approaches and architectures

## 📁 Project Structure

```
ai_model/
├── models/
│   ├── neural_network.py      # Custom NumPy neural network
│   └── pytorch_model.py       # PyTorch implementations
├── utils/
│   └── data_utils.py          # Data processing and visualization utilities
├── examples/
│   ├── basic_classification.py # Basic classification examples
│   └── pytorch_advanced.py    # Advanced PyTorch experiments
├── data/                      # Directory for datasets (auto-created)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd ai_model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import numpy, torch, sklearn; print('All dependencies installed successfully!')"
   ```

## 🎯 Quick Start

### Running the Basic Classification Example

```bash
cd ai_model
python examples/basic_classification.py
```

This will:
- Train neural networks on the Iris dataset
- Create a spiral dataset and visualize decision boundaries
- Compare different network architectures
- Generate performance plots and confusion matrices

### Running Advanced PyTorch Examples

```bash
python examples/pytorch_advanced.py
```

This will:
- Compare different optimizers (Adam, SGD, RMSprop)
- Study regularization effects (dropout, weight decay)
- Compare custom NumPy vs PyTorch implementations
- Analyze learning rate effects

## 📚 Core Components

### 1. Custom Neural Network (`models/neural_network.py`)

A complete neural network implementation using only NumPy, featuring:

- **Forward and backward propagation**
- **Multiple activation functions** (ReLU, Sigmoid, Tanh)
- **Configurable architectures**
- **Mini-batch training**
- **Training history tracking**
- **Model saving/loading**

**Example Usage:**
```python
from models.neural_network import NeuralNetwork

# Create a 3-layer network
nn = NeuralNetwork(
    layers=[4, 64, 32, 3],  # input_size, hidden1, hidden2, output_size
    learning_rate=0.01,
    activation='relu'
)

# Train the model
nn.train(X_train, y_train, epochs=500, batch_size=32)

# Make predictions
predictions = nn.predict(X_test)

# Evaluate performance
loss, accuracy = nn.evaluate(X_test, y_test)
```

### 2. PyTorch Neural Networks (`models/pytorch_model.py`)

Advanced implementations with modern ML features:

- **GPU acceleration**
- **Multiple optimizers** (Adam, SGD, RMSprop)
- **Regularization** (Dropout, Weight Decay)
- **Learning rate scheduling**
- **Batch normalization**
- **Convolutional networks** for image data

**Example Usage:**
```python
from models.pytorch_model import PyTorchNeuralNetwork

# Create model
model = PyTorchNeuralNetwork(
    input_size=20,
    hidden_sizes=[128, 64, 32],
    output_size=3,
    dropout_rate=0.3
)

# Train with validation
model.train_model(
    X_train, y_train, X_val, y_val,
    epochs=200, batch_size=32,
    optimizer='adam', learning_rate=0.001
)
```

### 3. Data Utilities (`utils/data_utils.py`)

Comprehensive data processing tools:

- **Dataset loading** (Iris, Wine, Breast Cancer, Synthetic)
- **Data preprocessing** (normalization, encoding)
- **Train/validation/test splitting**
- **Feature engineering** (polynomial features)
- **Visualization tools**
- **Performance evaluation**

**Example Usage:**
```python
from utils.data_utils import DataProcessor

# Initialize processor
processor = DataProcessor()

# Load dataset
X, y, feature_names = processor.load_sample_dataset('iris')

# Preprocess
X_processed, y_processed = processor.preprocess_data(X, y)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
    X_processed, y_processed
)

# Visualize
processor.visualize_data(X, y, feature_names)
```

## 🔬 Key Features

### Educational Focus
- **Step-by-step implementations** with detailed comments
- **Mathematical foundations** clearly explained in code
- **Comparison tools** to understand trade-offs between approaches

### Production Ready
- **Robust error handling** and input validation
- **Efficient implementations** with vectorized operations
- **Model persistence** with save/load functionality
- **Comprehensive logging** and progress tracking

### Advanced Features
- **Multiple activation functions** and their derivatives
- **Various optimization algorithms**
- **Regularization techniques** to prevent overfitting
- **Hyperparameter analysis** tools
- **Visualization utilities** for insights

## 📊 Datasets Included

The project comes with several built-in datasets:

1. **Iris Dataset** - Classic 3-class flower classification
2. **Wine Dataset** - Wine quality classification
3. **Breast Cancer** - Binary medical diagnosis
4. **Synthetic Datasets** - Customizable artificial data
5. **Spiral Dataset** - Non-linear pattern recognition

## 🧪 Experiments You Can Run

### 1. Architecture Comparison
Compare shallow vs deep networks:
```python
architectures = [
    [features, 32, classes],           # Shallow
    [features, 64, 32, classes],       # Medium  
    [features, 128, 64, 32, classes]   # Deep
]
```

### 2. Optimizer Analysis
Test different optimization algorithms:
```python
optimizers = ['adam', 'sgd', 'rmsprop']
# Compare convergence speed and final performance
```

### 3. Regularization Study
Prevent overfitting with:
- Dropout layers
- Weight decay (L2 regularization)
- Early stopping

### 4. Learning Rate Analysis
Find optimal learning rates:
```python
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
# Plot performance vs learning rate
```

## 📈 Performance Metrics

The project tracks comprehensive metrics:

- **Accuracy** - Overall correctness
- **Precision** - True positive rate
- **Recall** - Sensitivity
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Detailed classification results
- **Training Curves** - Loss and accuracy over time

## 🎓 Learning Objectives

By working with this project, you'll learn:

1. **Neural Network Fundamentals**
   - Forward and backward propagation
   - Gradient descent optimization
   - Activation functions and their purposes

2. **Modern ML Practices**
   - Data preprocessing and normalization
   - Train/validation/test splitting
   - Hyperparameter tuning
   - Model evaluation and selection

3. **Framework Comparison**
   - Advantages of custom implementations
   - Benefits of established frameworks (PyTorch)
   - Performance trade-offs

4. **Practical Skills**
   - Data visualization and analysis
   - Model debugging and improvement
   - Experiment design and execution

## 🚨 Common Issues & Solutions

### GPU Support
If you have a CUDA-compatible GPU:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
For large datasets:
- Reduce batch size
- Use gradient accumulation
- Enable data streaming

### Convergence Problems
If the model isn't learning:
- Check learning rate (try 0.001-0.01)
- Verify data preprocessing
- Ensure proper label encoding
- Add regularization if overfitting

## 🔧 Customization

### Adding New Datasets
```python
def load_custom_dataset():
    # Your data loading logic
    X = your_features
    y = your_labels
    feature_names = your_feature_names
    return X, y, feature_names
```

### Creating New Architectures
```python
# Custom network architectures
custom_nn = NeuralNetwork(
    layers=[input_size, hidden1, hidden2, output_size],
    learning_rate=0.01,
    activation='your_activation'
)
```

### New Activation Functions
Add to the `neural_network.py` file:
```python
def _your_activation(self, x):
    # Your activation function
    return processed_x

def _your_activation_derivative(self, x):
    # Derivative for backpropagation
    return derivative_x
```

## 📝 Next Steps

1. **Experiment with different datasets** - Try your own data
2. **Implement new features** - Add batch normalization, different optimizers
3. **Scale up** - Try larger networks and datasets
4. **Specialize** - Focus on specific domains (NLP, Computer Vision)
5. **Deploy** - Create web applications with your trained models

## 🤝 Contributing

This project is designed for learning and experimentation. Feel free to:
- Add new datasets and examples
- Implement additional ML algorithms
- Improve documentation and comments
- Create tutorials and guides

## 📚 Further Reading

- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Pattern Recognition and Machine Learning** by Christopher Bishop
- **PyTorch Documentation** - https://pytorch.org/docs/
- **Scikit-learn User Guide** - https://scikit-learn.org/stable/user_guide.html

## 🎉 Conclusion

This project provides a solid foundation for understanding and implementing neural networks. Whether you're a beginner learning the fundamentals or an experienced practitioner exploring new techniques, the combination of custom implementations and modern frameworks offers valuable insights into the world of artificial intelligence.

Start with the basic examples, experiment with different parameters, and gradually work your way up to more complex scenarios. Happy learning! 🚀