# X vs O Classification - Deep Learning Mini Project

This project implements three different classifiers to distinguish between X and O shapes written on a whiteboard, as part of a deep learning mini-project worth 5 points.

## Project Overview

The project contains three classifiers with specific requirements:

### Classifier 1: Manual Perceptron 🎯
- **Goal**: Single perceptron with manually selected weights (no training)
- **Scoring**: 1 point if better than random guessing
- **Purpose**: Understand what features the perceptron needs to learn

### Classifier 2: Multi-Layer Perceptron 🧠
- **Goal**: Trained MLP on X/O dataset
- **Scoring**: 2 points if accuracy ≥ 0.9×(best team MLP), otherwise 1 point if > random

### Classifier 3: Convolutional Neural Network 🔍
- **Goal**: Trained CNN on X/O dataset
- **Scoring**: 2 points if accuracy ≥ 0.9×(best team CNN), otherwise 1 point if > random

## Files Structure

```
📁 Mini Project Deep Learning/
├── 📄 requirements.txt              # Python dependencies
├── 📄 data_preprocessing.py         # Dataset loading and preprocessing
├── 📄 classifier1_manual_perceptron.py  # Manual perceptron implementation
├── 📄 classifier2_mlp.py           # Multi-layer perceptron with training
├── 📄 classifier3_cnn.py           # Convolutional neural network
├── 📄 main_evaluation.py           # Main evaluation script
└── 📄 README.md                    # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset (Optional)
If you have real whiteboard photos:
```
data/
├── X/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── O/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. Run Complete Evaluation
```bash
python main_evaluation.py
```

### 4. Run Individual Classifiers
```bash
# Manual perceptron only
python classifier1_manual_perceptron.py

# MLP only
python classifier2_mlp.py

# CNN only
python classifier3_cnn.py
```

## Features

### Data Preprocessing
- **Image Loading**: Supports JPG, PNG formats from directory structure
- **Preprocessing**: Converts to 28×28 grayscale, normalizes to [-1, 1]
- **Augmentation**: Random rotation, cropping, flipping, brightness/contrast adjustment
- **Synthetic Dataset**: Creates artificial X/O images when real data unavailable

### Classifier 1: Manual Perceptron
- **Weight Design**: Manually crafted weights based on visual features:
  - Positive weights for diagonal patterns (X-like)
  - Negative weights for circular patterns (O-like)
  - Corner detection for X shapes
- **Visualization**: Shows weight matrix and distribution
- **Analysis**: Compares mean X vs O images to understand distinguishing features

### Classifier 2: Multi-Layer Perceptron
- **Architecture**: Fully connected layers with BatchNorm and Dropout
- **Training**: Adam optimizer with learning rate scheduling
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Visualization**: Training history plots

### Classifier 3: Convolutional Neural Network
- **Architecture**: 3 convolutional blocks + fully connected classifier
- **Features**: BatchNorm, MaxPooling, Dropout for regularization
- **Visualization**: Feature maps from different convolutional layers
- **Efficiency**: Optimized for small image classification

### Evaluation Framework
- **Comprehensive Comparison**: Accuracy, model complexity, training time
- **Visualization**: Bar charts and scatter plots comparing all classifiers
- **Grading Estimation**: Based on project requirements
- **Performance Analysis**: Relative performance calculations

## Technical Details

### Image Processing Pipeline
1. **Loading**: Read images in grayscale
2. **Resizing**: Convert to 28×28 pixels
3. **Normalization**: Scale pixel values to [-1, 1] range
4. **Augmentation**: Apply random transformations (training only)

### Manual Perceptron Weight Design
The manual perceptron uses feature detectors based on geometric properties:
- **Diagonal Detectors**: High positive weights along main and anti-diagonals for X detection
- **Circle Detectors**: Negative weights in circular patterns for O detection
- **Corner Enhancement**: Positive weights in corners where X typically has activity
- **Center Suppression**: Negative weights in center region where O is typically empty

### Model Architectures

**MLP**: Input(784) → FC(256) → FC(128) → FC(64) → FC(2)
- BatchNorm and Dropout between layers
- ReLU activation functions
- ~200K parameters

**CNN**: 1@28×28 → 32@14×14 → 64@7×7 → 128@3×3 → FC(256) → FC(128) → FC(2)
- Convolutional blocks with BatchNorm and MaxPooling
- Dropout in fully connected layers
- ~300K parameters

## Results Interpretation

The evaluation script provides:
- **Accuracy Comparison**: How well each classifier performs
- **Complexity Analysis**: Parameter count and computational requirements
- **Training Efficiency**: Time required to train each model
- **Grade Estimation**: Expected points based on project requirements

## Recommendations for Real Data

1. **Consistent Writer**: Use the same person for all X and O samples
2. **Consistent Conditions**: Same lighting, whiteboard, marker color
3. **Sufficient Samples**: Aim for 50+ samples of each class minimum
4. **Data Quality**: Clear, well-lit photos with minimal background noise
5. **Preprocessing**: The pipeline handles resizing and normalization automatically

## Extensions and Improvements

### Possible Enhancements:
- **Data Augmentation**: More sophisticated augmentation techniques
- **Architecture Search**: Try different network architectures
- **Ensemble Methods**: Combine predictions from multiple classifiers
- **Transfer Learning**: Use pre-trained models as feature extractors
- **Real-time Inference**: Optimize models for live classification

### For Competition:
- **Hyperparameter Tuning**: Grid search or Bayesian optimization
- **Cross-validation**: More robust evaluation methodology
- **Test-time Augmentation**: Average predictions over multiple augmented versions
- **Model Distillation**: Train smaller models using larger model predictions

## Troubleshooting

### Common Issues:
- **No data directory**: The script will automatically use synthetic data
- **CUDA not available**: Models will run on CPU automatically
- **Memory issues**: Reduce batch size in DataLoader initialization
- **Import errors**: Ensure all dependencies are installed via requirements.txt

### Performance Issues:
- **Low accuracy on real data**: Check data quality and preprocessing
- **Overfitting**: Increase dropout rate or reduce model complexity
- **Slow training**: Use GPU if available, reduce epochs or model size
- **Manual perceptron failing**: Adjust weight patterns in `_initialize_manual_weights()`

## Authors
Team of 4 students - Deep Learning Course Mini Project

## License
Educational use only - Mini Project Assignment# MIni-project
