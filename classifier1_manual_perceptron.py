import numpy as np
import torch
from data_preprocessing import DataPreprocessor, create_sample_dataset
import matplotlib.pyplot as plt

class ManualPerceptron:
    """
    A single perceptron with manually selected weights.
    The goal is to understand what features distinguish X from O.
    """

    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = None
        self.bias = None
        self._initialize_manual_weights()

    def _initialize_manual_weights(self):
        """
        Manually design weights based on visual features that distinguish X from O.

        For X vs O classification:
        - X has diagonal patterns (main diagonal and anti-diagonal)
        - O has circular/ring patterns

        We'll create weight patterns that:
        1. Respond positively to diagonal structures (X-like)
        2. Respond negatively to circular structures (O-like)
        """
        # Assuming 28x28 images (784 features)
        img_size = int(np.sqrt(self.input_size))

        if img_size * img_size != self.input_size:
            # Fallback for non-square inputs
            self.weights = np.random.randn(self.input_size) * 0.1
            self.bias = 0.0
            return

        # Create 2D weight matrix for easier manipulation
        weight_matrix = np.zeros((img_size, img_size))

        # Design feature detectors
        center = img_size // 2

        # Simple strategy: detect diagonal patterns for X
        # X has strong diagonal structure, O has circular structure

        # Diagonal detectors (positive weight on diagonals for X classification)
        for i in range(img_size):
            for j in range(img_size):
                # Main diagonal (top-left to bottom-right)
                if abs(i - j) <= 2:
                    weight_matrix[i, j] = 1.0

                # Anti-diagonal (top-right to bottom-left)
                if abs(i + j - (img_size - 1)) <= 2:
                    weight_matrix[i, j] = 1.0

        # Flatten back to 1D
        self.weights = weight_matrix.flatten()

        # Adjusted bias (positive to help detect X)
        self.bias = -5.0

        print(f"Initialized manual perceptron with {len(self.weights)} weights")
        print(f"Weight range: [{self.weights.min():.3f}, {self.weights.max():.3f}]")
        print(f"Bias: {self.bias}")

    def forward(self, x):
        """Forward pass through perceptron"""
        # x can be a single sample or batch
        if len(x.shape) == 1:
            # Single sample
            return np.dot(x, self.weights) + self.bias
        else:
            # Batch
            return np.dot(x, self.weights) + self.bias

    def predict(self, x):
        """Make binary predictions"""
        output = self.forward(x)
        return (output > 0).astype(int)  # 1 for X, 0 for O

    def predict_proba(self, x):
        """Get prediction probabilities using sigmoid"""
        output = self.forward(x)
        # Apply sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-output))
        return probs

    def visualize_weights(self):
        """Visualize the weight matrix to understand what the perceptron learned"""
        img_size = int(np.sqrt(len(self.weights)))

        if img_size * img_size == len(self.weights):
            weight_matrix = self.weights.reshape(img_size, img_size)

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(weight_matrix, cmap='RdBu', vmin=-2, vmax=2)
            plt.colorbar()
            plt.title('Perceptron Weights Visualization')
            plt.xlabel('Pixel X')
            plt.ylabel('Pixel Y')

            plt.subplot(1, 2, 2)
            plt.hist(self.weights, bins=50, alpha=0.7)
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.title('Weight Distribution')

            plt.tight_layout()
            plt.savefig('manual_perceptron_weights.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            # Just show histogram for non-square inputs
            plt.figure(figsize=(6, 4))
            plt.hist(self.weights, bins=50, alpha=0.7)
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.title('Manual Perceptron Weight Distribution')
            plt.savefig('manual_perceptron_weights_hist.png', dpi=150, bbox_inches='tight')
            plt.show()

    def evaluate(self, X, y):
        """Evaluate the perceptron on test data"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        # Calculate per-class accuracy
        x_mask = y == 1
        o_mask = y == 0

        x_accuracy = np.mean(predictions[x_mask] == y[x_mask]) if np.any(x_mask) else 0
        o_accuracy = np.mean(predictions[o_mask] == y[o_mask]) if np.any(o_mask) else 0

        return {
            'accuracy': accuracy,
            'x_accuracy': x_accuracy,
            'o_accuracy': o_accuracy,
            'predictions': predictions
        }

def analyze_features(preprocessor, images, labels):
    """Analyze what features distinguish X from O in the dataset"""
    x_images = images[labels == 1]  # X images
    o_images = images[labels == 0]  # O images

    if len(x_images) == 0 or len(o_images) == 0:
        print("Warning: Need both X and O images for analysis")
        return

    # Convert to preprocessed format
    x_processed = []
    o_processed = []

    for img in x_images[:10]:  # Sample first 10
        processed = preprocessor.preprocess_image(img)
        x_processed.append(processed)

    for img in o_images[:10]:  # Sample first 10
        processed = preprocessor.preprocess_image(img)
        o_processed.append(processed)

    x_processed = np.array(x_processed)
    o_processed = np.array(o_processed)

    # Calculate mean images
    x_mean = np.mean(x_processed, axis=0)
    o_mean = np.mean(o_processed, axis=0)
    difference = x_mean - o_mean

    # Visualize
    img_size = int(np.sqrt(len(x_mean)))
    if img_size * img_size == len(x_mean):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(x_mean.reshape(img_size, img_size), cmap='gray')
        axes[0].set_title('Mean X Image')
        axes[0].axis('off')

        axes[1].imshow(o_mean.reshape(img_size, img_size), cmap='gray')
        axes[1].set_title('Mean O Image')
        axes[1].axis('off')

        axes[2].imshow(difference.reshape(img_size, img_size), cmap='RdBu')
        axes[2].set_title('Difference (X - O)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('feature_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    return x_mean, o_mean, difference

def main():
    print("=== Manual Perceptron Classifier (Classifier 1) ===")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(image_size=(28, 28), augmentation=False)

    # Load or create dataset
    try:
        import os
        if os.path.exists("data"):
            images, labels = preprocessor.load_images_from_directory("data")
            print(f"Loaded {len(images)} real images")
        else:
            print("No data directory found, using synthetic dataset...")
            images, labels = create_sample_dataset()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic dataset...")
        images, labels = create_sample_dataset()

    print(f"Dataset: {len(images)} images, {np.sum(labels)} X's, {len(labels) - np.sum(labels)} O's")

    # Analyze features
    print("\nAnalyzing dataset features...")
    analyze_features(preprocessor, images, labels)

    # Preprocess data for perceptron
    processed_data = []
    for img in images:
        processed = preprocessor.preprocess_image(img)
        processed_data.append(processed)

    X = np.array(processed_data)
    y = labels

    print(f"Processed data shape: {X.shape}")

    # Initialize manual perceptron
    perceptron = ManualPerceptron(input_size=X.shape[1])

    # Visualize weights
    print("\nVisualizing perceptron weights...")
    perceptron.visualize_weights()

    # Evaluate on full dataset
    print("\nEvaluating manual perceptron...")
    results = perceptron.evaluate(X, y)

    print(f"Overall Accuracy: {results['accuracy']:.3f}")
    print(f"X Classification Accuracy: {results['x_accuracy']:.3f}")
    print(f"O Classification Accuracy: {results['o_accuracy']:.3f}")

    # Random baseline
    random_acc = max(np.mean(y), 1 - np.mean(y))  # Majority class accuracy
    print(f"Random Baseline (majority class): {random_acc:.3f}")

    if results['accuracy'] > random_acc:
        print("✅ Perceptron performs better than random guessing!")
    else:
        print("❌ Perceptron does not beat random guessing. Consider adjusting weights.")

    # Show some predictions
    print(f"\nSample predictions:")
    sample_indices = np.random.choice(len(X), 10, replace=False)
    for i, idx in enumerate(sample_indices):
        pred = perceptron.predict(X[idx])
        actual = y[idx]
        prob = perceptron.predict_proba(X[idx])

        pred_label = 'X' if pred == 1 else 'O'
        actual_label = 'X' if actual == 1 else 'O'
        correct = "✓" if pred == actual else "✗"

        print(f"Sample {i+1}: Predicted={pred_label}, Actual={actual_label}, "
              f"Prob={prob:.3f}, {correct}")

    return perceptron, results

if __name__ == "__main__":
    perceptron, results = main()