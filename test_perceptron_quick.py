"""
Quick test of manual perceptron without visualizations
"""
import numpy as np
from data_preprocessing import DataPreprocessor
from classifier1_manual_perceptron import ManualPerceptron

def main():
    print("=== Testing Manual Perceptron (Classifier 1) ===\n")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(image_size=(28, 28), augmentation=False)

    # Load data
    images, labels = preprocessor.load_images_from_directory("data")
    print(f"Dataset: {len(images)} images, {np.sum(labels)} X's, {len(labels) - np.sum(labels)} O's")

    # Preprocess data for perceptron
    processed_data = []
    for img in images:
        processed = preprocessor.preprocess_image(img)
        processed_data.append(processed)

    X = np.array(processed_data)
    y = labels

    print(f"Processed data shape: {X.shape}\n")

    # Initialize manual perceptron
    perceptron = ManualPerceptron(input_size=X.shape[1])

    # Evaluate on full dataset
    print("Evaluating manual perceptron...\n")
    results = perceptron.evaluate(X, y)

    print(f"Overall Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"X Classification Accuracy: {results['x_accuracy']:.3f}")
    print(f"O Classification Accuracy: {results['o_accuracy']:.3f}")

    # Random baseline
    random_acc = max(np.mean(y), 1 - np.mean(y))  # Majority class accuracy
    print(f"\nRandom Baseline (majority class): {random_acc:.3f} ({random_acc*100:.1f}%)")

    if results['accuracy'] > random_acc:
        print("✅ Perceptron performs better than random guessing!")
    else:
        print("❌ Perceptron does not beat random guessing.")

    # Show some predictions
    print(f"\nSample predictions:")
    sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
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
