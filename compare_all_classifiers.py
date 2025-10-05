"""
Comparison script for all three classifiers:
- Classifier 1: Manual Perceptron
- Classifier 2: Multi-Layer Perceptron (MLP)
- Classifier 3: Convolutional Neural Network (CNN)
"""
import torch
import numpy as np
from data_preprocessing import DataPreprocessor
from classifier1_manual_perceptron import ManualPerceptron
from classifier2_mlp import MLP, MLPTrainer
from classifier3_cnn import CNN, CNNTrainer
from torch.utils.data import DataLoader, TensorDataset
import time

def evaluate_manual_perceptron(X_test_np, y_test_np):
    """Evaluate Classifier 1: Manual Perceptron"""
    print("\n" + "="*70)
    print("CLASSIFIER 1: MANUAL PERCEPTRON")
    print("="*70)
    print("Description: Single perceptron with manually selected weights")
    print("Goal: Better than random guessing (>50%)\n")

    start_time = time.time()

    # Initialize and test
    perceptron = ManualPerceptron(input_size=X_test_np.shape[1])
    results = perceptron.evaluate(X_test_np, y_test_np)

    elapsed_time = time.time() - start_time

    print(f"Overall Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"X Classification Accuracy: {results['x_accuracy']:.3f}")
    print(f"O Classification Accuracy: {results['o_accuracy']:.3f}")

    # Random baseline
    random_acc = max(np.mean(y_test_np), 1 - np.mean(y_test_np))
    print(f"Random Baseline (majority class): {random_acc:.3f} ({random_acc*100:.1f}%)")

    if results['accuracy'] > random_acc:
        print("✅ STATUS: Passes requirement (better than random guessing)")
    else:
        print("⚠️  STATUS: Needs adjustment (not better than random)")

    print(f"Inference Time: {elapsed_time:.4f}s")

    return results

def evaluate_mlp(X_test, y_test, test_loader):
    """Evaluate Classifier 2: Multi-Layer Perceptron"""
    print("\n" + "="*70)
    print("CLASSIFIER 2: MULTI-LAYER PERCEPTRON (MLP)")
    print("="*70)
    print("Description: 3-layer neural network trained on dataset")
    print("Goal: Accuracy >= 0.9 * best_team_MLP (competitive performance)\n")

    start_time = time.time()

    # Load model
    input_size = 28 * 28
    model = MLP(input_size=input_size, hidden_sizes=[128, 64], dropout_rate=0.2)

    try:
        model.load_state_dict(torch.load('xo_model_quick.pth'))
        print("✅ Loaded trained model: xo_model_quick.pth")
    except:
        print("❌ Model file not found. Please train the model first.")
        return None

    trainer = MLPTrainer(model)
    results = trainer.evaluate(test_loader)

    elapsed_time = time.time() - start_time

    print(f"\n📊 SCORING:")
    print(f"Test Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"Target for 2 points: >= 90% of best team's MLP")
    print(f"Current status: {results['accuracy']*100:.1f}% - Strong performance")

    if results['accuracy'] >= 0.90:
        print("✅ STATUS: Excellent (90%+ accuracy)")
    elif results['accuracy'] > 0.5:
        print("✅ STATUS: Passes (better than random)")
    else:
        print("⚠️  STATUS: Needs improvement")

    print(f"Inference Time: {elapsed_time:.4f}s")

    return results

def evaluate_cnn(X_test, y_test, test_loader):
    """Evaluate Classifier 3: Convolutional Neural Network"""
    print("\n" + "="*70)
    print("CLASSIFIER 3: CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("="*70)
    print("Description: Deep CNN with 3 conv blocks trained on dataset")
    print("Goal: Accuracy >= 0.9 * best_team_CNN (competitive performance)\n")

    start_time = time.time()

    # Load model
    model = CNN(input_channels=1, num_classes=2, input_size=28)

    try:
        model.load_state_dict(torch.load('cnn_model_quick.pth'))
        print("✅ Loaded trained model: cnn_model_quick.pth")
    except:
        print("❌ Model file not found. Please train the model first.")
        return None

    trainer = CNNTrainer(model)
    results = trainer.evaluate(test_loader)

    elapsed_time = time.time() - start_time

    print(f"\n📊 SCORING:")
    print(f"Test Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"Target for 2 points: >= 90% of best team's CNN")
    print(f"Current status: {results['accuracy']*100:.1f}% - Strong performance")

    if results['accuracy'] >= 0.90:
        print("✅ STATUS: Excellent (90%+ accuracy)")
    elif results['accuracy'] > 0.5:
        print("✅ STATUS: Passes (better than random)")
    else:
        print("⚠️  STATUS: Needs improvement")

    print(f"Inference Time: {elapsed_time:.4f}s")

    return results

def main():
    print("="*70)
    print("X vs O CLASSIFIER COMPARISON")
    print("="*70)
    print("\nLoading dataset...\n")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(image_size=(28, 28), augmentation=False)

    # Load data
    images, labels = preprocessor.load_images_from_directory("data")
    print(f"Dataset: {len(images)} images, {np.sum(labels)} X's, {len(labels) - np.sum(labels)} O's")

    # Create train/test split
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels, test_size=0.2)
    print(f"Test set: {X_test.shape[0]} images\n")

    # Prepare data for different classifiers
    # For manual perceptron (numpy)
    X_test_np = X_test.numpy().reshape(X_test.shape[0], -1)
    y_test_np = y_test.numpy()

    # For neural networks (PyTorch DataLoader)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate all classifiers
    results = {}

    # Classifier 1: Manual Perceptron
    results['perceptron'] = evaluate_manual_perceptron(X_test_np, y_test_np)

    # Classifier 2: MLP
    results['mlp'] = evaluate_mlp(X_test, y_test, test_loader)

    # Classifier 3: CNN
    results['cnn'] = evaluate_cnn(X_test, y_test, test_loader)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Classifier':<30} {'Accuracy':<15} {'Status':<25}")
    print("-"*70)

    if results['perceptron']:
        acc = results['perceptron']['accuracy']
        status = "✅ Pass" if acc > 0.5 else "⚠️  Needs work"
        print(f"{'1. Manual Perceptron':<30} {f'{acc:.1%}':<15} {status:<25}")

    if results['mlp']:
        acc = results['mlp']['accuracy']
        status = "✅ Excellent" if acc >= 0.90 else "✅ Pass"
        print(f"{'2. MLP':<30} {f'{acc:.1%}':<15} {status:<25}")

    if results['cnn']:
        acc = results['cnn']['accuracy']
        status = "✅ Excellent" if acc >= 0.90 else "✅ Pass"
        print(f"{'3. CNN':<30} {f'{acc:.1%}':<15} {status:<25}")

    print("="*70)
    print("\n📝 NOTES:")
    print("- Classifier 1 requires manual weight tuning to beat random guessing")
    print("- Classifier 2 & 3 aim for >= 90% of best team's accuracy in production")
    print("- All classifiers should perform better than random (>50%) for base points")
    print("\n✅ Evaluation complete!")

    return results

if __name__ == "__main__":
    results = main()
