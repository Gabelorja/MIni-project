"""
Test all three classifiers and report their performance
"""
import torch
import numpy as np
from data_preprocessing import DataPreprocessor
from classifier1_manual_perceptron import ManualPerceptron
from classifier2_mlp import MLP
from classifier3_cnn import CNN
from torch.utils.data import DataLoader, TensorDataset
import os

def test_perceptron():
    """Test Manual Perceptron (Classifier 1)"""
    print("="*60)
    print("TESTING CLASSIFIER 1: Manual Perceptron")
    print("="*60)

    # Load data
    preprocessor = DataPreprocessor(augmentation=False)
    images, labels = preprocessor.load_images_from_directory("data")

    # Create dataset
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels)

    # Flatten for perceptron
    X_train_flat = X_train.view(X_train.size(0), -1).numpy()
    X_test_flat = X_test.view(X_test.size(0), -1).numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    # Check if model exists
    if os.path.exists('perceptron_model.pth'):
        print("Loading existing model...")
        model = ManualPerceptron(input_size=X_train_flat.shape[1])
        checkpoint = torch.load('perceptron_model.pth', weights_only=False)
        model.weights = checkpoint['weights']
        model.bias = checkpoint['bias']
    else:
        print("No model found. Need to train first.")
        return None, 0.0

    # Test
    predictions = []
    for x in X_test_flat:
        pred = model.predict(x)
        predictions.append(pred)

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test_np) * 100

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {np.sum(predictions == y_test_np)}/{len(y_test_np)}")

    return model, accuracy

def test_mlp():
    """Test MLP (Classifier 2)"""
    print("\n" + "="*60)
    print("TESTING CLASSIFIER 2: MLP")
    print("="*60)

    # Load data
    preprocessor = DataPreprocessor(augmentation=False)
    images, labels = preprocessor.load_images_from_directory("data")

    # Create dataset
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels)

    # Flatten for MLP
    X_train_flat = X_train.view(X_train.size(0), -1)
    X_test_flat = X_test.view(X_test.size(0), -1)

    # Create test loader
    test_dataset = TensorDataset(X_test_flat, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Check if model exists
    if os.path.exists('xo_model_quick.pth'):
        print("Loading existing model...")
        model = MLP(input_size=X_train_flat.shape[1], hidden_sizes=[128, 64])
        model.load_state_dict(torch.load('xo_model_quick.pth'))
        model.eval()
    else:
        print("No model found. Need to train first.")
        return None, 0.0

    # Test
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {correct}/{total}")

    return model, accuracy

def test_cnn():
    """Test CNN (Classifier 3)"""
    print("\n" + "="*60)
    print("TESTING CLASSIFIER 3: CNN")
    print("="*60)

    # Load data
    preprocessor = DataPreprocessor(augmentation=False)
    images, labels = preprocessor.load_images_from_directory("data")

    # Create dataset
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels)

    # Create test loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Check if model exists
    if os.path.exists('cnn_model_quick.pth'):
        print("Loading existing model...")
        model = CNN(input_channels=1, num_classes=2, input_size=28)
        model.load_state_dict(torch.load('cnn_model_quick.pth'))
        model.eval()
    else:
        print("No model found. Need to train first.")
        return None, 0.0

    # Test
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {correct}/{total}")

    return model, accuracy

def main():
    print("\n🔍 TESTING ALL CLASSIFIERS")
    print("="*60 + "\n")

    results = {}

    # Test each classifier
    _, acc1 = test_perceptron()
    results['Perceptron'] = acc1

    _, acc2 = test_mlp()
    results['MLP'] = acc2

    _, acc3 = test_cnn()
    results['CNN'] = acc3

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, acc in results.items():
        status = "✅ GOOD" if acc >= 85 else "⚠️  NEEDS RETRAINING"
        print(f"{name:20s}: {acc:6.2f}%  {status}")

    print("\n" + "="*60)

    # Determine what needs retraining
    needs_retraining = [name for name, acc in results.items() if acc < 85 or acc == 0.0]

    if needs_retraining:
        print(f"\n⚠️  Models needing retraining: {', '.join(needs_retraining)}")
    else:
        print("\n✅ All models performing well!")

    return results, needs_retraining

if __name__ == "__main__":
    results, needs_retraining = main()
