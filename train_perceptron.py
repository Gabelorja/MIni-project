"""
Train Manual Perceptron (Classifier 1)
"""
import torch
import numpy as np
from data_preprocessing import DataPreprocessor
from classifier1_manual_perceptron import ManualPerceptron

def main():
    print("="*60)
    print("TRAINING CLASSIFIER 1: Manual Perceptron")
    print("="*60)

    # Load data
    preprocessor = DataPreprocessor(augmentation=True)
    images, labels = preprocessor.load_images_from_directory("data")
    print(f"\nDataset: {len(images)} images, {np.sum(labels)} X's, {len(labels) - np.sum(labels)} O's")

    # Create dataset
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Flatten for perceptron
    X_train_flat = X_train.view(X_train.size(0), -1).numpy()
    X_test_flat = X_test.view(X_test.size(0), -1).numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    # Initialize and train
    print(f"\n🚀 Training perceptron...")
    model = ManualPerceptron(input_size=X_train_flat.shape[1])
    train_acc, test_acc = model.train(
        X_train_flat, y_train_np,
        X_test_flat, y_test_np,
        learning_rate=0.001,
        epochs=100
    )

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Test predictions
    predictions = []
    for x in X_test_flat:
        pred = model.predict(x)
        predictions.append(pred)

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test_np) * 100

    print(f"Test Set Performance:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {np.sum(predictions == y_test_np)}/{len(y_test_np)}")

    # Save model
    torch.save({
        'weights': model.weights,
        'bias': model.bias,
        'input_size': model.input_size
    }, 'perceptron_model.pth')

    print("\n✅ Model saved as 'perceptron_model.pth'")

    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()
