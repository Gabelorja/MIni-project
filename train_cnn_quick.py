"""
Quick CNN training script with fewer epochs for faster results
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_preprocessing import DataPreprocessor
from classifier3_cnn import CNN, CNNTrainer
import os

def main():
    print("=== Quick CNN Training ===\n")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(image_size=(28, 28), augmentation=True)

    # Load data
    images, labels = preprocessor.load_images_from_directory("data")
    print(f"\nDataset: {len(images)} images, {np.sum(labels)} X's, {len(labels) - np.sum(labels)} O's")

    # Create train/test split
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = CNN(input_channels=1, num_classes=2, input_size=28)

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Train (fewer epochs for quick results)
    trainer = CNNTrainer(model)
    print("\n🚀 Training for 50 epochs (quick mode)...")
    trainer.train(train_loader, test_loader, epochs=50, lr=0.001, patience=10)

    # Evaluate
    print("\n" + "="*60)
    results = trainer.evaluate(test_loader)

    # Save model
    torch.save(model.state_dict(), 'cnn_model_quick.pth')
    print("\n✅ Model saved as 'cnn_model_quick.pth'")

    return model, trainer, results

if __name__ == "__main__":
    model, trainer, results = main()
