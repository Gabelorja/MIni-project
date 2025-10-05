import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import DataPreprocessor, create_sample_dataset
import os

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for X vs O classification
    """

    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Build layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 2))  # 2 classes: X and O

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

class MLPTrainer:
    """
    Trainer class for the MLP
    """

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Flatten data
            data = data.view(data.size(0), -1)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)

                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy, all_preds, all_targets

    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
        """Train the MLP with early stopping"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Training MLP for {epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_loss)

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:3d}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('mlp_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, predictions, targets = self.validate(test_loader, criterion)

        print(f"\n=== MLP Test Results ===")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        # Classification report
        print("\nClassification Report:")
        class_names = ['O', 'X']  # 0: O, 1: X
        print(classification_report(targets, predictions, target_names=class_names))

        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        print("Confusion Matrix:")
        print(f"      Predicted")
        print(f"         O    X")
        print(f"Actual O  {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"       X  {cm[1,0]:3d}  {cm[1,1]:3d}")

        return {
            'accuracy': test_acc / 100.0,  # Convert to fraction
            'loss': test_loss,
            'predictions': predictions,
            'targets': targets,
            'confusion_matrix': cm
        }

    def predict_single(self, image):
        """Predict class for a single image"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = torch.tensor(image, dtype=torch.float32)

            if len(image.shape) == 2:  # Add batch dimension
                image = image.unsqueeze(0)

            image = image.to(self.device)
            image = image.view(image.size(0), -1)  # Flatten

            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1)

            return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]

def main():
    print("=== Multi-Layer Perceptron Classifier (Classifier 2) ===")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize preprocessor with augmentation
    preprocessor = DataPreprocessor(image_size=(28, 28), augmentation=True)

    # Load or create dataset
    try:
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

    # Create train/test split with augmentation
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels)

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    input_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]  # Flattened image size
    model = MLP(input_size=input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3)

    print(f"Model architecture:")
    print(model)

    # Initialize trainer
    trainer = MLPTrainer(model, device)

    # Train model
    trainer.train(train_loader, test_loader, epochs=150, lr=0.001, patience=20)

    # Plot training history
    trainer.plot_training_history()

    # Evaluate on test set
    results = trainer.evaluate(test_loader)

    # Save model
    torch.save(model.state_dict(), 'mlp_model.pth')
    print("\nModel saved as 'mlp_model.pth'")

    # Test on some individual samples
    print("\n=== Individual Sample Predictions ===")
    test_samples = 5
    indices = np.random.choice(len(X_test), test_samples, replace=False)

    for i, idx in enumerate(indices):
        image = X_test[idx]
        actual = y_test[idx].item()
        prediction, probabilities = trainer.predict_single(image)

        actual_label = 'X' if actual == 1 else 'O'
        pred_label = 'X' if prediction == 1 else 'O'
        confidence = probabilities[prediction] * 100

        correct = "✓" if prediction == actual else "✗"

        print(f"Sample {i+1}: Actual={actual_label}, Predicted={pred_label}, "
              f"Confidence={confidence:.1f}%, {correct}")

    return model, trainer, results

if __name__ == "__main__":
    model, trainer, results = main()