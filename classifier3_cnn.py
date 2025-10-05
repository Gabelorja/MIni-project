import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import DataPreprocessor, create_sample_dataset
import os

class CNN(nn.Module):
    """
    Convolutional Neural Network for X vs O classification
    """

    def __init__(self, input_channels=1, num_classes=2, input_size=28):
        super(CNN, self).__init__()

        self.input_size = input_size

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 7x7 -> 3x3 (for 28x28 input)
        )

        # Calculate the size after convolutions
        # For 28x28 input: 28 -> 14 -> 7 -> 3
        conv_output_size = self._get_conv_output_size()

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output_size(self):
        """Calculate the output size after all convolutions"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_size, self.input_size)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """Extract feature maps from different layers for visualization"""
        features = {}

        x = self.conv1(x)
        features['conv1'] = x.clone()

        x = self.conv2(x)
        features['conv2'] = x.clone()

        x = self.conv3(x)
        features['conv3'] = x.clone()

        return features

class CNNTrainer:
    """
    Trainer class for the CNN
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

    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
        """Train the CNN with early stopping"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=7)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Training CNN for {epochs} epochs...")
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
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:3d}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                      f'LR: {current_lr:.6f}')

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
        ax1.set_title('CNN Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('CNN Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_feature_maps(self, sample_image, sample_label):
        """Visualize feature maps from different CNN layers"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(sample_image, np.ndarray):
                sample_image = torch.tensor(sample_image, dtype=torch.float32)

            if len(sample_image.shape) == 2:  # Add channel and batch dimensions
                sample_image = sample_image.unsqueeze(0).unsqueeze(0)
            elif len(sample_image.shape) == 3:  # Add batch dimension
                sample_image = sample_image.unsqueeze(0)

            sample_image = sample_image.to(self.device)

            # Get feature maps
            feature_maps = self.model.get_feature_maps(sample_image)

            # Create visualization
            fig, axes = plt.subplots(3, 8, figsize=(16, 6))

            # Original image
            orig_img = sample_image[0, 0].cpu().numpy()
            axes[0, 0].imshow(orig_img, cmap='gray')
            axes[0, 0].set_title(f'Original ({sample_label})')
            axes[0, 0].axis('off')

            # Hide unused subplots in first row
            for i in range(1, 8):
                axes[0, i].axis('off')

            # Conv1 feature maps (first 8 channels)
            conv1_maps = feature_maps['conv1'][0]
            for i in range(min(8, conv1_maps.shape[0])):
                if i < 7:
                    axes[1, i].imshow(conv1_maps[i].cpu().numpy(), cmap='viridis')
                    axes[1, i].set_title(f'Conv1-{i}')
                    axes[1, i].axis('off')
                else:
                    axes[1, i].axis('off')

            # Conv2 feature maps (first 8 channels)
            conv2_maps = feature_maps['conv2'][0]
            for i in range(min(8, conv2_maps.shape[0])):
                if i < 7:
                    axes[2, i].imshow(conv2_maps[i].cpu().numpy(), cmap='viridis')
                    axes[2, i].set_title(f'Conv2-{i}')
                    axes[2, i].axis('off')
                else:
                    axes[2, i].axis('off')

            plt.suptitle('CNN Feature Maps Visualization')
            plt.tight_layout()
            plt.savefig('cnn_feature_maps.png', dpi=150, bbox_inches='tight')
            plt.show()

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, predictions, targets = self.validate(test_loader, criterion)

        print(f"\n=== CNN Test Results ===")
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

            if len(image.shape) == 2:  # Add channel and batch dimensions
                image = image.unsqueeze(0).unsqueeze(0)
            elif len(image.shape) == 3:  # Add batch dimension
                image = image.unsqueeze(0)

            image = image.to(self.device)

            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1)

            return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]

def main():
    print("=== Convolutional Neural Network Classifier (Classifier 3) ===")

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
    model = CNN(input_channels=1, num_classes=2, input_size=28)

    print(f"CNN architecture:")
    print(model)

    # Initialize trainer
    trainer = CNNTrainer(model, device)

    # Train model
    trainer.train(train_loader, test_loader, epochs=100, lr=0.001, patience=15)

    # Plot training history
    trainer.plot_training_history()

    # Visualize feature maps
    sample_idx = np.random.choice(len(X_test))
    sample_image = X_test[sample_idx]
    sample_label = 'X' if y_test[sample_idx] == 1 else 'O'
    trainer.visualize_feature_maps(sample_image, sample_label)

    # Evaluate on test set
    results = trainer.evaluate(test_loader)

    # Save model
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("\nModel saved as 'cnn_model.pth'")

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