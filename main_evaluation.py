#!/usr/bin/env python3
"""
Main evaluation script for X vs O classification project.
This script runs all three classifiers and provides a comprehensive comparison.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tabulate import tabulate
import time

# Import our classifiers
from data_preprocessing import DataPreprocessor, create_sample_dataset
from classifier1_manual_perceptron import ManualPerceptron, analyze_features
from classifier2_mlp import MLP, MLPTrainer
from classifier3_cnn import CNN, CNNTrainer
from torch.utils.data import DataLoader, TensorDataset

class ProjectEvaluator:
    """
    Main evaluation class that runs and compares all three classifiers
    """

    def __init__(self, use_synthetic=True):
        self.use_synthetic = use_synthetic
        self.results = {}
        self.preprocessor = DataPreprocessor(image_size=(28, 28), augmentation=True)

    def load_dataset(self):
        """Load the dataset (real or synthetic)"""
        try:
            if not self.use_synthetic and os.path.exists("data"):
                images, labels = self.preprocessor.load_images_from_directory("data")
                print(f"✅ Loaded {len(images)} real images from 'data' directory")
                data_type = "real"
            else:
                if not self.use_synthetic:
                    print("⚠️  No 'data' directory found, falling back to synthetic dataset")
                else:
                    print("📊 Using synthetic dataset as requested")
                images, labels = create_sample_dataset()
                data_type = "synthetic"
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("📊 Falling back to synthetic dataset")
            images, labels = create_sample_dataset()
            data_type = "synthetic"

        print(f"Dataset info: {len(images)} images, {np.sum(labels)} X's, {len(labels) - np.sum(labels)} O's")
        return images, labels, data_type

    def evaluate_classifier1_manual_perceptron(self, images, labels):
        """Evaluate the manual perceptron"""
        print("\n" + "="*60)
        print("🎯 CLASSIFIER 1: MANUAL PERCEPTRON")
        print("="*60)

        start_time = time.time()

        # Preprocess data for perceptron (no augmentation)
        preprocessor_simple = DataPreprocessor(image_size=(28, 28), augmentation=False)
        processed_data = []
        for img in images:
            processed = preprocessor_simple.preprocess_image(img)
            processed_data.append(processed)

        X = np.array(processed_data)
        y = labels

        # Initialize and evaluate manual perceptron
        perceptron = ManualPerceptron(input_size=X.shape[1])
        results = perceptron.evaluate(X, y)

        training_time = time.time() - start_time

        # Store results
        self.results['manual_perceptron'] = {
            'accuracy': results['accuracy'],
            'x_accuracy': results['x_accuracy'],
            'o_accuracy': results['o_accuracy'],
            'training_time': training_time,
            'model_complexity': len(perceptron.weights),
            'predictions': results['predictions']
        }

        print(f"📊 Results:")
        print(f"   Overall Accuracy: {results['accuracy']:.3f}")
        print(f"   X Classification: {results['x_accuracy']:.3f}")
        print(f"   O Classification: {results['o_accuracy']:.3f}")
        print(f"   Training Time: {training_time:.2f}s")

        # Check if better than random
        random_baseline = max(np.mean(y), 1 - np.mean(y))
        if results['accuracy'] > random_baseline:
            print(f"   ✅ Better than random baseline ({random_baseline:.3f})")
        else:
            print(f"   ❌ Not better than random baseline ({random_baseline:.3f})")

        return perceptron

    def evaluate_classifier2_mlp(self, images, labels):
        """Evaluate the multi-layer perceptron"""
        print("\n" + "="*60)
        print("🧠 CLASSIFIER 2: MULTI-LAYER PERCEPTRON")
        print("="*60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()

        # Create dataset with augmentation
        X_train, y_train, X_test, y_test = self.preprocessor.create_dataset(images, labels)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model
        input_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        model = MLP(input_size=input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3)

        # Train model
        trainer = MLPTrainer(model, device)
        trainer.train(train_loader, test_loader, epochs=100, lr=0.001, patience=15)

        training_time = time.time() - start_time

        # Evaluate
        results = trainer.evaluate(test_loader)

        # Store results
        self.results['mlp'] = {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'training_time': training_time,
            'model_complexity': sum(p.numel() for p in model.parameters()),
            'predictions': results['predictions'],
            'targets': results['targets']
        }

        print(f"📊 Results:")
        print(f"   Test Accuracy: {results['accuracy']:.3f}")
        print(f"   Test Loss: {results['loss']:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Model Parameters: {self.results['mlp']['model_complexity']:,}")

        return model, trainer

    def evaluate_classifier3_cnn(self, images, labels):
        """Evaluate the convolutional neural network"""
        print("\n" + "="*60)
        print("🔍 CLASSIFIER 3: CONVOLUTIONAL NEURAL NETWORK")
        print("="*60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()

        # Create dataset with augmentation
        X_train, y_train, X_test, y_test = self.preprocessor.create_dataset(images, labels)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model
        model = CNN(input_channels=1, num_classes=2, input_size=28)

        # Train model
        trainer = CNNTrainer(model, device)
        trainer.train(train_loader, test_loader, epochs=80, lr=0.001, patience=12)

        training_time = time.time() - start_time

        # Evaluate
        results = trainer.evaluate(test_loader)

        # Store results
        self.results['cnn'] = {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'training_time': training_time,
            'model_complexity': sum(p.numel() for p in model.parameters()),
            'predictions': results['predictions'],
            'targets': results['targets']
        }

        print(f"📊 Results:")
        print(f"   Test Accuracy: {results['accuracy']:.3f}")
        print(f"   Test Loss: {results['loss']:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Model Parameters: {self.results['cnn']['model_complexity']:,}")

        return model, trainer

    def compare_results(self):
        """Create a comprehensive comparison of all classifiers"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE COMPARISON")
        print("="*60)

        # Create comparison table
        table_data = []
        for name, results in self.results.items():
            display_name = {
                'manual_perceptron': 'Manual Perceptron',
                'mlp': 'Multi-Layer Perceptron',
                'cnn': 'Convolutional NN'
            }[name]

            accuracy = results['accuracy']
            complexity = results.get('model_complexity', 'N/A')
            training_time = results['training_time']

            if isinstance(complexity, int):
                complexity_str = f"{complexity:,}"
            else:
                complexity_str = str(complexity)

            table_data.append([
                display_name,
                f"{accuracy:.3f}",
                complexity_str,
                f"{training_time:.1f}s"
            ])

        headers = ["Classifier", "Accuracy", "Parameters", "Training Time"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Find best performer
        best_classifier = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_classifier]['accuracy']

        print(f"\n🏆 Best Performer: {best_classifier.replace('_', ' ').title()}")
        print(f"   Best Accuracy: {best_accuracy:.3f}")

        # Calculate relative performance
        print(f"\n📈 Performance Analysis:")
        for name, results in self.results.items():
            relative_performance = (results['accuracy'] / best_accuracy) * 100
            display_name = name.replace('_', ' ').title()

            if name == best_classifier:
                print(f"   {display_name}: {relative_performance:.1f}% (BEST)")
            else:
                print(f"   {display_name}: {relative_performance:.1f}% of best")

        # Grade estimation based on project requirements
        print(f"\n🎓 Estimated Grades:")
        random_baseline = 0.5  # For binary classification

        for name, results in self.results.items():
            accuracy = results['accuracy']
            display_name = name.replace('_', ' ').title()

            if name == 'manual_perceptron':
                if accuracy > random_baseline:
                    grade = "1 point (better than random)"
                else:
                    grade = "0 points (not better than random)"
            else:
                # For MLP and CNN: 2 points if accuracy >= 0.9 * best_accuracy, else 1 point if > random
                threshold = 0.9 * best_accuracy
                if accuracy >= threshold:
                    grade = "2 points (within 90% of best)"
                elif accuracy > random_baseline:
                    grade = "1 point (better than random)"
                else:
                    grade = "0 points (not better than random)"

            print(f"   {display_name}: {grade}")

    def create_visualization(self):
        """Create visualization comparing all classifiers"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Accuracy comparison
        names = [name.replace('_', ' ').title() for name in self.results.keys()]
        accuracies = [self.results[name]['accuracy'] for name in self.results.keys()]
        colors = ['red', 'blue', 'green']

        bars = ax1.bar(names, accuracies, color=colors[:len(names)])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classifier Accuracy Comparison')
        ax1.set_ylim([0, 1])

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        # Add random baseline line
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
        ax1.legend()

        # 2. Model complexity comparison
        complexities = []
        complex_names = []
        for name in self.results.keys():
            complexity = self.results[name].get('model_complexity')
            if isinstance(complexity, int):
                complexities.append(complexity)
                complex_names.append(name.replace('_', ' ').title())

        if complexities:
            ax2.bar(complex_names, complexities, color=colors[:len(complex_names)])
            ax2.set_ylabel('Number of Parameters')
            ax2.set_title('Model Complexity Comparison')
            ax2.set_yscale('log')

        # 3. Training time comparison
        training_times = [self.results[name]['training_time'] for name in self.results.keys()]
        ax3.bar(names, training_times, color=colors[:len(names)])
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time Comparison')

        # 4. Performance vs Complexity scatter plot
        if len(complexities) >= 2:
            scatter_accuracies = [self.results[name]['accuracy'] for name in self.results.keys()
                                if isinstance(self.results[name].get('model_complexity'), int)]
            ax4.scatter(complexities, scatter_accuracies, c=colors[:len(complexities)], s=100)

            for i, name in enumerate(complex_names):
                ax4.annotate(name, (complexities[i], scatter_accuracies[i]),
                           xytext=(5, 5), textcoords='offset points')

            ax4.set_xlabel('Model Complexity (Parameters)')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Accuracy vs Model Complexity')
            ax4.set_xscale('log')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor complexity analysis',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Accuracy vs Model Complexity')

        plt.tight_layout()
        plt.savefig('classifier_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("📊 Comparison visualization saved as 'classifier_comparison.png'")

def main():
    """Main evaluation function"""
    print("🚀 X vs O Classification Project - Complete Evaluation")
    print("="*60)

    # Initialize evaluator
    evaluator = ProjectEvaluator(use_synthetic=True)  # Set to False if you have real data

    # Load dataset
    images, labels, data_type = evaluator.load_dataset()

    print(f"\n📁 Using {data_type} dataset")
    print(f"   Total images: {len(images)}")
    print(f"   X images: {np.sum(labels)}")
    print(f"   O images: {len(labels) - np.sum(labels)}")

    # Evaluate all classifiers
    try:
        # Classifier 1: Manual Perceptron
        perceptron = evaluator.evaluate_classifier1_manual_perceptron(images, labels)

        # Classifier 2: Multi-Layer Perceptron
        mlp, mlp_trainer = evaluator.evaluate_classifier2_mlp(images, labels)

        # Classifier 3: Convolutional Neural Network
        cnn, cnn_trainer = evaluator.evaluate_classifier3_cnn(images, labels)

        # Compare results
        evaluator.compare_results()

        # Create visualization
        evaluator.create_visualization()

        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Generated files:")
        print(f"   - classifier_comparison.png (comparison chart)")
        print(f"   - mlp_model.pth (trained MLP model)")
        print(f"   - cnn_model.pth (trained CNN model)")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()