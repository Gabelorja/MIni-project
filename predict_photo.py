"""
Simple script to classify X or O from a photo and show confidence percentage
Usage: python predict_photo.py <path_to_image>
"""
import torch
import numpy as np
from PIL import Image
import sys
import os
from classifier3_cnn_improved import ImprovedCNN
from classifier3_cnn import CNN
from data_preprocessing import DataPreprocessor

def load_and_preprocess_image(image_path, image_size=(28, 28)):
    """Load and preprocess an image for classification"""
    try:
        # Load image
        img = Image.open(image_path)

        # Convert to grayscale
        img = img.convert('L')

        # Resize to model input size
        img = img.resize(image_size, Image.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0

        # Invert if needed (model expects black marks on white background)
        # Check if background is dark
        if np.mean(img_array) < 0.5:
            img_array = 1.0 - img_array

        # Convert to torch tensor
        img_tensor = torch.tensor(img_array, dtype=torch.float32)

        # Add channel and batch dimensions
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        return img_tensor, img_array

    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None


def predict_image(image_path, model_path='improved_cnn_model.pth', use_improved=True):
    """
    Predict X or O from an image and show confidence

    Args:
        image_path: Path to the image file
        model_path: Path to the trained model (default: improved_cnn_model.pth)
        use_improved: Whether to use improved CNN (default: True)
    """
    print("="*60)
    print("X vs O Classifier - Photo Prediction")
    print("="*60)

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return

    print(f"\n📷 Loading image: {image_path}")

    # Load and preprocess image
    img_tensor, img_array = load_and_preprocess_image(image_path)

    if img_tensor is None:
        return

    print("✅ Image loaded and preprocessed")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        print("\nAvailable models:")

        if os.path.exists('improved_cnn_model.pth'):
            print("  - improved_cnn_model.pth (Improved CNN)")
        if os.path.exists('cnn_model.pth'):
            print("  - cnn_model.pth (Standard CNN)")
        if os.path.exists('cnn_model_quick.pth'):
            print("  - cnn_model_quick.pth (Quick-trained CNN)")

        print("\nPlease train a model first or specify a valid model path.")
        return

    # Load model
    print(f"🔄 Loading model: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_improved:
        model = ImprovedCNN(input_channels=1, num_classes=2, input_size=28)
    else:
        model = CNN(input_channels=1, num_classes=2, input_size=28)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("✅ Model loaded successfully")

    # Make prediction
    print("\n🔍 Classifying...")

    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1)

    # Get results
    pred_class = prediction.cpu().numpy()[0]
    probs = probabilities.cpu().numpy()[0]

    pred_label = 'X' if pred_class == 1 else 'O'
    o_confidence = probs[0] * 100
    x_confidence = probs[1] * 100

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    if pred_label == 'X':
        print(f"\n🎯 PREDICTION: ❌ X")
        print(f"\n📊 Confidence Breakdown:")
        print(f"   X: {x_confidence:.2f}% {'█' * int(x_confidence/2)}")
        print(f"   O: {o_confidence:.2f}% {'█' * int(o_confidence/2)}")
    else:
        print(f"\n🎯 PREDICTION: 🔵 O")
        print(f"\n📊 Confidence Breakdown:")
        print(f"   O: {o_confidence:.2f}% {'█' * int(o_confidence/2)}")
        print(f"   X: {x_confidence:.2f}% {'█' * int(x_confidence/2)}")

    # Confidence level interpretation
    max_confidence = max(x_confidence, o_confidence)

    print(f"\n💪 Confidence Level: ", end="")
    if max_confidence >= 95:
        print("Very High ⭐⭐⭐⭐⭐")
    elif max_confidence >= 85:
        print("High ⭐⭐⭐⭐")
    elif max_confidence >= 70:
        print("Medium ⭐⭐⭐")
    elif max_confidence >= 60:
        print("Low ⭐⭐")
    else:
        print("Very Low ⭐ (Uncertain)")

    print("="*60)

    # Save a visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Show preprocessed image
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title('Preprocessed Image')
        ax1.axis('off')

        # Show confidence bars
        classes = ['O', 'X']
        confidences = [o_confidence, x_confidence]
        colors = ['#17a2b8', '#28a745']

        bars = ax2.barh(classes, confidences, color=colors)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title(f'Prediction: {pred_label} ({max_confidence:.1f}%)')
        ax2.set_xlim(0, 100)

        # Add percentage labels on bars
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(conf + 2, i, f'{conf:.1f}%', va='center')

        plt.tight_layout()

        output_path = 'prediction_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Visualization saved as: {output_path}")
        plt.close()

    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python predict_photo.py <path_to_image> [model_path]")
        print("\nExample:")
        print("  python predict_photo.py my_drawing.png")
        print("  python predict_photo.py my_drawing.png improved_cnn_model.pth")
        print("\nYou can also drag and drop an image file onto this script!")

        # Interactive mode
        print("\n" + "="*60)
        image_path = input("Enter image path (or 'q' to quit): ").strip()

        if image_path.lower() == 'q':
            return

        # Remove quotes if user pasted a path with quotes
        image_path = image_path.strip('"').strip("'")

        if not image_path:
            print("No image path provided. Exiting.")
            return

        # Check for model preference
        model_choice = input("\nUse improved model? (y/n, default=y): ").strip().lower()
        use_improved = model_choice != 'n'

        if use_improved and os.path.exists('improved_cnn_model.pth'):
            model_path = 'improved_cnn_model.pth'
        elif os.path.exists('cnn_model_quick.pth'):
            model_path = 'cnn_model_quick.pth'
        elif os.path.exists('cnn_model.pth'):
            model_path = 'cnn_model.pth'
        else:
            print("\n❌ No trained model found!")
            print("Please train a model first by running:")
            print("  python classifier3_cnn_improved.py")
            print("or")
            print("  python classifier3_cnn.py")
            return

        predict_image(image_path, model_path, use_improved)

    else:
        image_path = sys.argv[1]

        # Optional model path
        if len(sys.argv) >= 3:
            model_path = sys.argv[2]
            use_improved = 'improved' in model_path.lower()
        else:
            # Auto-detect best available model
            if os.path.exists('improved_cnn_model.pth'):
                model_path = 'improved_cnn_model.pth'
                use_improved = True
            elif os.path.exists('cnn_model_quick.pth'):
                model_path = 'cnn_model_quick.pth'
                use_improved = False
            elif os.path.exists('cnn_model.pth'):
                model_path = 'cnn_model.pth'
                use_improved = False
            else:
                print("❌ No trained model found!")
                print("Please train a model first.")
                return

        predict_image(image_path, model_path, use_improved)


if __name__ == "__main__":
    main()
