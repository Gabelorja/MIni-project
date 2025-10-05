import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess_simple(img_bgr, out_size=28, cutoff=90):
    """
    Advanced preprocessing for X/O images:
    1. Convert to grayscale
    2. Apply threshold to get black/white
    3. Crop to bounding box
    4. Center on square canvas
    5. Resize to output size

    Args:
        img_bgr: Input image (BGR format from cv2)
        out_size: Output image size (default 28x28)
        cutoff: Threshold value (default 90, adjust if needed)

    Returns:
        Preprocessed grayscale image
    """
    # 1) Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) Fixed threshold → black/white
    _, bw = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY_INV)

    # 3) Crop to bounding box of content
    coords = cv2.findNonZero(bw)
    if coords is None:
        return np.zeros((out_size, out_size), np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    crop = bw[y:y+h, x:x+w]

    # 4) Center on square canvas
    side = max(w, h)
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h)//2, (side - w)//2
    canvas[y0:y0+h, x0:x0+w] = crop

    # 5) Resize down
    out = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return out

class DataPreprocessor:
    def __init__(self, image_size=(28, 28), augmentation=True):
        self.image_size = image_size
        self.augmentation = augmentation

        # Basic transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

        # Augmentation transforms
        if augmentation:
            self.augment_transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Slightly larger for random crop
                transforms.RandomRotation(15),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def load_images_from_directory(self, data_dir):
        """
        Load images from directory structure:
        data_dir/
            X/ (or x/)
                image1.jpg
                image2.jpg
                ...
            O/ (or o/)
                image1.jpg
                image2.jpg
                ...

        Also supports clean_x/ and clean_o/ naming from preprocessing.
        """
        images = []
        labels = []

        # Support multiple directory naming conventions
        class_mappings = [
            (['X', 'x', 'clean_x'], 1),  # X variants
            (['O', 'o', 'clean_o'], 0)   # O variants
        ]

        for class_variants, label in class_mappings:
            class_dir = None

            # Find which variant exists
            for variant in class_variants:
                test_dir = os.path.join(data_dir, variant)
                if os.path.exists(test_dir):
                    class_dir = test_dir
                    break

            if class_dir is None:
                print(f"Warning: No directory found for class variants {class_variants} in {data_dir}")
                continue

            print(f"Loading images from: {class_dir}")
            count = 0

            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        # Load image in color first (for preprocessing)
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is not None:
                            # Apply advanced preprocessing
                            img = preprocess_simple(img_bgr, out_size=28, cutoff=90)
                            if img is not None and img.size > 0:
                                images.append(img)
                                labels.append(label)
                                count += 1
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

            label_name = 'X' if label == 1 else 'O'
            print(f"  Loaded {count} {label_name} images")

        if len(images) == 0:
            print(f"ERROR: No images found in {data_dir}")
            print("Expected structure: data/X/ and data/O/ (or data/x/ and data/o/)")
            print("Or: data/clean_x/ and data/clean_o/")

        return np.array(images), np.array(labels)

    def preprocess_image(self, image):
        """Preprocess a single image for manual perceptron"""
        # Convert to PIL Image for transforms
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert('L')
        else:
            pil_img = image.convert('L')

        # Apply basic transform
        tensor_img = self.basic_transform(pil_img)

        # Flatten for perceptron
        flattened = tensor_img.view(-1).numpy()

        return flattened

    def create_dataset(self, images, labels, test_size=0.2, augment_factor=3):
        """Create train/test split with optional augmentation"""
        # Initial train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Convert to tensors
        train_data = []
        train_labels = []

        # Process training data
        for img, label in zip(X_train, y_train):
            pil_img = Image.fromarray(img).convert('L')

            # Add original image
            tensor_img = self.basic_transform(pil_img)
            train_data.append(tensor_img)
            train_labels.append(label)

            # Add augmented versions
            if self.augmentation:
                for _ in range(augment_factor):
                    aug_img = self.augment_transform(pil_img)
                    train_data.append(aug_img)
                    train_labels.append(label)

        # Process test data (no augmentation)
        test_data = []
        test_labels = []
        for img, label in zip(X_test, y_test):
            pil_img = Image.fromarray(img).convert('L')
            tensor_img = self.basic_transform(pil_img)
            test_data.append(tensor_img)
            test_labels.append(label)

        return (torch.stack(train_data), torch.tensor(train_labels, dtype=torch.long),
                torch.stack(test_data), torch.tensor(test_labels, dtype=torch.long))

    def visualize_samples(self, images, labels, num_samples=8):
        """Visualize sample images from the dataset"""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()

        indices = np.random.choice(len(images), num_samples, replace=False)

        for i, idx in enumerate(indices):
            img = images[idx]
            label = 'X' if labels[idx] == 1 else 'O'

            if len(img.shape) == 3:  # If tensor with channel dimension
                img = img.squeeze().numpy()

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        plt.show()

def create_sample_dataset():
    """Create a sample dataset for testing (when real images aren't available)"""
    np.random.seed(42)

    # Create synthetic X images (diagonal lines)
    X_images = []
    for _ in range(50):
        img = np.zeros((28, 28), dtype=np.uint8)
        # Draw diagonal lines for X
        for i in range(28):
            for j in range(28):
                # Main diagonal
                if abs(i - j) < 3:
                    img[i, j] = 255
                # Anti-diagonal
                if abs(i + j - 27) < 3:
                    img[i, j] = 255
        # Add some noise
        noise = np.random.normal(0, 20, (28, 28))
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        X_images.append(img)

    # Create synthetic O images (circle)
    O_images = []
    for _ in range(50):
        img = np.zeros((28, 28), dtype=np.uint8)
        # Draw circle
        center = (14, 14)
        radius = 10
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if abs(dist - radius) < 2:
                    img[i, j] = 255
        # Add some noise
        noise = np.random.normal(0, 20, (28, 28))
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        O_images.append(img)

    # Combine datasets
    images = np.array(X_images + O_images)
    labels = np.array([1] * len(X_images) + [0] * len(O_images))

    return images, labels

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()

    # Try to load real data, fall back to synthetic
    if os.path.exists("data"):
        images, labels = preprocessor.load_images_from_directory("data")
        print(f"Loaded {len(images)} real images")
    else:
        print("No data directory found, creating synthetic dataset...")
        images, labels = create_sample_dataset()
        print(f"Created {len(images)} synthetic images")

    # Create dataset splits
    X_train, y_train, X_test, y_test = preprocessor.create_dataset(images, labels)

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    # Visualize samples
    preprocessor.visualize_samples(X_train, y_train)