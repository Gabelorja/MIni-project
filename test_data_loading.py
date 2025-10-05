"""
Test script to verify data loading works correctly
"""
import os
import numpy as np
from data_preprocessing import DataPreprocessor, create_sample_dataset

def test_data_loading():
    print("=== Testing Data Loading ===\n")

    preprocessor = DataPreprocessor(image_size=(28, 28), augmentation=False)

    # Try to load from local data directory
    if os.path.exists("data"):
        print("✓ Found 'data' directory")
        try:
            images, labels = preprocessor.load_images_from_directory("data")

            if len(images) > 0:
                print(f"\n✓ Successfully loaded {len(images)} images!")
                print(f"  - X images: {np.sum(labels == 1)}")
                print(f"  - O images: {np.sum(labels == 0)}")
                print(f"  - Image shape: {images[0].shape}")
                print(f"  - Labels shape: {labels.shape}")

                # Show data distribution
                print(f"\n✓ Data distribution:")
                print(f"  - X: {np.sum(labels == 1) / len(labels) * 100:.1f}%")
                print(f"  - O: {np.sum(labels == 0) / len(labels) * 100:.1f}%")

                return True
            else:
                print("\n✗ No images loaded. Check that:")
                print("  1. Images are in data/X/ and data/O/ directories")
                print("  2. Images are .jpg, .png, .jpeg, .bmp, or .gif format")
                return False

        except Exception as e:
            print(f"\n✗ Error loading images: {e}")
            return False
    else:
        print("✗ 'data' directory not found")
        print("\nTo use your own images, create this structure:")
        print("  data/")
        print("    X/  (put X images here)")
        print("    O/  (put O images here)")
        print("\nOr use lowercase:")
        print("  data/")
        print("    x/  (put X images here)")
        print("    o/  (put O images here)")
        return False

def test_synthetic_data():
    print("\n\n=== Testing Synthetic Data (Fallback) ===\n")

    images, labels = create_sample_dataset()
    print(f"✓ Created {len(images)} synthetic images")
    print(f"  - X images: {np.sum(labels == 1)}")
    print(f"  - O images: {np.sum(labels == 0)}")
    print(f"  - Image shape: {images[0].shape}")

    return True

if __name__ == "__main__":
    real_data_loaded = test_data_loading()

    if not real_data_loaded:
        print("\n" + "="*50)
        print("Testing synthetic data as fallback...")
        test_synthetic_data()
        print("\n⚠ Using synthetic data. For better results, add real images to data/X/ and data/O/")
    else:
        print("\n" + "="*50)
        print("✓ Data loading successful! You can now train your models.")
