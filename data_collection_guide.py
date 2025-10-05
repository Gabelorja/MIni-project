#!/usr/bin/env python3
"""
Data Collection Guide and Helper Functions
This script helps teams collect and organize their X vs O dataset
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

class DataCollectionHelper:
    """
    Helper class for collecting and organizing X vs O dataset
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directory structure"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "X"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "O"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)

        print(f"📁 Directory structure created:")
        print(f"   {self.data_dir}/")
        print(f"   ├── X/          (X images go here)")
        print(f"   ├── O/          (O images go here)")
        print(f"   └── raw/        (original photos)")

    def create_collection_checklist(self):
        """Generate a data collection checklist"""
        checklist = """
        📋 DATA COLLECTION CHECKLIST FOR X vs O CLASSIFICATION

        PREPARATION:
        □ Choose one team member as the consistent writer
        □ Use the same whiteboard and marker throughout
        □ Ensure good, consistent lighting
        □ Use a camera/phone with decent resolution
        □ Clear the whiteboard completely between samples

        COLLECTION GUIDELINES:
        □ Collect at least 50 samples of X and 50 samples of O
        □ Vary the size slightly (but keep reasonably consistent)
        □ Vary the position on the whiteboard
        □ Include some slight variations in drawing style
        □ Take photos from same distance and angle
        □ Ensure the shape fills a reasonable portion of the image
        □ Avoid shadows, reflections, or background distractions

        PHOTO REQUIREMENTS:
        □ Good contrast between marker and whiteboard
        □ Shape is clearly visible and well-formed
        □ Minimal background noise (other writing, smudges)
        □ Consistent camera angle (straight-on, not tilted)
        □ Adequate resolution (at least 800x600 pixels)

        FILE ORGANIZATION:
        □ Name files descriptively (e.g., X_001.jpg, O_001.jpg)
        □ Keep raw photos in data/raw/ directory
        □ Sort processed images into data/X/ and data/O/
        □ Document any special conditions or variations

        QUALITY CHECKS:
        □ Review all images before training
        □ Remove blurry, unclear, or mislabeled images
        □ Ensure balanced dataset (similar number of X's and O's)
        □ Test preprocessing on a few samples first

        RECOMMENDED VARIATIONS TO INCLUDE:
        □ Different positions on whiteboard (center, corners, edges)
        □ Slight size variations (80%-120% of typical size)
        □ Minor style variations (thick/thin lines, slightly different angles)
        □ Different lighting conditions (if possible)

        ADVANCED COLLECTION TIPS:
        □ Take multiple photos of same shape from slightly different angles
        □ Include some "challenging" cases (slightly imperfect shapes)
        □ Consider different marker types if available
        □ Document metadata (time, conditions, variations)

        AVOID:
        □ Completely different writing styles mid-collection
        □ Extreme size variations that would confuse classification
        □ Poor lighting that makes shapes unclear
        □ Tilted or angled photos
        □ Including multiple shapes in one photo
        □ Using different colored markers
        """

        with open("data_collection_checklist.txt", "w") as f:
            f.write(checklist)

        print("📋 Checklist saved as 'data_collection_checklist.txt'")
        print("\nKey points for successful data collection:")
        print("1. 🎯 CONSISTENCY: Same person, same conditions")
        print("2. 📸 QUALITY: Good lighting, clear shapes, minimal background")
        print("3. 📊 QUANTITY: At least 50 samples per class")
        print("4. 🔄 VARIETY: Different positions and slight variations")
        print("5. 🧹 CLEANLINESS: Clear whiteboard between samples")

    def analyze_raw_photos(self, raw_dir=None):
        """Analyze raw photos and provide feedback"""
        if raw_dir is None:
            raw_dir = os.path.join(self.data_dir, "raw")

        if not os.path.exists(raw_dir):
            print(f"❌ Raw directory {raw_dir} not found")
            return

        image_files = [f for f in os.listdir(raw_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print("❌ No images found in raw directory")
            return

        print(f"📊 Analyzing {len(image_files)} raw images...")

        # Analysis metrics
        sizes = []
        brightnesses = []
        contrasts = []

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        sample_images = []

        for i, filename in enumerate(image_files[:6]):  # Analyze first 6 images
            img_path = os.path.join(raw_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                sizes.append(img.shape)
                brightness = np.mean(img)
                contrast = np.std(img)
                brightnesses.append(brightness)
                contrasts.append(contrast)
                sample_images.append((img, filename))

        # Display sample images
        for i, (img, filename) in enumerate(sample_images):
            if i < 6:
                row, col = i // 3, i % 3
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(f'{filename}\nBright: {brightnesses[i]:.1f}')
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig('raw_image_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Print analysis results
        if sizes:
            print(f"\n📏 Image Size Analysis:")
            unique_sizes = list(set(sizes))
            print(f"   Found {len(unique_sizes)} different sizes:")
            for size in unique_sizes:
                count = sizes.count(size)
                print(f"   - {size[1]}x{size[0]}: {count} images")

        if brightnesses:
            print(f"\n💡 Image Quality Analysis:")
            print(f"   Average brightness: {np.mean(brightnesses):.1f} (0-255)")
            print(f"   Brightness std: {np.std(brightnesses):.1f}")
            print(f"   Average contrast: {np.mean(contrasts):.1f}")
            print(f"   Contrast std: {np.std(contrasts):.1f}")

            # Recommendations
            avg_brightness = np.mean(brightnesses)
            brightness_std = np.std(brightnesses)

            print(f"\n💡 Recommendations:")
            if avg_brightness < 100:
                print("   ⚠️  Images seem dark - consider better lighting")
            elif avg_brightness > 200:
                print("   ⚠️  Images seem too bright - reduce lighting/exposure")
            else:
                print("   ✅ Brightness levels look good")

            if brightness_std > 30:
                print("   ⚠️  Inconsistent brightness - try to maintain consistent lighting")
            else:
                print("   ✅ Consistent brightness across images")

            if np.mean(contrasts) < 20:
                print("   ⚠️  Low contrast - shapes may be hard to distinguish")
            else:
                print("   ✅ Good contrast for shape detection")

    def crop_and_organize_images(self, raw_dir=None):
        """Interactive tool to crop and organize images"""
        if raw_dir is None:
            raw_dir = os.path.join(self.data_dir, "raw")

        print("🔧 Interactive Image Organization Tool")
        print("Instructions:")
        print("- Each image will be displayed")
        print("- Press 'x' to label as X")
        print("- Press 'o' to label as O")
        print("- Press 's' to skip")
        print("- Press 'q' to quit")
        print("- Close window to continue to next image")

        if not os.path.exists(raw_dir):
            print(f"❌ Raw directory {raw_dir} not found")
            return

        image_files = [f for f in os.listdir(raw_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print("❌ No images found in raw directory")
            return

        processed_count = {'X': 0, 'O': 0, 'skipped': 0}

        for filename in image_files:
            img_path = os.path.join(raw_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Display image
            display_img = cv2.resize(img, (800, 600))
            cv2.imshow(f'Classify: {filename}', display_img)

            print(f"\nClassifying: {filename}")
            print("Press: 'x' for X, 'o' for O, 's' to skip, 'q' to quit")

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('x'):
                # Save as X
                save_path = os.path.join(self.data_dir, "X", filename)
                cv2.imwrite(save_path, img)
                processed_count['X'] += 1
                print(f"✅ Saved as X: {filename}")
            elif key == ord('o'):
                # Save as O
                save_path = os.path.join(self.data_dir, "O", filename)
                cv2.imwrite(save_path, img)
                processed_count['O'] += 1
                print(f"✅ Saved as O: {filename}")
            elif key == ord('s'):
                processed_count['skipped'] += 1
                print(f"⏭️  Skipped: {filename}")
            else:
                print(f"❓ Unknown command, skipping: {filename}")
                processed_count['skipped'] += 1

        print(f"\n📊 Processing Complete:")
        print(f"   X images: {processed_count['X']}")
        print(f"   O images: {processed_count['O']}")
        print(f"   Skipped: {processed_count['skipped']}")

    def validate_dataset(self):
        """Validate the collected dataset"""
        print("🔍 Dataset Validation")

        x_dir = os.path.join(self.data_dir, "X")
        o_dir = os.path.join(self.data_dir, "O")

        x_files = [f for f in os.listdir(x_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(x_dir) else []
        o_files = [f for f in os.listdir(o_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(o_dir) else []

        print(f"📊 Dataset Summary:")
        print(f"   X images: {len(x_files)}")
        print(f"   O images: {len(o_files)}")
        print(f"   Total: {len(x_files) + len(o_files)}")

        # Minimum requirements check
        min_samples = 30  # Minimum recommended
        print(f"\n✅ Requirements Check:")

        if len(x_files) >= min_samples:
            print(f"   X samples: {len(x_files)} ≥ {min_samples} ✅")
        else:
            print(f"   X samples: {len(x_files)} < {min_samples} ❌ (Need more X samples)")

        if len(o_files) >= min_samples:
            print(f"   O samples: {len(o_files)} ≥ {min_samples} ✅")
        else:
            print(f"   O samples: {len(o_files)} < {min_samples} ❌ (Need more O samples)")

        # Balance check
        total_samples = len(x_files) + len(o_files)
        if total_samples > 0:
            x_ratio = len(x_files) / total_samples
            balance_ok = 0.3 <= x_ratio <= 0.7  # Between 30%-70%

            print(f"   Balance: {x_ratio:.1%} X, {1-x_ratio:.1%} O", end="")
            if balance_ok:
                print(" ✅")
            else:
                print(" ⚠️  (Consider more balanced dataset)")

        # Create dataset metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'x_samples': len(x_files),
            'o_samples': len(o_files),
            'total_samples': total_samples,
            'x_files': x_files,
            'o_files': o_files
        }

        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Metadata saved to {self.data_dir}/metadata.json")

        return len(x_files) >= min_samples and len(o_files) >= min_samples and balance_ok

def main():
    """Main function for data collection helper"""
    print("🎯 X vs O Data Collection Helper")
    print("=" * 50)

    helper = DataCollectionHelper()

    while True:
        print("\nSelect an option:")
        print("1. 📋 Generate collection checklist")
        print("2. 📊 Analyze raw photos")
        print("3. 🔧 Organize images interactively")
        print("4. 🔍 Validate final dataset")
        print("5. 🚪 Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            helper.create_collection_checklist()
        elif choice == '2':
            helper.analyze_raw_photos()
        elif choice == '3':
            helper.crop_and_organize_images()
        elif choice == '4':
            is_valid = helper.validate_dataset()
            if is_valid:
                print("🎉 Dataset looks good! Ready for training.")
            else:
                print("⚠️  Dataset needs improvement before training.")
        elif choice == '5':
            print("👋 Goodbye! Good luck with your project.")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()