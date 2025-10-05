"""
Interactive image collection tool for X vs O dataset
Allows you to draw X's and O's and save them for training
"""
import cv2
import numpy as np
import os
from datetime import datetime

class ImageCollector:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        self.canvas_size = 400
        self.drawing = False
        self.last_point = None

        # Create directories
        os.makedirs(os.path.join(save_dir, "X"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "O"), exist_ok=True)

        self.reset_canvas()

    def reset_canvas(self):
        """Create a blank white canvas"""
        self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255

    def draw(self, event, x, y, flags, param):
        """Mouse callback for drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.canvas, self.last_point, (x, y), (0, 0, 0), 8)
                self.last_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def save_image(self, label):
        """Save current canvas as image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, label, filename)

        cv2.imwrite(filepath, self.canvas)
        print(f"✓ Saved: {filepath}")

    def run(self):
        """Main collection loop"""
        window_name = "Draw X or O"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.draw)

        print("\n" + "="*60)
        print("IMAGE COLLECTION TOOL")
        print("="*60)
        print("\nInstructions:")
        print("  - Draw X or O with your mouse")
        print("  - Press 'X' to save as X image")
        print("  - Press 'O' to save as O image")
        print("  - Press 'C' to clear canvas")
        print("  - Press 'Q' to quit")
        print("\nTip: Draw 20-50 examples of each for good results!")
        print("="*60 + "\n")

        x_count = 0
        o_count = 0

        while True:
            # Display canvas with instructions
            display = self.canvas.copy()
            cv2.putText(display, f"X: {x_count} | O: {o_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, "Press: X=save X | O=save O | C=clear | Q=quit", (10, 380),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                self.reset_canvas()
                print("Canvas cleared")
            elif key == ord('x') or key == ord('X'):
                self.save_image('X')
                x_count += 1
                self.reset_canvas()
            elif key == ord('o') or key == ord('O'):
                self.save_image('O')
                o_count += 1
                self.reset_canvas()

        cv2.destroyAllWindows()

        print(f"\n✓ Collection complete!")
        print(f"  - X images: {x_count}")
        print(f"  - O images: {o_count}")
        print(f"  - Total: {x_count + o_count}")
        print(f"\nImages saved to: {self.save_dir}/X/ and {self.save_dir}/O/")
        print("\nNext step: Run 'python test_data_loading.py' to verify")


def collect_from_file():
    """Guide user to manually add image files"""
    print("\n" + "="*60)
    print("MANUAL FILE COLLECTION")
    print("="*60)
    print("\nTo add your own image files:")
    print("\n1. Place your X images in: data/X/")
    print("   - Take photos of X's drawn on paper/whiteboard")
    print("   - Or use digital X drawings")
    print("   - Supported: .jpg, .png, .jpeg, .bmp, .gif")
    print("\n2. Place your O images in: data/O/")
    print("   - Take photos of O's drawn on paper/whiteboard")
    print("   - Or use digital O drawings")
    print("\n3. Aim for at least 20-50 images per class")
    print("\n4. Run: python test_data_loading.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Choose collection method:")
    print("1. Draw X's and O's (recommended)")
    print("2. Manual file upload guide")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        collector = ImageCollector()
        collector.run()
    elif choice == "2":
        collect_from_file()
    else:
        print("Invalid choice. Run script again.")
