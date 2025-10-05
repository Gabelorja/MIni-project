import numpy as np
import cv2
from typing import Tuple, List
import math

class BlockBasedClassifier:
    """
    A classifier that uses block-based analysis with dot product to detect X vs O patterns
    """

    def __init__(self, grid_size: int = 20, black_threshold: float = 0.3):
        """
        Initialize the block-based classifier

        Args:
            grid_size: Number of blocks per dimension (grid_size x grid_size total blocks)
            black_threshold: Threshold below which a pixel is considered "black" (0-1 range)
        """
        self.grid_size = grid_size
        self.total_blocks = grid_size * grid_size
        self.black_threshold = black_threshold

        # Pre-compute pattern templates for dot product comparison
        self.circle_template = self._create_circle_template()
        self.x_template = self._create_x_template()

    def _create_circle_template(self) -> np.ndarray:
        """Create a template for circle/O pattern detection"""
        template = np.zeros((self.grid_size, self.grid_size))
        center = self.grid_size // 2
        radius = center - 1

        # Create a circle pattern
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distance = math.sqrt((i - center)**2 + (j - center)**2)
                # Mark the circle boundary
                if radius - 2 <= distance <= radius + 1:
                    template[i, j] = 1.0

        return template

    def _create_x_template(self) -> np.ndarray:
        """Create a template for X pattern detection"""
        template = np.zeros((self.grid_size, self.grid_size))

        # Create X pattern with main diagonal and anti-diagonal
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Main diagonal (top-left to bottom-right)
                if abs(i - j) <= 1:
                    template[i, j] = 1.0
                # Anti-diagonal (top-right to bottom-left)
                if abs(i + j - (self.grid_size - 1)) <= 1:
                    template[i, j] = 1.0

        return template

    def _extract_block_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract block-based features from image

        Args:
            image: Input image (grayscale, 0-1 range)

        Returns:
            Binary feature map indicating black blocks
        """
        h, w = image.shape
        block_h = h // self.grid_size
        block_w = w // self.grid_size

        feature_map = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract block
                start_y = i * block_h
                end_y = min((i + 1) * block_h, h)
                start_x = j * block_w
                end_x = min((j + 1) * block_w, w)

                block = image[start_y:end_y, start_x:end_x]

                # Calculate average intensity in block
                avg_intensity = np.mean(block)

                # Mark as black if below threshold
                feature_map[i, j] = 1.0 if avg_intensity < self.black_threshold else 0.0

        return feature_map

    def _compute_pattern_score(self, feature_map: np.ndarray, template: np.ndarray) -> float:
        """
        Compute dot product score between feature map and template

        Args:
            feature_map: Binary feature map from image
            template: Template pattern to match against

        Returns:
            Normalized dot product score
        """
        # Flatten both arrays for dot product
        feature_flat = feature_map.flatten()
        template_flat = template.flatten()

        # Compute dot product
        dot_product = np.dot(feature_flat, template_flat)

        # Normalize by template magnitude
        template_magnitude = np.linalg.norm(template_flat)
        feature_magnitude = np.linalg.norm(feature_flat)

        if template_magnitude == 0 or feature_magnitude == 0:
            return 0.0

        # Cosine similarity
        similarity = dot_product / (template_magnitude * feature_magnitude)

        return similarity

    def classify(self, image: np.ndarray) -> Tuple[str, float, dict]:
        """
        Classify image as X or O using block-based dot product analysis

        Args:
            image: Input image (can be RGB or grayscale)

        Returns:
            Tuple of (prediction, confidence, debug_info)
        """
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Normalize to 0-1 range
        image = image.astype(np.float32) / 255.0

        # Invert if needed (assuming dark marks on light background)
        if np.mean(image) > 0.5:
            image = 1.0 - image

        # Extract block features
        feature_map = self._extract_block_features(image)

        # Compute pattern scores
        circle_score = self._compute_pattern_score(feature_map, self.circle_template)
        x_score = self._compute_pattern_score(feature_map, self.x_template)

        # Additional heuristics
        center_region = self._get_center_region_score(feature_map)
        corner_regions = self._get_corner_regions_score(feature_map)

        # Adjust scores based on heuristics
        # O typically has empty center and filled boundary
        adjusted_circle_score = circle_score + (1.0 - center_region) * 0.3

        # X typically has filled center and corners
        adjusted_x_score = x_score + center_region * 0.2 + corner_regions * 0.2

        # Make prediction
        if adjusted_circle_score > adjusted_x_score:
            prediction = "O"
            confidence = adjusted_circle_score / (adjusted_circle_score + adjusted_x_score)
        else:
            prediction = "X"
            confidence = adjusted_x_score / (adjusted_circle_score + adjusted_x_score)

        debug_info = {
            'feature_map': feature_map,
            'circle_score': circle_score,
            'x_score': x_score,
            'adjusted_circle_score': adjusted_circle_score,
            'adjusted_x_score': adjusted_x_score,
            'center_region': center_region,
            'corner_regions': corner_regions,
            'total_black_blocks': np.sum(feature_map)
        }

        return prediction, confidence, debug_info

    def _get_center_region_score(self, feature_map: np.ndarray) -> float:
        """Get score for center region activity"""
        center = self.grid_size // 2
        radius = max(1, self.grid_size // 6)

        center_sum = 0
        center_count = 0

        for i in range(max(0, center - radius), min(self.grid_size, center + radius + 1)):
            for j in range(max(0, center - radius), min(self.grid_size, center + radius + 1)):
                center_sum += feature_map[i, j]
                center_count += 1

        return center_sum / center_count if center_count > 0 else 0

    def _get_corner_regions_score(self, feature_map: np.ndarray) -> float:
        """Get score for corner regions activity (typical for X)"""
        corner_size = max(1, self.grid_size // 5)
        corners = [
            (0, 0),  # top-left
            (0, self.grid_size - corner_size),  # top-right
            (self.grid_size - corner_size, 0),  # bottom-left
            (self.grid_size - corner_size, self.grid_size - corner_size)  # bottom-right
        ]

        corner_scores = []
        for start_i, start_j in corners:
            corner_sum = 0
            corner_count = 0

            for i in range(start_i, min(self.grid_size, start_i + corner_size)):
                for j in range(start_j, min(self.grid_size, start_j + corner_size)):
                    corner_sum += feature_map[i, j]
                    corner_count += 1

            if corner_count > 0:
                corner_scores.append(corner_sum / corner_count)

        return np.mean(corner_scores) if corner_scores else 0

    def visualize_analysis(self, image: np.ndarray) -> dict:
        """
        Visualize the analysis process for debugging

        Returns:
            Dictionary with visualization data
        """
        prediction, confidence, debug_info = self.classify(image)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'feature_map': debug_info['feature_map'],
            'circle_template': self.circle_template,
            'x_template': self.x_template,
            'scores': {
                'circle': debug_info['circle_score'],
                'x': debug_info['x_score'],
                'adjusted_circle': debug_info['adjusted_circle_score'],
                'adjusted_x': debug_info['adjusted_x_score']
            }
        }

# Test the classifier
if __name__ == "__main__":
    # Create classifier with 20x20 grid (400 blocks)
    classifier = BlockBasedClassifier(grid_size=20)

    # Test with a simple synthetic image
    test_image = np.ones((280, 280)) * 255  # White background

    # Draw a simple X
    for i in range(280):
        test_image[i, i] = 0  # Main diagonal
        test_image[i, 279-i] = 0  # Anti-diagonal

    result = classifier.classify(test_image)
    print(f"Test X classification: {result[0]} (confidence: {result[1]:.3f})")

    # Draw a simple O
    test_image2 = np.ones((280, 280)) * 255  # White background
    center = 140
    radius = 60
    for i in range(280):
        for j in range(280):
            dist = math.sqrt((i-center)**2 + (j-center)**2)
            if radius-5 <= dist <= radius+5:
                test_image2[i, j] = 0

    result2 = classifier.classify(test_image2)
    print(f"Test O classification: {result2[0]} (confidence: {result2[1]:.3f})")