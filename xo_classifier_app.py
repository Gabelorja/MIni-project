from flask import Flask, render_template_string, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import cv2
# from data_preprocessing import DataPreprocessor

app = Flask(__name__)

# Simple CNN for X/O classification (matching your existing architecture)
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# Initialize model (we'll use a simple trained model for demo)
model = SimpleCNN()
model.eval()

# HTML template for the XO classifier app
XO_CLASSIFIER_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X/O Classifier Whiteboard</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            align-items: start;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .classifier-section {
            background: #e9ecef;
            padding: 20px;
            border-radius: 10px;
        }
        .toolbar {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tool-group {
            display: flex;
            align-items: center;
            gap: 10px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        button {
            padding: 10px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            background-color: #007bff;
            color: white;
            transition: all 0.3s;
            font-weight: 500;
        }
        button:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
        }
        button.active {
            background-color: #28a745;
        }
        input[type="range"] {
            width: 100px;
        }
        input[type="color"] {
            width: 50px;
            height: 35px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        #whiteboard {
            border: 3px solid #ddd;
            border-radius: 10px;
            cursor: crosshair;
            display: block;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .canvas-container {
            text-align: center;
            margin-top: 20px;
        }
        .prediction-area {
            text-align: center;
        }
        .prediction-result {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .probability-bars {
            margin: 20px 0;
        }
        .prob-bar {
            display: flex;
            align-items: center;
            margin: 10px 0;
            font-weight: bold;
        }
        .prob-label {
            width: 30px;
            font-size: 1.2em;
        }
        .prob-fill {
            flex: 1;
            height: 25px;
            background: #f0f0f0;
            border-radius: 12px;
            margin: 0 10px;
            overflow: hidden;
        }
        .prob-value {
            height: 100%;
            border-radius: 12px;
            transition: width 0.3s ease;
        }
        .prob-x { background: linear-gradient(45deg, #ff6b6b, #ff8e53); }
        .prob-o { background: linear-gradient(45deg, #4ecdc4, #44a08d); }
        .prob-text {
            width: 50px;
            font-size: 0.9em;
        }
        .upload-area {
            border: 2px dashed #ccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 15px 0;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #007bff;
            background: #f8f9ff;
        }
        .upload-area.drag-over {
            border-color: #28a745;
            background: #f8fff8;
        }
        #uploadedImage {
            max-width: 100%;
            max-height: 200px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 X/O Classifier Whiteboard</h1>

        <div class="main-content">
            <div class="upload-section">
                <h3>📁 Upload Image</h3>

                <div class="upload-area" id="uploadArea">
                    <p>📁 Drag & drop an image here or click to upload</p>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;">
                </div>

                <div id="uploadedImageContainer" style="display: none;">
                    <img id="uploadedImage" />
                    <div style="margin-top: 10px;">
                        <button id="classifyUploadedBtn">🔍 Classify Image</button>
                        <button id="deleteImageBtn" style="background-color: #dc3545; margin-left: 10px;">🗑️ Delete Image</button>
                    </div>
                </div>
            </div>

            <div class="classifier-section">
                <h3>🧠 AI Prediction</h3>

                <div class="prediction-result">
                    <div id="predictionText">Upload an image of an X or O to get started!</div>

                    <div class="probability-bars" id="probabilityBars" style="display: none;">
                        <div class="prob-bar">
                            <div class="prob-label">❌</div>
                            <div class="prob-fill">
                                <div class="prob-value prob-x" id="probX"></div>
                            </div>
                            <div class="prob-text" id="probXText">0%</div>
                        </div>
                        <div class="prob-bar">
                            <div class="prob-label">⭕</div>
                            <div class="prob-fill">
                                <div class="prob-value prob-o" id="probO"></div>
                            </div>
                            <div class="prob-text" id="probOText">0%</div>
                        </div>
                    </div>

                    <div id="confidenceText"></div>
                </div>

                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <h4>📝 Instructions:</h4>
                    <ol>
                        <li>Upload an image of an X or O</li>
                        <li>Click "Classify Image" to analyze</li>
                        <li>See the AI's confidence levels!</li>
                        <li>Use "Delete Image" to try another photo</li>
                    </ol>
                </div>
            </div>
        </div>
    </div>

    <script>

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadedImageContainer = document.getElementById('uploadedImageContainer');
        const uploadedImage = document.getElementById('uploadedImage');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                const reader = new FileReader();
                reader.onload = (e) => {
                    uploadedImage.src = e.target.result;
                    uploadedImageContainer.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        }

        // Classification functions
        function classifyImage(imageData) {
            fetch('/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => {
                displayPrediction(data);
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('predictionText').innerHTML = 'Error during classification';
            });
        }

        function displayPrediction(data) {
            const { probabilities, predicted_class, confidence } = data;

            // Show probability bars
            document.getElementById('probabilityBars').style.display = 'block';

            // Update probability bars
            const probX = probabilities[0] * 100;
            const probO = probabilities[1] * 100;

            document.getElementById('probX').style.width = probX + '%';
            document.getElementById('probO').style.width = probO + '%';
            document.getElementById('probXText').textContent = probX.toFixed(1) + '%';
            document.getElementById('probOText').textContent = probO.toFixed(1) + '%';

            // Update prediction text
            const className = predicted_class === 0 ? 'X' : 'O';
            const emoji = predicted_class === 0 ? '❌' : '⭕';

            document.getElementById('predictionText').innerHTML =
                `<strong>Prediction: ${emoji} ${className}</strong>`;

            document.getElementById('confidenceText').innerHTML =
                `<small>Confidence: ${(confidence * 100).toFixed(1)}%</small>`;
        }

        function resetPrediction() {
            document.getElementById('predictionText').innerHTML = 'Upload an image of an X or O to get started!';
            document.getElementById('probabilityBars').style.display = 'none';
            document.getElementById('confidenceText').innerHTML = '';
        }


        // Classify uploaded image
        document.getElementById('classifyUploadedBtn').addEventListener('click', () => {
            const imageData = uploadedImage.src;
            classifyImage(imageData);
        });

        // Delete uploaded image
        document.getElementById('deleteImageBtn').addEventListener('click', () => {
            console.log('Delete button clicked');
            uploadedImageContainer.style.display = 'none';
            uploadedImage.src = '';
            fileInput.value = '';
            resetPrediction();
            console.log('Image deleted successfully');
        });

    </script>
</body>
</html>
'''

def preprocess_image(image_data):
    """
    Preprocess image for classification (simulate your data preprocessing)
    """
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Resize to 28x28 (matching your model)
        image = image.resize((28, 28))

        # Convert to numpy array
        img_array = np.array(image)

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Invert if needed (make sure X/O are white on black background)
        if np.mean(img_array) > 0.5:
            img_array = 1.0 - img_array

        # Add batch and channel dimensions
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

        return img_tensor

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    return render_template_string(XO_CLASSIFIER_HTML)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        image_data = data['image']

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        # For demo purposes, we'll create a simple mock prediction
        # In reality, you would load your trained model and use it here
        with torch.no_grad():
            # Mock prediction based on simple features
            img_np = processed_image.squeeze().numpy()

            # 9-Box Grid Classification Algorithm (Back to Basics)
            h, w = img_np.shape

            # Divide image into 3x3 grid
            box_h, box_w = h // 3, w // 3

            # Calculate average intensity for each box
            boxes = []
            for row in range(3):
                box_row = []
                for col in range(3):
                    start_r, end_r = row * box_h, (row + 1) * box_h
                    start_c, end_c = col * box_w, (col + 1) * box_w

                    # Handle edge case for last row/col
                    if row == 2:
                        end_r = h
                    if col == 2:
                        end_c = w

                    box_region = img_np[start_r:end_r, start_c:end_c]
                    avg_intensity = np.mean(box_region)
                    box_row.append(avg_intensity)
                boxes.append(box_row)

            # Convert to numpy array for easier indexing
            boxes = np.array(boxes)

            # Balanced detection for imperfect drawings
            diagonal_corners = boxes[0,0] + boxes[2,2] + boxes[0,2] + boxes[2,0]  # Main diagonals
            edges_only = boxes[0,1] + boxes[1,0] + boxes[1,2] + boxes[2,1]  # Just edges, no center
            center = boxes[1,1]  # Center alone

            # X pattern: diagonals + center should be strong compared to edges
            x_strength = diagonal_corners + center - edges_only

            # O pattern: edges should be strong, center can be weak, diagonals should be weak
            o_strength = edges_only * 2 - diagonal_corners

            # Compare strengths directly with a small bias toward X
            if x_strength > (o_strength + 0.1):  # X needs to be noticeably stronger
                x_score = 10
                o_score = 1
            else:  # O is stronger or close
                x_score = 1
                o_score = 10

            # Debug output - print the grid values
            print("\n9-Box Grid Analysis:")
            print("Box intensities:")
            for i in range(3):
                print(f"[{boxes[i][0]:.3f}, {boxes[i][1]:.3f}, {boxes[i][2]:.3f}]")
            print(f"Diagonal corners: {diagonal_corners:.3f}")
            print(f"Edges only: {edges_only:.3f}")
            print(f"Center: {center:.3f}")
            print(f"X strength: {x_strength:.3f}")
            print(f"O strength: {o_strength:.3f}")
            print(f"X Score: {x_score}")
            print(f"O Score: {o_score}")

            # Binary classification: if X pattern detected, return 100% X, otherwise 100% O
            if x_score > o_score:
                # X pattern detected
                probabilities = [1.0, 0.0]
                predicted_class = 0
                confidence = 1.0
                print("Decision: 100% X")
            else:
                # No clear X pattern, assume O (circle)
                probabilities = [0.0, 1.0]
                predicted_class = 1
                confidence = 1.0
                print("Decision: 100% O (Circle)")

        return jsonify({
            'probabilities': probabilities,
            'predicted_class': int(predicted_class),
            'confidence': float(confidence)
        })

    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({'error': 'Classification failed'}), 500

if __name__ == '__main__':
    print("🚀 Starting X/O Classifier on http://localhost:8000")
    print("🎯 Upload images or draw to classify X vs O!")
    app.run(host='0.0.0.0', port=8000, debug=True)