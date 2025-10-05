from flask import Flask, render_template_string, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import cv2
from block_classifier import BlockBasedClassifier
import matplotlib.pyplot as plt
import json

app = Flask(__name__)

# Initialize our block-based classifier
classifier = BlockBasedClassifier(grid_size=25, black_threshold=0.4)  # 625 blocks

# HTML template for the improved XO classifier app
IMPROVED_XO_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Block-Based X/O Classifier</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .drawing-section {
            text-align: center;
        }
        .analysis-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        #drawingCanvas {
            border: 3px solid #ddd;
            border-radius: 10px;
            cursor: crosshair;
            background: white;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .canvas-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            font-size: 14px;
        }
        .btn-primary {
            background: #007bff;
            color: white;
        }
        .btn-primary:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        .btn-danger:hover {
            background: #c82333;
            transform: translateY(-2px);
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-success:hover {
            background: #1e7e34;
            transform: translateY(-2px);
        }
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            margin: 15px 0;
            border-radius: 10px;
            text-align: center;
        }
        .prediction-x {
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .prediction-o {
            background: #d1ecf1;
            color: #0c5460;
            border: 2px solid #b8daff;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .analysis-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .analysis-item h4 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
        }
        .score-bar {
            background: #e9ecef;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        .score-fill {
            height: 100%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        .score-x { background: #28a745; }
        .score-o { background: #17a2b8; }
        .feature-map {
            width: 100%;
            max-width: 200px;
            margin: 10px auto;
            display: block;
            border-radius: 5px;
        }
        .debug-info {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }
        .upload-section {
            margin: 20px 0;
            text-align: center;
        }
        .file-input {
            margin: 10px;
        }
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            .analysis-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Block-Based X/O Classifier</h1>
        <p class="subtitle">
            Advanced pattern recognition using {{ grid_size }}×{{ grid_size }} grid analysis ({{ total_blocks }} blocks) with dot product matching
        </p>

        <div class="main-content">
            <div class="drawing-section">
                <h3>Draw or Upload</h3>
                <canvas id="drawingCanvas" width="300" height="300"></canvas>

                <div class="canvas-controls">
                    <button class="btn-success" onclick="classifyDrawing()">🔍 Classify</button>
                    <button class="btn-danger" onclick="clearCanvas()">🗑️ Clear</button>
                    <button class="btn-primary" onclick="toggleBrush()">✏️ <span id="brushText">Thick</span></button>
                </div>

                <div class="upload-section">
                    <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="uploadImage()">
                    <br>
                    <label for="fileInput" class="btn-primary" style="cursor: pointer; display: inline-block; margin-top: 10px;">
                        📁 Upload Image
                    </label>
                </div>
            </div>

            <div class="analysis-section">
                <h3>Analysis Results</h3>
                <div id="predictionResult" class="prediction-result" style="display: none;">
                    <span id="predictionText"></span>
                </div>

                <div class="analysis-grid">
                    <div class="analysis-item">
                        <h4>🔵 Circle Pattern Score</h4>
                        <div class="score-bar">
                            <div id="circleScore" class="score-fill score-o" style="width: 0%">0%</div>
                        </div>
                    </div>

                    <div class="analysis-item">
                        <h4>❌ X Pattern Score</h4>
                        <div class="score-bar">
                            <div id="xScore" class="score-fill score-x" style="width: 0%">0%</div>
                        </div>
                    </div>

                    <div class="analysis-item">
                        <h4>📊 Block Analysis</h4>
                        <canvas id="featureMapCanvas" class="feature-map" width="200" height="200"></canvas>
                        <div class="debug-info">
                            <div>Black blocks: <span id="blackBlocks">-</span></div>
                            <div>Center activity: <span id="centerActivity">-</span></div>
                            <div>Corner activity: <span id="cornerActivity">-</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('drawingCanvas');
        const ctx = canvas.getContext('2d');
        const featureMapCanvas = document.getElementById('featureMapCanvas');
        const featureMapCtx = featureMapCanvas.getContext('2d');

        let isDrawing = false;
        let brushSize = 8;
        let isBrushThick = true;

        // Set up canvas
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = brushSize;
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Mouse events for drawing
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        // Touch events for mobile
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouch);
        canvas.addEventListener('touchend', stopDrawing);

        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }

        function draw(e) {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            ctx.globalCompositeOperation = 'source-over';
            ctx.beginPath();
            ctx.arc(x, y, brushSize/2, 0, Math.PI * 2);
            ctx.fill();
        }

        function stopDrawing() {
            isDrawing = false;
        }

        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' :
                                            e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }

        function clearCanvas() {
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            clearResults();
        }

        function toggleBrush() {
            isBrushThick = !isBrushThick;
            brushSize = isBrushThick ? 8 : 3;
            ctx.lineWidth = brushSize;
            document.getElementById('brushText').textContent = isBrushThick ? 'Thick' : 'Thin';
        }

        function clearResults() {
            document.getElementById('predictionResult').style.display = 'none';
            document.getElementById('circleScore').style.width = '0%';
            document.getElementById('circleScore').textContent = '0%';
            document.getElementById('xScore').style.width = '0%';
            document.getElementById('xScore').textContent = '0%';
            document.getElementById('blackBlocks').textContent = '-';
            document.getElementById('centerActivity').textContent = '-';
            document.getElementById('cornerActivity').textContent = '-';

            // Clear feature map
            featureMapCtx.fillStyle = '#f8f9fa';
            featureMapCtx.fillRect(0, 0, featureMapCanvas.width, featureMapCanvas.height);
        }

        function classifyDrawing() {
            const imageData = canvas.toDataURL('image/png');

            fetch('/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => updateResults(data))
            .catch(error => console.error('Error:', error));
        }

        function uploadImage() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];

            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = new Image();
                    img.onload = function() {
                        // Clear canvas and draw uploaded image
                        ctx.fillStyle = '#fff';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);

                        // Scale image to fit canvas while maintaining aspect ratio
                        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
                        const x = (canvas.width - img.width * scale) / 2;
                        const y = (canvas.height - img.height * scale) / 2;

                        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

                        // Automatically classify
                        classifyDrawing();
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        }

        function updateResults(data) {
            // Update prediction
            const predictionResult = document.getElementById('predictionResult');
            const predictionText = document.getElementById('predictionText');

            predictionResult.style.display = 'block';
            predictionText.textContent = `Prediction: ${data.prediction} (${(data.confidence * 100).toFixed(1)}% confident)`;

            if (data.prediction === 'X') {
                predictionResult.className = 'prediction-result prediction-x';
            } else {
                predictionResult.className = 'prediction-result prediction-o';
            }

            // Update scores
            const circleScorePercent = (data.debug_info.adjusted_circle_score * 100).toFixed(1);
            const xScorePercent = (data.debug_info.adjusted_x_score * 100).toFixed(1);

            document.getElementById('circleScore').style.width = `${circleScorePercent}%`;
            document.getElementById('circleScore').textContent = `${circleScorePercent}%`;

            document.getElementById('xScore').style.width = `${xScorePercent}%`;
            document.getElementById('xScore').textContent = `${xScorePercent}%`;

            // Update debug info
            document.getElementById('blackBlocks').textContent = data.debug_info.total_black_blocks;
            document.getElementById('centerActivity').textContent = (data.debug_info.center_region * 100).toFixed(1) + '%';
            document.getElementById('cornerActivity').textContent = (data.debug_info.corner_regions * 100).toFixed(1) + '%';

            // Draw feature map
            drawFeatureMap(data.debug_info.feature_map);
        }

        function drawFeatureMap(featureMap) {
            const gridSize = featureMap.length;
            const cellSize = featureMapCanvas.width / gridSize;

            featureMapCtx.clearRect(0, 0, featureMapCanvas.width, featureMapCanvas.height);

            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const value = featureMap[i][j];
                    const color = value > 0 ? '#000' : '#fff';

                    featureMapCtx.fillStyle = color;
                    featureMapCtx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

                    // Draw grid lines
                    featureMapCtx.strokeStyle = '#ccc';
                    featureMapCtx.lineWidth = 0.5;
                    featureMapCtx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
                }
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(IMPROVED_XO_HTML,
                                grid_size=classifier.grid_size,
                                total_blocks=classifier.total_blocks)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        image_data = data['image']

        # Remove the data:image/png;base64, prefix
        image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array
        image_array = np.array(image.convert('RGB'))

        # Classify using our block-based classifier
        prediction, confidence, debug_info = classifier.classify(image_array)

        # Convert numpy arrays to lists for JSON serialization
        debug_info_serializable = {}
        for key, value in debug_info.items():
            if isinstance(value, np.ndarray):
                debug_info_serializable[key] = value.tolist()
            else:
                debug_info_serializable[key] = float(value) if isinstance(value, np.number) else value

        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'debug_info': debug_info_serializable
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Block-Based X/O Classifier on http://localhost:8000")
    print("🎯 Features:")
    print(f"   📊 Grid Analysis: {classifier.grid_size}×{classifier.grid_size} = {classifier.total_blocks} blocks")
    print("   🔍 Pattern Detection: Circle & X templates with dot product matching")
    print("   📈 Advanced Heuristics: Center region & corner analysis")
    print("   🎨 Interactive Drawing: Real-time classification")

    app.run(host='0.0.0.0', port=8000, debug=True)