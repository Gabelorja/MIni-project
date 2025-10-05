from flask import Flask, render_template_string, request, jsonify
import json
import os

app = Flask(__name__)

# HTML template for the whiteboard
WHITEBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Learning Whiteboard</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
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
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            background-color: #007bff;
            color: white;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
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
            border: 2px solid #ddd;
            border-radius: 5px;
            cursor: crosshair;
            display: block;
            margin: 0 auto;
            background-color: white;
        }
        .canvas-container {
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Deep Learning Whiteboard</h1>

        <div class="toolbar">
            <div class="tool-group">
                <label>Tool:</label>
                <button id="penTool" class="active">✏️ Pen</button>
                <button id="eraserTool">🧽 Eraser</button>
                <button id="lineTool">📏 Line</button>
                <button id="rectTool">⬜ Rectangle</button>
                <button id="circleTool">⭕ Circle</button>
            </div>

            <div class="tool-group">
                <label>Color:</label>
                <input type="color" id="colorPicker" value="#000000">
            </div>

            <div class="tool-group">
                <label>Size:</label>
                <input type="range" id="brushSize" min="1" max="50" value="3">
                <span id="sizeDisplay">3px</span>
            </div>

            <div class="tool-group">
                <button id="clearBtn">🗑️ Clear</button>
                <button id="saveBtn">💾 Save</button>
                <button id="loadBtn">📁 Load</button>
            </div>
        </div>

        <div class="canvas-container">
            <canvas id="whiteboard" width="1000" height="600"></canvas>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('whiteboard');
        const ctx = canvas.getContext('2d');

        let isDrawing = false;
        let startX, startY;
        let currentTool = 'pen';
        let currentColor = '#000000';
        let currentSize = 3;
        let savedImageData = null;

        // Tool selection
        document.querySelectorAll('[id$="Tool"]').forEach(tool => {
            tool.addEventListener('click', () => {
                document.querySelectorAll('[id$="Tool"]').forEach(t => t.classList.remove('active'));
                tool.classList.add('active');
                currentTool = tool.id.replace('Tool', '');

                if (currentTool === 'eraser') {
                    canvas.style.cursor = 'crosshair';
                } else {
                    canvas.style.cursor = 'crosshair';
                }
            });
        });

        // Color and size controls
        document.getElementById('colorPicker').addEventListener('change', (e) => {
            currentColor = e.target.value;
        });

        document.getElementById('brushSize').addEventListener('input', (e) => {
            currentSize = e.target.value;
            document.getElementById('sizeDisplay').textContent = currentSize + 'px';
        });

        // Drawing functions
        function startDrawing(e) {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;

            if (currentTool === 'pen') {
                ctx.beginPath();
                ctx.moveTo(startX, startY);
            } else if (currentTool !== 'pen') {
                savedImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            }
        }

        function draw(e) {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;

            ctx.lineWidth = currentSize;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            if (currentTool === 'pen') {
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = currentColor;
                ctx.lineTo(currentX, currentY);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(currentX, currentY);
            } else if (currentTool === 'eraser') {
                ctx.globalCompositeOperation = 'destination-out';
                ctx.beginPath();
                ctx.arc(currentX, currentY, currentSize, 0, 2 * Math.PI);
                ctx.fill();
            } else {
                // Shape tools
                ctx.putImageData(savedImageData, 0, 0);
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = currentColor;

                if (currentTool === 'line') {
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(currentX, currentY);
                    ctx.stroke();
                } else if (currentTool === 'rect') {
                    ctx.beginPath();
                    ctx.rect(startX, startY, currentX - startX, currentY - startY);
                    ctx.stroke();
                } else if (currentTool === 'circle') {
                    const radius = Math.sqrt(Math.pow(currentX - startX, 2) + Math.pow(currentY - startY, 2));
                    ctx.beginPath();
                    ctx.arc(startX, startY, radius, 0, 2 * Math.PI);
                    ctx.stroke();
                }
            }
        }

        function stopDrawing() {
            if (isDrawing) {
                isDrawing = false;
                ctx.beginPath();
            }
        }

        // Event listeners
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        // Touch support for mobile
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });

        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });

        canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            canvas.dispatchEvent(mouseEvent);
        });

        // Utility buttons
        document.getElementById('clearBtn').addEventListener('click', () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });

        document.getElementById('saveBtn').addEventListener('click', () => {
            const imageData = canvas.toDataURL();
            const link = document.createElement('a');
            link.download = 'whiteboard_' + new Date().getTime() + '.png';
            link.href = imageData;
            link.click();
        });

        document.getElementById('loadBtn').addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = (e) => {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        const img = new Image();
                        img.onload = () => {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(img, 0, 0);
                        };
                        img.src = event.target.result;
                    };
                    reader.readAsDataURL(file);
                }
            };
            input.click();
        });

        // Initialize canvas
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    </script>
</body>
</html>
'''

@app.route('/')
def whiteboard():
    return render_template_string(WHITEBOARD_HTML)

if __name__ == '__main__':
    print("🚀 Starting Deep Learning Whiteboard on http://localhost:6000")
    print("📝 Features: Drawing, shapes, colors, save/load, touch support")
    app.run(host='0.0.0.0', port=6000, debug=True)