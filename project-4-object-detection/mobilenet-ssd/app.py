# =======================================================
# APP.PY — AI Object Detection (MobileNet-SSD + Flask + ngrok)
# =======================================================

import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok
import os

app = Flask(__name__)

# =========================================================
# AI ENGINE SETUP (Loads only once when server starts)
# =========================================================
print("⏳ Loading AI Model into Memory...")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

# Ensure models exist before starting
if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print("❌ ERROR: Model files not found in 'models' folder!")
    exit()

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
print("✅ AI Model Loaded & Ready!")

# =========================================================
# INTEGRATED HTML + CSS + JS TEMPLATE (Professional AI Theme)
# =========================================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DecodeLabs | AI Optic Nerve</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0f18;
            --glass-bg: rgba(20, 30, 48, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
            --accent: #00ff88; 
            --accent-hover: #00cc6a;
            --text-main: #ffffff;
            --text-muted: #8a9bb3;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top, #1a2a3a, var(--bg-dark));
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }

        .main-wrapper {
            display: flex;
            flex-direction: column;
            gap: 20px;
            width: 100%;
            max-width: 900px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 40px;
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 30px 60px rgba(0,0,0,0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 20px;
        }

        .header h1 {
            font-size: 36px;
            font-weight: 800;
            margin: 0 0 5px 0;
            letter-spacing: -0.5px;
        }

        .header p.subtitle {
            color: var(--accent);
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 0;
        }

        .upload-area {
            border: 2px dashed var(--glass-border);
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(0,0,0,0.2);
        }

        .upload-area:hover {
            border-color: var(--accent);
            background: rgba(0, 255, 136, 0.05);
        }

        .upload-area p { color: var(--text-muted); font-size: 15px; margin-bottom: 15px;}
        
        input[type="file"] { display: none; }

        .primary-btn {
            padding: 14px 28px;
            background: var(--accent);
            color: #000;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(0, 255, 136, 0.2);
        }

        .primary-btn:hover { background: var(--accent-hover); transform: translateY(-2px); }

        .preview-container {
            display: none;
            text-align: center;
            margin-top: 20px;
        }

        .preview-container img {
            max-width: 100%;
            max-height: 500px;
            border-radius: 12px;
            border: 2px solid var(--accent);
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.2);
        }

        .loader {
            display: none;
            text-align: center;
            margin-top: 20px;
            color: var(--accent);
            font-weight: 600;
            font-size: 16px;
        }
    </style>
</head>
<body>

<div class="main-wrapper">
    <div class="header">
        <h1>AI Optic Nerve</h1>
        <p class="subtitle">MobileNet-SSD Vision System</p>
    </div>

    <label class="upload-area" id="upload-label">
        <p>Drag & Drop an image here or click to browse (JPG, JPEG, PNG)</p>
        <div class="primary-btn">Select Image</div>
        <input type="file" id="imageInput" accept="image/png, image/jpeg, image/jpg" onchange="processImage()">
    </label>

    <div class="loader" id="loader">🧠 AI is analyzing the image... Please wait.</div>

    <div class="preview-container" id="preview-container">
        <h3 style="color: var(--text-muted); margin-bottom: 15px;">Detection Result:</h3>
        <img id="resultImage" src="" alt="AI Processed Image">
    </div>
</div>

<script>
    async function processImage() {
        const fileInput = document.getElementById('imageInput');
        if (!fileInput.files || fileInput.files.length === 0) return;

        // Hide upload area, show loader
        document.getElementById('upload-label').style.display = 'none';
        document.getElementById('loader').style.display = 'block';
        document.getElementById('preview-container').style.display = 'none';

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert("Error: " + data.error);
                resetUI();
                return;
            }

            // Display the returned base64 image
            document.getElementById('resultImage').src = "data:image/jpeg;base64," + data.image_base64;
            
            document.getElementById('loader').style.display = 'none';
            document.getElementById('preview-container').style.display = 'block';
            
            // Allow user to click image to reset and upload another
            document.getElementById('resultImage').onclick = resetUI;

        } catch (error) {
            alert("Failed to connect to AI Engine!");
            resetUI();
        }
    }

    function resetUI() {
        document.getElementById('imageInput').value = '';
        document.getElementById('upload-label').style.display = 'block';
        document.getElementById('loader').style.display = 'none';
        document.getElementById('preview-container').style.display = 'none';
    }
</script>

</body>
</html>
"""

# =========================================================
# FLASK ROUTES
# =========================================================
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files['image']
    
    # 1. Read the image file stream into OpenCV format
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    (h, w) = image.shape[:2]

    # 2. Convert to Blob & Pass through Network
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # 3. Process Detections (Threshold = 0.80)
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.80:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            
            # Draw green rectangle and text
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 4. Encode the final image to Base64 to send to Frontend
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image_base64": image_base64})

# =========================================================
# START SERVER & NGROK
# =========================================================
if __name__ == '__main__':
    # Make sure to keep your token secure!
    ngrok.set_auth_token("34vanTj6wOahEuYzqikHCgVsvUh_55byjA96RB6ZdYoGDF7Cy")
    print("🚀 Starting AI Web Server...")
    
    # Disconnect any old ngrok sessions to avoid errors
    ngrok.kill() 
    
    public_url = ngrok.connect(5000).public_url
    print("\n" + "="*50)
    print(f"🌍 LIVE PUBLIC URL: {public_url}")
    print("="*50 + "\n")
    
    app.run(port=5000)
