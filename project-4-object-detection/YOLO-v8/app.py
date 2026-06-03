# =======================================================
# APP_IMAGE.PY — YOLOv8 Image Detection (Flask + Ngrok)
# =======================================================

import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from pyngrok import ngrok

app = Flask(__name__)

print("⏳ Loading YOLOv8 Nano Model...")
model = YOLO("yolov8n.pt")
print("✅ Model Loaded!")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tech GenZ | YOLOv8 Image Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0a0f18; color: #fff; text-align: center; padding: 40px; }
        .container { background: #14202d; padding: 40px; border-radius: 20px; max-width: 700px; margin: auto; box-shadow: 0 10px 30px rgba(0, 255, 136, 0.1); border: 1px solid rgba(255,255,255,0.1); }
        h1 { margin-bottom: 5px; font-weight: 800; }
        p { color: #00ff88; margin-bottom: 30px; letter-spacing: 1px; font-weight: 600; }
        .upload-btn { background: #00ff88; color: #000; padding: 15px 30px; border-radius: 10px; cursor: pointer; font-weight: bold; display: inline-block; transition: 0.3s; }
        .upload-btn:hover { background: #00cc6a; transform: translateY(-2px); }
        input[type="file"] { display: none; }
        #loader { display: none; color: #00ff88; font-weight: bold; margin-top: 20px; }
        #preview-section { display: none; margin-top: 30px; }
        img { max-width: 100%; border-radius: 12px; border: 2px solid #00ff88; box-shadow: 0 10px 20px rgba(0, 255, 136, 0.2); }
        .reset-btn { background: #ff3366; color: white; padding: 12px 25px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; margin-top: 20px; transition: 0.3s; }
        .reset-btn:hover { background: #cc2952; }
    </style>
</head>
<body>
<div class="container">
    <h1>YOLOv8 Vision Scanner</h1>
    <p>AI OBJECT DETECTION ENGINE</p>

    <label class="upload-btn" id="upload-label">
        Upload Image
        <input type="file" id="imageInput" accept="image/*" onchange="uploadImage()">
    </label>

    <div id="loader">🧠 YOLOv8 is analyzing the image...</div>

    <div id="preview-section">
        <h3 style="color: #8a9bb3;">Detection Result:</h3>
        <img id="resImg" src="">
        <br>
        <button class="reset-btn" onclick="location.reload()">Process Another Image</button>
    </div>
</div>

<script>
    async function uploadImage() {
        const fileInput = document.getElementById('imageInput');
        if (!fileInput.files.length) return;

        document.getElementById('upload-label').style.display = 'none';
        document.getElementById('loader').style.display = 'block';

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        
        try {
            const res = await fetch('/detect', { method: 'POST', body: formData });
            const data = await res.json();
            
            document.getElementById('loader').style.display = 'none';
            document.getElementById('resImg').src = "data:image/jpeg;base64," + data.image_base64;
            document.getElementById('preview-section').style.display = 'block';
        } catch (e) {
            alert("Error processing image!");
            location.reload();
        }
    }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image'].read()
    image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    
    # YOLO prediction
    results = model(image, verbose=False)[0]
    
    # Draw boxes
    res_img = results.plot()
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', res_img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({"image_base64": image_base64})

if __name__ == '__main__':
    ngrok.kill()
    ngrok.set_auth_token("34vanTj6wOahEuYzqikHCgVsvUh_55byjA96RB6ZdYoGDF7Cy")
    url = ngrok.connect(5000).public_url
    print(f"\n🌍 IMAGE APP LIVE URL: {url}\n")
    app.run(port=5000)
