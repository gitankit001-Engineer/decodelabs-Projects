# =========================================================
# SECTION 1: LIBRARIES & SETUP
# =========================================================

# 1. Import necessary libraries
import cv2
import numpy as np
import os

print("✅ Libraries successfully imported!")

# 2. Define the list of objects MobileNet-SSD can detect
# Index 0 is 'background', the rest are actual objects
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 3. Assign random colors to each class for drawing bounding boxes (Dabbe)
# np.random.uniform creates random RGB color codes between 0 and 255
np.random.seed(42) # Setting a seed for reproducibility of colors
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print(f"✅ Loaded {len(CLASSES)} classes for detection.")
print("-" * 40)

# =========================================================
# SECTION 2: LOADING THE AI MODEL
# =========================================================

prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

print("⏳ Loading AI Model from disk...")

# Read the network into memory
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

print("✅ AI Model loaded successfully!")
print("-" * 40)

# =========================================================
# SECTION 3: IMAGE PRE-PROCESSING & PREDICTION
# =========================================================

image_path = "img/test3.jpg"  
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Error: Could not read image {image_path}. Check path!")
    exit()

# Get image dimensions (Height and Width)
(h, w) = image.shape[:2]

print("⏳ Converting image to Blob...")
# Convert image to a 4D Blob (Scaling and Resizing for AI)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Pass the blob through the neural network
print("🧠 AI is analyzing the image...")
net.setInput(blob)
detections = net.forward()

print(f"✅ Analysis complete! AI found {detections.shape[2]} potential objects.")
print("-" * 40)

# =========================================================
# SECTION 4: THE GATEKEEPER & DECODING THE MATRIX
# =========================================================

print("🔍 Filtering results (80% Confidence Threshold)...")

# Loop over the 100 detections
for i in np.arange(0, detections.shape[2]):
    # Extract the confidence (probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the confidence is greater than 80%
    if confidence > 0.80:
        idx = int(detections[0, 0, i, 1])

        # Compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = f"{CLASSES[idx]}: {f'{confidence * 100:.2f}%'}"
        print(f"🎯 Validated: {label}")
        
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # Position the text label right above the box
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

print("🖼️ Displaying final image...")

print("🖼️ Preparing to save final image...")

base_name = "final_output"
ext = ".jpg"
counter = 1
filename = f"{base_name}{ext}"

while os.path.exists(filename):
    filename = f"{base_name}_{counter}{ext}"
    counter += 1

# Naye naam se image save karo
cv2.imwrite(filename, image)
print(f"✅ Image successfully saved as: {filename}")



cv2.imshow("DecodeLabs - AI Optic Nerve", image)
cv2.waitKey(0) # Press any key on the image window to close it
cv2.destroyAllWindows()
