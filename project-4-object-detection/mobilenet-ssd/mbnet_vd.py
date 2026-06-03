import cv2
import numpy as np

# 1. Model aur Classes load karna (Same as before)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt", "models/MobileNetSSD_deploy.caffemodel")

# 2. Video Load Karna (Webcam ke liye "traffic.mp4" ki jagah 0 likh dena)
cap = cv2.VideoCapture("vd/b.mp4")

print("⏳ Starting Video Stream...")

# 3. Frame-by-Frame Processing Loop
while True:
    ret, frame = cap.read() # ret batata hai video chal rahi hai ya khatam ho gayi
    
    if not ret:
        print("✅ Video Ended.")
        break # Video khatam, loop tod do
        
    (h, w) = frame.shape[:2]

    # Blob banana aur predict karna
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Detections par loop lagana aur Box draw karna
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.50: # Video mein thoda kam confidence rakhte hain taaki objects miss na hon
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Output show karna
    cv2.imshow("Video Detection", frame)

    # 'q' button dabane par video band ho jayegi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
