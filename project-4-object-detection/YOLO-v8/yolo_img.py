from ultralytics import YOLO
import cv2

# 1. Nano model load karo (Pehli baar run karne par yeh 6MB ki file auto-download karega)
model = YOLO("yolov8n.pt") 

# 2. Seedha video par predict karo (show=True se automatically boxes draw hokar video chalegi!)
results = model.predict(source="yolo/a.webp", show=True, conf=0.50)
print("✅ Prediction complete! Check the displayed image for results.")
for r in results:
    im_array = r.plot()  # Yeh bounding boxes wali image bana dega
    cv2.imshow("YOLOv8 Detection", im_array)
# Optional: Agar screen par hold karna ho video khatam hone ke baad
cv2.waitKey(0)
cv2.destroyAllWindows()
