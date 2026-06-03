# from ultralytics import YOLO
# import cv2

# model = YOLO("yolov8n.pt")

# # List of class IDs for vehicles in YOLO (car=2, motorcycle=3, bus=5, truck=7)
# VEHICLE_CLASSES = [2, 3, 5, 7]

# cap = cv2.VideoCapture("n.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Model ko frame do, lekin display manually karenge
#     results = model(frame)[0]
    
#     vehicle_count = 0
    
#     # Check what objects YOLO found
#     for box in results.boxes:
#         class_id = int(box.cls[0]) # Object ka ID kya hai
        
#         # Agar object vehicle hai, toh count badhao
#         if class_id in VEHICLE_CLASSES:
#             vehicle_count += 1
            
#     # --- DYNAMIC TRAFFIC LIGHT LOGIC ---
#     if vehicle_count >= 15:
#         traffic_status = "HEAVY TRAFFIC"
#         green_timer = 60 # seconds
#         color = (0, 0, 255) # Red text warning
#     elif vehicle_count >= 5:
#         traffic_status = "NORMAL TRAFFIC"
#         green_timer = 30
#         color = (0, 255, 255) # Yellow
#     else:
#         traffic_status = "LOW TRAFFIC"
#         green_timer = 15
#         color = (0, 255, 0) # Green

#     # Draw the count and timer on the screen
#     cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
#     cv2.putText(frame, f"Status: {traffic_status}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
#     cv2.putText(frame, f"Allocated Green Time: {green_timer}s", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

#     cv2.imshow("Smart Traffic Management", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# List of vehicle IDs in YOLO (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

cap = cv2.VideoCapture("yolo/b.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video Finished!")
        break

    # YOLO prediction on current frame
    results = model(frame, verbose=False)[0] # verbose=False se terminal saaf rahega
    
    vehicle_count = 0
    
    # Draw boxes using YOLO's plot() function
    frame = results.plot()
    
    # Count vehicles
    for box in results.boxes:
        if int(box.cls[0]) in VEHICLE_CLASSES:
            vehicle_count += 1
            
    # Logic for Traffic Light
    if vehicle_count >= 15:
        traffic_status, green_timer, color = "HEAVY", 60, (0, 0, 255)
    elif vehicle_count >= 5:
        traffic_status, green_timer, color = "NORMAL", 30, (0, 255, 255)
    else:
        traffic_status, green_timer, color = "LOW", 15, (0, 255, 0)

    # UI Display
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {traffic_status}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Green Time: {green_timer}s", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Smart Traffic Management", frame)

    # ESC key code (ASCII value 27)
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 User stopped the video.")
        break

cap.release()
cv2.destroyAllWindows()
