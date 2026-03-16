import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO('/home/max/Desktop/tennis/tennis-v5/notebook/yolov8x.pt')  # Load a pre-trained YOLOv8 model

# Open the video stream
cap = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')  # Use 0 for webcam or replace with video file path
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_time_elapsed = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()
    # Perform detection
    results = model(frame)
    end_time = time.time()
    total_time_elapsed += (end_time - start_time)*1000

    # Filter for persons
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            if box.cls == 0:  # Class ID 0 is for persons in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Person Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Time taken average: {total_time_elapsed/total_frames:.2f}ms")