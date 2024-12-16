# Img-recog
YOLOv5 Object Detection with Live Video Feed and MongoDB Storage
This project demonstrates real-time object detection using the YOLOv5 model, with results from a live video feed captured using OpenCV. The detected frames are stored in a MongoDB database using GridFS for efficient management.

Features
Real-time Object Detection: Utilizes YOLOv5 for accurate and fast object detection.
Live Video Feed: Processes frames from your device's camera using OpenCV.
MongoDB Integration: Stores detected frames directly in a MongoDB database with GridFS for future analysis or retrieval.
Prerequisites
Before running the code, ensure the following requirements are met:

Hardware
A computer with a camera (or an external webcam).
Sufficient processing power (GPU recommended for faster inference).
Software
Python 3.8 or higher installed on your machine.
Required libraries and dependencies installed (listed below).
Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/oddproblem/img-recog.git
cd yolov5-mongodb-integration
2. Install Dependencies
Install the required Python libraries using pip:

bash
Copy code
pip install ultralytics opencv-python pymongo gridfs matplotlib numpy
3. Setup MongoDB
Install and start MongoDB on your machine (MongoDB Installation Guide).
Ensure the database is running at mongodb://localhost:27017/.
4. Download YOLOv5 Model
The code uses the yolov5s.pt model by default.
Download the YOLOv5 models from the YOLOv5 GitHub Repository.
How to Run
Ensure MongoDB is running locally:

bash
Copy code
mongod
Run the Python script:

bash
Copy code
python yolov5_mongodb.py
A live feed window will open, displaying the detected objects with bounding boxes.

Detected frames are stored in the MongoDB database under the image_database GridFS collection.
Press the q key to stop the live video feed.

How It Works
YOLOv5 Model Initialization
The script loads a pre-trained YOLOv5 model (yolov5s.pt) for object detection.

Live Video Feed
The device's camera feed is processed frame-by-frame using OpenCV.

Object Detection

Each frame is analyzed with YOLOv5 for object detection.
Detected objects are highlighted with bounding boxes and labels.
Frame Storage in MongoDB

Frames with detected objects are encoded as JPEG images.
The encoded images are uploaded to a MongoDB GridFS collection.
Code Explanation
Here’s a quick breakdown of the core logic:

Object Detection
YOLOv5 is used to detect objects in each frame, with bounding boxes drawn around the detected objects.
Labels and confidence scores are displayed.
Database Integration
MongoDB with GridFS is used to store frames efficiently.
Each frame is saved with a unique filename for easy retrieval.
Live Feed Display
Frames are displayed in a window, with bounding boxes for detected objects.
Optionally, detected frames can also be visualized using Matplotlib.
Example Output
A live video feed window showcasing detected objects with bounding boxes and labels.
Detected frames are stored in MongoDB for further analysis.
Notes
If no objects are detected in a frame, the frame is skipped.
Ensure your system camera is functional and properly connected.
For GPU acceleration, install torch with CUDA support.
Future Improvements
Add a feature to retrieve and display stored frames from the MongoDB database.
Enhance detection performance by fine-tuning the YOLOv5 model on a custom dataset.
Add support for video file input instead of live camera feed.


## Code

Here is the Python code for real-time object detection and MongoDB integration:

```python
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pymongo import MongoClient
import gridfs
import os
import matplotlib.pyplot as plt  # Added for displaying images

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["image_database"]
fs = gridfs.GridFS(db)

# Load the YOLOv5 model (small version for faster inference)
model = YOLO("yolov5s.pt")  # You can use other versions like 'yolov5m', 'yolov5l', or 'yolov5x'

# Open the device camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_counter = 0  # To keep track of saved frames

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform detection on the current frame
    results = model(frame)  # Perform inference using the frame as input

    # Get predictions (boxes, class labels, and confidences)
    boxes = results[0].boxes  # Detected bounding boxes
    labels = results[0].names  # Class names
    probs = boxes.conf  # Confidence scores

    # Check if any objects are detected
    if len(boxes) > 0:
        # Draw bounding boxes on the frame
        for box, label, prob in zip(boxes.xyxy, boxes.cls, probs):
            x1, y1, x2, y2 = map(int, box.tolist())  # Get the coordinates of the bounding box
            label_name = labels[int(label)]  # Get the class name
            confidence = prob.item()  # Get the confidence score

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the rectangle (green color)
            cv2.putText(frame, f'{label_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the frame with detections as a JPEG image in memory
        _, buffer = cv2.imencode(".jpg", frame)

        # Upload the frame to MongoDB
        fs.put(buffer.tobytes(), filename=f"detected_frame_{frame_counter}.jpg")
        print(f"Frame {frame_counter} uploaded to MongoDB.")
        frame_counter += 1

        # Display the frame in the console (additional functionality)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format for Matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(frame_rgb)
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print("No objects detected.")

    # Display the frame in a window
    cv2.imshow("YOLO Detection", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open OpenCV windows
cap.release()
cv2.destroyAllWindows()


