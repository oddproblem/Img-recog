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



