# 🚀 Advanced YOLO Object Detection System

This project implements an advanced object detection system using **YOLO (You Only Look Once)**. The system provides real-time object detection capabilities for **images**, **videos**, and **live streams**. The YOLOv8 model (and other variants) is used to detect and annotate objects. The user interface is built with **Streamlit**, which allows easy integration and interaction with uploaded media files and live camera feeds.

## 🛠️ Features

- **Image Detection**: Upload an image, and the system will detect and annotate objects.
- **Video Detection**: Upload a video, and the system will process and display the video with object annotations in real-time.
- **Live Stream Detection**: Stream from a webcam or IP camera, and detect objects live.
- **Custom Model Support**: Upload your own YOLO model for specialized detection tasks.
- **Real-time Annotation**: Annotations are displayed directly on the video frames.

## ⚙️ Technologies Used

- **YOLOv8 (You Only Look Once)**: State-of-the-art object detection model.
- **Streamlit**: A framework for building interactive web applications.
- **OpenCV**: Library for computer vision tasks such as video capture and frame manipulation.
- **PyTorch**: Framework for training and inference with YOLO models.
- **NumPy**: Numerical operations, particularly for image processing.

## 🎮 Usage
1. Run the Streamlit App
To start the web application, run the following command:

```bash 
streamlit run app.py
```
2. Interface Overview
Detection Mode: Choose from Image Detection, Video Detection, or Live Stream Detection.
Model Selection: Choose from YOLOv8 variants or upload a custom YOLO model.
Confidence Threshold: Adjust the confidence level for object detection.
Live Stream: Select your camera source (default webcam, IP camera, or custom video source).
3. Supported Media
Images: Upload .jpg, .png, .jpeg images.
Videos: Upload .mp4, .avi videos.
Live Stream: Connect to a webcam or IP camera stream.


### Clone the Repository

```bash
git clone https://github.com/mahmouddbelo/advanced-yolo-detection.git
cd advanced-yolo-detection
```
