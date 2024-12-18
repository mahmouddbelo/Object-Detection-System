import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
import torch

class AdvancedObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence = confidence
    
    def detect_objects(self, image):
        results = self.model(image, conf=self.confidence)
        annotated_image = results[0].plot()
        return annotated_image, results[0].boxes
    
    def process_video(self, input_path, output_path=None):      
        cap = cv2.VideoCapture(input_path)       
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp4', prefix='detected_')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = self.model(frame, conf=self.confidence)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        cap.release()
        out.release()
        return output_path
    
    def live_stream_detection(self, source=0):
        """
        Perform live stream object detection
        
        Args:
            source (int/str): Camera source (0 for default camera)
        
        Yields:
            np.ndarray: Annotated frames
        """
        cap = cv2.VideoCapture(source)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Detect objects
            results = self.model(frame, conf=self.confidence)
            annotated_frame = results[0].plot()
            
            yield annotated_frame
        
        cap.release()


def main():
    st.set_page_config(page_title="Advanced YOLO Detection", page_icon="🤖", layout="wide")
    
    st.title("🚀 Advanced YOLO Object Detection System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Detection Configuration")
        
        # Detection mode selection
        detection_mode = st.selectbox(
            "Select Detection Mode",
            [
                "Image Detection", 
                "Video Detection",  
                "Live Stream Detection"
            ]
        )
        
        # Model selection
        model_type = st.selectbox(
            "YOLO Model", 
            ["YOLOv8 Nano", "YOLOv8 Small", "YOLOv8 Medium", "Custom Model"]
        )
        
        # Confidence threshold
        confidence = st.slider(
            "Detection Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05
        )
    
    # Initialize detector
    try:
        model_paths = {
            "YOLOv8 Nano": "yolov8n.pt",
            "YOLOv8 Small": "yolov8s.pt",
            "YOLOv8 Medium": "yolov8m.pt",
            "Custom Model": st.sidebar.file_uploader("Upload Custom Model", type=['pt'])
        }
        
        model_path = model_paths[model_type] if isinstance(model_paths[model_type], str) else model_paths[model_type].name
        detector = AdvancedObjectDetector(model_path, confidence)
    except Exception as e:
        st.error(f"Model Initialization Error: {e}")
        return
    
    # Detection mode implementations
    if detection_mode == "Video Detection":
        st.subheader("🎥 Video Object Detection")
        
        # Video upload
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        
        if uploaded_video is not None:
            # Save temporary video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(uploaded_video.getvalue())
                input_video_path = tmp_video.name
            
            # Process video
            st.write("Processing video...")
            output_video_path = detector.process_video(input_video_path)
            
            # Provide download
            with open(output_video_path, 'rb') as video_file:
                st.download_button(
                    label="Download Processed Video",
                    data=video_file.read(),
                    file_name="detected_video.mp4",
                    mime="video/mp4"
                )
    
    elif detection_mode == "Live Stream Detection":
        st.subheader("🖥️ Live Stream Object Detection")
        
        # Camera source selection
        camera_source = st.selectbox(
            "Select Camera", 
            ["Default Camera", "IP Camera", "Custom Source"]
        )
        
        if camera_source == "Default Camera":
            source = 0
        elif camera_source == "IP Camera":
            ip_camera = st.text_input("Enter IP Camera URL")
            source = ip_camera if ip_camera else 0
        else:
            custom_source = st.text_input("Enter Custom Video Source")
            source = custom_source if custom_source else 0
        
        # Start live detection
        if st.button("Start Live Detection"):
            live_stream_container = st.empty()
            
            for frame in detector.live_stream_detection(source):
                live_stream_container.image(frame, channels="BGR")
    
    
    elif detection_mode == "Image Detection":
        uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_image is not None:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Detect objects
            annotated_image, results = detector.detect_objects(image)
            
            # Display results
            st.image(annotated_image, channels="BGR", caption="Detected Objects")
            
            # Show detection details
            st.subheader("Detection Results")
            for box in results:
                cls = detector.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                st.write(f"- {cls}: {conf:.2f}")       

if __name__ == "__main__":
    main()