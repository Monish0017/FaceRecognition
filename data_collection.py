import os
import cv2
import json
import numpy as np
from pathlib import Path
import mediapipe as mp
from datetime import datetime
from scipy import spatial
from onnx_embedder import get_face_embedding as get_onnx_face_embedding

# Configure paths
BASE_DIR = Path("d:/Face Recognition")
DATA_DIR = BASE_DIR / "face"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Constants for face validation
FACE_MIN_SIZE = 50  # Minimum width and height for valid face detection
MIN_CONF_THRESHOLD = 0.5  # Minimum confidence for face detection

# Initialize MediaPipe for face detection and landmarks
print("Initializing MediaPipe models...")
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define key landmark indices for embedding
KEY_LANDMARKS = [
    1,    # Nose tip
    33,   # Left eye inner corner
    263,  # Right eye inner corner
    61,   # Left mouth corner
    291,  # Right mouth corner
    199,  # Upper lip
    164,  # Lower lip
    4,    # Forehead center
    5,    # Chin
    93,   # Left eye center
    323   # Right eye center
]

# Initialize face detector and landmark detector
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=MIN_CONF_THRESHOLD)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def align_and_crop_face(image, landmarks, target_size=(160, 160)):
    """Aligns and crops a face using facial landmarks"""
    # Get key points
    left_eye = np.mean([landmarks[i] for i in [362, 385, 387, 263, 373]], axis=0)
    right_eye = np.mean([landmarks[i] for i in [33, 160, 158, 133, 153]], axis=0)
    
    # Convert relative to absolute coordinates
    h, w = image.shape[:2]
    left_eye = (left_eye[0] * w, left_eye[1] * h)
    right_eye = (right_eye[0] * w, right_eye[1] * h)
    
    # Calculate center point between eyes
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    
    # Calculate angle for alignment
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Get rotation matrix
    center = (int(eye_center[0]), int(eye_center[1]))
    M = cv2.getRotationMatrix2D(center, angle, 1)
    
    # Apply rotation
    height, width = image.shape[:2]
    aligned = cv2.warpAffine(image, M, (width, height))
    
    # Calculate face size and crop area
    eye_dist = np.linalg.norm([right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]])
    crop_width = int(3.5 * eye_dist)
    crop_height = int(crop_width * target_size[1] / target_size[0])
    
    # Calculate crop coordinates
    x = int(eye_center[0] - crop_width//2)
    y = int(eye_center[1] - crop_height//3)  # Place eyes at 1/3 from top
    
    # Ensure crop region is within image bounds
    x = max(0, min(x, width - crop_width))
    y = max(0, min(y, height - crop_height))
    
    # Crop the face
    if x + crop_width <= width and y + crop_height <= height:
        cropped = aligned[y:y+crop_height, x:x+crop_width]
        # Resize to target size
        resized = cv2.resize(cropped, target_size)
        return resized
    else:
        # Fallback to simple resize if crop coordinates are invalid
        return cv2.resize(aligned, target_size)

def get_face_embedding(face_img):
    """Extract face embedding using ONNX model"""
    try:
        # Use ONNX model for embedding
        embedding = get_onnx_face_embedding(face_img)
        
        if embedding is None:
            print("Failed to get ONNX embedding")
            return None
        
        return embedding
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_face(img):
    """Detect face using MediaPipe"""
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run face detection
    with mp_face_detection.FaceDetection(
        min_detection_confidence=MIN_CONF_THRESHOLD) as face_detector:
        
        detection_results = face_detector.process(img_rgb)
        
        if detection_results.detections and len(detection_results.detections) > 0:
            # Get the detection with highest confidence
            detection = detection_results.detections[0]
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
            
            # Ensure coordinates are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Check minimum size
            if (x2 - x1) >= FACE_MIN_SIZE and (y2 - y1) >= FACE_MIN_SIZE:
                # Add padding
                pad_w = int((x2 - x1) * 0.2)
                pad_h = int((y2 - y1) * 0.2)
                
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_h)
                x2_pad = min(w, x2 + pad_w)
                y2_pad = min(h, y2 + pad_h)
                
                # Extract face
                face_img = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                return face_img, (x1_pad, y1_pad, x2_pad, y2_pad)
    
    return None, None

def capture_and_process():
    """Capture images from webcam and process them"""
    person_name = input("Enter person name: ")
    person_file = DATA_DIR / f"{person_name}.json"
    
    # Initialize or load existing data
    if person_file.exists():
        with open(person_file, "r") as f:
            person_data = json.load(f)
    else:
        person_data = {"name": person_name, "faces": []}
    
    # Create a folder for raw images
    raw_images_dir = DATA_DIR / person_name
    raw_images_dir.mkdir(exist_ok=True)
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Ready to capture images. Press SPACE to capture, ESC to quit.")
    count = len(person_data["faces"])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
            
        # Create a display frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press SPACE to capture, ESC to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Pre-process frame to detect face
        try:
            # Run face detection for preview
            face_img, bbox = detect_face(frame)
            if face_img is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face Detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No Face Detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            cv2.putText(display_frame, f"Detection Error: {str(e)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show current count
        cv2.putText(display_frame, f"Captures: {count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Data Collection", display_frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 32:  # SPACE key
            # Run face detection for capture
            face_img, bbox = detect_face(frame)
            
            if face_img is not None:
                # Process face
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_file = f"{person_name}_{timestamp}_{count}.jpg"
                image_path = raw_images_dir / image_file
                
                try:
                    # Save face image
                    cv2.imwrite(str(image_path), face_img)
                    
                    # Get embedding
                    embedding = get_face_embedding(face_img)
                    if embedding is None:
                        print("Warning: Failed to get valid embedding, skipping this image")
                        if os.path.exists(str(image_path)):
                            os.remove(str(image_path))  # Remove the invalid image
                        continue
                    
                    # Get face landmarks for pose estimation
                    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    mesh_results = face_mesh.process(rgb_img)
                    
                    # Default pose value
                    pose = (0, 0, 0)  # yaw, pitch, roll
                    
                    # Calculate pose if landmarks are available
                    if mesh_results.multi_face_landmarks:
                        landmarks = mesh_results.multi_face_landmarks[0].landmark
                        left_eye = np.mean([(landmarks[33].x, landmarks[33].y), 
                                           (landmarks[263].x, landmarks[263].y)], axis=0)
                        right_eye = np.mean([(landmarks[362].x, landmarks[362].y), 
                                            (landmarks[263].x, landmarks[263].y)], axis=0)
                        nose = landmarks[1]
                        
                        # Calculate simple pose estimation
                        dx = right_eye[0] - left_eye[0]
                        dy = right_eye[1] - left_eye[1]
                        yaw = np.degrees(np.arctan2(dy, dx))
                        
                        nose_pos = (nose.x, nose.y)
                        eye_center = ((left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2)
                        dx_nose = nose_pos[0] - eye_center[0]
                        dy_nose = nose_pos[1] - eye_center[1]
                        
                        pitch = np.degrees(np.arctan2(dy_nose, dx_nose))
                        roll = 0  # We don't calculate roll here
                        
                        pose = (float(yaw), float(pitch), float(roll))
                    
                    # Add to person data
                    face_data = {
                        "image": str(image_path),
                        "embedding": embedding.tolist(),
                        "pose": pose,
                        "timestamp": timestamp,
                        "bbox": bbox
                    }
                    person_data["faces"].append(face_data)
                    
                    # Save to JSON
                    with open(person_file, "w") as f:
                        json.dump(person_data, f, indent=2)
                    
                    print(f"Captured image #{count+1} for {person_name}")
                    count += 1
                    
                    # Display success message and the face that was captured
                    success_display = np.zeros((400, 400, 3), dtype=np.uint8)
                    face_resized = cv2.resize(face_img, (300, 300))
                    h, w = face_resized.shape[:2]
                    success_display[50:50+h, 50:50+w] = face_resized
                    
                    cv2.putText(success_display, "Face Captured!", (60, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(success_display, f"Pose: Yaw={pose[0]:.1f}, Pitch={pose[1]:.1f}", 
                               (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow("Captured Face", success_display)
                    cv2.waitKey(1000)  # Show success message for 1 second
                    cv2.destroyWindow("Captured Face")
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("No valid face detected!")
                cv2.putText(display_frame, "No Valid Face Detected!", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Data Collection", display_frame)
                cv2.waitKey(500)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total images captured for {person_name}: {count}")


if __name__ == "__main__":
    try:
        capture_and_process()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
