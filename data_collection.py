import os
import cv2
import json
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from keras_facenet import FaceNet
import mediapipe as mp
from datetime import datetime
from skimage.feature import local_binary_pattern, hog  # Correct import statement
from scipy.stats import entropy

# Configure paths
BASE_DIR = Path("d:/Face Recognition")
DATA_DIR = BASE_DIR / "face"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Initialize models - with model downloading if needed
print("Loading models...")
try:
    # Try to use the face model if available
    yolo = YOLO("yolov8n-face.pt")
    using_face_model = True
except FileNotFoundError:
    print("Face detection model not found. Downloading yolov8n model...")
    # Use the standard model that comes with ultralytics
    yolo = YOLO("yolov8n.pt")
    using_face_model = False
    print("Model downloaded successfully.")

print("Initializing FaceNet...")
embedder = FaceNet()
print("Initializing MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
pose_estimator = mp_face_mesh.FaceMesh(static_image_mode=True)

# Constants for face validation
FACE_MIN_SIZE = 80  # Minimum width and height for valid face detection
MIN_CONF_THRESHOLD = 0.5  # Minimum confidence for face detection

def detect_face(img):
    """Detect face using YOLOv8 and return the cropped face with improved validation"""
    if using_face_model:
        # If using face-specific model, don't filter by class
        results = yolo(img)
    else:
        # If using general model, filter for person class
        results = yolo(img, classes=[0])  # Class 0 is person in COCO dataset
    
    for r in results:
        boxes = r.boxes
        if not boxes or len(boxes) == 0:
            return None
        
        # Get the highest confidence detection
        conf = boxes.conf
        if len(conf) > 0:
            valid_detections = []
            
            # Collect all detections with confidence above threshold
            for i in range(len(conf)):
                if conf[i] >= MIN_CONF_THRESHOLD:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    # Make sure coordinates are within image bounds
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Ensure minimum face size and valid box
                    if x2 > x1 and y2 > y1 and (x2 - x1) >= FACE_MIN_SIZE and (y2 - y1) >= FACE_MIN_SIZE:
                        # Capture a bit more context around the face (20% padding)
                        pad_x = int((x2 - x1) * 0.2)
                        pad_y = int((y2 - y1) * 0.2)
                        
                        # Apply padding but stay within image bounds
                        x1_pad = max(0, x1 - pad_x)
                        y1_pad = max(0, y1 - pad_y)
                        x2_pad = min(w, x2 + pad_x)
                        y2_pad = min(h, y2 + pad_y)
                        
                        face_img = img[y1_pad:y2_pad, x1_pad:x2_pad]
                        
                        # Verify face using MediaPipe face mesh as additional validation
                        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        with mp_face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=1,
                            min_detection_confidence=0.5) as face_mesh:
                            
                            results = face_mesh.process(img_rgb)
                            if results.multi_face_landmarks:
                                # If MediaPipe confirms it's a face, add to valid detections
                                valid_detections.append((face_img, conf[i].item()))
            
            # Sort by confidence and return the best face
            if valid_detections:
                valid_detections.sort(key=lambda x: x[1], reverse=True)
                return valid_detections[0][0]
    
    return None

def estimate_pose(face_img):
    """Estimate face pose using MediaPipe"""
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    res = pose_estimator.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    
    lm = res.multi_face_landmarks[0].landmark
    left_eye = lm[33]
    right_eye = lm[263]
    nose = lm[1]
    
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = np.degrees(np.arctan2(nose.z, nose.y))
    
    return (yaw, pitch)

def get_custom_embedding(face_img, size=(160, 160)):
    """Create a custom face embedding using multiple methods"""
    try:
        # Resize face to standard dimensions
        face = cv2.resize(face_img, size)
        
        # Convert to grayscale for feature extraction
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features (Histogram of Oriented Gradients)
        hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=False,
                        block_norm='L2-Hys')
        
        # Extract LBP features (Local Binary Pattern)
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        # Get color histograms for each channel
        color_features = []
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([face], [i], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_features.extend(hist)
        
        # Calculate image moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Combine all features
        combined_features = np.concatenate([
            hog_features,
            lbp_hist,
            np.array(color_features),
            np.log(np.abs(hu_moments) + 1e-10)  # Log transform Hu moments
        ])
        
        # Normalize the combined feature vector
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
        
        return combined_features
    except Exception as e:
        print(f"Error generating custom embedding: {e}")
        return None

def get_embedding(face_img):
    """Get face embedding using a combination of FaceNet and custom features"""
    try:
        # Verify minimum size
        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            print("Face too small for reliable embedding")
            return None
        
        # Normalize the face with better preprocessing
        face = cv2.resize(face_img, (160, 160))
        
        # Convert to RGB if needed
        if len(face.shape) == 2:  # Grayscale
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 1:  # Single channel
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 4:  # RGBA
            face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)
        
        # Normalize pixel values
        face_normalized = face.astype("float32") / 255.0
        
        # Get both FaceNet and custom embeddings
        try:
            facenet_embedding = embedder.embeddings([face_normalized])[0]
            
            # Check if FaceNet embedding is valid
            if np.std(facenet_embedding) < 0.05:
                print("Warning: Low variance in FaceNet embedding, may be unreliable")
                facenet_valid = False
            else:
                facenet_valid = True
                
            # Debug FaceNet embedding
            embedding_min = np.min(facenet_embedding)
            embedding_max = np.max(facenet_embedding)
            embedding_mean = np.mean(facenet_embedding)
            print(f"FaceNet stats - min: {embedding_min:.4f}, max: {embedding_max:.4f}, mean: {embedding_mean:.4f}")
        except:
            print("FaceNet embedding failed, using zeros")
            facenet_embedding = np.zeros(512)
            facenet_valid = False
        
        # Get custom embedding
        custom_embedding = get_custom_embedding(face_img)
        if custom_embedding is None:
            return None
            
        print(f"Custom embedding shape: {custom_embedding.shape}, std: {np.std(custom_embedding):.4f}")
        
        # Use custom embedding if FaceNet isn't valid
        if not facenet_valid:
            print("Using only custom embedding")
            return custom_embedding.tolist()
        
        # Return FaceNet embedding (still the best when it works properly)
        return facenet_embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

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
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
            
        # Display the frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press SPACE to capture, ESC to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Check for face pre-emptively to show status
        face_detected = detect_face(frame) is not None
        if face_detected:
            cv2.putText(display_frame, "Face Detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No Face Detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 32:  # SPACE key
            face = detect_face(frame)
            if face is not None:
                # Process face
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_file = f"{person_name}_{timestamp}_{count}.jpg"
                image_path = raw_images_dir / image_file
                
                try:
                    # Save raw face image
                    cv2.imwrite(str(image_path), face)
                    
                    # Get pose
                    pose = estimate_pose(face)
                    if pose is None:
                        pose = (0, 0)
                        print("Warning: Could not estimate pose, using default (0,0)")
                    
                    # Get embedding with validation
                    embedding = get_embedding(face)
                    if embedding is None:
                        print("Warning: Failed to get valid embedding, skipping this image")
                        os.remove(str(image_path))  # Remove the invalid image
                        continue
                    
                    # Add to person data
                    face_data = {
                        "image": str(image_path),
                        "embedding": embedding,
                        "pose": pose,
                        "timestamp": timestamp
                    }
                    person_data["faces"].append(face_data)
                    
                    # Save to JSON
                    with open(person_file, "w") as f:
                        json.dump(person_data, f, indent=2)
                    
                    print(f"Captured image #{count+1} for {person_name}")
                    count += 1
                    
                    # Display success message and the face that was captured
                    success_display = np.zeros((300, 300, 3), dtype=np.uint8)
                    face_resized = cv2.resize(face, (200, 200))
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
            else:
                print("No valid face detected!")
                # Visual feedback for no face
                cv2.putText(display_frame, "No Valid Face Detected!", (10, 90), 
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
