# Suppress TensorFlow and MediaPipe warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from keras_facenet import FaceNet
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern, hog
from scipy.stats import entropy

# Configure paths
BASE_DIR = Path("d:/Face Recognition")
DATA_DIR = BASE_DIR / "face"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Initialize models with fallback
print("Loading models...")
try:
    # Try to use the face model if available
    yolo = YOLO("yolov8n-face.pt")
    using_face_model = True
except FileNotFoundError:
    print("Face detection model not found. Downloading standard yolov8n model...")
    # Use the standard model that comes with ultralytics
    yolo = YOLO("yolov8n.pt")
    using_face_model = False
    print("Model downloaded successfully.")

print("Initializing FaceNet...")
embedder = FaceNet()
print("Initializing MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
pose_estimator = mp_face_mesh.FaceMesh(static_image_mode=True)

# Constants
POSE_THRESHOLD = 15  # Maximum pose difference in degrees
SIMILARITY_THRESHOLD = 0.6  # Increased minimum similarity score for better accuracy
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
            return None, None
        
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
                                valid_detections.append((face_img, (x1_pad, y1_pad, x2_pad, y2_pad), conf[i].item()))
            
            # Sort by confidence and return the best face
            if valid_detections:
                valid_detections.sort(key=lambda x: x[2], reverse=True)
                return valid_detections[0][0], valid_detections[0][1]
    
    return None, None

def estimate_pose(face_img):
    """Estimate face pose using MediaPipe"""
    try:
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
    except Exception as e:
        print(f"Error estimating pose: {e}")
        return None

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
        return np.zeros(500)  # Return a zero array of appropriate size

def get_embedding(face_img):
    """Get face embedding using a combination of FaceNet and custom features"""
    try:
        # Verify minimum size
        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            print("Face too small for reliable embedding")
            return np.zeros(512)
        
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
        print(f"Custom embedding shape: {custom_embedding.shape}, std: {np.std(custom_embedding):.4f}")
        
        # Use custom embedding if FaceNet isn't valid
        if not facenet_valid:
            print("Using only custom embedding")
            return custom_embedding
        
        # Return FaceNet embedding (still the best when it works properly)
        return facenet_embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(512)  # Return zero embedding as fallback

def load_face_database():
    """Load all face data from JSON files"""
    database = {}
    try:
        for json_file in DATA_DIR.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    person_data = json.load(f)
                    database[person_data["name"]] = person_data["faces"]
                    print(f"Loaded data for: {person_data['name']} ({len(person_data['faces'])} faces)")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    except Exception as e:
        print(f"Error scanning database directory: {e}")
    
    return database

def recognize_face(face_img, pose, database):
    """Recognize face by comparing with database using improved matching"""
    if face_img is None:
        return "No face detected", 0
    
    if not database:
        return "No data in database", 0
    
    try:
        # Get test embedding
        test_embedding = get_embedding(face_img)
        
        # Check if embedding is valid (non-zero)
        if np.all(test_embedding == 0):
            return "Invalid face", 0
        
        best_match = {"name": "Unknown", "similarity": 0, "weighted_score": 0}
        all_similarities = {}  # Store all similarities for debugging
        all_pose_similarities = {}  # Store pose similarity info for debugging
        
        print(f"Test embedding shape: {test_embedding.shape}, std: {np.std(test_embedding):.4f}")
        
        # First evaluate ALL faces in the database before making a decision
        for person_name, faces in database.items():
            # Filter faces by pose similarity
            pose_filtered_faces = []
            pose_similarities = []
            
            for face in faces:
                face_pose = face["pose"]
                yaw_diff = abs(face_pose[0] - pose[0])
                pitch_diff = abs(face_pose[1] - pose[1])
                pose_sim = 1.0 - min((yaw_diff + pitch_diff) / (POSE_THRESHOLD * 2), 1.0)
                pose_similarities.append(pose_sim)
                
                if yaw_diff <= POSE_THRESHOLD and pitch_diff <= POSE_THRESHOLD:
                    pose_filtered_faces.append((face, pose_sim))
            
            # If no faces match the pose, use the top 3 faces with highest pose similarity
            if not pose_filtered_faces:
                pose_similarities_with_idx = [(i, ps) for i, ps in enumerate(pose_similarities)]
                pose_similarities_with_idx.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in pose_similarities_with_idx[:3]]
                if top_indices:  # Make sure we actually have faces to use
                    pose_filtered_faces = [(faces[idx], pose_similarities[idx]) for idx in top_indices]
            
            # Find the best matching face for this person with weighted scoring
            person_best_similarity = 0
            person_best_weighted_score = 0
            person_best_pose = None
            
            face_similarities = []
            
            for (face, pose_sim) in pose_filtered_faces:
                # Get stored embedding and ensure it's a numpy array
                if "embedding" not in face:
                    continue
                
                face_embedding = np.array(face["embedding"])
                
                # Check if embeddings have compatible dimensions
                if face_embedding.shape != test_embedding.shape:
                    print(f"Warning: Embedding shape mismatch: {face_embedding.shape} vs {test_embedding.shape}")
                    continue
                
                # Compute cosine similarity with explicit formula to debug
                dot_product = np.dot(test_embedding, face_embedding)
                norm_test = np.linalg.norm(test_embedding)
                norm_face = np.linalg.norm(face_embedding)
                
                if norm_test > 0 and norm_face > 0:
                    # Use higher precision for similarity calculation
                    similarity = float(dot_product) / (float(norm_test) * float(norm_face))
                    # Ensure similarity is between 0 and 1
                    similarity = max(0, min(1, similarity))
                else:
                    similarity = 0
                
                # Debug individual comparisons
                print(f"  {person_name} face compare - similarity: {similarity:.6f}")
                
                face_similarities.append((similarity, face["pose"]))
                
                # Weight similarity by pose similarity for better matching
                weighted_score = similarity * (0.7 + 0.3 * pose_sim)  # 70% embedding + 30% pose influence
                
                if weighted_score > person_best_weighted_score:
                    person_best_weighted_score = weighted_score
                    person_best_similarity = similarity
                    person_best_pose = face["pose"]
            
            # Store the best similarity for this person
            if face_similarities:
                all_similarities[person_name] = person_best_similarity
                all_pose_similarities[person_name] = person_best_pose
                
                # Update the global best match if this person is better
                if person_best_weighted_score > best_match.get("weighted_score", 0):
                    best_match = {
                        "name": person_name,
                        "similarity": person_best_similarity,
                        "weighted_score": person_best_weighted_score,
                        "pose": person_best_pose
                    }
        
        # Apply threshold with stricter criteria
        if best_match["similarity"] < SIMILARITY_THRESHOLD:
            best_match["name"] = "Unknown"
        
        # Print all similarities for debugging
        print("Similarities:", ", ".join([f"{name}: {sim:.6f}" for name, sim in all_similarities.items()]))
            
        return best_match["name"], best_match["similarity"]
    except Exception as e:
        print(f"Error during face recognition: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0

def real_time_recognition():
    """Perform real-time face recognition using webcam with detailed debugging visualization"""
    try:
        print("Loading face database...")
        database = load_face_database()
        print(f"Loaded {len(database)} people from database")
        
        if not database:
            print("No faces in database. Please run data_collection.py first.")
            input("Press Enter to exit...")
            return
        
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            input("Press Enter to exit...")
            return
        
        print("Starting face recognition...")
        print("Press ESC to exit")
        
        # Initialize variables
        fps_counter = 0
        fps_start_time = cv2.getTickCount()
        debug_mode = True
        recognition_history = []
        HISTORY_SIZE = 10  # Number of frames to keep in history

        # Main loop
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from webcam")
                    break
                
                # Create a debug overlay frame
                debug_frame = frame.copy()
                
                # FPS calculation
                fps_counter += 1
                if fps_counter >= 10:
                    current_time = cv2.getTickCount()
                    fps = 10 / ((current_time - fps_start_time) / cv2.getTickFrequency())
                    fps_start_time = current_time
                    fps_counter = 0
                else:
                    fps = 0
                
                # Display FPS
                if fps > 0:
                    cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Run face detection
                face, bbox = detect_face(frame)
                
                # If a face is detected, process it
                if face is not None:
                    # Get pose and display it
                    pose = estimate_pose(face) or (0, 0)
                    
                    # Draw pose information
                    cv2.putText(debug_frame, f"Yaw: {pose[0]:.1f}°, Pitch: {pose[1]:.1f}°", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Add face validation check
                    face_validation = "Valid Face"
                    img_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    with mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        min_detection_confidence=0.5) as face_mesh:
                        
                        results = face_mesh.process(img_rgb)
                        if not results.multi_face_landmarks:
                            face_validation = "Invalid Face"
                    
                    # Display face validation result
                    validation_color = (0, 255, 0) if face_validation == "Valid Face" else (0, 0, 255)
                    cv2.putText(debug_frame, face_validation, (10, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, validation_color, 2)
                    
                    # Only proceed with recognition if it's a valid face
                    if face_validation == "Valid Face":
                        # Recognize face with detailed debugging
                        best_name, best_sim = recognize_face(face, pose, database)
                        
                        # Update recognition history
                        recognition_history.append((best_name, best_sim))
                        if len(recognition_history) > HISTORY_SIZE:
                            recognition_history.pop(0)
                        
                        # Get the most frequent name in the last N frames
                        name_counts = {}
                        total_similarities = {}
                        for rec_name, rec_sim in recognition_history:
                            if rec_name not in name_counts:
                                name_counts[rec_name] = 0
                                total_similarities[rec_name] = 0
                            name_counts[rec_name] += 1
                            total_similarities[rec_name] += rec_sim
                        
                        # Find the name with the most occurrences in history
                        stable_name = "Unknown"
                        max_count = 0
                        for name, count in name_counts.items():
                            if count > max_count and name != "Unknown":
                                max_count = count
                                stable_name = name
                        
                        # Only use stable name if it appears in at least half the frames
                        if max_count >= HISTORY_SIZE // 2:
                            display_name = stable_name
                            # Get the average similarity for this name
                            avg_sim = sum([s for n, s in recognition_history if n == stable_name]) / max_count
                        else:
                            display_name = best_name
                            avg_sim = best_sim
                        
                        # Draw bounding box and results
                        x1, y1, x2, y2 = bbox
                        
                        # Choose color based on confidence
                        if avg_sim > 0.7:
                            color = (0, 255, 0)  # Green for high confidence
                        elif avg_sim > SIMILARITY_THRESHOLD:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 0, 255)  # Red for low confidence
                        
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Display name and confidence
                        label = f"{display_name}: {avg_sim:.3f}"
                        cv2.putText(debug_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Display debugging information
                        debug_y = 110  # Starting y position for debug text
                        
                        # Display similarity information
                        if debug_mode:
                            cv2.putText(debug_frame, "Similarity scores:", 
                                       (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                            debug_y += 25
                            
                            # Sort by similarity
                            sorted_similarities = sorted(
                                [(name, total_similarities[name]/name_counts[name]) 
                                 for name in name_counts if name != "Unknown"],
                                key=lambda x: x[1], reverse=True
                            )
                            
                            for person, sim in sorted_similarities:
                                if sim < SIMILARITY_THRESHOLD * 0.8:  # Don't show very low scores
                                    continue
                                    
                                color = (0, 255, 0) if person == display_name else (0, 200, 200)
                                
                                # Add an indicator for the selected person
                                prefix = "* " if person == display_name else "  "
                                
                                text = f"{prefix}{person}: {sim:.3f}"
                                cv2.putText(debug_frame, text, (10, debug_y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                debug_y += 20
                                
                                # Don't overflow the screen
                                if debug_y > frame.shape[0] - 50:
                                    cv2.putText(debug_frame, "...", (10, debug_y), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                                    break
                    else:
                        cv2.putText(debug_frame, "Face validation failed", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(debug_frame, "No Face Detected", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Reset recognition history when no face is detected
                    recognition_history = []
                
                # Add instructions
                cv2.putText(debug_frame, "Press ESC to exit, D to toggle debug info", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display the appropriate frame based on debug mode
                cv2.imshow("Face Recognition", debug_frame)
                
                # Handle key presses safely
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('d') or key == ord('D'):  # Toggle debug mode
                    debug_mode = not debug_mode
                    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
            except Exception as e:
                print(f"Error in frame processing loop: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next frame rather than exiting
                continue
        
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        print("Face recognition terminated")
        
    except Exception as e:
        print(f"Error in real_time_recognition: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        print("Starting face recognition application...")
        real_time_recognition()
    except Exception as e:
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
