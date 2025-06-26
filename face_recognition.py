# Suppress warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import cv2
import json
import numpy as np
from pathlib import Path
import mediapipe as mp
from datetime import datetime
from scipy import spatial
import time

# Import the ONNX face embedder
from onnx_embedder import get_face_embedding as get_onnx_face_embedding

# Configure paths
BASE_DIR = Path("d:/Face Recognition")
DATA_DIR = BASE_DIR / "face"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Constants
SIMILARITY_THRESHOLD = 0.3  # Face recognition similarity threshold
FACE_MIN_SIZE = 50  # Minimum width and height for valid face detection
MIN_CONF_THRESHOLD = 0.4  # Minimum confidence for face detection

# Initialize MediaPipe for face detection and landmarks
print("Initializing MediaPipe models...")
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize face detector and landmark detector
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=MIN_CONF_THRESHOLD)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define key landmark indices for embedding (must match data_collection.py)
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

def align_and_crop_face(image, landmarks, target_size=(160, 160)):
    """Aligns and crops a face using facial landmarks"""
    # Get key points (using MediaPipe format)
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

def get_face_embedding(face_img, face_landmarks=None):
    """Extract face embedding using ONNX model"""
    try:
        if face_img is None:
            return None
        
        # Use ONNX model for embedding
        embedding = get_onnx_face_embedding(face_img)
        
        if embedding is None:
            print("Failed to get ONNX embedding")
            return None
        
        return embedding
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

def detect_and_align_face(img):
    """Detect face using MediaPipe, align and crop it"""
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # MediaPipe face detection
    with mp_face_detection.FaceDetection(
        min_detection_confidence=MIN_CONF_THRESHOLD) as face_detector:
        
        detection_results = face_detector.process(img_rgb)
        
        if detection_results.detections and len(detection_results.detections) > 0:
            # Get the first detection with highest confidence
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
            if (x2 - x1) < FACE_MIN_SIZE or (y2 - y1) < FACE_MIN_SIZE:
                return None, None, None
            
            # Add padding
            pad_w = int((x2 - x1) * 0.2)
            pad_h = int((y2 - y1) * 0.2)
            
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(w, x2 + pad_w)
            y2_pad = min(h, y2 + pad_h)
            
            # Get face region
            face_region = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            
            # Get facial landmarks for alignment
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:
                
                mesh_results = face_mesh.process(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                
                if mesh_results.multi_face_landmarks and len(mesh_results.multi_face_landmarks) > 0:
                    # Resize to standard size
                    aligned_face = cv2.resize(face_region, (160, 160))
                    return aligned_face, (x1_pad, y1_pad, x2_pad, y2_pad), mesh_results.multi_face_landmarks[0]
                else:
                    # Just resize if no landmarks
                    aligned_face = cv2.resize(face_region, (160, 160))
                    return aligned_face, (x1_pad, y1_pad, x2_pad, y2_pad), None
    
    # If no face detected
    return None, None, None

def load_face_database():
    """Load all face data from JSON files"""
    database = {}
    try:
        # Create a directory if it doesn't exist
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True)
            print(f"Created directory {DATA_DIR}")
            return database
            
        for json_file in DATA_DIR.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    person_data = json.load(f)
                    # Convert embeddings to numpy arrays
                    for face in person_data["faces"]:
                        if "embedding" in face:
                            face["embedding"] = np.array(face["embedding"])
                    database[person_data["name"]] = person_data["faces"]
                    print(f"Loaded data for: {person_data['name']} ({len(person_data['faces'])} faces)")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    except Exception as e:
        print(f"Error scanning database directory: {e}")
    
    return database

def recognize_face(face_img, face_landmarks, database):
    """Recognize face by comparing with database"""
    if face_img is None:
        return "No face detected", 0
    
    if not database:
        return "No data in database", 0
    
    try:
        # Get test embedding
        test_embedding = get_face_embedding(face_img, face_landmarks)
        
        # Check if embedding is valid
        if test_embedding is None or len(test_embedding) == 0:
            return "Invalid embedding", 0
        
        best_match = {"name": "Unknown", "similarity": 0}
        all_similarities = {}  # Store similarities for debugging
        
        # First evaluate ALL faces in the database
        for person_name, faces in database.items():
            person_best_similarity = 0
            
            for face in faces:
                # Get stored embedding
                if "embedding" not in face:
                    continue
                
                face_embedding = face["embedding"]
                
                # Convert to numpy array if it's a list
                if isinstance(face_embedding, list):
                    face_embedding = np.array(face_embedding)
                
                # Check if embeddings have compatible dimensions
                if face_embedding.shape != test_embedding.shape:
                    print(f"Warning: Embedding shape mismatch: {face_embedding.shape} vs {test_embedding.shape}")
                    continue
                
                # Calculate cosine similarity
                similarity = 1 - spatial.distance.cosine(test_embedding, face_embedding)
                
                # Keep track of best similarity for this person
                if similarity > person_best_similarity:
                    person_best_similarity = similarity
            
            # Store best similarity for this person
            all_similarities[person_name] = person_best_similarity
            
            # Update global best match if better than current
            if person_best_similarity > best_match["similarity"]:
                best_match = {
                    "name": person_name,
                    "similarity": person_best_similarity
                }
        
        # Apply threshold
        if best_match["similarity"] < SIMILARITY_THRESHOLD:
            best_match["name"] = "Unknown"
        
        # Print all similarities for debugging
        similarities_str = ", ".join([f"{name}: {sim:.4f}" for name, sim in all_similarities.items()])
        print(f"Similarities: {similarities_str}")
        print(f"Best match: {best_match['name']} with similarity: {best_match['similarity']:.4f}")
            
        return best_match["name"], best_match["similarity"]
    
    except Exception as e:
        print(f"Error during face recognition: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0

def real_time_recognition():
    """Perform real-time face recognition using webcam"""
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
        print("Press ESC to exit, D to toggle debug info")
        
        # Initialize variables
        fps_counter = 0
        fps_start_time = time.time()
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
                    elapsed_time = time.time() - fps_start_time
                    fps = fps_counter / elapsed_time
                    fps_counter = 0
                    fps_start_time = time.time()
                    current_fps = fps
                else:
                    current_fps = 0
                
                # Display FPS
                if current_fps > 0:
                    cv2.putText(debug_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Run face detection and alignment
                aligned_face, bbox, face_obj = detect_and_align_face(frame)
                
                # If a face is detected, process it
                if aligned_face is not None and bbox is not None:
                    x1, y1, x2, y2 = bbox
                    
                    # Recognize face
                    best_name, best_sim = recognize_face(aligned_face, face_obj, database)
                    
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
                    
                    # Find the name with the most occurrences and highest average similarity
                    stable_name = "Unknown"
                    max_count = 0
                    max_avg_sim = 0
                    for name, count in name_counts.items():
                        if name == "Unknown" or name == "Error":
                            continue
                        
                        if count >= HISTORY_SIZE // 3:  # At least 1/3 of frames
                            avg_sim = total_similarities[name] / count
                            if avg_sim > max_avg_sim:
                                max_avg_sim = avg_sim
                                stable_name = name
                                max_count = count
                    
                    # Only use stable name if it appears enough times
                    if max_count >= HISTORY_SIZE // 3:
                        display_name = stable_name
                        # Get the average similarity for this name
                        avg_sim = total_similarities[stable_name] / max_count
                    else:
                        display_name = best_name
                        avg_sim = best_sim
                    
                    # Draw face bounding box
                    color = (0, 0, 255)  # Default to red (unknown)
                    if avg_sim > 0.7 and display_name != "Unknown":
                        color = (0, 255, 0)  # Green for high confidence
                    elif avg_sim > SIMILARITY_THRESHOLD and display_name != "Unknown":
                        color = (0, 255, 255)  # Yellow for medium confidence
                        
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Display name and similarity score
                    label = f"{display_name}: {avg_sim:.3f}"
                    cv2.putText(debug_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Show aligned face in corner if debug mode
                    if debug_mode:
                        # Display the aligned face in the corner
                        h, w = aligned_face.shape[:2]
                        display_size = (128, 128)
                        aligned_display = cv2.resize(aligned_face, display_size)
                        
                        # Create a small frame to display the aligned face
                        debug_frame[10:10+display_size[1], debug_frame.shape[1]-10-display_size[0]:debug_frame.shape[1]-10] = aligned_display
                        
                        # Add border around the aligned face
                        cv2.rectangle(debug_frame, 
                                     (debug_frame.shape[1]-10-display_size[0], 10), 
                                     (debug_frame.shape[1]-10, 10+display_size[1]), 
                                     (255, 255, 255), 2)
                        
                        # Add text "Aligned Face"
                        cv2.putText(debug_frame, "Aligned Face", 
                                   (debug_frame.shape[1]-10-display_size[0], 10-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Display similarity information
                        debug_y = 110  # Starting y position for debug text
                        cv2.putText(debug_frame, "Similarity scores:", 
                                   (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                        debug_y += 25
                        
                        # Sort by similarity
                        sorted_similarities = sorted(
                            [(name, total_similarities[name]/name_counts[name]) 
                             for name in name_counts if name not in ["Unknown", "Error"]],
                            key=lambda x: x[1], reverse=True
                        )
                        
                        for person, sim in sorted_similarities:
                            if sim < SIMILARITY_THRESHOLD * 0.7:  # Don't show very low scores
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
        # Check if models need to be downloaded
        if not os.path.exists(os.path.expanduser('~/.insightface/models/buffalo_l')):
            print("First-time setup: Downloading InsightFace models...")
            print("This may take a few minutes...")
        
        print("Starting face recognition application...")
        real_time_recognition()
    except Exception as e:
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
