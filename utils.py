import cv2
import numpy as np
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis
from scipy import spatial
import os

# Initialize InsightFace models
print("Loading InsightFace models...")
# Using CPU provider as default
face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Initialize MediaPipe for backup face landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def detect_face(img):
    """Detect face using InsightFace and return the cropped and aligned face"""
    faces = face_analyzer.get(img)
    
    if not faces:
        return None, None
    
    # Get the highest confidence detection
    faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
    face = faces[0]
    
    if face.det_score < 0.5:  # Minimum confidence threshold
        return None, None
    
    # Get bounding box
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    
    # Add padding (20% each side)
    h, w = img.shape[:2]
    pad_w = int((x2 - x1) * 0.2)
    pad_h = int((y2 - y1) * 0.2)
    
    # Apply padding but stay within image bounds
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(w, x2 + pad_w)
    y2_pad = min(h, y2 + pad_h)
    
    # Get the face region
    face_img = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # If landmarks are available, align the face
    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
        # Use InsightFace's built-in alignment if possible
        aligned_face = align_face(face_img, face)
    else:
        # Simple resize if alignment not possible
        aligned_face = cv2.resize(face_img, (112, 112))
    
    return aligned_face, (x1_pad, y1_pad, x2_pad, y2_pad)

def align_face(face_img, face_obj=None):
    """Align face using landmarks"""
    try:
        # Just resize to 112x112 for ArcFace if no landmarks
        if face_obj is None or not hasattr(face_obj, 'landmark_2d_106'):
            return cv2.resize(face_img, (112, 112))
        
        # Get landmarks
        landmarks = face_obj.landmark_2d_106
        
        if landmarks is None or len(landmarks) == 0:
            return cv2.resize(face_img, (112, 112))
        
        # Get key points for alignment
        left_eye = np.mean(landmarks[[33, 34, 35, 36, 37]], axis=0)  # Left eye points
        right_eye = np.mean(landmarks[[42, 43, 44, 45, 46]], axis=0)  # Right eye points
        
        # Calculate angle for alignment
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get center point between eyes
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Get rotation matrix
        center = (int(eye_center[0]), int(eye_center[1]))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        
        # Apply rotation
        height, width = face_img.shape[:2]
        aligned = cv2.warpAffine(face_img, M, (width, height))
        
        # Calculate face size and crop area
        eye_dist = np.linalg.norm(right_eye - left_eye)
        crop_width = int(3.5 * eye_dist)
        crop_height = int(crop_width)  # Square crop for 112x112
        
        # Calculate crop coordinates
        x = int(eye_center[0] - crop_width//2)
        y = int(eye_center[1] - crop_height//3)  # Place eyes at 1/3 from top
        
        # Ensure crop region is within image bounds
        x = max(0, min(x, width - crop_width))
        y = max(0, min(y, height - crop_height))
        
        # Crop and resize
        if x + crop_width <= width and y + crop_height <= height:
            cropped = aligned[y:y+crop_height, x:x+crop_width]
            resized = cv2.resize(cropped, (112, 112))
            return resized
        else:
            return cv2.resize(aligned, (112, 112))
    except Exception as e:
        print(f"Error in face alignment: {e}")
        return cv2.resize(face_img, (112, 112))

def estimate_pose(face_img):
    """Estimate face pose using MediaPipe"""
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return (0, 0, 0)  # Default values
    
    # Extract landmarks
    lm = res.multi_face_landmarks[0].landmark
    left_eye = lm[33]
    right_eye = lm[263]
    nose = lm[1]
    
    # Calculate pose
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    dz = right_eye.z - left_eye.z
    
    # Calculate yaw, pitch, roll
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = np.degrees(np.arctan2(nose.z, nose.y))
    roll = np.degrees(np.arctan2(dz, dx))
    
    return (yaw, pitch, roll)

def get_embedding(face_img, face_obj=None):
    """Get face embedding using ArcFace"""
    if face_obj and hasattr(face_obj, 'embedding') and face_obj.embedding is not None:
        # Use the embedding directly from the face object
        return face_obj.embedding
    
    # Ensure face is 112x112 for ArcFace
    if face_img.shape[:2] != (112, 112):
        face_img = cv2.resize(face_img, (112, 112))
    
    # Use InsightFace to extract embedding
    try:
        faces = face_analyzer.get(face_img)
        if faces and len(faces) > 0:
            # Return the embedding of the highest confidence face
            return faces[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
    
    # Return None if embedding extraction failed
    return None

def compare_embeddings(emb1, emb2):
    """Compare two face embeddings using cosine similarity"""
    # Ensure embeddings are valid
    if emb1 is None or emb2 is None:
        return 0
    
    # Calculate cosine similarity (1 - cosine distance)
    similarity = 1 - spatial.distance.cosine(emb1, emb2)
    return similarity
