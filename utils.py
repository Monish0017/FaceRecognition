import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from keras_facenet import FaceNet

# Initialize models
yolo = YOLO("yolov8n-face.pt")  # Using YOLOv8 face detection model
embedder = FaceNet()
mp_face_mesh = mp.solutions.face_mesh
pose_estimator = mp_face_mesh.FaceMesh(static_image_mode=True)

def detect_face(img):
    """Detect face using YOLOv8 and return the cropped face"""
    results = yolo(img)
    for r in results:
        boxes = r.boxes
        if not boxes:
            continue
        # Get the highest confidence detection
        conf = boxes.conf
        if len(conf) > 0:
            best_idx = conf.argmax().item()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            return img[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None

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

def get_embedding(face_img):
    """Get face embedding using FaceNet"""
    face = cv2.resize(face_img, (160, 160)).astype("float32") / 255.0
    return embedder.embeddings([face])[0]
