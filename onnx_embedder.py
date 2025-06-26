import os
import numpy as np
import cv2
import importlib.util
from pathlib import Path
import sys

# Configure paths
BASE_DIR = Path("d:/Face Recognition")
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Check if onnxruntime is installed
onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None

if not onnxruntime_available:
    print("ERROR: onnxruntime is not installed. Please install it using:")
    print("    pip install onnxruntime")
    print("Alternatively, install all dependencies using:")
    print("    pip install -r requirements.txt")

# Path to the local OpenVINO ArcFace model file that was manually downloaded
ARCFACE_MODEL_PATH = BASE_DIR / "arcfaceresnet100-8.onnx"
# Alternative location in models directory
ARCFACE_MODEL_PATH_ALT = MODEL_DIR / "arcfaceresnet100-8.onnx"
# Default path for other models if needed
DEFAULT_MODEL_PATH = MODEL_DIR / "face_embedder.onnx"

class DummyEmbedder:
    """Fallback embedder when ONNX is not available"""
    def __init__(self):
        self.input_shape = (112, 112, 3)
        self.session = None  # Explicitly set session to None
        print("WARNING: Using fallback pixel-based embedder. Quality will be reduced.")
        
    def get_embedding(self, face_image):
        # Create a simple embedding from pixels as a fallback
        # This will have poor quality but allows the app to run
        if face_image.shape[:2] != (112, 112):
            face_image = cv2.resize(face_image, (112, 112))
        
        # Convert to grayscale and flatten to vector
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
            
        # Reduce dimensionality and normalize
        small = cv2.resize(gray, (16, 16)).flatten()
        embedding = small.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

if not onnxruntime_available:
    # Use dummy embedder if ONNX runtime is not available
    embedder = DummyEmbedder()
    get_face_embedding = embedder.get_embedding
else:
    import onnxruntime
    
    class ONNXFaceEmbedder:
        def __init__(self):
            """Initialize the ONNX face embedder"""
            # Initialize session to None first to avoid attribute errors
            self.session = None
            self.input_shape = (112, 112, 3)
            self.input_name = "data"  # Default input name for ArcFace model
            
            # Check all possible model locations in order of preference
            model_paths = [
                ARCFACE_MODEL_PATH,                # Root directory
                ARCFACE_MODEL_PATH_ALT,            # Models subdirectory
                DEFAULT_MODEL_PATH,                # Generic face_embedder.onnx
                # Add any additional .onnx files found in model directory
                *list(MODEL_DIR.glob("*.onnx"))
            ]
            
            model_path = None
            
            # Find the first existing model file
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print(f"ERROR: ArcFace model not found.")
                print(f"Please ensure the file 'arcfaceresnet100-8.onnx' exists at one of:")
                print(f"  - {ARCFACE_MODEL_PATH} (root directory)")
                print(f"  - {ARCFACE_MODEL_PATH_ALT} (models directory)")
                print("Using fallback embedder with reduced accuracy.")
                return
                
            try:
                # Create ONNX inference session
                print(f"Loading ArcFace ONNX model from: {model_path}")
                session_options = onnxruntime.SessionOptions()
                session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Try multiple provider options in case one fails
                providers_options = [
                    ['CPUExecutionProvider'],
                    None  # Default providers
                ]
                
                # Try each provider option
                for providers in providers_options:
                    try:
                        if providers:
                            self.session = onnxruntime.InferenceSession(
                                str(model_path), 
                                providers=providers,
                                sess_options=session_options
                            )
                        else:
                            self.session = onnxruntime.InferenceSession(
                                str(model_path),
                                sess_options=session_options
                            )
                        break  # Stop trying providers if successful
                    except Exception as e:
                        print(f"Failed with provider {providers}: {e}")
                        continue
                
                if self.session is None:
                    raise RuntimeError("Failed to create ONNX session with any provider")
                
                # Get model metadata
                model_inputs = self.session.get_inputs()
                if model_inputs and len(model_inputs) > 0:
                    self.input_name = model_inputs[0].name
                    print(f"Model input name: {self.input_name}")
                    
                    # Get input shape (ignoring the batch dimension)
                    if len(model_inputs[0].shape) >= 3:
                        self.input_shape = model_inputs[0].shape[2:]
                        if len(self.input_shape) == 2:
                            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)
                        
                print(f"OpenVINO ArcFace model loaded successfully, input shape: {self.input_shape}")
                print(f"Model expects input tensor named: '{self.input_name}'")
            except Exception as e:
                print(f"Error loading ONNX model: {e}")
                print("Using fallback embedder with reduced accuracy.")
                self.session = None
        
        def get_embedding(self, face_image):
            """Get face embedding from the ONNX model"""
            # Check if session exists and is valid
            if self.session is None:
                # Fall back to pixel-based embedding if model loading failed
                return DummyEmbedder().get_embedding(face_image)
            
            try:    
                # Preprocess image to match ArcFace model requirements
                # Resize to model input size (typically 112x112)
                if face_image is None:
                    return None
                
                if face_image.shape[:2] != self.input_shape[:2]:
                    face_image = cv2.resize(face_image, (self.input_shape[1], self.input_shape[0]))
                
                # Convert to RGB if needed
                if len(face_image.shape) == 2:  # Grayscale
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
                elif face_image.shape[2] == 4:  # RGBA
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
                elif face_image.shape[2] == 3:  # BGR (OpenCV default)
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values for ArcFace
                # Convert to float32 and normalize to [0, 1]
                face_image = face_image.astype(np.float32) / 255.0
                
                # Apply specific normalization for ArcFace model
                # Subtract mean and divide by std
                mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                face_image = (face_image - mean) / std
                
                # Convert to NCHW format (batch, channels, height, width)
                input_tensor = np.transpose(face_image, (2, 0, 1))[np.newaxis, ...]
                
                # Run inference
                outputs = self.session.run(None, {self.input_name: input_tensor})
                embedding = outputs[0][0]  # Extract the embedding from the output
                
                # Normalize embedding (L2 norm)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    
                return embedding
            except Exception as e:
                print(f"Error running inference: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to pixel-based embedding
                return DummyEmbedder().get_embedding(face_image)

# Create global instance
try:
    embedder = ONNXFaceEmbedder()
    
    # Function to get embedding from face image
    def get_face_embedding(face_img):
        """Get face embedding using ONNX model"""
        return embedder.get_embedding(face_img)
except Exception as e:
    print(f"Error initializing ONNX embedder: {e}")
    embedder = DummyEmbedder()
    get_face_embedding = embedder.get_embedding
