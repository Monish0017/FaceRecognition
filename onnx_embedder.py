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
requests_available = importlib.util.find_spec("requests") is not None

if not onnxruntime_available:
    print("ERROR: onnxruntime is not installed. Please install it using:")
    print("    pip install onnxruntime")
    print("Alternatively, install all dependencies using:")
    print("    pip install -r requirements.txt")

if not requests_available:
    print("WARNING: requests module not found, will skip automatic downloads.")
    print("Please install manually using: pip install requests")

# Model URLs - updated to reliable sources
FACENET_MODEL_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
# Alternative models if the above doesn't work
ALT_FACENET_URL = "https://github.com/timesler/facenet-pytorch/raw/master/models/20180402-114759-vggface2.onnx"
# Update to a working MobileFaceNet URL (using a reliable source)
MOBILEFACENET_MODEL_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"

# Directly from Microsoft ONNX Model Zoo
ARCFACE_MODEL_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"

# Default model choice - use ArcFace as it's reliable and maintained
DEFAULT_MODEL_URL = ARCFACE_MODEL_URL  
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
        def __init__(self, model_path=None):
            """Initialize the ONNX face embedder"""
            # Initialize session to None first to avoid attribute errors
            self.session = None
            self.input_shape = (112, 112, 3)
            self.input_name = "input"  # Default input name
            
            if model_path is None:
                model_path = DEFAULT_MODEL_PATH
            
            # If model doesn't exist, try to download it or use a local file
            if not os.path.exists(model_path):
                # Try to find any .onnx file in models directory as a fallback
                onnx_files = list(MODEL_DIR.glob("*.onnx"))
                if onnx_files:
                    print(f"Using existing model: {onnx_files[0]}")
                    model_path = onnx_files[0]
                elif requests_available:
                    try:
                        self.download_model(DEFAULT_MODEL_URL, model_path)
                    except Exception as e:
                        print(f"ERROR downloading model: {e}")
                        print(f"You need to manually download a face recognition ONNX model.")
                        print(f"1. Try this URL: {ARCFACE_MODEL_URL}")
                        print(f"2. Or this URL: {ALT_FACENET_URL}")
                        print(f"3. Save the downloaded file to: {model_path}")
                        print(f"4. Or place any face embedding ONNX model in: {MODEL_DIR}")
                        
                        # Try to find any ONNX file as last resort
                        onnx_files = list(Path(".").glob("*.onnx"))
                        if onnx_files:
                            print(f"Found local ONNX file: {onnx_files[0]}")
                            model_path = onnx_files[0]
                        else:
                            print("No ONNX models found. Using fallback embedder.")
                            # We're already initialized with defaults, just return
                            return
                else:
                    print("You need to manually download the model since 'requests' is not available.")
                    print(f"1. Try this URL: {ARCFACE_MODEL_URL}")
                    print(f"2. Or this URL: {ALT_FACENET_URL}")
                    print(f"3. Save the downloaded file to: {model_path}")
                    # Using fallback embedder, already initialized with defaults
                    return
            
            # Only try to load the model if the file exists
            if os.path.exists(model_path):
                try:
                    # Create ONNX inference session
                    print(f"Loading ONNX model from: {model_path}")
                    self.session = onnxruntime.InferenceSession(str(model_path), 
                                                        providers=['CPUExecutionProvider'])
                    
                    # Get model metadata
                    model_inputs = self.session.get_inputs()
                    if model_inputs and len(model_inputs) > 0:
                        self.input_name = model_inputs[0].name
                        
                        # Get input shape (ignoring the batch dimension)
                        if len(model_inputs[0].shape) >= 3:
                            self.input_shape = model_inputs[0].shape[2:]
                            if len(self.input_shape) == 2:
                                self.input_shape = (self.input_shape[0], self.input_shape[1], 3)
                            
                    print(f"ONNX Face Embedder loaded, expected input shape: {self.input_shape}")
                except Exception as e:
                    print(f"Error loading ONNX model: {e}")
                    print("Using fallback embedder with reduced accuracy.")
                    # self.session already initialized as None
            else:
                print(f"Model file not found at: {model_path}")
                print("Using fallback embedder with reduced accuracy.")
        
        def download_model(self, url, output_path):
            """Download model from URL"""
            import requests
            print(f"Downloading face embedding model from {url}...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download the model
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save the model to disk
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Model downloaded to {output_path}")
        
        def get_embedding(self, face_image):
            """Get face embedding from the ONNX model"""
            # Check if session exists and is valid
            if self.session is None:
                # Fall back to pixel-based embedding if model loading failed
                return DummyEmbedder().get_embedding(face_image)
            
            try:    
                # Preprocess image to match the model input requirements
                # Resize
                if face_image.shape[:2] != self.input_shape[:2]:
                    face_image = cv2.resize(face_image, (self.input_shape[1], self.input_shape[0]))
                
                # Ensure RGB format
                if len(face_image.shape) == 2:
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
                elif face_image.shape[2] == 4:
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
                elif face_image.shape[2] == 3:
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values to [0, 1]
                face_image = face_image.astype(np.float32) / 255.0
                
                # Prepare input tensor (NCHW format for most ONNX models)
                input_tensor = np.transpose(face_image, (2, 0, 1))[None, :, :, :]
                
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
