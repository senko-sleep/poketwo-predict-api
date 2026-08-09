"""
Simple Pokemon Image Recognition Module
Clean, straightforward image recognition for Pokemon.
Supports both ONNX model prediction and embedding-based event Pokemon detection.
"""
import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Tuple, Optional

class PokemonRecognizer:
    """Clean image recognition for Pokemon with event Pokemon support."""
    
    def __init__(self, model_path: str = None, labels_path: str = None, 
                 embedding_index_path: str = None, embedding_meta_path: str = None,
                 event_label_config_path: str = None, event_labels_path: str = None):
        """Initialize the recognizer with model, labels, and embedding index."""
        self.model_path = model_path or "models/pokemon_cnn_v2.onnx"
        self.labels_path = labels_path or "models/labels_v2.json"
        self.embedding_index_path = embedding_index_path or "models/event_embedding_index.npz"
        self.embedding_meta_path = embedding_meta_path or "models/event_embedding_meta.json"
        self.event_label_config_path = event_label_config_path or "models/event_label_config.json"
        self.event_labels_path = event_labels_path or "models/event_labels.json"
        
        self.session = None
        self.labels = []
        self.input_size = 224
        
        # Embedding index for event Pokemon
        self.embeddings = {}
        self.embedding_labels = []
        self.embedding_meta = {}
        self.event_label_config = {}
        self.event_labels = {}
        
        self._load_model()
        self._load_labels()
        self._load_embeddings()
        self._load_event_label_config()
        self._load_event_labels()

        # Sanity-check that the model's output size matches the number
        # of labels we loaded. A mismatch here is the #1 cause of every
        # prediction coming back as "unknown".
        self._validate_model_label_alignment()

    def _validate_model_label_alignment(self):
        """Warn loudly if the ONNX model's output class count doesn't match labels."""
        try:
            output_shape = self.session.get_outputs()[0].shape
            # Last dim is usually the class count; may be dynamic (None/str) in some exports
            model_classes = output_shape[-1] if isinstance(output_shape[-1], int) else None
            if model_classes is not None and model_classes != len(self.labels):
                print(
                    f"WARNING: model '{self.model_path}' outputs {model_classes} classes, "
                    f"but {len(self.labels)} labels were loaded from '{self.labels_path}'. "
                    f"This mismatch will cause predictions to fall back to 'unknown' whenever "
                    f"the top class index is >= {len(self.labels)}. "
                    f"Make sure labels_v2.json was exported alongside this exact model version."
                )
            else:
                print(f"Model/label alignment OK: {model_classes if model_classes is not None else '?'} classes.")
        except Exception as e:
            print(f"Could not validate model/label alignment: {e}")
    
    def _load_model(self):
        """Load the ONNX model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.session = ort.InferenceSession(
            self.model_path, 
            providers=['CPUExecutionProvider']
        )
        print(f"Model loaded: {self.model_path}")
    
    def _load_labels(self):
        """Load labels from JSON file."""
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")
        
        with open(self.labels_path, 'r') as f:
            labels_data = json.load(f)
        
        # Ensure labels are in list format
        if isinstance(labels_data, dict):
            # Guard against non-contiguous / missing keys instead of raising
            # a raw KeyError deep inside a dict comprehension.
            missing = [str(i) for i in range(len(labels_data)) if str(i) not in labels_data]
            if missing:
                print(
                    f"WARNING: labels_v2.json is a dict but is missing contiguous keys "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                    f"Labels will be loaded out of order or incompletely, which can make "
                    f"predictions map to the wrong Pokemon or fall back to 'unknown'."
                )
            self.labels = [labels_data.get(str(i), "unknown") for i in range(len(labels_data))]
        else:
            self.labels = labels_data
        
        print(f"Loaded {len(self.labels)} labels from file")
        
        # Truncate labels to match model output size if model is already loaded
        if self.session is not None:
            output_shape = self.session.get_outputs()[0].shape
            model_classes = output_shape[-1] if isinstance(output_shape[-1], int) else None
            if model_classes is not None and len(self.labels) > model_classes:
                print(f"WARNING: Truncating labels from {len(self.labels)} to {model_classes} to match model output")
                self.labels = self.labels[:model_classes]
                print(f"Loaded {len(self.labels)} labels (truncated to match model)")
    
    def _load_embeddings(self):
        """Load embedding index for event Pokemon."""
        if not os.path.exists(self.embedding_index_path):
            print("No embedding index found - event Pokemon disabled")
            return
        
        try:
            data = np.load(self.embedding_index_path)
            
            # Check if it's the new matrix format (embeddings + labels)
            if 'embeddings' in data.files:
                self.embeddings = data['embeddings']  # Shape: (n_embeddings, embedding_dim)
                if 'labels' in data.files:
                    labels_arr = data['labels']
                    # Convert numpy string arrays to Python strings
                    if labels_arr.dtype.kind in ['U', 'S', 'O']:
                        self.embedding_labels = [str(label) if label is not None else None for label in labels_arr]
                    else:
                        self.embedding_labels = labels_arr.tolist()
                print(f"Loaded {len(self.embedding_labels)} embeddings (matrix format)")
            else:
                # Old format: individual arrays per embedding
                self.embeddings = {k: data[k] for k in data.files if k != 'labels'}
                if 'labels' in data.files:
                    self.embedding_labels = data['labels'].tolist()
                print(f"Loaded {len(self.embeddings)} embeddings (legacy format)")
            
            # Load metadata
            if os.path.exists(self.embedding_meta_path):
                with open(self.embedding_meta_path, 'r') as f:
                    self.embedding_meta = json.load(f)
                print(f"Embedding metadata: {self.embedding_meta}")
        except Exception as e:
            print(f"Warning: Could not load embeddings: {e}")
    
    def _load_event_label_config(self):
        """Load event label configuration for per-label thresholds."""
        if not os.path.exists(self.event_label_config_path):
            print("No event label config found - using default thresholds")
            return
        
        try:
            with open(self.event_label_config_path, 'r') as f:
                self.event_label_config = json.load(f)
            print(f"Loaded event label config for {len(self.event_label_config)} labels")
        except Exception as e:
            print(f"Warning: Could not load event label config: {e}")
    
    def _load_event_labels(self):
        """Load event labels metadata."""
        if not os.path.exists(self.event_labels_path):
            print("No event labels metadata found")
            return
        
        try:
            with open(self.event_labels_path, 'r') as f:
                self.event_labels = json.load(f)
            print(f"Loaded event labels metadata for {len(self.event_labels)} labels")
        except Exception as e:
            print(f"Warning: Could not load event labels: {e}")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for model input."""
        import io
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to input size
        image = image.resize((self.input_size, self.input_size), Image.LANCZOS)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array.astype(np.float32)
        
        # Add batch dimension
        return img_array[np.newaxis, ...]
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract embedding from image using the model."""
        if self.session is None:
            return None
        
        try:
            # Preprocess
            input_data = self.preprocess_image(image_bytes)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})
            
            # Use the logits as embedding
            embedding = outputs[0][0]
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def _embeddings_empty(self) -> bool:
        """Numpy-safe check for 'no embeddings loaded', works for both
        the matrix format (np.ndarray) and the legacy dict format."""
        if self.embeddings is None:
            return True
        if isinstance(self.embeddings, np.ndarray):
            return self.embeddings.size == 0
        # dict / legacy format
        return len(self.embeddings) == 0
    
    def predict_with_embeddings(self, image_bytes: bytes, onnx_confidence: float = 0.0) -> Optional[Tuple[str, float]]:
        """Predict using embedding index for event Pokemon with per-label thresholds."""
        # NOTE: never use `if not self.embeddings:` here - when embeddings is a
        # numpy array with more than one element, numpy raises:
        #   ValueError: The truth value of an array with more than one element
        #   is ambiguous. Use a.any() or a.all()
        # Use the dedicated helper instead, which handles both the matrix
        # (np.ndarray) format and the legacy dict format safely.
        if self._embeddings_empty():
            return None
        
        # Extract embedding from input image
        try:
            query_embedding = self.extract_embedding(image_bytes)
            if query_embedding is None:
                return None
        except Exception as e:
            print(f"DEBUG: Error in extract_embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Check if embeddings is a matrix (new format) or dict (old format)
        if isinstance(self.embeddings, np.ndarray):
            # Matrix format: (n_embeddings, embedding_dim)
            embeddings_matrix = self.embeddings
            labels = self.embedding_labels
            
            # Normalize all embeddings
            emb_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            emb_norms[emb_norms == 0] = 1  # Avoid division by zero
            normalized_embeddings = embeddings_matrix / emb_norms
            
            # Calculate cosine similarity with all embeddings
            similarities = np.dot(normalized_embeddings, query_embedding)
            
            # Find best match
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])
            best_label_val = labels[best_idx]
            # Ensure we get a proper Python string, not numpy string
            best_label = str(best_label_val) if best_label_val is not None else None
            if isinstance(best_label, np.ndarray):
                best_label = str(best_label.item()) if best_label.size == 1 else None
            
            # Group similarities by label
            label_similarities = {}
            for label, sim in zip(labels, similarities):
                label_val = label
                label_str = str(label_val) if label_val is not None else None
                if isinstance(label_str, np.ndarray):
                    label_str = str(label_str.item()) if label_str.size == 1 else None
                sim_float = float(sim)
                if label_str is not None and (label_str not in label_similarities or sim_float > label_similarities[label_str]):
                    label_similarities[label_str] = sim_float
        else:
            # Legacy format: dict of individual embeddings
            best_label = None
            best_similarity = 0.0
            label_similarities = {}
            
            for key, embedding in self.embeddings.items():
                # Normalize stored embedding
                emb_norm = np.linalg.norm(embedding)
                if emb_norm > 0:
                    embedding = embedding / emb_norm
                
                similarity = self.cosine_similarity(query_embedding, embedding)
                
                # Extract label from key (format: "label_0", "label_1", etc.)
                label = key.rsplit('_', 1)[0]
                
                # Track best similarity per label
                if label not in label_similarities or similarity > label_similarities[label]:
                    label_similarities[label] = similarity
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_label = label
        
        # Ensure best_label is a proper Python string
        best_label_str = str(best_label) if best_label is not None else None
        if best_label_str is not None and isinstance(best_label_str, str) and len(best_label_str) > 0:
            try:
                if best_label_str in self.event_label_config:
                    config = self.event_label_config[best_label_str]
                    onnx_ceiling = config.get('onnx_ceiling', 0.95)
                    min_sim = config.get('min_sim', 0.93)
                    min_margin = config.get('min_margin', 0.005)
                    
                    print(f"DEBUG: Event check - label={best_label_str}, onnx_conf={onnx_confidence:.4f}, ceiling={onnx_ceiling:.4f}, sim={best_similarity:.4f}")
                    
                    # Check if ONNX confidence is below ceiling (allow event override)
                    if onnx_confidence >= onnx_ceiling:
                        print(f"DEBUG: ONNX too confident ({onnx_confidence:.4f} >= {onnx_ceiling:.4f}), skipping event override")
                        return None  # ONNX is too confident, don't override
                    
                    # Check if similarity meets minimum threshold
                    if best_similarity < min_sim:
                        print(f"DEBUG: Similarity too low ({best_similarity:.4f} < {min_sim:.4f}), skipping event override")
                        return None  # Similarity too low
                    
                    # Check margin between best and second best
                    sorted_similarities = sorted(label_similarities.values(), reverse=True)
                    if len(sorted_similarities) > 1:
                        margin = sorted_similarities[0] - sorted_similarities[1]
                        if margin < min_margin:
                            return None  # Too close to second best
                    
                    return best_label_str, float(best_similarity)
            except Exception as e:
                print(f"Error checking event label config: {e}")
                pass
        
        # Fallback for labels without config (use default thresholds)
        if best_label_str is not None and isinstance(best_label_str, str) and len(best_label_str) > 0 and best_similarity > 0.85:
            sorted_similarities = sorted(label_similarities.values(), reverse=True)
            if len(sorted_similarities) > 1:
                margin = (sorted_similarities[0] - sorted_similarities[1]) / sorted_similarities[1]
                if margin < 0.1:
                    return None
            
            return best_label_str, float(best_similarity)
        
        return None
    
    def predict(self, image_bytes: bytes) -> Tuple[str, float]:
        """Predict Pokemon from image bytes.
        
        Priority:
        1. Get ONNX prediction first
        2. Check embedding index for event Pokemon (using ONNX confidence)
        3. Fall back to ONNX model if no event match
        """
        # Get ONNX prediction first
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        input_data = self.preprocess_image(image_bytes)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {input_name: input_data})
        
        # Get prediction
        logits = outputs[0][0]
        exp_predictions = np.exp(logits - np.max(logits))
        probabilities = exp_predictions / np.sum(exp_predictions)
        top_index = np.argmax(probabilities)
        confidence = float(probabilities[top_index])
        
        # Get Pokemon name
        if top_index < len(self.labels):
            pokemon_name = self.labels[top_index]
        else:
            # This branch means the model produced more output classes
            # than we have labels for. Log it clearly instead of silently
            # returning "unknown" so the real cause (label/model mismatch)
            # is visible in your logs.
            print(
                f"WARNING: predicted class index {top_index} is out of range for "
                f"{len(self.labels)} labels (model output has {len(probabilities)} classes). "
                f"Check that '{self.labels_path}' matches '{self.model_path}'."
            )
            pokemon_name = "unknown"
        
        # Try embedding-based prediction for event Pokemon (using ONNX confidence)
        embedding_result = self.predict_with_embeddings(image_bytes, onnx_confidence=confidence)
        if embedding_result:
            print(f"DEBUG: Event override - ONNX predicted {pokemon_name} ({confidence:.2f}), event predicted {embedding_result[0]} ({embedding_result[1]:.2f})")
            return embedding_result
        
        # Fall back to ONNX prediction
        return pokemon_name, confidence