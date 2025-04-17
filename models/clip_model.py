# models/clip_model.py - Scene & clothing classification with optimized CLIP
import os
import torch
import logging
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from models.base_model import BaseModel, ModelError, ModelNotFoundError, ModelInitializationError, ModelInferenceError

# Logger setup
logger = logging.getLogger("auto-tag")

# Predefined categories for classification
SCENE_CATEGORIES = ["indoor", "outdoor"]
ROOMTYPES = ["kitchen", "bathroom", "bedroom", "living room", "office"]
CLOTHING = ["dressed", "naked"]

class CLIPModel(BaseModel):
    """CLIP model for scene and clothing classification"""
    
    def __init__(self):
        """Initialize CLIP model"""
        super().__init__("clip")
        self.preprocess = None
        self.tokenizer = None
        self.model_file = os.path.join(self.model_dir, "clip_vit_b32.pth")
        self.model_architecture = "ViT-B-32"
    
    def initialize(self) -> bool:
        """Initialize the CLIP model
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.initialized:
            return True
        
        try:
            # Check if model file exists
            if not self.verify_model_file(self.model_file):
                raise ModelNotFoundError(f"CLIP model not found: {self.model_file}")
            
            # Import CLIP dynamically (reduces dependencies)
            try:
                import open_clip
            except ImportError:
                raise ModelInitializationError("open_clip_torch is not installed")
            
            # Load model and transformer
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_architecture,
                pretrained=self.model_file
            )
            self.model = self.model.to(self.device).eval()
            
            # Load tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_architecture)
            
            self.initialized = True
            logger.info(f"CLIP model successfully initialized on {self.device}")
            return True
            
        except ModelNotFoundError as e:
            logger.error(str(e))
            return False
        except ModelInitializationError as e:
            logger.error(f"CLIP initialization error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing CLIP model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the CLIP model
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        return self.initialize()
    
    def load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and prepare image for CLIP
        
        Args:
            image_path: Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor, or None if loading failed
        """
        if not self.initialized:
            raise ModelInitializationError("CLIP model not initialized")
        
        try:
            image = Image.open(image_path).convert("RGB")
            return self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def classify(self, image_tensor: torch.Tensor, label_list: List[str], topk: int = 1) -> List[Tuple[str, float]]:
        """Classify image against text prompts
        
        Args:
            image_tensor: Preprocessed image tensor
            label_list: List of text labels to classify against
            topk: Number of top results to return
            
        Returns:
            List of (label, probability) tuples, sorted by probability
        """
        if not self.initialized:
            raise ModelInitializationError("CLIP model not initialized")
        
        if image_tensor is None:
            raise ModelInferenceError("Invalid image tensor")
        
        try:
            with torch.no_grad():
                # Create text prompts
                text_inputs = self.tokenizer([f"a photo of {label}" for label in label_list]).to(self.device)
                
                # Encode image and text
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = similarity.squeeze().cpu().numpy()
                
                # Sort by probability
                results = list(zip(label_list, probs))
                sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                
                return sorted_results[:topk]
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise ModelInferenceError(f"CLIP inference failed: {e}")
    
    def analyze_scene_clothing(self, image_path: str) -> Dict[str, Tuple[str, float]]:
        """Analyze scene and clothing with CLIP
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with scene, roomtype, and clothing classifications
        """
        # Initialize model if needed
        if not self.initialized:
            if not self.initialize():
                return {
                    "scene": ("unknown", 0.0),
                    "roomtype": ("unknown", 0.0),
                    "clothing": ("unknown", 0.0)
                }
        
        try:
            # Load and prepare image
            image_tensor = self.load_image(image_path)
            if image_tensor is None:
                return {
                    "scene": ("unknown", 0.0),
                    "roomtype": ("unknown", 0.0),
                    "clothing": ("unknown", 0.0)
                }
            
            # Classify scene, room type, and clothing
            scene = self.classify(image_tensor, SCENE_CATEGORIES)[0]
            room = self.classify(image_tensor, ROOMTYPES)[0]
            clothing = self.classify(image_tensor, CLOTHING)[0]
            
            return {
                "scene": scene,
                "roomtype": room,
                "clothing": clothing
            }
        except ModelError as e:
            logger.error(f"Model error during scene/clothing analysis: {e}")
            return {
                "scene": ("unknown", 0.0),
                "roomtype": ("unknown", 0.0),
                "clothing": ("unknown", 0.0)
            }
        except Exception as e:
            logger.error(f"Unexpected error during scene/clothing analysis: {e}")
            return {
                "scene": ("unknown", 0.0),
                "roomtype": ("unknown", 0.0),
                "clothing": ("unknown", 0.0)
            }

# Singleton instance
_clip_model = None

def get_clip_model() -> CLIPModel:
    """Get or initialize CLIP model singleton
    
    Returns:
        CLIPModel: Initialized CLIP model instance
    """
    global _clip_model
    
    if _clip_model is None:
        _clip_model = CLIPModel()
        _clip_model.initialize()
    
    return _clip_model

def analyze_scene_clothing(image_path: str) -> Dict[str, Tuple[str, float]]:
    """Analyze scene and clothing with CLIP (convenience function)
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with scene, roomtype, and clothing classifications
    """
    model = get_clip_model()
    return model.analyze_scene_clothing(image_path)