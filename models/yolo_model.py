# models/yolo_model.py - Person detection and counting with YOLOv8
import os
import torch
import logging
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from models.base_model import BaseModel, ModelError, ModelNotFoundError, ModelInitializationError, ModelInferenceError

# Logger setup
logger = logging.getLogger("auto-tag")

# Minimum size of a "usable" person in pixels
MIN_PERSON_HEIGHT = 40

class YOLOModel(BaseModel):
    """YOLOv8 model for person detection and counting"""
    
    def __init__(self):
        """Initialize YOLOv8 model"""
        super().__init__("yolov8")
        self.model_file = os.path.join(self.model_dir, "yolov8n.pt")
        self.min_person_height = self.config.get("tagging.min_person_height", MIN_PERSON_HEIGHT)
    
    def initialize(self) -> bool:
        """Initialize the YOLOv8 model
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.initialized:
            return True
        
        try:
            # Check if model file exists
            if not self.verify_model_file(self.model_file):
                raise ModelNotFoundError(f"YOLOv8 model not found: {self.model_file}")
            
            # Import YOLO dynamically (reduces dependencies)
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ModelInitializationError("ultralytics is not installed")
            
            # Load model
            self.model = YOLO(self.model_file)
            
            self.initialized = True
            logger.info(f"YOLOv8 model successfully initialized")
            return True
            
        except ModelNotFoundError as e:
            logger.error(str(e))
            return False
        except ModelInitializationError as e:
            logger.error(f"YOLOv8 initialization error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing YOLOv8 model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the YOLOv8 model
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        return self.initialize()
    
    def count_people(self, image_path: str, min_person_height: Optional[int] = None) -> str:
        """Count people in the image
        
        Args:
            image_path: Path to the image file
            min_person_height: Minimum height of a person to count (in pixels)
            
        Returns:
            str: "none", "solo", or "group" based on the number of people detected
        """
        if not self.initialized:
            if not self.initialize():
                return "none"
        
        if min_person_height is None:
            min_person_height = self.min_person_height
        
        try:
            # Run inference
            results = self.model(image_path)
            
            # Count people (class 0 in COCO)
            count = 0
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    cls = int(box.cls[0].item())
                    if cls == 0:  # Person
                        height = int(box.xywh[0][3].item())  # Box height
                        if height >= min_person_height:
                            count += 1
            
            # Categorize the result
            if count == 0:
                return "none"
            elif count == 1:
                return "solo"
            else:
                return "group"
                
        except Exception as e:
            logger.error(f"Error counting people in {image_path}: {e}")
            return "none"
    
    def detect_objects(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Detect objects in the image
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of dictionaries with detection information
        """
        if not self.initialized:
            if not self.initialize():
                return []
        
        try:
            # Run inference
            results = self.model(image_path, conf=conf_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Get class name if available
                    cls_name = result.names[cls_id] if hasattr(result, 'names') else f"class_{cls_id}"
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
            
            return detections
                
        except Exception as e:
            logger.error(f"Error detecting objects in {image_path}: {e}")
            return []

# Singleton instance
_yolo_model = None

def get_yolo_model() -> YOLOModel:
    """Get or initialize YOLOv8 model singleton
    
    Returns:
        YOLOModel: Initialized YOLOv8 model instance
    """
    global _yolo_model
    
    if _yolo_model is None:
        _yolo_model = YOLOModel()
        _yolo_model.initialize()
    
    return _yolo_model

def count_people(image_path: str, min_person_height: Optional[int] = None) -> str:
    """Count people in the image (convenience function)
    
    Args:
        image_path: Path to the image file
        min_person_height: Minimum height of a person to count (in pixels)
        
    Returns:
        str: "none", "solo", or "group" based on the number of people detected
    """
    model = get_yolo_model()
    return model.count_people(image_path, min_person_height)

def detect_objects(image_path: str, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """Detect objects in the image (convenience function)
    
    Args:
        image_path: Path to the image file
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of dictionaries with detection information
    """
    model = get_yolo_model()
    return model.detect_objects(image_path, conf_threshold)