# models/base_model.py - Base class for all AI models
import os
import logging
import torch
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_config

# Logger setup
logger = logging.getLogger("auto-tag")

class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class ModelNotFoundError(ModelError):
    """Exception raised when a model file is not found"""
    pass

class ModelInitializationError(ModelError):
    """Exception raised when a model fails to initialize"""
    pass

class ModelInferenceError(ModelError):
    """Exception raised when model inference fails"""
    pass

class BaseModel(ABC):
    """Base class for all AI models in AUTO-TAG"""
    
    def __init__(self, model_name: str):
        """Initialize the base model
        
        Args:
            model_name: Name of the model (used for file paths and logging)
        """
        self.model_name = model_name
        self.model = None
        self.initialized = False
        self.config = get_config()
        
        # Determine device (CPU/GPU)
        self.device = "cuda" if torch.cuda.is_available() and self.config["hardware.use_gpu"] else "cpu"
        if self.device == "cuda":
            cuda_device_id = self.config["hardware.cuda_device_id"]
            torch.cuda.set_device(cuda_device_id)
        
        # Set model directory
        self.models_dir = self.config["paths.models_dir"]
        self.model_dir = os.path.join(self.models_dir, self.model_name)
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized"""
        return self.initialized
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model from file
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        pass
    
    def verify_model_file(self, file_path: str, expected_hash: Optional[str] = None) -> bool:
        """Verify that a model file exists and optionally check its hash
        
        Args:
            file_path: Path to the model file
            expected_hash: Expected SHA-256 hash of the file (optional)
            
        Returns:
            bool: True if the file exists and hash matches (if provided)
        """
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            return False
        
        if expected_hash:
            file_hash = self._calculate_file_hash(file_path)
            if file_hash != expected_hash:
                logger.error(f"Hash verification failed for {file_path}")
                logger.error(f"Expected: {expected_hash}")
                logger.error(f"Got: {file_hash}")
                return False
            logger.debug(f"Hash verification successful for {file_path}")
        
        return True
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: SHA-256 hash as hexadecimal string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def cleanup(self) -> None:
        """Clean up resources used by the model"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self) -> None:
        """Destructor to ensure resources are cleaned up"""
        self.cleanup()