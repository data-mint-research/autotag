# config_loader.py - Refactored to use YAML configuration
import os
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Logger setup
logger = logging.getLogger("auto-tag")

# Default configuration structure matching settings.yaml
DEFAULT_CONFIG = {
    "paths": {
        "input_folder": "./data/input",
        "output_folder": "./data/output",
        "models_dir": "./models",
    },
    "tagging": {
        "mode": "append",
        "min_confidence_percent": 80,
        "min_face_size": 40,
        "min_person_height": 40,
    },
    "hardware": {
        "use_gpu": True,
        "cuda_device_id": 0,  # -1 for CPU, 0+ for specific GPU
        "num_workers": 4,
        "batch_size": 0,  # 0 for auto-detection based on GPU memory
        "gpu": {
            "mixed_precision": True,  # Enable mixed precision (FP16/BF16)
            "tensor_cores": True,     # Enable tensor cores for RTX GPUs
            "num_streams": 4,         # Number of CUDA streams for parallel operations
            "memory_fraction": 0.8,   # Fraction of GPU memory to use (0.0-1.0)
            "multi_gpu": False,       # Enable multi-GPU processing
        },
    },
    "models": {
        "auto_download": True,
        "force_update": False,
        "offline_mode": False,
    },
    "processing": {
        "subdirectories": True,
        "max_recursion_depth": 0,
        "checkpointing": {
            "enabled": True,
            "interval_minutes": 5,
            "auto_resume": True,
        },
        "batch": {
            "dynamic_sizing": True,   # Dynamically adjust batch size based on GPU memory
            "progress_tracking": True,
            "eta_estimation": True,
        },
    },
    "minio": {
        "enabled": False,
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "secure": False,
        "input_bucket": "images",
        "output_bucket": "tagged-images",
    },
    "online_identity": {
        "enabled": False,
        "service": "open_face",
        "api_key": "",
    },
    "logging": {
        "level": "INFO",
        "max_size_mb": 10,
        "backup_count": 3,
    }
}

# Legacy key mapping for backward compatibility
LEGACY_KEY_MAPPING = {
    "INPUT_FOLDER": ["paths", "input_folder"],
    "OUTPUT_FOLDER": ["paths", "output_folder"],
    "MODELS_DIR": ["paths", "models_dir"],
    "TAG_MODE": ["tagging", "mode"],
    "MIN_CONFIDENCE_PERCENT": ["tagging", "min_confidence_percent"],
    "MIN_FACE_SIZE": ["tagging", "min_face_size"],
    "MIN_PERSON_HEIGHT": ["tagging", "min_person_height"],
    "USE_GPU": ["hardware", "use_gpu"],
    "CUDA_DEVICE_ID": ["hardware", "cuda_device_id"],
    "NUM_WORKERS": ["hardware", "num_workers"],
    "BATCH_SIZE": ["hardware", "batch_size"],
    "MIXED_PRECISION": ["hardware", "gpu", "mixed_precision"],
    "TENSOR_CORES": ["hardware", "gpu", "tensor_cores"],
    "NUM_CUDA_STREAMS": ["hardware", "gpu", "num_streams"],
    "GPU_MEMORY_FRACTION": ["hardware", "gpu", "memory_fraction"],
    "MULTI_GPU": ["hardware", "gpu", "multi_gpu"],
    "AUTO_DOWNLOAD_MODELS": ["models", "auto_download"],
    "FORCE_MODEL_UPDATE": ["models", "force_update"],
    "OFFLINE_MODE": ["models", "offline_mode"],
    "PROCESS_SUBDIRECTORIES": ["processing", "subdirectories"],
    "MAX_RECURSION_DEPTH": ["processing", "max_recursion_depth"],
    "CHECKPOINTING_ENABLED": ["processing", "checkpointing", "enabled"],
    "CHECKPOINT_INTERVAL": ["processing", "checkpointing", "interval_minutes"],
    "AUTO_RESUME": ["processing", "checkpointing", "auto_resume"],
    "DYNAMIC_BATCH_SIZING": ["processing", "batch", "dynamic_sizing"],
    "PROGRESS_TRACKING": ["processing", "batch", "progress_tracking"],
    "ETA_ESTIMATION": ["processing", "batch", "eta_estimation"],
    "MINIO_ENDPOINT": ["minio", "endpoint"],
    "MINIO_ACCESS_KEY": ["minio", "access_key"],
    "MINIO_SECRET_KEY": ["minio", "secret_key"],
    "MINIO_SECURE": ["minio", "secure"],
    "MINIO_INPUT_BUCKET": ["minio", "input_bucket"],
    "MINIO_OUTPUT_BUCKET": ["minio", "output_bucket"],
    "ONLINE_IDENTITY_ENABLED": ["online_identity", "enabled"],
    "ONLINE_IDENTITY_SERVICE": ["online_identity", "service"],
    "ONLINE_IDENTITY_API_KEY": ["online_identity", "api_key"],
    "LOG_LEVEL": ["logging", "level"],
    "MAX_LOG_SIZE_MB": ["logging", "max_size_mb"],
    "LOG_BACKUP_COUNT": ["logging", "backup_count"],
}

class Config:
    """Load and manage configuration from YAML file"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with optional YAML file path"""
        self.config = DEFAULT_CONFIG.copy()
        self.legacy_config = {}  # For backward compatibility
        
        # Set base directory
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # If no config file specified, look for one
        if config_file is None:
            config_file = os.path.join(self.base_dir, "config", "settings.yaml")
            
            # If settings.yaml doesn't exist, look for template
            if not os.path.exists(config_file):
                template_file = os.path.join(self.base_dir, "config", "settings.yaml.template")
                if os.path.exists(template_file):
                    logger.warning(f"settings.yaml not found, copying template")
                    self._copy_template(template_file, config_file)
        
        # Load configuration from YAML file
        if os.path.exists(config_file):
            self._load_from_yaml(config_file)
            logger.info(f"Configuration loaded from {config_file}")
        else:
            logger.warning(f"No configuration file found at {config_file}, using defaults")
        
        # Convert paths to absolute paths
        self._convert_paths()
        
        # Ensure required directories exist
        self._ensure_directories()
        
        # Build legacy config for backward compatibility
        self._build_legacy_config()
    
    def _copy_template(self, template_file: str, target_file: str) -> None:
        """Copy template configuration file to target location"""
        try:
            with open(template_file, 'r') as source:
                content = source.read()
            
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            with open(target_file, 'w') as target:
                target.write(content)
                
            logger.info(f"Template configuration copied to {target_file}")
        except Exception as e:
            logger.error(f"Error copying template configuration: {e}")
    
    def _load_from_yaml(self, config_file: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            if yaml_config:
                # Update configuration with values from YAML
                self._update_nested_dict(self.config, yaml_config)
        except Exception as e:
            logger.error(f"Error loading YAML configuration: {e}")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary with another dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _convert_paths(self) -> None:
        """Convert relative paths to absolute paths"""
        paths = self.config["paths"]
        
        for key, path_value in paths.items():
            # If it's a relative path, make it absolute
            if not os.path.isabs(path_value):
                paths[key] = os.path.abspath(os.path.join(self.base_dir, path_value))
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist"""
        paths = self.config["paths"]
        
        for key, directory in paths.items():
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Directory exists or was created: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
    
    def _build_legacy_config(self) -> None:
        """Build legacy configuration dictionary for backward compatibility"""
        for legacy_key, path in LEGACY_KEY_MAPPING.items():
            value = self.config
            try:
                for key in path:
                    value = value[key]
                self.legacy_config[legacy_key] = value
            except (KeyError, TypeError):
                # Use default if path doesn't exist in config
                temp = DEFAULT_CONFIG
                for key in path:
                    temp = temp[key]
                self.legacy_config[legacy_key] = temp
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        # First try legacy key for backward compatibility
        if key in self.legacy_config:
            return self.legacy_config.get(key, default)
        
        # If not a legacy key, try to find in nested structure
        if "." in key:
            # Handle nested keys like "paths.input_folder"
            parts = key.split(".")
            value = self.config
            try:
                for part in parts:
                    value = value[part]
                return value
            except (KeyError, TypeError):
                return default
        
        # Try top-level key
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-like access: config['key']"""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Enable 'in' operator: 'key' in config"""
        if key in self.legacy_config:
            return True
        
        if "." in key:
            parts = key.split(".")
            value = self.config
            try:
                for part in parts:
                    value = value[part]
                return True
            except (KeyError, TypeError):
                return False
        
        return key in self.config
    
    def to_dict(self) -> Dict:
        """Return complete configuration as dictionary"""
        return self.config.copy()
    
    def to_legacy_dict(self) -> Dict:
        """Return legacy configuration as dictionary"""
        return self.legacy_config.copy()

# Global configuration instance for easy import
config = Config()

# For direct import of values
def get_config():
    """Get configuration singleton"""
    return config

# Access methods for commonly used configuration values
def get_input_folder():
    return config["paths.input_folder"]

def get_output_folder():
    return config["paths.output_folder"]

def get_models_dir():
    return config["paths.models_dir"]

def get_tag_mode():
    return config["tagging.mode"]

def use_gpu():
    return config["hardware.use_gpu"]

def get_gpu_device_id():
    return config["hardware.cuda_device_id"]

def use_mixed_precision():
    return config["hardware.gpu.mixed_precision"]

def use_tensor_cores():
    return config["hardware.gpu.tensor_cores"]

def get_cuda_streams():
    return config["hardware.gpu.num_streams"]

def get_gpu_memory_fraction():
    return config["hardware.gpu.memory_fraction"]

def use_multi_gpu():
    return config["hardware.gpu.multi_gpu"]

def should_process_subdirectories():
    return config["processing.subdirectories"]

def get_max_recursion_depth():
    return config["processing.max_recursion_depth"]

def is_checkpointing_enabled():
    return config["processing.checkpointing.enabled"]

def get_checkpoint_interval():
    return config["processing.checkpointing.interval_minutes"]

def should_auto_resume():
    return config["processing.checkpointing.auto_resume"]

def use_dynamic_batch_sizing():
    return config["processing.batch.dynamic_sizing"]

def is_online_identity_enabled():
    return config["online_identity.enabled"]

# When script is run directly, display loaded configuration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = Config()
    
    # Display loaded configuration
    print("\n=== AUTO-TAG Configuration ===\n")
    
    # Display in YAML format
    print("# Structured Configuration:")
    print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))
    
    # Display legacy keys for backward compatibility
    print("\n# Legacy Keys (for backward compatibility):")
    for key, value in sorted(config.to_legacy_dict().items()):
        if key.endswith('KEY') or key.endswith('SECRET'):
            # Hide keys and secrets
            print(f"{key}: {'*' * len(str(value))}")
        else:
            print(f"{key}: {value}")