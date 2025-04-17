# models/model_manager.py - Robust model management system
import os
import sys
import json
import hashlib
import logging
import time
import requests
import subprocess
import re
import tempfile
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from urllib.parse import urlparse
from packaging import version

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_config

# Try to import Azure Blob Storage client
try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Logger setup
logger = logging.getLogger("auto-tag")

class ModelCatalog:
    """Model catalog for managing model metadata and download sources"""
    
    def __init__(self, catalog_path: Optional[str] = None):
        """Initialize model catalog
        
        Args:
            catalog_path: Path to the model catalog JSON file (optional)
        """
        self.config = get_config()
        
        # Set default catalog path if not provided
        if catalog_path is None:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
            catalog_path = os.path.join(config_dir, "model_catalog.json")
        
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()
    
    def _load_catalog(self) -> Dict:
        """Load or create the model catalog
        
        Returns:
            Dict: Model catalog data
        """
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model catalog: {e}")
        
        # Create default catalog if not found
        default_catalog = self._create_default_catalog()
        self._save_catalog(default_catalog)
        return default_catalog
    
    def _create_default_catalog(self) -> Dict:
        """Create default model catalog
        
        Returns:
            Dict: Default model catalog data
        """
        return {
            "models": {
                "clip": {
                    "filename": "clip_vit_b32.pth",
                    "size": 354355280,
                    "sha256": "a4ccb0c288dd8c53e8ef99417d08e3731ecf29c9e39297a45f37c56e5366ca6e",
                    "version": "1.0.0",
                    "min_version": "1.0.0",
                    "sources": {
                        "github": "https://github.com/openai/CLIP/releases/download/v1.0/clip_vit_b32.pth",
                        "huggingface": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin",
                        "azure": "https://autotagmodels.blob.core.windows.net/models/clip/clip_vit_b32.pth",
                        "git_lfs": "https://github.com/auto-tag/model-repo.git:models/clip/clip_vit_b32.pth",
                        "local": "./offline_models/clip_vit_b32.pth"
                    },
                    "required": True,
                    "description": "CLIP ViT-B/32 model for scene and clothing classification",
                    "compatibility": {
                        "frameworks": ["pytorch>=1.7.0"],
                        "platforms": ["windows", "linux", "macos"]
                    }
                },
                "yolov8": {
                    "filename": "yolov8n.pt",
                    "size": 6246000,
                    "sha256": "6dbb68b8a5d19992f5a5e3b99d1ba466893dcf618bd5e8c0fe551705eb1f6315",
                    "version": "8.0.0",
                    "min_version": "8.0.0",
                    "sources": {
                        "github": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        "huggingface": "https://huggingface.co/Ultralytics/yolov8/resolve/main/yolov8n.pt",
                        "azure": "https://autotagmodels.blob.core.windows.net/models/yolo/yolov8n.pt",
                        "git_lfs": "https://github.com/auto-tag/model-repo.git:models/yolo/yolov8n.pt",
                        "local": "./offline_models/yolov8n.pt"
                    },
                    "required": True,
                    "description": "YOLOv8 nano model for person detection and counting",
                    "compatibility": {
                        "frameworks": ["pytorch>=1.8.0"],
                        "platforms": ["windows", "linux", "macos"]
                    }
                },
                "facenet": {
                    "filename": "facenet_model.pth",
                    "size": 89456789,
                    "sha256": "5e4c2578ffeff9e1dde7d0d10e025c4319b13e4d058577cf430c8df5cf613c45",
                    "version": "2.5.2",
                    "min_version": "2.5.0",
                    "sources": {
                        "github": "https://github.com/timesler/facenet-pytorch/releases/download/v2.5.2/20180402-114759-vggface2.pt",
                        "huggingface": "https://huggingface.co/timesler/facenet-pytorch/resolve/main/20180402-114759-vggface2.pt",
                        "azure": "https://autotagmodels.blob.core.windows.net/models/facenet/facenet_model.pth",
                        "git_lfs": "https://github.com/auto-tag/model-repo.git:models/facenet/facenet_model.pth",
                        "local": "./offline_models/facenet_model.pth"
                    },
                    "required": True,
                    "description": "FaceNet model for face analysis (age, gender, mood)",
                    "compatibility": {
                        "frameworks": ["pytorch>=1.7.0"],
                        "platforms": ["windows", "linux", "macos"]
                    }
                }
            },
            "offline_package": {
                "url": "https://github.com/your-org/autotag-models/releases/download/v1.0/all_models.zip",
                "size": 450057069,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "version": "1.0.0"
            },
            "version": "2.0.0"
        }
    
    def _save_catalog(self, catalog: Dict) -> None:
        """Save model catalog to file
        
        Args:
            catalog: Model catalog data to save
        """
        try:
            os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
            with open(self.catalog_path, 'w') as f:
                json.dump(catalog, f, indent=2)
            logger.debug(f"Model catalog saved to {self.catalog_path}")
        except Exception as e:
            logger.error(f"Error saving model catalog: {e}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Model information, or None if not found
        """
        return self.catalog.get("models", {}).get(model_name)
    
    def get_all_models(self) -> Dict:
        """Get information for all models
        
        Returns:
            Dict: All model information
        """
        return self.catalog.get("models", {})
    
    def get_required_models(self) -> Dict:
        """Get information for all required models
        
        Returns:
            Dict: Required model information
        """
        return {name: info for name, info in self.catalog.get("models", {}).items() 
                if info.get("required", False)}
    
    def get_catalog_version(self) -> str:
        """Get catalog version
        
        Returns:
            str: Catalog version
        """
        return self.catalog.get("version", "unknown")


class ModelManager:
    """Manager for downloading, verifying, and managing AI models"""
    
    def __init__(self):
        """Initialize model manager"""
        self.config = get_config()
        self.catalog = ModelCatalog()
        self.models_dir = self.config["paths.models_dir"]
        self.offline_mode = self.config["models.offline_mode"]
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check if Git LFS is available
        self.git_lfs_available = self._check_git_lfs()
        
        # Check if Azure Blob Storage client is available
        self.azure_available = AZURE_AVAILABLE
    
    def _check_git_lfs(self) -> bool:
        """Check if Git LFS is installed and available
        
        Returns:
            bool: True if Git LFS is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode == 0 and "git-lfs" in result.stdout:
                logger.debug(f"Git LFS is available: {result.stdout.strip()}")
                return True
            else:
                logger.warning("Git LFS is not available or not properly installed")
                return False
        except Exception as e:
            logger.warning(f"Error checking Git LFS: {e}")
            return False
    
    def verify_hash(self, file_path: str, expected_hash: str) -> bool:
        """Verify SHA-256 hash of a file
        
        Args:
            file_path: Path to the file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            bool: True if hash matches, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        try:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            
            if actual_hash != expected_hash:
                logger.error(f"Hash verification failed for {file_path}")
                logger.error(f"Expected: {expected_hash}")
                logger.error(f"Got: {actual_hash}")
                return False
            
            logger.debug(f"Hash verification successful for {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error verifying hash: {e}")
            return False
    
    def download_with_progress(self, url: str, dest_path: str, expected_size: Optional[int] = None) -> bool:
        """Download file with progress bar
        
        Args:
            url: URL to download from
            dest_path: Destination path
            expected_size: Expected file size in bytes (optional)
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0 and expected_size:
                total_size = expected_size
            
            with open(dest_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc=os.path.basename(dest_path)
            ) as progress:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    
    def download_from_azure(self, blob_url: str, dest_path: str, expected_size: Optional[int] = None) -> bool:
        """Download file from Azure Blob Storage
        
        Args:
            blob_url: Azure Blob Storage URL
            dest_path: Destination path
            expected_size: Expected file size in bytes (optional)
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.azure_available:
            logger.error("Azure Blob Storage client is not available. Install with: pip install azure-storage-blob")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Parse the URL to get the account name, container name, and blob path
            parsed_url = urlparse(blob_url)
            account_name = parsed_url.netloc.split('.')[0]
            path_parts = parsed_url.path.strip('/').split('/')
            container_name = path_parts[0]
            blob_path = '/'.join(path_parts[1:])
            
            # Create a blob service client
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=None  # Using public access
            )
            
            # Get a blob client
            blob_client = blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_path
            )
            
            # Get blob properties to determine size
            properties = blob_client.get_blob_properties()
            total_size = properties.size
            
            # Download with progress bar
            with open(dest_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc=os.path.basename(dest_path)
            ) as progress:
                download_stream = blob_client.download_blob()
                for chunk in download_stream.chunks():
                    f.write(chunk)
                    progress.update(len(chunk))
            
            return True
        except ResourceNotFoundError:
            logger.error(f"Azure blob not found: {blob_url}")
            return False
        except ServiceRequestError as e:
            logger.error(f"Azure service request error: {e}")
            return False
        except Exception as e:
            logger.error(f"Azure download failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    
    def download_from_git_lfs(self, git_lfs_url: str, dest_path: str) -> bool:
        """Download file using Git LFS
        
        Args:
            git_lfs_url: Git LFS URL in format "repo_url:file_path"
            dest_path: Destination path
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.git_lfs_available:
            logger.error("Git LFS is not available. Install with: git lfs install")
            return False
        
        try:
            # Parse the Git LFS URL
            repo_url, file_path = git_lfs_url.split(':', 1)
            
            # Create a temporary directory for the Git clone
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Cloning Git LFS repository: {repo_url}")
                
                # Clone the repository with LFS
                clone_result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, temp_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if clone_result.returncode != 0:
                    logger.error(f"Git clone failed: {clone_result.stderr}")
                    return False
                
                # Pull LFS files
                lfs_result = subprocess.run(
                    ["git", "lfs", "pull", "--include", file_path, "--exclude", ""],
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if lfs_result.returncode != 0:
                    logger.error(f"Git LFS pull failed: {lfs_result.stderr}")
                    return False
                
                # Copy the file to the destination
                source_path = os.path.join(temp_dir, file_path)
                if not os.path.exists(source_path):
                    logger.error(f"File not found in Git LFS repository: {file_path}")
                    return False
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Copy the file
                with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                
                logger.info(f"File successfully downloaded from Git LFS: {os.path.basename(dest_path)}")
                return True
                
        except Exception as e:
            logger.error(f"Git LFS download failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    
    def verify_model(self, file_path: str, expected_hash: Optional[str] = None,
                    expected_version: Optional[str] = None) -> bool:
        """Verify model file hash and version
        
        Args:
            file_path: Path to the model file
            expected_hash: Expected SHA-256 hash
            expected_version: Expected model version
            
        Returns:
            bool: True if verification passes, False otherwise
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            return False
        
        # Verify hash if provided
        if expected_hash and not self.verify_hash(file_path, expected_hash):
            return False
        
        # Verify version if provided
        if expected_version:
            try:
                # For now, we don't have a way to extract version from model files
                # This would need to be implemented based on the specific model format
                # For now, we'll just log that we're skipping version verification
                logger.debug(f"Version verification not implemented for {file_path}")
                # In a real implementation, you would extract the version from the model file
                # and compare it with the expected version
            except Exception as e:
                logger.error(f"Error verifying model version: {e}")
        
        return True
    
    def check_version_compatibility(self, model_version: str, min_required_version: str) -> bool:
        """Check if model version meets minimum required version
        
        Args:
            model_version: Current model version
            min_required_version: Minimum required version
            
        Returns:
            bool: True if version is compatible, False otherwise
        """
        try:
            return version.parse(model_version) >= version.parse(min_required_version)
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return False
    
    def download_with_retries(self, model_name: str, max_retries: int = 3) -> bool:
        """Download model with retries and multiple sources
        
        Args:
            model_name: Name of the model to download
            max_retries: Maximum number of retry attempts per source
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        model_info = self.catalog.get_model_info(model_name)
        if not model_info:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        filename = model_info["filename"]
        expected_hash = model_info.get("sha256")
        expected_size = model_info.get("size")
        expected_version = model_info.get("version")
        sources = model_info.get("sources", {})
        
        if not sources:
            logger.error(f"No download sources for {model_name}")
            return False
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        dest_path = os.path.join(model_dir, filename)
        
        # Check if valid model already exists
        if os.path.exists(dest_path) and self.verify_model(dest_path, expected_hash, expected_version):
            logger.info(f"✓ Model {model_name} already exists and is valid")
            return True
        
        # If in offline mode, try to copy from local source
        if self.offline_mode:
            local_path = sources.get("local")
            if local_path and os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as src, open(dest_path, "wb") as dst:
                        dst.write(src.read())
                    
                    if self.verify_model(dest_path, expected_hash, expected_version):
                        logger.info(f"✓ Model {model_name} copied from local storage")
                        return True
                except Exception as e:
                    logger.error(f"Error copying from local storage: {e}")
            
            logger.error(f"Model {model_name} not found in offline mode")
            return False
        
        # Define source priority and download methods
        source_methods = {
            "local": None,  # Skip in online mode
            "azure": self.download_from_azure if self.azure_available else None,
            "git_lfs": self.download_from_git_lfs if self.git_lfs_available else None,
            "github": self.download_with_progress,
            "huggingface": self.download_with_progress
        }
        
        # Try each source with retries
        for source_name, download_method in source_methods.items():
            if source_name not in sources or download_method is None:
                continue
            
            url = sources[source_name]
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Downloading {model_name} from {source_name} (attempt {attempt+1}/{max_retries})...")
                    
                    success = False
                    if source_name == "azure":
                        success = download_method(url, dest_path, expected_size)
                    elif source_name == "git_lfs":
                        success = download_method(url, dest_path)
                    else:
                        success = download_method(url, dest_path, expected_size)
                    
                    if success and self.verify_model(dest_path, expected_hash, expected_version):
                        logger.info(f"✓ Model {model_name} successfully downloaded from {source_name}")
                        return True
                    else:
                        logger.error(f"Verification failed for {model_name} from {source_name}")
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                except Exception as e:
                    logger.error(f"Download error ({source_name}, attempt {attempt+1}/{max_retries}): {e}")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
        
        logger.error(f"Failed to download model {model_name} from all sources")
        return False
    
    def check_model(self, model_name: str, force_download: bool = False) -> bool:
        """Check if a model is available and download if needed
        
        Args:
            model_name: Name of the model to check
            force_download: Force download even if model exists
            
        Returns:
            bool: True if model is available, False otherwise
        """
        model_info = self.catalog.get_model_info(model_name)
        if not model_info:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        filename = model_info["filename"]
        expected_hash = model_info.get("sha256")
        expected_version = model_info.get("version")
        min_required_version = model_info.get("min_version")
        model_dir = os.path.join(self.models_dir, model_name)
        model_path = os.path.join(model_dir, filename)
        
        # Check if model exists and is valid
        if not force_download and os.path.exists(model_path):
            if self.verify_model(model_path, expected_hash, expected_version):
                # Check version compatibility if minimum version is specified
                if min_required_version and expected_version:
                    if self.check_version_compatibility(expected_version, min_required_version):
                        logger.info(f"✓ Model {model_name} is valid and meets minimum version requirement")
                    else:
                        logger.warning(f"Model {model_name} version {expected_version} is below minimum required version {min_required_version}")
                        return self.download_with_retries(model_name)
                else:
                    logger.info(f"✓ Model {model_name} is valid")
                return True
        
        # Download model if needed
        return self.download_with_retries(model_name)
    
    def check_all_models(self, force_download: bool = False) -> bool:
        """Check all required models and download if needed
        
        Args:
            force_download: Force download even if models exist
            
        Returns:
            bool: True if all required models are available, False otherwise
        """
        required_models = self.catalog.get_required_models()
        if not required_models:
            logger.warning("No required models found in catalog")
            return True
        
        success = True
        missing_models = []
        
        for model_name in required_models:
            logger.info(f"Checking model: {model_name}")
            if not self.check_model(model_name, force_download):
                missing_models.append(model_name)
                success = False
        
        if success:
            logger.info("✓ All required models are available")
        else:
            logger.error(f"× Missing required models: {', '.join(missing_models)}")
        
        return success
    
    def list_models(self) -> Dict:
        """List all models with their status
        
        Returns:
            Dict: Model status information
        """
        all_models = self.catalog.get_all_models()
        model_status = {}
        
        for model_name, model_info in all_models.items():
            filename = model_info["filename"]
            expected_hash = model_info.get("sha256")
            expected_version = model_info.get("version")
            min_required_version = model_info.get("min_version")
            model_dir = os.path.join(self.models_dir, model_name)
            model_path = os.path.join(model_dir, filename)
            
            # Check if model exists and is valid
            if os.path.exists(model_path):
                if self.verify_model(model_path, expected_hash, expected_version):
                    # Check version compatibility if minimum version is specified
                    if min_required_version and expected_version:
                        if self.check_version_compatibility(expected_version, min_required_version):
                            status = "valid"
                        else:
                            status = "outdated"
                    else:
                        status = "valid"
                else:
                    status = "invalid"
            else:
                status = "missing"
            
            model_status[model_name] = {
                "status": status,
                "path": model_path,
                "required": model_info.get("required", False),
                "size": model_info.get("size"),
                "version": expected_version,
                "min_version": min_required_version,
                "description": model_info.get("description", "")
            }
        
        return model_status


# Singleton instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get or initialize model manager singleton
    
    Returns:
        ModelManager: Initialized model manager instance
    """
    global _model_manager
    
    if _model_manager is None:
        _model_manager = ModelManager()
    
    return _model_manager

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="AUTO-TAG Model Manager")
    parser.add_argument("--check", action="store_true", help="Check all required models")
    parser.add_argument("--force", action="store_true", help="Force download even if models exist")
    parser.add_argument("--list", action="store_true", help="List all models with their status")
    parser.add_argument("--download", type=str, help="Download a specific model")
    parser.add_argument("--sources", action="store_true", help="Show available download sources")
    args = parser.parse_args()
    
    # Initialize model manager
    manager = get_model_manager()
    
    # Process commands
    if args.list:
        model_status = manager.list_models()
        print("\nModel Status:")
        print("============")
        for model_name, status in model_status.items():
            status_str = status["status"].upper()
            if status["status"] == "valid":
                status_str = f"\033[92m{status_str}\033[0m"  # Green
            elif status["status"] == "invalid":
                status_str = f"\033[91m{status_str}\033[0m"  # Red
            elif status["status"] == "missing":
                status_str = f"\033[93m{status_str}\033[0m"  # Yellow
            elif status["status"] == "outdated":
                status_str = f"\033[95m{status_str}\033[0m"  # Purple
            
            required = "[REQUIRED]" if status["required"] else "[OPTIONAL]"
            size_mb = status["size"] / (1024 * 1024) if status["size"] else 0
            version_info = f"v{status['version']}" if status.get("version") else "unknown version"
            
            print(f"{model_name}: {status_str} {required} ({size_mb:.2f} MB) - {version_info}")
            print(f"  Path: {status['path']}")
            if status.get("description"):
                print(f"  Description: {status['description']}")
            if status.get("min_version"):
                print(f"  Minimum required version: {status['min_version']}")
        print()
    
    elif args.sources:
        all_models = manager.catalog.get_all_models()
        print("\nAvailable Download Sources:")
        print("=========================")
        for model_name, model_info in all_models.items():
            print(f"\n{model_name}:")
            sources = model_info.get("sources", {})
            
            # Check which sources are available
            if manager.git_lfs_available and "git_lfs" in sources:
                git_lfs_status = "\033[92m[AVAILABLE]\033[0m"  # Green
            elif "git_lfs" in sources:
                git_lfs_status = "\033[93m[NOT CONFIGURED]\033[0m"  # Yellow
            else:
                git_lfs_status = ""
                
            if manager.azure_available and "azure" in sources:
                azure_status = "\033[92m[AVAILABLE]\033[0m"  # Green
            elif "azure" in sources:
                azure_status = "\033[93m[NOT CONFIGURED]\033[0m"  # Yellow
            else:
                azure_status = ""
            
            # Display sources with their status
            for source_name, url in sources.items():
                status = ""
                if source_name == "git_lfs":
                    status = git_lfs_status
                elif source_name == "azure":
                    status = azure_status
                
                print(f"  {source_name}: {url} {status}")
        print()
    
    elif args.download:
        success = manager.check_model(args.download, force_download=args.force)
        if success:
            print(f"\n✓ Model {args.download} is available")
        else:
            print(f"\n× Failed to download model {args.download}")
    
    elif args.check or not (args.list or args.download or args.sources):
        success = manager.check_all_models(force_download=args.force)
        if success:
            print("\n✓ All required models are available")
        else:
            print("\n× Some required models are missing")
    
    sys.exit(0 if success else 1)