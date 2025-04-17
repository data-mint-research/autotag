#!/usr/bin/env python
"""
ExifTool Downloader and Packager

This script downloads the latest ExifTool release and extracts the appropriate
binaries for each platform (Windows, macOS, Linux). It's used during the build
process to prepare the vendored ExifTool binaries.
"""

import os
import sys
import platform
import shutil
import tempfile
import zipfile
import tarfile
import subprocess
import argparse
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('exiftool-downloader')

# ExifTool download URLs
EXIFTOOL_WINDOWS_URL = "https://exiftool.org/exiftool-12.70.zip"
EXIFTOOL_UNIX_URL = "https://exiftool.org/Image-ExifTool-12.70.tar.gz"

def download_file(url: str, output_path: str) -> bool:
    """
    Download a file from a URL to a local path.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url} to {output_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Download completed: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def extract_windows_exiftool(zip_path: str, output_dir: str) -> Optional[str]:
    """
    Extract ExifTool from the Windows ZIP package.
    
    Args:
        zip_path: Path to the downloaded ZIP file
        output_dir: Directory to extract to
        
    Returns:
        Path to the extracted exiftool.exe, or None if extraction failed
    """
    try:
        logger.info(f"Extracting Windows ExifTool from {zip_path}")
        
        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the exiftool(-k).exe file
            exiftool_path = None
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower() == "exiftool(-k).exe":
                        exiftool_path = os.path.join(root, file)
                        break
                if exiftool_path:
                    break
            
            if not exiftool_path:
                logger.error("exiftool(-k).exe not found in the extracted files")
                return None
            
            # Copy and rename to exiftool.exe
            output_path = os.path.join(output_dir, "exiftool.exe")
            shutil.copy2(exiftool_path, output_path)
            logger.info(f"Extracted Windows ExifTool to {output_path}")
            
            return output_path
    except Exception as e:
        logger.error(f"Error extracting Windows ExifTool: {e}")
        return None

def extract_unix_exiftool(tar_path: str, output_dir: str) -> Optional[str]:
    """
    Extract ExifTool from the Unix tarball and prepare the executable.
    
    Args:
        tar_path: Path to the downloaded tar.gz file
        output_dir: Directory to extract to
        
    Returns:
        Path to the prepared exiftool executable, or None if extraction failed
    """
    try:
        logger.info(f"Extracting Unix ExifTool from {tar_path}")
        
        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the tar.gz file
            with tarfile.open(tar_path, 'r:gz') as tar_ref:
                tar_ref.extractall(temp_dir)
            
            # Find the exiftool script
            exiftool_dir = None
            for item in os.listdir(temp_dir):
                if item.startswith("Image-ExifTool-"):
                    exiftool_dir = os.path.join(temp_dir, item)
                    break
            
            if not exiftool_dir:
                logger.error("Image-ExifTool directory not found in the extracted files")
                return None
            
            exiftool_script = os.path.join(exiftool_dir, "exiftool")
            if not os.path.exists(exiftool_script):
                logger.error(f"exiftool script not found in {exiftool_dir}")
                return None
            
            # Copy to output directory
            if platform.system() == "Darwin":  # macOS
                output_path = os.path.join(output_dir, "exiftool-macos")
            else:  # Linux
                output_path = os.path.join(output_dir, "exiftool-linux")
            
            shutil.copy2(exiftool_script, output_path)
            os.chmod(output_path, 0o755)  # Make executable
            
            logger.info(f"Extracted Unix ExifTool to {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"Error extracting Unix ExifTool: {e}")
        return None

def download_and_extract_exiftool(output_dir: str) -> Dict[str, Optional[str]]:
    """
    Download and extract ExifTool for the current platform.
    
    Args:
        output_dir: Directory to save the extracted ExifTool
        
    Returns:
        Dictionary with paths to the extracted ExifTool for each platform
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "windows": None,
        "macos": None,
        "linux": None
    }
    
    system = platform.system()
    
    # Download and extract for Windows
    if system == "Windows":
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
            temp_file.close()
            if download_file(EXIFTOOL_WINDOWS_URL, temp_file.name):
                results["windows"] = extract_windows_exiftool(temp_file.name, output_dir)
            os.unlink(temp_file.name)
    
    # Download and extract for Unix (macOS/Linux)
    else:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            temp_file.close()
            if download_file(EXIFTOOL_UNIX_URL, temp_file.name):
                if system == "Darwin":  # macOS
                    results["macos"] = extract_unix_exiftool(temp_file.name, output_dir)
                else:  # Linux
                    results["linux"] = extract_unix_exiftool(temp_file.name, output_dir)
            os.unlink(temp_file.name)
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download and package ExifTool for different platforms")
    parser.add_argument("--output-dir", default=".", help="Directory to save the extracted ExifTool")
    args = parser.parse_args()
    
    output_dir = os.path.abspath(args.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    results = download_and_extract_exiftool(output_dir)
    
    # Print results
    logger.info("ExifTool extraction results:")
    for platform_name, path in results.items():
        if path:
            logger.info(f"  {platform_name}: {path}")
        else:
            logger.info(f"  {platform_name}: Not extracted")
    
    # Check if at least one platform was successful
    if not any(results.values()):
        logger.error("Failed to extract ExifTool for any platform")
        return 1
    
    logger.info("ExifTool download and extraction completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())