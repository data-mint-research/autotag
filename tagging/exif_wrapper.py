"""
ExifTool Wrapper Module

This module provides a wrapper for ExifTool with vendoring capabilities.
It handles platform-specific detection and extraction of ExifTool.
"""

import os
import sys
import logging
import platform
import shutil
import tempfile
import zipfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# Configure logging
logger = logging.getLogger('auto-tag.exif')

class ExifToolVendor:
    """
    Handles the vendoring of ExifTool, including platform-specific detection and extraction.
    """
    
    def __init__(self, vendor_dir: Optional[str] = None):
        """
        Initialize the ExifTool vendor.
        
        Args:
            vendor_dir: Optional directory to store vendored ExifTool. If None, uses default location.
        """
        self.system = platform.system()
        self.vendor_dir = vendor_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vendor')
        self.bin_dir = os.path.join(self.vendor_dir, 'bin')
        
        # Create vendor directories if they don't exist
        os.makedirs(self.bin_dir, exist_ok=True)
        
        # Platform-specific paths
        if self.system == "Windows":
            self.exiftool_path = os.path.join(self.bin_dir, 'exiftool.exe')
        else:  # Linux/Mac
            self.exiftool_path = os.path.join(self.bin_dir, 'exiftool')
    
    def find_system_exiftool(self) -> Optional[str]:
        """
        Find ExifTool in the system path.
        
        Returns:
            Path to ExifTool if found, None otherwise.
        """
        try:
            if self.system == "Windows":
                # Try to find exiftool.exe in PATH
                result = subprocess.run(["where", "exiftool"], 
                                       capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().split('\n')[0]
            else:
                # Linux/Mac: Try to find exiftool in PATH
                result = subprocess.run(["which", "exiftool"], 
                                       capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Error finding system ExifTool: {e}")
        
        return None
    
    def extract_vendored_exiftool(self) -> bool:
        """
        Extract the vendored ExifTool for the current platform.
        
        Returns:
            True if extraction was successful, False otherwise.
        """
        try:
            # Check if we already have the vendored ExifTool
            if os.path.exists(self.exiftool_path):
                logger.debug(f"Vendored ExifTool already exists at {self.exiftool_path}")
                return True
            
            # Platform-specific extraction
            if self.system == "Windows":
                # For Windows, we would extract exiftool.exe from our resources
                # In a real implementation, this would extract from an embedded resource
                # For now, we'll just check if it's in the package directory
                package_exiftool = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              '..', 'resources', 'exiftool.exe')
                
                if os.path.exists(package_exiftool):
                    shutil.copy2(package_exiftool, self.exiftool_path)
                    logger.info(f"Extracted vendored ExifTool to {self.exiftool_path}")
                    return True
                else:
                    logger.warning("Vendored ExifTool not found in package resources")
                    return False
            
            elif self.system == "Darwin":  # macOS
                # For macOS, we would extract the macOS version
                package_exiftool = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              '..', 'resources', 'exiftool-macos')
                
                if os.path.exists(package_exiftool):
                    shutil.copy2(package_exiftool, self.exiftool_path)
                    # Make it executable
                    os.chmod(self.exiftool_path, 0o755)
                    logger.info(f"Extracted vendored ExifTool to {self.exiftool_path}")
                    return True
                else:
                    logger.warning("Vendored ExifTool not found in package resources")
                    return False
            
            elif self.system == "Linux":
                # For Linux, we would extract the Linux version
                package_exiftool = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              '..', 'resources', 'exiftool-linux')
                
                if os.path.exists(package_exiftool):
                    shutil.copy2(package_exiftool, self.exiftool_path)
                    # Make it executable
                    os.chmod(self.exiftool_path, 0o755)
                    logger.info(f"Extracted vendored ExifTool to {self.exiftool_path}")
                    return True
                else:
                    logger.warning("Vendored ExifTool not found in package resources")
                    return False
            
            else:
                logger.error(f"Unsupported platform: {self.system}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting vendored ExifTool: {e}")
            return False
    
    def get_exiftool_path(self) -> Optional[str]:
        """
        Get the path to ExifTool, trying multiple sources.
        
        Returns:
            Path to ExifTool if found, None otherwise.
        """
        # First, check if we already have the vendored ExifTool
        if os.path.exists(self.exiftool_path):
            logger.debug(f"Using vendored ExifTool at {self.exiftool_path}")
            return self.exiftool_path
        
        # Next, try to find ExifTool in the system path
        system_exiftool = self.find_system_exiftool()
        if system_exiftool:
            logger.debug(f"Using system ExifTool at {system_exiftool}")
            return system_exiftool
        
        # Finally, try to extract the vendored ExifTool
        if self.extract_vendored_exiftool():
            logger.debug(f"Using newly extracted vendored ExifTool at {self.exiftool_path}")
            return self.exiftool_path
        
        # If all else fails, return None
        logger.error("ExifTool not found")
        return None
    
    def test_exiftool(self, exiftool_path: Optional[str] = None) -> bool:
        """
        Test if ExifTool is working.
        
        Args:
            exiftool_path: Path to ExifTool to test. If None, uses the result of get_exiftool_path().
            
        Returns:
            True if ExifTool is working, False otherwise.
        """
        path = exiftool_path or self.get_exiftool_path()
        if not path:
            return False
        
        try:
            result = subprocess.run([path, "-ver"], 
                                   capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"ExifTool version: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"ExifTool test failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error testing ExifTool: {e}")
            return False


class TagWriter:
    """
    Writes tags to image files using ExifTool.
    """
    
    def __init__(self, vendor_dir: Optional[str] = None):
        """
        Initialize the tag writer.
        
        Args:
            vendor_dir: Optional directory to store vendored ExifTool. If None, uses default location.
        """
        self.vendor = ExifToolVendor(vendor_dir)
        self.exiftool_path = self.vendor.get_exiftool_path()
        
        if not self.exiftool_path:
            logger.error("ExifTool not available. Tags cannot be written.")
        else:
            logger.info(f"Using ExifTool at {self.exiftool_path}")
            
        # Test ExifTool
        if self.exiftool_path and not self.vendor.test_exiftool(self.exiftool_path):
            logger.error("ExifTool test failed. Tags may not be written correctly.")
    
    def write_tags(self, image_path: str, tags: List[str], mode: str = "append") -> bool:
        """
        Write tags to image metadata using ExifTool.
        
        Args:
            image_path: Path to the image file
            tags: List of tags to write
            mode: Tag writing mode ("append" or "overwrite")
            
        Returns:
            True if successful, False otherwise
        """
        if not self.exiftool_path:
            logger.error("ExifTool not available. Tags cannot be written.")
            return False
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False
        
        try:
            # Format tags as a comma-separated list
            tag_list = ",".join(tags)
            
            # Build the ExifTool command
            if mode == "overwrite":
                cmd = [
                    self.exiftool_path,
                    f"-XMP-digiKam:TagsList={tag_list}",
                    "-overwrite_original",
                    image_path
                ]
            elif mode == "append":
                cmd = [
                    self.exiftool_path,
                    f"-XMP-digiKam:TagsList+={tag_list}",
                    "-overwrite_original",
                    image_path
                ]
            else:
                logger.error(f"Invalid mode: {mode}. Must be 'append' or 'overwrite'.")
                return False
            
            # Execute the command
            logger.debug(f"Executing ExifTool command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # Check the result
            if result.returncode != 0:
                logger.error(f"ExifTool error: {result.stderr}")
                return False
            
            logger.info(f"Successfully wrote {len(tags)} tags to {os.path.basename(image_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing tags to {image_path}: {e}")
            return False
    
    def read_tags(self, image_path: str) -> List[str]:
        """
        Read existing tags from image metadata using ExifTool.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of tags
        """
        if not self.exiftool_path or not os.path.exists(image_path):
            return []
        
        try:
            # Build the ExifTool command
            cmd = [
                self.exiftool_path,
                "-XMP-digiKam:TagsList",
                "-s",  # Short output format
                "-j",  # JSON output
                image_path
            ]
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # Check the result
            if result.returncode != 0:
                logger.error(f"ExifTool error: {result.stderr}")
                return []
            
            # Parse the JSON output
            import json
            data = json.loads(result.stdout)
            
            if not data or not isinstance(data, list) or len(data) == 0:
                return []
            
            # Extract the tags
            tags_str = data[0].get("TagsList", "")
            if not tags_str:
                return []
            
            # Split the comma-separated list
            tags = [tag.strip() for tag in tags_str.split(",")]
            return tags
            
        except Exception as e:
            logger.error(f"Error reading tags from {image_path}: {e}")
            return []


# Create a singleton instance for easy import
default_tag_writer = None

def get_tag_writer(vendor_dir: Optional[str] = None) -> TagWriter:
    """
    Get the default TagWriter instance.
    
    Args:
        vendor_dir: Optional directory to store vendored ExifTool. If None, uses default location.
        
    Returns:
        TagWriter instance
    """
    global default_tag_writer
    if default_tag_writer is None:
        default_tag_writer = TagWriter(vendor_dir)
    return default_tag_writer