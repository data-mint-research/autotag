#!/usr/bin/env python
"""
AUTO-TAG Build and Packaging Script

This script handles the PyInstaller packaging process for AUTO-TAG:
1. Downloads and extracts ExifTool for all platforms
2. Creates the vendor directory structure
3. Runs PyInstaller to create the distribution package
4. Packages the result into a ZIP file for distribution
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse
import logging
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("build.log")
    ]
)
logger = logging.getLogger('build')

def setup_environment():
    """Set up the build environment"""
    logger.info("Setting up build environment")
    
    # Create necessary directories
    os.makedirs("dist", exist_ok=True)
    os.makedirs("build", exist_ok=True)
    os.makedirs("vendor", exist_ok=True)
    os.makedirs("vendor/bin", exist_ok=True)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        logger.info(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        logger.error("PyInstaller not found. Please install it with: pip install pyinstaller")
        return False
    
    return True

def download_exiftool():
    """Download and extract ExifTool for all platforms"""
    logger.info("Downloading and extracting ExifTool")
    
    # Run the ExifTool downloader script
    try:
        result = subprocess.run(
            [sys.executable, "resources/download_exiftool.py", "--output-dir", "resources"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        
        # Check if the files were created
        windows_exiftool = os.path.join("resources", "exiftool.exe")
        macos_exiftool = os.path.join("resources", "exiftool-macos")
        linux_exiftool = os.path.join("resources", "exiftool-linux")
        
        if os.path.exists(windows_exiftool):
            logger.info(f"Windows ExifTool downloaded: {windows_exiftool}")
        else:
            logger.warning("Windows ExifTool not downloaded")
        
        if os.path.exists(macos_exiftool):
            logger.info(f"macOS ExifTool downloaded: {macos_exiftool}")
        else:
            logger.warning("macOS ExifTool not downloaded")
        
        if os.path.exists(linux_exiftool):
            logger.info(f"Linux ExifTool downloaded: {linux_exiftool}")
        else:
            logger.warning("Linux ExifTool not downloaded")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading ExifTool: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def run_pyinstaller(onefile=False):
    """Run PyInstaller to create the distribution package"""
    logger.info(f"Running PyInstaller (onefile={onefile})")
    
    # Determine the target name
    target = "AUTO-TAG-onefile" if onefile else "AUTO-TAG"
    
    # Run PyInstaller
    try:
        cmd = ["pyinstaller", "auto-tag.spec"]
        if onefile:
            cmd.append("--onefile")
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("PyInstaller completed successfully")
        logger.debug(result.stdout)
        
        # Check if the output was created
        dist_path = os.path.join("dist", target)
        if os.path.exists(dist_path):
            logger.info(f"Distribution created: {dist_path}")
            return dist_path
        else:
            logger.error(f"Distribution not created: {dist_path}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running PyInstaller: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return None

def create_zip_package(dist_path, onefile=False):
    """Create a ZIP package of the distribution"""
    logger.info("Creating ZIP package")
    
    # Determine the ZIP file name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    package_type = "onefile" if onefile else "dir"
    system_name = platform.system().lower()
    zip_name = f"AUTO-TAG-{package_type}-{system_name}-{timestamp}.zip"
    zip_path = os.path.join("dist", zip_name)
    
    try:
        if onefile:
            # For onefile, just zip the executable
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                exe_name = "AUTO-TAG-onefile.exe" if system_name == "windows" else "AUTO-TAG-onefile"
                exe_path = os.path.join("dist", exe_name)
                zipf.write(exe_path, os.path.basename(exe_path))
                
                # Add README and other documentation
                if os.path.exists("README.md"):
                    zipf.write("README.md", "README.md")
        else:
            # For directory distribution, zip the entire directory
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(dist_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(dist_path))
                        zipf.write(file_path, arcname)
        
        logger.info(f"ZIP package created: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"Error creating ZIP package: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AUTO-TAG Build and Packaging Script")
    parser.add_argument("--onefile", action="store_true", help="Create a single executable file")
    parser.add_argument("--skip-exiftool", action="store_true", help="Skip downloading ExifTool")
    parser.add_argument("--clean", action="store_true", help="Clean build directories before building")
    args = parser.parse_args()
    
    logger.info("Starting AUTO-TAG build process")
    
    # Clean build directories if requested
    if args.clean:
        logger.info("Cleaning build directories")
        shutil.rmtree("build", ignore_errors=True)
        shutil.rmtree("dist", ignore_errors=True)
    
    # Set up the build environment
    if not setup_environment():
        logger.error("Failed to set up build environment")
        return 1
    
    # Download ExifTool if not skipped
    if not args.skip_exiftool:
        if not download_exiftool():
            logger.error("Failed to download ExifTool")
            return 1
    else:
        logger.info("Skipping ExifTool download")
    
    # Run PyInstaller
    dist_path = run_pyinstaller(args.onefile)
    if not dist_path:
        logger.error("Failed to run PyInstaller")
        return 1
    
    # Create ZIP package
    zip_path = create_zip_package(dist_path, args.onefile)
    if not zip_path:
        logger.error("Failed to create ZIP package")
        return 1
    
    logger.info("Build process completed successfully")
    logger.info(f"Distribution package: {zip_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())