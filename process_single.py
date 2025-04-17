#!/usr/bin/env python
# process_single.py - Process a single image with AUTO-TAG
import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import configuration
from config_loader import get_config

# Import models
from models.clip_model import get_clip_model, analyze_scene_clothing
from models.yolo_model import get_yolo_model, count_people
from models.gpu_utils import optimize_for_gpu, cleanup_gpu

# Import tagging module
from tagging.exif_wrapper import get_tag_writer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/auto-tag.log")
    ]
)
logger = logging.getLogger('auto-tag')

def setup_environment():
    """Set up the environment for processing"""
    # Create necessary directories
    config = get_config()
    os.makedirs(config["paths.input_folder"], exist_ok=True)
    os.makedirs(config["paths.output_folder"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Optimize GPU if available
    if config["hardware.use_gpu"]:
        device_id = config["hardware.cuda_device_id"]
        optimize_for_gpu(device_id)

def generate_tags(clip_result, people_result):
    """Generate tags from model results
    
    Args:
        clip_result: Results from CLIP model
        people_result: Results from YOLOv8 model
        
    Returns:
        List of tags
    """
    tags = []
    
    # Add scene tags
    if clip_result and "scene" in clip_result:
        scene_tag, confidence = clip_result["scene"]
        tags.append(f"scene/{scene_tag}")
    
    # Add room type tags
    if clip_result and "roomtype" in clip_result:
        room_tag, confidence = clip_result["roomtype"]
        tags.append(f"roomtype/{room_tag}")
    
    # Add clothing tags
    if clip_result and "clothing" in clip_result:
        clothing_tag, confidence = clip_result["clothing"]
        tags.append(f"clothing/{clothing_tag}")
    
    # Add people count tags
    if people_result:
        tags.append(f"people/{people_result}")
    
    return tags

def process_image(image_path: str) -> Dict[str, Any]:
    """Process a single image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dict with processing results
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return {"success": False, "error": "Image file not found"}

    logger.info(f"Processing image: {os.path.basename(image_path)}")
    start_time = time.time()
    
    try:
        # Analyze with CLIP model
        clip_result = analyze_scene_clothing(image_path)
        
        # Count people with YOLOv8 model
        people_result = count_people(image_path)
        
        # Generate tags
        tags = generate_tags(clip_result, people_result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return results
        return {
            "success": True,
            "tags": tags,
            "clip_result": clip_result,
            "people_result": people_result,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"success": False, "error": str(e)}
    finally:
        # Clean up GPU resources
        cleanup_gpu()

def write_tags_to_file(image_path: str, tags: List[str], mode: str = "append") -> bool:
    """Write tags to image file using ExifTool
    
    Args:
        image_path: Path to the image file
        tags: List of tags to write
        mode: Tag writing mode ("append" or "overwrite")
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not tags:
        logger.warning(f"No tags to write to {os.path.basename(image_path)}")
        return True
    
    logger.info(f"Writing {len(tags)} tags to {os.path.basename(image_path)} (mode: {mode})")
    logger.debug(f"Tags: {tags}")
    
    # Get the tag writer instance
    tag_writer = get_tag_writer()
    
    # Write tags to file
    success = tag_writer.write_tags(image_path, tags, mode)
    
    if success:
        logger.info(f"Successfully wrote tags to {os.path.basename(image_path)}")
    else:
        logger.error(f"Failed to write tags to {os.path.basename(image_path)}")
    
    return success

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AUTO-TAG Single Image Processing")
    parser.add_argument("--input", required=True, help="Path to the image file")
    parser.add_argument("--output", help="Path to save the tagged image (optional)")
    parser.add_argument("--tag-mode", choices=["append", "overwrite"], default="append", 
                        help="Tag writing mode (append or overwrite)")
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Process image
    result = process_image(args.input)
    
    if result["success"]:
        # Write tags to file
        tags = result["tags"]
        success = write_tags_to_file(args.input, tags, args.tag_mode)
        
        # Copy to output location if specified
        if success and args.output:
            try:
                import shutil
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                shutil.copy2(args.input, args.output)
                logger.info(f"Copied to: {args.output}")
            except Exception as e:
                logger.error(f"Error copying file: {e}")
        
        # Print results
        print(f"\n✓ Image successfully processed: {os.path.basename(args.input)}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Tags added: {len(tags)}")
        print(f"Tags: {', '.join(tags)}")
        return 0
    else:
        print(f"\n× Error processing image: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())