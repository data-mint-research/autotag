#!/usr/bin/env python
# batch_process.py - Advanced batch processing with AUTO-TAG
import os
import sys
import argparse
import logging
import time
import json
import queue
import threading
import signal
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Set, Deque
from collections import deque
from tqdm import tqdm
import numpy as np

# Import configuration
from config_loader import get_config

# Import process_single functionality
from process_single import process_image, write_tags_to_file, setup_environment

# Import GPU utilities
from models.gpu_utils import (
    check_gpu_availability,
    get_optimal_batch_size,
    optimize_for_gpu,
    cleanup_gpu,
    MultiGPUManager,
    get_optimal_device_config
)

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/auto-tag.log")
    ]
)
logger = logging.getLogger('auto-tag')

# Global variables for checkpointing and graceful shutdown
_shutdown_requested = False
_checkpoint_lock = threading.Lock()
_processed_files = set()
_failed_files = set()
_current_checkpoint_file = None

def signal_handler(sig, frame):
    """Handle interrupt signals for graceful shutdown"""
    global _shutdown_requested
    logger.warning("Shutdown requested. Completing current tasks and saving checkpoint...")
    _shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def find_images(folder_path: str, recursive: bool = False, max_depth: int = 0,
               current_depth: int = 0, file_extensions: Set[str] = None) -> List[str]:
    """Find all images in the specified folder, optionally recursively
    
    Args:
        folder_path: Path to the folder
        recursive: Whether to search recursively
        max_depth: Maximum recursion depth (0 means no limit)
        current_depth: Current recursion depth
        file_extensions: Set of file extensions to include
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return []
    
    # Default file extensions if not specified
    if file_extensions is None:
        file_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Max depth 0 means no limit
    if max_depth == 0:
        max_depth = float('inf')
    
    image_files = []
    
    # Get all files in current directory
    try:
        items = os.listdir(folder_path)
        
        # Find images in current directory
        for item in items:
            item_path = os.path.join(folder_path, item)
            
            # If it's a file with an image extension
            if os.path.isfile(item_path) and any(item.lower().endswith(ext) for ext in file_extensions):
                image_files.append(item_path)
            
            # If it's a directory and we're searching recursively
            elif os.path.isdir(item_path) and recursive and current_depth < max_depth:
                # Recursively find images in subdirectories
                sub_images = find_images(
                    item_path,
                    recursive=recursive,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    file_extensions=file_extensions
                )
                image_files.extend(sub_images)
    
    except Exception as e:
        logger.error(f"Error searching {folder_path}: {e}")
    
    return image_files

def load_checkpoint(checkpoint_path: str) -> Tuple[Set[str], Set[str]]:
    """Load checkpoint file to resume processing
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (processed_files, failed_files)
    """
    processed_files = set()
    failed_files = set()
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found: {checkpoint_path}")
        return processed_files, failed_files
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            
        processed_files = set(checkpoint_data.get('processed_files', []))
        failed_files = set(checkpoint_data.get('failed_files', []))
        
        logger.info(f"Loaded checkpoint: {len(processed_files)} processed, {len(failed_files)} failed")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
    
    return processed_files, failed_files

def save_checkpoint(checkpoint_path: str, processed_files: Set[str], failed_files: Set[str],
                   total_files: int, metadata: Dict = None) -> None:
    """Save checkpoint file to allow resuming processing
    
    Args:
        checkpoint_path: Path to the checkpoint file
        processed_files: Set of processed file paths
        failed_files: Set of failed file paths
        total_files: Total number of files to process
        metadata: Additional metadata to save
    """
    with _checkpoint_lock:
        try:
            checkpoint_data = {
                'timestamp': time.time(),
                'datetime': datetime.datetime.now().isoformat(),
                'processed_files': list(processed_files),
                'failed_files': list(failed_files),
                'total_files': total_files,
                'progress_percent': (len(processed_files) + len(failed_files)) / total_files * 100 if total_files > 0 else 0
            }
            
            # Add additional metadata if provided
            if metadata:
                checkpoint_data.update(metadata)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            # Save checkpoint
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

def process_image_wrapper(args):
    """Wrapper function for process_image to use with ThreadPoolExecutor
    
    Args:
        args: Tuple of (image_path, config, tag_mode, device_id)
        
    Returns:
        Dict with processing results
    """
    global _processed_files, _failed_files, _shutdown_requested
    
    image_path, config, tag_mode, device_id = args
    
    # Skip if shutdown requested
    if _shutdown_requested:
        return image_path, {"success": False, "error": "Shutdown requested", "skipped": True}
    
    # Skip if already processed
    if image_path in _processed_files:
        return image_path, {"success": True, "tags": [], "skipped": True, "reason": "Already processed"}
    
    try:
        # Set CUDA device if specified
        if device_id is not None and device_id >= 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(device_id)
            except Exception as e:
                logger.warning(f"Failed to set CUDA device {device_id}: {e}")
        
        # Process the image
        start_time = time.time()
        result = process_image(image_path)
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        # If successful, write tags
        if result["success"]:
            tags = result["tags"]
            
            # Write tags to file
            tag_success = write_tags_to_file(image_path, tags, tag_mode)
            result["tag_success"] = tag_success
            
            # If output folder is specified and different from input folder
            output_folder = config["paths.output_folder"]
            input_folder = config["paths.input_folder"]
            
            if output_folder and output_folder != os.path.dirname(image_path):
                try:
                    # Create relative path structure
                    if image_path.startswith(input_folder):
                        rel_path = os.path.relpath(image_path, input_folder)
                        output_path = os.path.join(output_folder, rel_path)
                        
                        # Ensure target directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Copy the file (with tags)
                        import shutil
                        shutil.copy2(image_path, output_path)
                        result["copied_to"] = output_path
                except Exception as e:
                    logger.error(f"Error copying file: {e}")
            
            # Add to processed files
            with _checkpoint_lock:
                _processed_files.add(image_path)
        else:
            # Add to failed files
            with _checkpoint_lock:
                _failed_files.add(image_path)
        
        return image_path, result
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        
        # Add to failed files
        with _checkpoint_lock:
            _failed_files.add(image_path)
        
        return image_path, {"success": False, "error": str(e)}

def estimate_eta(start_time: float, processed_count: int, total_count: int) -> str:
    """Estimate time remaining based on current progress
    
    Args:
        start_time: Start time in seconds
        processed_count: Number of items processed
        total_count: Total number of items to process
        
    Returns:
        String with estimated time remaining
    """
    if processed_count == 0:
        return "Unknown"
    
    elapsed = time.time() - start_time
    items_per_second = processed_count / elapsed
    
    if items_per_second == 0:
        return "Unknown"
    
    remaining_items = total_count - processed_count
    seconds_remaining = remaining_items / items_per_second
    
    # Format time remaining
    if seconds_remaining < 60:
        return f"{seconds_remaining:.0f} seconds"
    elif seconds_remaining < 3600:
        return f"{seconds_remaining/60:.1f} minutes"
    else:
        return f"{seconds_remaining/3600:.1f} hours"

def create_work_queue(image_files: List[str], batch_size: int,
                     processed_files: Set[str] = None) -> List[List[str]]:
    """Create batches of work for processing
    
    Args:
        image_files: List of image file paths
        batch_size: Size of each batch
        processed_files: Set of already processed files to skip
        
    Returns:
        List of batches, where each batch is a list of file paths
    """
    # Filter out already processed files
    if processed_files:
        remaining_files = [f for f in image_files if f not in processed_files]
    else:
        remaining_files = image_files
    
    # Create batches
    batches = []
    for i in range(0, len(remaining_files), batch_size):
        batch = remaining_files[i:i+batch_size]
        batches.append(batch)
    
    return batches

def batch_process(input_folder: Optional[str] = None, recursive: Optional[bool] = None, 
                 max_workers: Optional[int] = None, dry_run: bool = False,
                 checkpoint_file: Optional[str] = None, resume: bool = False,
                 batch_size: Optional[int] = None, device_id: Optional[int] = None,
                 checkpoint_interval: int = 5) -> List[Dict]:
    """Process all images in the specified folder with advanced features
    
    Args:
        input_folder: Path to the input folder
        recursive: Whether to process subdirectories
        max_workers: Number of parallel workers
        dry_run: If True, only list images without processing
        checkpoint_file: Path to checkpoint file for resumable processing
        resume: Whether to resume from checkpoint
        batch_size: Size of processing batches (0 for auto)
        device_id: CUDA device ID to use (-1 for CPU, None for auto)
        checkpoint_interval: Minutes between checkpoint saves
        
    Returns:
        List of processing results
    """
    global _processed_files, _failed_files, _current_checkpoint_file, _shutdown_requested
    
    # Load configuration
    config = get_config()
    
    # Use default values from configuration if not specified
    if input_folder is None:
        input_folder = config["paths.input_folder"]
    
    if recursive is None:
        recursive = config["processing.subdirectories"]
    
    if max_workers is None:
        max_workers = config["hardware.num_workers"]
    
    # Set up GPU if available
    gpu_info = check_gpu_availability()
    using_gpu = gpu_info["available"] and device_id != -1
    
    if using_gpu:
        # Auto-detect optimal device if not specified
        if device_id is None:
            # Find best device (prefer RTX 5090)
            best_device = 0
            best_memory = 0
            rtx_5090_found = False
            
            for device in gpu_info["devices"]:
                # Check if this is an RTX 5090
                if device.get("is_rtx_5090", False) and not rtx_5090_found:
                    best_device = device["id"]
                    rtx_5090_found = True
                    logger.info(f"RTX 5090 found at device {best_device}")
                # Otherwise, choose the device with the most memory
                elif not rtx_5090_found and device.get("total_memory_mb", 0) > best_memory:
                    best_device = device["id"]
                    best_memory = device.get("total_memory_mb", 0)
            
            device_id = best_device
        
        # Optimize GPU settings
        logger.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
        optimize_for_gpu(device_id=device_id)
    else:
        logger.info("Using CPU for processing")
        device_id = -1
    
    # Find all images in the specified folder
    logger.info(f"Searching for images in {input_folder} {'(recursive)' if recursive else ''}...")
    image_files = find_images(input_folder, recursive=recursive)
    
    if not image_files:
        logger.warning(f"No images found in {input_folder}")
        return []
    
    logger.info(f"Found: {len(image_files)} images")
    
    # If dry run, just return the list of found images
    if dry_run:
        return image_files
    
    # Set up checkpoint file
    if checkpoint_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_file = f"checkpoints/batch_checkpoint_{timestamp}.json"
    
    _current_checkpoint_file = checkpoint_file
    
    # Create checkpoints directory
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    
    # Load checkpoint if resuming
    if resume and os.path.exists(checkpoint_file):
        _processed_files, _failed_files = load_checkpoint(checkpoint_file)
        logger.info(f"Resuming from checkpoint: {len(_processed_files)} processed, {len(_failed_files)} failed")
    else:
        _processed_files = set()
        _failed_files = set()
    
    # Determine optimal batch size if not specified
    if batch_size is None or batch_size <= 0:
        if using_gpu:
            # Use dynamic batch sizing based on GPU memory
            # Assume average image size of 1280x720 for estimation
            batch_size = get_optimal_batch_size(720, 1280, "medium", device_id=device_id)
            logger.info(f"Using dynamic batch size: {batch_size} based on GPU memory")
        else:
            # Default batch size for CPU
            batch_size = 4
            logger.info(f"Using default CPU batch size: {batch_size}")
    
    # Get tag mode from configuration
    tag_mode = config["tagging.mode"]
    
    # Create work queue
    work_batches = create_work_queue(image_files, batch_size, _processed_files)
    logger.info(f"Created {len(work_batches)} work batches of size {batch_size}")
    
    # Initialize results list and tracking variables
    results = []
    start_time = time.time()
    last_checkpoint_time = start_time
    processed_count = len(_processed_files)
    total_count = len(image_files)
    
    # Create progress bar
    progress_bar = tqdm(
        total=total_count,
        initial=processed_count,
        desc="Processing images",
        unit="image",
        ncols=100
    )
    
    # Process images in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each batch
        for batch_idx, batch in enumerate(work_batches):
            if _shutdown_requested:
                logger.warning("Shutdown requested, stopping processing")
                break
            
            # Create arguments for each image in the batch
            args_list = [(image_path, config, tag_mode, device_id) for image_path in batch]
            
            # Submit batch for processing
            futures = [executor.submit(process_image_wrapper, args) for args in args_list]
            
            # Process results as they complete
            for future in futures:
                if _shutdown_requested:
                    break
                
                try:
                    image_path, result = future.result()
                    
                    # Skip already processed files
                    if result.get("skipped", False):
                        continue
                    
                    # Add to results
                    results.append({
                        "path": image_path,
                        "success": result.get("success", False),
                        "tags": result.get("tags", []) if result.get("success", False) else [],
                        "error": result.get("error", None) if not result.get("success", False) else None,
                        "processing_time": result.get("processing_time", 0)
                    })
                    
                    # Update progress
                    processed_count = len(_processed_files) + len(_failed_files)
                    progress_bar.update(1)
                    
                    # Update progress bar description with ETA
                    eta = estimate_eta(start_time, processed_count, total_count)
                    progress_bar.set_description(f"Processing images (ETA: {eta})")
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
            
            # Save checkpoint periodically
            current_time = time.time()
            if (current_time - last_checkpoint_time) / 60 >= checkpoint_interval:
                save_checkpoint(
                    checkpoint_file,
                    _processed_files,
                    _failed_files,
                    total_count,
                    {
                        "batch_index": batch_idx,
                        "total_batches": len(work_batches),
                        "elapsed_time": current_time - start_time,
                        "eta": estimate_eta(start_time, processed_count, total_count)
                    }
                )
                last_checkpoint_time = current_time
                logger.info(f"Checkpoint saved: {processed_count}/{total_count} images processed")
    
    # Close progress bar
    progress_bar.close()
    
    # Save final results
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Processing completed: {len(results)} images in {elapsed_time:.2f} seconds")
    
    # Create results report
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Successful: {success_count}/{len(results)} images")
    
    # Calculate statistics
    if results:
        processing_times = [r.get("processing_time", 0) for r in results if r.get("processing_time", 0) > 0]
        if processing_times:
            avg_time = np.mean(processing_times)
            min_time = np.min(processing_times)
            max_time = np.max(processing_times)
            logger.info(f"Processing time per image: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"batch_results_{timestamp}.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "input_folder": input_folder,
            "recursive": recursive,
            "total_images": len(image_files),
            "processed_images": len(_processed_files),
            "failed_images": len(_failed_files),
            "success_count": success_count,
            "elapsed_time": elapsed_time,
            "gpu_used": using_gpu,
            "device_id": device_id if using_gpu else None,
            "batch_size": batch_size,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {result_file}")
    
    # Clean up GPU resources
    if using_gpu:
        cleanup_gpu(device_id)
    
    return results

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AUTO-TAG Batch Processing")
    parser.add_argument("--input", help="Input folder with images")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Only list images, no processing")
    parser.add_argument("--checkpoint", help="Checkpoint file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, help="Batch size (0 for auto)")
    parser.add_argument("--device", type=int, help="CUDA device ID (-1 for CPU, default: auto)")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Minutes between checkpoints")
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Run batch processing
    results = batch_process(
        input_folder=args.input,
        recursive=args.recursive, 
        max_workers=args.workers,
        dry_run=args.dry_run,
        checkpoint_file=args.checkpoint,
        resume=args.resume,
        batch_size=args.batch_size,
        device_id=args.device,
        checkpoint_interval=args.checkpoint_interval
    )
    
    if args.dry_run:
        print("\nFound images:")
        for image_path in results:
            print(f"  {image_path}")
        print(f"\nTotal: {len(results)} images")
    else:
        # Print summary
        success_count = sum(1 for r in results if r["success"])
        print(f"\nProcessing completed: {len(results)} images")
        print(f"Successful: {success_count}/{len(results)} images")
    
    return 0 if results is not None else 1

if __name__ == "__main__":
    sys.exit(main())