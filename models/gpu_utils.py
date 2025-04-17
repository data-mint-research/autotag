# models/gpu_utils.py - Advanced GPU optimization utilities for RTX 5090
import os
import logging
import gc
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

# Logger setup
logger = logging.getLogger("auto-tag")

# Global variables for CUDA streams and memory tracking
_cuda_streams = {}
_memory_locks = {}
_active_devices = set()

def check_gpu_availability() -> Dict[str, Union[bool, str, int, List[Dict]]]:
    """Check GPU availability and return detailed information
    
    Returns:
        Dict with GPU information:
        - available: Whether CUDA is available
        - device_count: Number of available GPUs
        - cuda_version: CUDA version if available
        - devices: List of device information dictionaries
    """
    result = {
        "available": False,
        "device_count": 0,
        "cuda_version": None,
        "devices": []
    }
    
    try:
        if torch.cuda.is_available():
            result["available"] = True
            result["device_count"] = torch.cuda.device_count()
            result["cuda_version"] = torch.version.cuda
            
            # Get detailed information for each device
            for device_id in range(result["device_count"]):
                device_props = torch.cuda.get_device_properties(device_id)
                
                # Get memory information
                try:
                    torch.cuda.set_device(device_id)
                    memory_total = device_props.total_memory
                    memory_reserved = torch.cuda.memory_reserved(device_id)
                    memory_allocated = torch.cuda.memory_allocated(device_id)
                    memory_free = memory_total - memory_allocated
                    
                    # Check if device supports tensor cores
                    has_tensor_cores = (
                        device_props.major >= 7 or
                        (device_props.major == 7 and device_props.minor >= 0)
                    )
                    
                    # Check if device is RTX 5090
                    is_rtx_5090 = "RTX 5090" in device_props.name
                    
                    device_info = {
                        "id": device_id,
                        "name": device_props.name,
                        "compute_capability": f"{device_props.major}.{device_props.minor}",
                        "total_memory_mb": memory_total / (1024 * 1024),
                        "free_memory_mb": memory_free / (1024 * 1024),
                        "multi_processor_count": device_props.multi_processor_count,
                        "has_tensor_cores": has_tensor_cores,
                        "is_rtx_5090": is_rtx_5090,
                        "max_threads_per_block": device_props.max_threads_per_block,
                        "max_shared_memory_per_block": device_props.max_shared_memory_per_block,
                    }
                    result["devices"].append(device_info)
                except Exception as e:
                    logger.warning(f"Could not get GPU memory information for device {device_id}: {e}")
                    result["devices"].append({
                        "id": device_id,
                        "name": device_props.name,
                        "compute_capability": f"{device_props.major}.{device_props.minor}",
                        "error": str(e)
                    })
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
    
    return result

def get_cuda_stream(device_id: int = 0, stream_id: int = 0) -> torch.cuda.Stream:
    """Get or create a CUDA stream for the specified device
    
    Args:
        device_id: CUDA device ID
        stream_id: Stream ID for the device
        
    Returns:
        torch.cuda.Stream: CUDA stream
    """
    global _cuda_streams
    
    key = (device_id, stream_id)
    if key not in _cuda_streams:
        with torch.cuda.device(device_id):
            _cuda_streams[key] = torch.cuda.Stream(device=device_id)
    
    return _cuda_streams[key]

def optimize_for_gpu(device_id: int = 0, enable_mixed_precision: bool = True,
                    enable_tensor_cores: bool = True, num_streams: int = 4) -> bool:
    """Configure PyTorch for optimal performance on RTX 5090 with advanced optimizations
    
    Args:
        device_id: CUDA device ID to use
        enable_mixed_precision: Whether to enable mixed precision (FP16/BF16)
        enable_tensor_cores: Whether to enable tensor cores
        num_streams: Number of CUDA streams to create
        
    Returns:
        bool: True if GPU optimization was successful, False otherwise
    """
    global _active_devices, _memory_locks
    
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available - using CPU")
        return False
    
    try:
        # Set CUDA device
        torch.cuda.set_device(device_id)
        _active_devices.add(device_id)
        
        # Create memory lock for this device
        if device_id not in _memory_locks:
            _memory_locks[device_id] = threading.RLock()
        
        # Enable TF32 for faster computation (RTX 30xx and newer)
        torch.backends.cuda.matmul.allow_tf32 = enable_tensor_cores
        torch.backends.cudnn.allow_tf32 = enable_tensor_cores
        
        # Set cuDNN benchmark mode for optimized performance
        torch.backends.cudnn.benchmark = True
        
        # Disable gradient calculation for inference
        torch.set_grad_enabled(False)
        
        # Initialize CUDA streams for this device
        for i in range(num_streams):
            get_cuda_stream(device_id, i)
        
        # Enable automatic mixed precision if requested
        if enable_mixed_precision:
            # Check if device supports BF16
            device_props = torch.cuda.get_device_properties(device_id)
            has_bf16 = device_props.major >= 8 or (device_props.major == 7 and device_props.minor >= 5)
            
            if has_bf16:
                # BF16 is preferred for RTX 5090
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                logger.info("BF16 mixed precision enabled")
            else:
                # Fall back to FP16
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                logger.info("FP16 mixed precision enabled")
        
        # Pre-allocate some CUDA memory to avoid fragmentation
        with torch.cuda.device(device_id):
            # Allocate and immediately free to initialize CUDA context
            dummy = torch.zeros(1024, 1024, device='cuda')
            del dummy
            torch.cuda.empty_cache()
            gc.collect()
        
        device_name = torch.cuda.get_device_name(device_id)
        logger.info(f"Advanced GPU optimization enabled: {device_name}")
        
        # Log optimization settings
        logger.debug(f"Device: {device_name}")
        logger.debug(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        logger.debug(f"Tensor cores enabled: {enable_tensor_cores}")
        logger.debug(f"Mixed precision enabled: {enable_mixed_precision}")
        logger.debug(f"CUDA streams: {num_streams}")
        logger.debug(f"cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
        
        return True
    except Exception as e:
        logger.error(f"Error optimizing for GPU: {e}")
        return False

def cleanup_gpu(device_id: Optional[int] = None) -> None:
    """Free GPU memory after processing with advanced cleanup
    
    Args:
        device_id: Specific device to clean up, or None for all active devices
    """
    global _active_devices, _memory_locks
    
    if not torch.cuda.is_available():
        return
    
    devices_to_clean = [device_id] if device_id is not None else list(_active_devices)
    
    for dev_id in devices_to_clean:
        if dev_id in _memory_locks:
            with _memory_locks[dev_id]:
                try:
                    # Set device context
                    torch.cuda.set_device(dev_id)
                    
                    # Force completion of all ongoing GPU operations
                    torch.cuda.synchronize(dev_id)
                    
                    # Clear CUDA cache and run garbage collection
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Log memory status after cleanup
                    allocated = torch.cuda.memory_allocated(dev_id) / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(dev_id) / (1024 * 1024)
                    logger.debug(f"Device {dev_id} after cleanup: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
                except Exception as e:
                    logger.warning(f"Error during GPU cleanup for device {dev_id}: {e}")

def get_optimal_batch_size(image_height: int, image_width: int, model_complexity: str = "medium",
                          memory_fraction: float = 0.8, device_id: int = 0) -> int:
    """Calculate optimal batch size based on available GPU memory and image size
    
    Args:
        image_height: Height of the images in pixels
        image_width: Width of the images in pixels
        model_complexity: Complexity of the model ("low", "medium", "high", "very_high")
        memory_fraction: Fraction of available memory to use (0.0-1.0)
        device_id: CUDA device ID to use
        
    Returns:
        int: Optimal batch size
    """
    if not torch.cuda.is_available():
        return 1  # Default to 1 for CPU
    
    try:
        # Get available GPU memory for the specific device
        with torch.cuda.device(device_id):
            device_props = torch.cuda.get_device_properties(device_id)
            total_memory = device_props.total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            available_memory = total_memory - allocated_memory
            
            # Convert to MB for easier calculations
            available_memory_mb = available_memory / (1024 * 1024)
            
            # Apply memory fraction limit
            available_memory_mb *= memory_fraction
            
            # Estimate memory per image based on resolution and model complexity
            # These are more accurate estimates for RTX 5090
            bytes_per_pixel = {
                "low": 4,      # Simple models
                "medium": 16,  # Medium complexity models (CLIP, YOLOv8n)
                "high": 32,    # Complex models (large CLIP, YOLOv8x)
                "very_high": 64 # Very complex models or multi-model pipelines
            }.get(model_complexity.lower(), 16)
            
            # For mixed precision, adjust memory requirements
            if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction or \
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
                bytes_per_pixel = bytes_per_pixel * 0.6  # ~40% memory savings with mixed precision
            
            # Calculate memory needed per image (with overhead)
            memory_per_image_mb = (image_height * image_width * bytes_per_pixel * 1.2) / (1024 * 1024)
            
            # Calculate batch size
            batch_size = max(1, int(available_memory_mb / memory_per_image_mb))
            
            # Cap batch size at reasonable limits based on model complexity and GPU capabilities
            is_rtx_5090 = "RTX 5090" in device_props.name
            
            if is_rtx_5090:
                # Higher limits for RTX 5090
                if model_complexity.lower() == "high":
                    batch_size = min(batch_size, 32)
                elif model_complexity.lower() == "medium":
                    batch_size = min(batch_size, 64)
                elif model_complexity.lower() == "very_high":
                    batch_size = min(batch_size, 16)
                else:
                    batch_size = min(batch_size, 128)
            else:
                # Standard limits for other GPUs
                if model_complexity.lower() == "high":
                    batch_size = min(batch_size, 16)
                elif model_complexity.lower() == "medium":
                    batch_size = min(batch_size, 32)
                elif model_complexity.lower() == "very_high":
                    batch_size = min(batch_size, 8)
                else:
                    batch_size = min(batch_size, 64)
            
            logger.debug(f"Calculated optimal batch size: {batch_size} for {image_height}x{image_width} images on device {device_id}")
            return batch_size
    except Exception as e:
        logger.warning(f"Error calculating optimal batch size: {e}")
        # Default batch sizes based on model complexity
        defaults = {"low": 16, "medium": 8, "high": 4, "very_high": 2}
        return defaults.get(model_complexity.lower(), 8)

def setup_mixed_precision(model_type: str = "clip") -> torch.amp.autocast:
    """Set up mixed precision for the specified model type
    
    Args:
        model_type: Type of model ("clip", "yolo", "facenet")
        
    Returns:
        torch.amp.autocast: Autocast context manager
    """
    if not torch.cuda.is_available():
        # Return a dummy context manager if CUDA is not available
        return torch.autocast("cpu", enabled=False)
    
    # Check if device supports BF16
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    has_bf16 = device_props.major >= 8 or (device_props.major == 7 and device_props.minor >= 5)
    
    # Determine precision type based on model and hardware
    if model_type.lower() == "clip":
        # CLIP models work well with BF16 on supported hardware
        dtype = torch.bfloat16 if has_bf16 else torch.float16
    elif model_type.lower() == "yolo":
        # YOLO models are optimized for FP16
        dtype = torch.float16
    else:
        # Default to BF16 if supported, otherwise FP16
        dtype = torch.bfloat16 if has_bf16 else torch.float16
    
    logger.debug(f"Using mixed precision {dtype} for {model_type} model")
    return torch.amp.autocast(device_type="cuda", dtype=dtype)

def benchmark_inference(model_func, sample_input, iterations: int = 10,
                       warmup_iterations: int = 3, device_id: int = 0) -> Dict[str, float]:
    """Benchmark inference performance with detailed metrics
    
    Args:
        model_func: Function that runs the model inference
        sample_input: Sample input for the model
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
        device_id: CUDA device ID to use
        
    Returns:
        Dict with benchmark results:
        - avg_time_ms: Average inference time in milliseconds
        - min_time_ms: Minimum inference time in milliseconds
        - max_time_ms: Maximum inference time in milliseconds
        - std_dev_ms: Standard deviation of inference time
        - throughput: Throughput in inferences per second
        - memory_used_mb: Peak memory used during inference in MB
    """
    result = {
        "avg_time_ms": 0,
        "min_time_ms": 0,
        "max_time_ms": 0,
        "std_dev_ms": 0,
        "throughput": 0,
        "memory_used_mb": 0
    }
    
    if torch.cuda.is_available():
        # Set device context
        torch.cuda.set_device(device_id)
        
        # Record initial memory usage
        torch.cuda.synchronize(device_id)
        initial_memory = torch.cuda.memory_allocated(device_id)
        peak_memory = initial_memory
        
        # Warm-up runs
        for _ in range(warmup_iterations):
            torch.cuda.synchronize(device_id)
            model_func(sample_input)
            torch.cuda.synchronize(device_id)
            
            # Update peak memory
            current_memory = torch.cuda.memory_allocated(device_id)
            peak_memory = max(peak_memory, current_memory)
        
        # Benchmark runs
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize(device_id)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            model_func(sample_input)
            end.record()
            
            torch.cuda.synchronize(device_id)
            times.append(start.elapsed_time(end))
            
            # Update peak memory
            current_memory = torch.cuda.memory_allocated(device_id)
            peak_memory = max(peak_memory, current_memory)
        
        # Calculate memory used
        memory_used = peak_memory - initial_memory
        result["memory_used_mb"] = memory_used / (1024 * 1024)
    else:
        # CPU benchmarking
        import time
        
        # Warm-up runs
        for _ in range(warmup_iterations):
            model_func(sample_input)
        
        # Benchmark runs
        times = []
        for _ in range(iterations):
            start_time = time.time()
            model_func(sample_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    result["avg_time_ms"] = np.mean(times)
    result["min_time_ms"] = np.min(times)
    result["max_time_ms"] = np.max(times)
    result["std_dev_ms"] = np.std(times)
    result["throughput"] = 1000 / result["avg_time_ms"]  # inferences per second
    
    return result

def get_optimal_device_config() -> Dict[str, Any]:
    """Detect and return optimal device configuration for the system
    
    Returns:
        Dict with optimal configuration:
        - use_gpu: Whether to use GPU
        - device_id: Optimal CUDA device ID
        - mixed_precision: Whether to use mixed precision
        - tensor_cores: Whether to use tensor cores
        - num_streams: Optimal number of CUDA streams
        - num_workers: Optimal number of worker threads/processes
        - batch_size: Recommended default batch size
    """
    config = {
        "use_gpu": False,
        "device_id": 0,
        "mixed_precision": False,
        "tensor_cores": False,
        "num_streams": 1,
        "num_workers": max(1, multiprocessing.cpu_count() // 2),
        "batch_size": 1
    }
    
    if not torch.cuda.is_available():
        logger.info("No GPU available, using CPU configuration")
        return config
    
    # Get GPU information
    gpu_info = check_gpu_availability()
    
    if not gpu_info["available"] or gpu_info["device_count"] == 0:
        logger.info("No GPU available, using CPU configuration")
        return config
    
    # We have at least one GPU
    config["use_gpu"] = True
    
    # Find the best GPU (prefer RTX 5090)
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
    
    config["device_id"] = best_device
    
    # Get device properties
    device_props = torch.cuda.get_device_properties(best_device)
    
    # Determine if we can use tensor cores
    has_tensor_cores = (device_props.major >= 7)
    config["tensor_cores"] = has_tensor_cores
    
    # Determine if we can use mixed precision
    has_fp16 = (device_props.major >= 6)
    has_bf16 = (device_props.major >= 8) or (device_props.major == 7 and device_props.minor >= 5)
    config["mixed_precision"] = has_fp16 or has_bf16
    
    # Determine optimal number of CUDA streams
    # Use more streams for GPUs with more SMs
    sm_count = device_props.multi_processor_count
    config["num_streams"] = min(16, max(2, sm_count // 4))
    
    # Determine optimal number of workers
    # Balance between CPU cores and GPU capabilities
    cpu_count = multiprocessing.cpu_count()
    config["num_workers"] = min(cpu_count, max(4, sm_count // 2))
    
    # Determine default batch size (for 1080p images with medium complexity)
    config["batch_size"] = get_optimal_batch_size(1080, 1920, "medium", 0.7, best_device)
    
    logger.info(f"Optimal device configuration detected: GPU {best_device} ({device_props.name})")
    logger.debug(f"Configuration: {config}")
    
    return config

class MultiGPUManager:
    """Manager for multi-GPU processing with load balancing"""
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """Initialize the multi-GPU manager
        
        Args:
            device_ids: List of CUDA device IDs to use, or None to use all available GPUs
        """
        self.available = torch.cuda.is_available()
        
        if not self.available:
            logger.warning("CUDA is not available, MultiGPUManager will operate in CPU-only mode")
            self.device_ids = []
            self.num_devices = 0
            return
        
        # If no device IDs are specified, use all available GPUs
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            # Validate device IDs
            valid_ids = []
            for device_id in device_ids:
                if 0 <= device_id < torch.cuda.device_count():
                    valid_ids.append(device_id)
                else:
                    logger.warning(f"Invalid device ID: {device_id}, ignoring")
            
            self.device_ids = valid_ids
        
        self.num_devices = len(self.device_ids)
        
        if self.num_devices == 0:
            logger.warning("No valid GPU devices found, MultiGPUManager will operate in CPU-only mode")
            return
        
        # Initialize device information
        self.device_info = {}
        for device_id in self.device_ids:
            props = torch.cuda.get_device_properties(device_id)
            self.device_info[device_id] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mb": props.total_memory / (1024 * 1024),
                "multi_processor_count": props.multi_processor_count,
                "max_threads_per_block": props.max_threads_per_block,
                "current_load": 0.0  # Load factor (0.0-1.0)
            }
        
        # Initialize each device
        for device_id in self.device_ids:
            # Initialize device with optimizations
            optimize_for_gpu(device_id)
        
        logger.info(f"MultiGPUManager initialized with {self.num_devices} GPUs: {self.device_ids}")
    
    def get_next_device(self) -> int:
        """Get the next available device with the lowest load
        
        Returns:
            int: Device ID, or -1 if no GPUs are available
        """
        if not self.available or self.num_devices == 0:
            return -1
        
        # Find device with lowest load
        min_load = float('inf')
        best_device = -1
        
        for device_id in self.device_ids:
            load = self.device_info[device_id]["current_load"]
            if load < min_load:
                min_load = load
                best_device = device_id
        
        return best_device
    
    def update_device_load(self, device_id: int, load_factor: float) -> None:
        """Update the load factor for a device
        
        Args:
            device_id: Device ID
            load_factor: Load factor (0.0-1.0)
        """
        if not self.available or device_id not in self.device_ids:
            return
        
        self.device_info[device_id]["current_load"] = max(0.0, min(1.0, load_factor))
    
    def execute_on_optimal_device(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function on the optimal device
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function
        """
        if not self.available or self.num_devices == 0:
            # Execute on CPU
            return func(*args, **kwargs)
        
        # Get the next available device
        device_id = self.get_next_device()
        
        # Update load before execution
        self.update_device_load(device_id, self.device_info[device_id]["current_load"] + 0.1)
        
        try:
            # Set device context
            with torch.cuda.device(device_id):
                # Execute function
                result = func(*args, **kwargs)
            
            return result
        finally:
            # Update load after execution
            self.update_device_load(device_id, self.device_info[device_id]["current_load"] - 0.1)
    
    def parallel_execute(self, func: Callable, inputs: List[Any], batch_size: Optional[int] = None) -> List[Any]:
        """Execute a function in parallel across multiple GPUs
        
        Args:
            func: Function to execute
            inputs: List of inputs to process
            batch_size: Batch size, or None to determine automatically
            
        Returns:
            List[Any]: Results for each input
        """
        if not self.available or self.num_devices == 0 or len(inputs) == 0:
            # Process sequentially on CPU
            return [func(x) for x in inputs]
        
        # Determine batch size if not specified
        if batch_size is None:
            # Simple heuristic: divide inputs evenly across devices
            batch_size = max(1, len(inputs) // (self.num_devices * 2))
        
        # Create batches
        batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
        
        # Process batches in parallel
        results = []
        
        # Use ProcessPoolExecutor for CPU-bound operations
        with ProcessPoolExecutor(max_workers=self.num_devices) as executor:
            # Submit each batch to the executor
            futures = []
            
            for i, batch in enumerate(batches):
                # Assign each batch to a specific device
                device_id = self.device_ids[i % self.num_devices]
                
                # Create a wrapper function that sets the device
                def process_batch_on_device(batch_inputs, device):
                    torch.cuda.set_device(device)
                    return [func(x) for x in batch_inputs]
                
                # Submit the batch
                future = executor.submit(process_batch_on_device, batch, device_id)
                futures.append(future)
            
            # Collect results
            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)
        
        return results
    
    def cleanup(self) -> None:
        """Clean up resources used by the MultiGPUManager"""
        if not self.available:
            return
        
        for device_id in self.device_ids:
            cleanup_gpu(device_id)
        
        logger.info("MultiGPUManager resources cleaned up")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Advanced GPU Utilities for RTX 5090")
    parser.add_argument("--info", action="store_true", help="Display detailed GPU information")
    parser.add_argument("--optimize", action="store_true", help="Apply GPU optimizations")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID to use")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--multi-gpu", action="store_true", help="Test multi-GPU capabilities")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision")
    parser.add_argument("--tensor-cores", action="store_true", help="Enable tensor cores")
    parser.add_argument("--auto-config", action="store_true", help="Detect optimal configuration")
    args = parser.parse_args()
    
    # Process commands
    if args.info or not (args.optimize or args.benchmark or args.multi_gpu or args.auto_config):
        gpu_info = check_gpu_availability()
        
        if gpu_info["available"]:
            print("\nGPU Information:")
            print("===============")
            print(f"CUDA Version: {gpu_info['cuda_version']}")
            print(f"Device Count: {gpu_info['device_count']}")
            
            # Display information for each device
            for i, device in enumerate(gpu_info["devices"]):
                print(f"\nDevice {device['id']}:")
                print(f"  Name: {device['name']}")
                print(f"  Compute Capability: {device['compute_capability']}")
                print(f"  Total Memory: {device['total_memory_mb']:.2f} MB")
                print(f"  Free Memory: {device['free_memory_mb']:.2f} MB")
                print(f"  Multi-Processor Count: {device['multi_processor_count']}")
                print(f"  Has Tensor Cores: {device['has_tensor_cores']}")
                print(f"  Is RTX 5090: {device.get('is_rtx_5090', False)}")
            
            # Calculate optimal batch sizes for different scenarios
            device_id = args.device
            print(f"\nRecommended Batch Sizes (Device {device_id}):")
            print("==============================================")
            print(f"Small images (640x480), low complexity: {get_optimal_batch_size(480, 640, 'low', device_id=device_id)}")
            print(f"Medium images (1280x720), medium complexity: {get_optimal_batch_size(720, 1280, 'medium', device_id=device_id)}")
            print(f"Large images (1920x1080), high complexity: {get_optimal_batch_size(1080, 1920, 'high', device_id=device_id)}")
            print(f"Very large images (3840x2160), very high complexity: {get_optimal_batch_size(2160, 3840, 'very_high', device_id=device_id)}")
            
            # Show mixed precision capabilities
            device_props = torch.cuda.get_device_properties(device_id)
            has_bf16 = device_props.major >= 8 or (device_props.major == 7 and device_props.minor >= 5)
            has_fp16 = device_props.major >= 6
            
            print("\nMixed Precision Support:")
            print("=======================")
            print(f"FP16 Support: {has_fp16}")
            print(f"BF16 Support: {has_bf16}")
        else:
            print("\nNo GPU available. Using CPU mode.")
    
    if args.optimize:
        success = optimize_for_gpu(
            device_id=args.device,
            enable_mixed_precision=args.mixed_precision,
            enable_tensor_cores=args.tensor_cores
        )
        if success:
            print(f"\nAdvanced GPU optimizations applied successfully for device {args.device}.")
        else:
            print(f"\nFailed to apply GPU optimizations for device {args.device}.")
    
    if args.benchmark:
        if torch.cuda.is_available():
            print("\nRunning GPU benchmarks...")
            
            # Create a dummy model function for benchmarking
            def dummy_model(input_tensor):
                # Simulate a model with matrix multiplications
                a = torch.matmul(input_tensor, input_tensor.transpose(0, 1))
                b = torch.matmul(a, input_tensor)
                return torch.nn.functional.softmax(b, dim=1)
            
            # Create sample inputs of different sizes
            sample_sizes = [(32, 1024), (64, 2048), (128, 4096)]
            
            for batch_size, feature_dim in sample_sizes:
                print(f"\nBenchmarking with input size: {batch_size}x{feature_dim}")
                
                # Create input tensor
                sample_input = torch.randn(batch_size, feature_dim, device=f"cuda:{args.device}")
                
                # Run benchmark with and without mixed precision
                results_fp32 = benchmark_inference(
                    dummy_model, sample_input,
                    iterations=20, warmup_iterations=5, device_id=args.device
                )
                
                print(f"FP32 Results:")
                print(f"  Average Time: {results_fp32['avg_time_ms']:.2f} ms")
                print(f"  Min/Max Time: {results_fp32['min_time_ms']:.2f}/{results_fp32['max_time_ms']:.2f} ms")
                print(f"  Throughput: {results_fp32['throughput']:.2f} inferences/sec")
                print(f"  Memory Used: {results_fp32['memory_used_mb']:.2f} MB")
                
                # Test with mixed precision if requested
                if args.mixed_precision:
                    # Set up mixed precision context
                    with setup_mixed_precision():
                        results_mixed = benchmark_inference(
                            dummy_model, sample_input,
                            iterations=20, warmup_iterations=5, device_id=args.device
                        )
                    
                    print(f"Mixed Precision Results:")
                    print(f"  Average Time: {results_mixed['avg_time_ms']:.2f} ms")
                    print(f"  Min/Max Time: {results_mixed['min_time_ms']:.2f}/{results_mixed['max_time_ms']:.2f} ms")
                    print(f"  Throughput: {results_mixed['throughput']:.2f} inferences/sec")
                    print(f"  Memory Used: {results_mixed['memory_used_mb']:.2f} MB")
                    
                    # Calculate speedup
                    speedup = results_fp32['avg_time_ms'] / results_mixed['avg_time_ms']
                    memory_saving = 1.0 - (results_mixed['memory_used_mb'] / results_fp32['memory_used_mb'])
                    
                    print(f"  Speedup: {speedup:.2f}x")
                    print(f"  Memory Saving: {memory_saving*100:.2f}%")
        else:
            print("\nNo GPU available for benchmarking.")
    
    if args.multi_gpu:
        if torch.cuda.device_count() > 1:
            print("\nTesting Multi-GPU capabilities...")
            
            # Initialize MultiGPUManager
            mgpu = MultiGPUManager()
            
            # Create a test function
            def test_function(x):
                # Get current device
                device_id = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_id)
                
                # Simulate some work
                time.sleep(0.1)
                
                return f"Processed on device {device_id} ({device_name}): {x}"
            
            # Test with a few inputs
            inputs = list(range(10))
            
            print(f"Processing {len(inputs)} items across {mgpu.num_devices} GPUs...")
            results = mgpu.parallel_execute(test_function, inputs)
            
            print("\nResults:")
            for result in results:
                print(f"  {result}")
            
            # Clean up
            mgpu.cleanup()
        else:
            print("\nMultiple GPUs not available for testing.")
    
    if args.auto_config:
        print("\nDetecting optimal device configuration...")
        config = get_optimal_device_config()
        
        print("\nOptimal Configuration:")
        print("=====================")
        print(f"Use GPU: {config['use_gpu']}")
        if config['use_gpu']:
            print(f"Device ID: {config['device_id']}")
            print(f"Mixed Precision: {config['mixed_precision']}")
            print(f"Tensor Cores: {config['tensor_cores']}")
            print(f"CUDA Streams: {config['num_streams']}")
            print(f"Worker Threads: {config['num_workers']}")
            print(f"Recommended Batch Size: {config['batch_size']}")