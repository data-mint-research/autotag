# AUTO-TAG

AUTO-TAG is an AI-powered image tagging system that automatically analyzes and tags images with structured metadata. The system uses multiple AI models to detect various aspects of images, including scene classification, person detection, face analysis, and clothing detection.

## Features

- **Scene Classification**: Identify indoor/outdoor scenes and room types
- **Person Detection**: Count people and categorize as solo or group
- **Face Analysis**: Detect age, gender, and mood (coming soon)
- **Clothing Detection**: Classify clothing status
- **Batch Processing**: Process entire folders of images with multi-threading
- **GPU Acceleration**: Optimized for NVIDIA GPUs, especially RTX 5090
- **Structured Tags**: Organized tag hierarchy for better organization
- **MinIO Integration**: Process images from network storage (coming soon)
- **ExifTool Integration**: Robust metadata writing with cross-platform support
- **Standalone Distribution**: Easy installation for end-users without Python knowledge
- **Development Mode**: Full access to source code and Conda environment for developers

## System Requirements

- Windows 11 Pro
- 64 GB RAM recommended
- NVIDIA GPU with CUDA support (RTX 5090 recommended)
- 50 GB free disk space (including models)
- Python 3.10 or newer

## Installation

AUTO-TAG offers two installation modes:

1. **End-User Mode**: A standalone installation that doesn't require Python knowledge
2. **Development Mode**: Full access to source code with a Conda environment for developers

### End-User Installation

The easiest way to install AUTO-TAG for regular users:

1. Download the latest release package from the [Releases page](https://github.com/your-org/autotag/releases)
2. Extract the ZIP file to a location of your choice
3. Run the `install.ps1` script with the `-EndUserMode` parameter:

```powershell
.\install.ps1 -EndUserMode
```

This will:
1. Install AUTO-TAG to `%LOCALAPPDATA%\AUTO-TAG`
2. Create desktop and Start Menu shortcuts
3. Register the application for uninstallation via Windows Control Panel

### Developer Installation

For developers who want to modify the code or contribute to the project:

```powershell
# Clone the repository
git clone https://github.com/your-org/autotag.git
cd autotag

# Run the installation script in development mode
.\install.ps1 -DevMode
```

For users experiencing issues with Conda initialization, especially if your user profile path contains spaces or special characters, you can specify a custom Conda installation path:

```powershell
# Install with a custom Conda path (no spaces or special characters)
.\install.ps1 -DevMode -CustomCondaPath "C:\Miniconda3"
```

The development installation will:
1. Install Miniconda if not already installed (to a custom path if specified)
2. Properly initialize Conda with robust path handling
3. Create a Conda environment with all required dependencies
4. Set up the project structure
5. Create desktop and Start Menu shortcuts

### Manual Installation

If you prefer to install manually:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to a path without spaces or special characters
2. Clone the repository: `git clone https://github.com/your-org/autotag.git`
3. Navigate to the project directory: `cd autotag`
4. Create the Conda environment: `conda env create -f environment.yml`
5. Activate the environment: `conda activate autotag`
6. Run the setup script: `python setup.py develop`

### Uninstallation

To uninstall AUTO-TAG:

```powershell
# For end-user installations
.\uninstall.ps1

# Or use the installation script with the -Uninstall parameter
.\install.ps1 -Uninstall
```

For development installations, you can also remove the Conda environment:

```powershell
conda env remove -n autotag
```

## Usage

### Interactive Mode

The easiest way to use AUTO-TAG is through the interactive menu:

```powershell
# For end-user installations, use the desktop shortcut or Start Menu entry

# For development installations
.\start.ps1
```

This will present a menu with the following options:
1. Tag single image
2. Process folder
3. Process MinIO buckets
4. Check/download models
5. View/edit settings
6. Show version info
7. Exit

### Command Line Mode

You can also use AUTO-TAG directly from the command line:

```powershell
# Process a single image
.\start.ps1 image C:\path\to\image.jpg

# Process a folder
.\start.ps1 folder C:\path\to\folder

# Process a folder recursively
.\start.ps1 folder C:\path\to\folder -Recursive

# Check and download models
.\start.ps1 models

# View and edit settings
.\start.ps1 settings

# Show version information
.\start.ps1 version

# Enable debug logging
.\start.ps1 -Debug
```

### Python API

For developers, you can use AUTO-TAG programmatically:

```python
# Process a single image
from process_single import process_image, write_tags_to_file

result = process_image("path/to/image.jpg")
print(result["tags"])

# Write tags to the image
write_tags_to_file("path/to/image.jpg", result["tags"], mode="append")

# Process a folder of images
from batch_process import batch_process

results = batch_process(
    input_folder="path/to/folder",
    recursive=True,
    max_workers=4
)
```

## Configuration

AUTO-TAG uses a YAML configuration file located at `config/settings.yaml`. You can edit this file directly or use the settings option in the interactive menu.

Key configuration options:

```yaml
# Paths
paths:
  input_folder: "./data/input"
  output_folder: "./data/output"
  models_dir: "./models"

# Hardware Settings
hardware:
  use_gpu: true
  cuda_device_id: 0
  num_workers: 4
  batch_size: 8

# Model Settings
models:
  auto_download: true
  force_update: false
  offline_mode: false

# Tagging Settings
tagging:
  mode: "append"
  exiftool_path: ""  # Leave empty to use auto-detection
```

### Advanced Configuration

#### ExifTool Configuration

AUTO-TAG uses ExifTool for writing metadata to images. The system will:

1. Try to use a system-installed ExifTool if available
2. Fall back to the vendored ExifTool included with AUTO-TAG
3. Allow specifying a custom ExifTool path in the configuration

To specify a custom ExifTool path:

```yaml
# Tagging Settings
tagging:
  exiftool_path: "C:/path/to/your/exiftool.exe"
```

#### GPU Optimization

For optimal performance with NVIDIA GPUs:

```yaml
# Hardware Settings
hardware:
  use_gpu: true
  cuda_device_id: 0  # Use the first GPU
  enable_tf32: true  # Enable TensorFloat-32 for faster computation on RTX cards
  optimize_memory: true
```

## Tag Schema

AUTO-TAG generates structured tags in the following categories:

1. `scene/indoor|outdoor`: Whether the image is indoor or outdoor
2. `roomtype/kitchen|bathroom|bedroom|living_room|office`: Type of room (for indoor scenes)
3. `clothing/dressed|naked`: Clothing status
4. `people/solo|group`: Whether the image contains a single person or multiple people

Additional tag categories coming soon:
- `person/NAME`: Person identification
- `gender/male|female`: Gender classification
- `age/20s|30s|...`: Age estimation
- `mood/happy|unhappy|...`: Mood detection

### Metadata Format

AUTO-TAG writes tags to the XMP-digiKam:TagsList field in the image metadata. This format is compatible with:

- digiKam
- Adobe Lightroom
- Adobe Bridge
- XnView
- Many other photo management applications

The tags are written in a hierarchical format (e.g., `scene/indoor`) that allows for better organization and filtering in photo management software.

## Distribution and Packaging

### Creating a Distribution Package

Developers can create a standalone distribution package for end-users:

```powershell
# Run the build script
python build_package.py

# Create a single-file executable
python build_package.py --onefile

# Skip downloading ExifTool (use existing)
python build_package.py --skip-exiftool

# Clean build directories before building
python build_package.py --clean
```

The build script will:
1. Download and package ExifTool for all supported platforms
2. Run PyInstaller to create the distribution package
3. Package the result into a ZIP file for distribution

The distribution package will be created in the `dist` directory.

### Offline Mode

AUTO-TAG can operate in offline mode without internet access. To enable offline mode:

1. Download the model files manually
2. Place them in the `models` directory with the correct structure:
   - `models/clip/clip_vit_b32.pth`
   - `models/yolov8/yolov8n.pt`
   - `models/facenet/facenet_model.pth`
3. Set `offline_mode: true` in the configuration file

You can also use the installation script with the `-OfflineMode` parameter:

```powershell
.\install.ps1 -OfflineMode -ModelsPath "C:\path\to\downloaded\models"
```

## Developer Guide

### Project Structure

```
AUTO-TAG/
├── config/                 # Configuration files
│   ├── model_catalog.json  # Model metadata and download URLs
│   └── settings.yaml       # User configuration
├── models/                 # AI models
│   ├── base_model.py       # Base model interface
│   ├── clip_model.py       # CLIP model implementation
│   ├── yolo_model.py       # YOLOv8 model implementation
│   ├── gpu_utils.py        # GPU optimization utilities
│   └── model_manager.py    # Model download and management
├── tagging/                # Tagging functionality
│   ├── __init__.py         # Package initialization
│   └── exif_wrapper.py     # ExifTool wrapper for metadata
├── resources/              # Application resources
│   ├── download_exiftool.py # ExifTool downloader
│   └── icon.ico            # Application icon
├── vendor/                 # Vendored dependencies
│   └── bin/                # Binary executables (ExifTool)
├── auto-tag.spec           # PyInstaller specification
├── build_package.py        # Distribution packaging script
├── process_single.py       # Single image processing
├── batch_process.py        # Batch processing
├── install.ps1             # Installation script
├── start.ps1               # Startup script
├── uninstall.ps1           # Uninstallation script (generated)
└── environment.yml         # Conda environment specification
```

### Adding New Models

To add a new AI model:

1. Create a new model class that inherits from `BaseModel` in `models/base_model.py`
2. Implement the required methods: `load_model()`, `predict()`, and `cleanup()`
3. Add the model to the model catalog in `config/model_catalog.json`
4. Update the tag generation logic in `process_single.py`

### ExifTool Integration

The ExifTool wrapper in `tagging/exif_wrapper.py` provides:

1. Automatic detection of system-installed ExifTool
2. Platform-specific vendored ExifTool fallback
3. Tag reading and writing functionality
4. Error handling and logging

## Troubleshooting

### Conda Installation Issues

If you encounter issues with Conda initialization during installation:

1. **User Profile Path Issues**:
   - If your Windows username or profile path contains spaces or special characters, use the `-CustomCondaPath` parameter:
     ```powershell
     .\install.ps1 -DevMode -CustomCondaPath "C:\Miniconda3"
     ```
   - This installs Miniconda to a path without spaces, avoiding common initialization problems

2. **Manual Conda Initialization**:
   - If Conda initialization fails, you may need to manually initialize Conda
   - Run `conda init powershell` from an administrator PowerShell prompt
   - Restart your PowerShell session

3. **Conda Environment Not Found**:
   - If you get "conda: command not found" or similar errors:
     - Ensure Conda is in your PATH environment variable
     - Try reinstalling Miniconda

4. **PowerShell Execution Policy**:
   - If scripts fail to run due to execution policy restrictions:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
   - Then run the installation script again

### Models Not Downloading

If models fail to download:
1. Check your internet connection
2. Try running `.\start.ps1 models` to manually download models
3. Enable offline mode and download models manually
4. Check the model catalog in `config/model_catalog.json` for correct URLs

### GPU Not Detected

If your GPU is not being detected:
1. Ensure you have the latest NVIDIA drivers installed
2. Check that CUDA is properly installed
3. Set `use_gpu: false` in the configuration to use CPU mode
4. Run with debug logging: `.\start.ps1 -Debug`

### ExifTool Issues

If you encounter problems with ExifTool:
1. Check if ExifTool is installed on your system
2. Verify that the vendored ExifTool is properly extracted
3. Set a custom ExifTool path in the configuration
4. Check the logs for specific error messages

### Installation Log

If you encounter any installation issues, check the installation log:
```powershell
Get-Content .\install.log
```
This log contains detailed information about each step of the installation process and can help identify where problems occurred.

## License

[MIT License](LICENSE)

## Acknowledgments

- [ExifTool](https://exiftool.org/) by Phil Harvey
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [CLIP](https://github.com/openai/CLIP) for scene classification
- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [PyInstaller](https://pyinstaller.org/) for application packaging