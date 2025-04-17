# AUTO-TAG User Guide

This guide provides step-by-step instructions for common workflows in AUTO-TAG. It's designed to help you get the most out of the application and understand its features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Tagging a Single Image](#tagging-a-single-image)
3. [Processing a Folder of Images](#processing-a-folder-of-images)
4. [Working with Models](#working-with-models)
5. [Configuring AUTO-TAG](#configuring-auto-tag)
6. [Understanding Tags](#understanding-tags)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. **End-User Installation**:
   - Download the latest release package from the [Releases page](https://github.com/your-org/autotag/releases)
   - Extract the ZIP file to a location of your choice
   - Run the `install.ps1` script:
     ```powershell
     .\install.ps1 -EndUserMode
     ```
   - Follow the on-screen instructions to complete the installation

2. **Developer Installation**:
   - Clone the repository and navigate to the project directory
   - Run the installation script in development mode:
     ```powershell
     .\install.ps1 -DevMode
     ```
   - If your Windows username contains spaces or special characters, use a custom Conda path:
     ```powershell
     .\install.ps1 -DevMode -CustomCondaPath "C:\Miniconda3"
     ```
   - This avoids common Conda initialization issues by installing to a path without spaces

3. **Starting AUTO-TAG**:
   - Use the desktop shortcut created during installation
   - Or use the Start Menu entry: Start → Programs → AUTO-TAG → AUTO-TAG
   - Or run directly from the installation directory:
     ```powershell
     %LOCALAPPDATA%\AUTO-TAG\AUTO-TAG.exe
     ```
   - For development installations, use:
     ```powershell
     .\start.ps1
     ```

### First Run

When you first run AUTO-TAG, it will:

1. Create the necessary directory structure
2. Generate a default configuration file
3. Check for required AI models and download them if needed

This initial setup may take a few minutes, especially if models need to be downloaded.

## Tagging a Single Image

To tag a single image:

1. Start AUTO-TAG
2. From the main menu, select option `1. Tag single image`
3. In the file selection dialog, browse to and select the image you want to tag
4. AUTO-TAG will process the image and display the results:
   - The tags that were added
   - The processing time
   - Any errors or warnings

Example output:
```
✓ Image successfully processed: vacation.jpg
Processing time: 1.25 seconds
Tags added: 3
Tags: scene/outdoor, people/group, clothing/dressed
```

By default, tags are appended to any existing tags in the image. To overwrite existing tags instead:

```powershell
.\start.ps1 image C:\path\to\image.jpg --tag-mode overwrite
```

## Processing a Folder of Images

To process an entire folder of images:

1. Start AUTO-TAG
2. From the main menu, select option `2. Process folder`
3. In the folder selection dialog, browse to and select the folder containing your images
4. When prompted, choose whether to include subfolders
5. AUTO-TAG will process all images in the folder and display progress:
   - A progress bar showing completion percentage
   - Estimated time remaining
   - Success/failure counts

For large folders, batch processing can take some time. AUTO-TAG automatically:

- Uses multiple threads for parallel processing
- Optimizes GPU memory usage
- Creates checkpoints to allow resuming if interrupted

To process a folder from the command line:

```powershell
# Process a folder
.\start.ps1 folder C:\path\to\folder

# Process a folder recursively (including subfolders)
.\start.ps1 folder C:\path\to\folder -Recursive
```

## Working with Models

AUTO-TAG uses AI models for image analysis. To manage these models:

1. Start AUTO-TAG
2. From the main menu, select option `4. Check/download models`

This will:
- Check if all required models are present
- Verify model integrity
- Download any missing models
- Update models if newer versions are available

### Offline Mode

If you need to use AUTO-TAG without internet access:

1. First, download all required models on a computer with internet access:
   ```powershell
   .\start.ps1 models
   ```

2. Copy the entire `models` directory to your offline computer

3. Enable offline mode in the configuration:
   - Edit `config\settings.yaml`
   - Set `models.offline_mode: true`

## Configuring AUTO-TAG

To view or edit AUTO-TAG settings:

1. Start AUTO-TAG
2. From the main menu, select option `5. View/edit settings`
3. The current configuration will be displayed
4. When prompted, choose whether to edit the settings
5. If you choose to edit, the settings file will open in Notepad

### Key Configuration Options

#### Paths

```yaml
paths:
  input_folder: "./data/input"    # Default input folder
  output_folder: "./data/output"  # Default output folder
  models_dir: "./models"          # Where models are stored
```

#### Hardware Settings

```yaml
hardware:
  use_gpu: true           # Set to false to use CPU only
  cuda_device_id: 0       # GPU device ID (0 for first GPU)
  num_workers: 4          # Number of parallel workers
  batch_size: 8           # Batch size for processing
```

#### Model Settings

```yaml
models:
  auto_download: true     # Automatically download missing models
  force_update: false     # Force update of existing models
  offline_mode: false     # Operate without internet access
```

#### Tagging Settings

```yaml
tagging:
  mode: "append"          # "append" or "overwrite"
  exiftool_path: ""       # Custom ExifTool path (leave empty for auto)
```

## Understanding Tags

AUTO-TAG generates structured tags in the following categories:

### Scene Tags

- `scene/indoor`: Indoor scenes
- `scene/outdoor`: Outdoor scenes

### Room Type Tags (for indoor scenes)

- `roomtype/kitchen`
- `roomtype/bathroom`
- `roomtype/bedroom`
- `roomtype/living_room`
- `roomtype/office`

### People Tags

- `people/solo`: Single person
- `people/group`: Multiple people

### Clothing Tags

- `clothing/dressed`
- `clothing/naked`

### Viewing Tags

To view the tags that have been added to an image:

1. Use a photo management application that supports XMP metadata
2. Or use ExifTool directly:
   ```powershell
   exiftool -XMP-digiKam:TagsList -s -s -s C:\path\to\image.jpg
   ```

## Troubleshooting

### Installation Issues

#### Conda Initialization Problems

If you encounter issues with Conda initialization during installation:

1. **User Profile Path Issues**:
   - If your Windows username or profile path contains spaces or special characters, this can cause Conda initialization problems
   - Solution: Use the `-CustomCondaPath` parameter during installation:
     ```powershell
     .\install.ps1 -DevMode -CustomCondaPath "C:\Miniconda3"
     ```
   - This installs Miniconda to a path without spaces, avoiding common initialization issues

2. **Manual Conda Initialization**:
   - If Conda initialization fails, you may need to manually initialize Conda:
     ```powershell
     conda init powershell
     ```
   - Then restart your PowerShell session
   - This will properly set up Conda in your PowerShell profile

3. **Check Conda Installation**:
   - If you're having issues with Conda:
     * Verify Conda is in your PATH environment variable
     * Check if your user profile path contains spaces or special characters
     * Ensure PowerShell execution policy is set correctly
     * Verify Conda initialization is in your PowerShell profile

4. **PowerShell Execution Policy**:
   - If scripts fail to run due to execution policy restrictions:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
   - Then run the installation script again

5. **Installation Log**:
   - Check the installation log for detailed information:
     ```powershell
     Get-Content .\install.log
     ```

### Common Issues

#### Models Not Downloading

If models fail to download:

1. Check your internet connection
2. Try running `.\start.ps1 models` to manually download models
3. Check the log file at `logs\auto-tag.log` for specific errors
4. Try downloading the models manually and placing them in the `models` directory

#### GPU Not Detected

If your GPU is not being detected:

1. Ensure you have the latest NVIDIA drivers installed
2. Check that CUDA is properly installed
3. Set `use_gpu: false` in the configuration to use CPU mode
4. Run with debug logging: `.\start.ps1 -Debug`

#### Tags Not Being Written

If tags are not being written to images:

1. Check if the image format supports metadata (JPEG, TIFF, PNG do; some others don't)
2. Verify that you have write permissions for the image files
3. Check if ExifTool is properly installed or vendored
4. Try running with debug logging: `.\start.ps1 -Debug`

### Getting Help

If you encounter issues not covered in this guide:

1. Check the log file at `logs\auto-tag.log` for detailed error information
2. Run AUTO-TAG with debug logging: `.\start.ps1 -Debug`
3. Check the [GitHub Issues](https://github.com/your-org/autotag/issues) for similar problems
4. Submit a new issue with:
   - A description of the problem
   - Steps to reproduce
   - Log file contents
   - System information (OS, GPU, etc.)

## Advanced Usage

### Command Line Options

AUTO-TAG supports various command line options:

```powershell
# Show help
.\start.ps1 -Help

# Process a single image
.\start.ps1 image C:\path\to\image.jpg

# Process a folder recursively
.\start.ps1 folder C:\path\to\folder -Recursive

# Skip model check on startup
.\start.ps1 -SkipModelCheck

# Enable debug logging
.\start.ps1 -Debug

# Show version information
.\start.ps1 version
```

### Batch Processing Options

For batch processing, you can customize:

```powershell
# Specify number of parallel workers
.\start.ps1 folder C:\path\to\folder --workers 8

# Resume from a checkpoint
.\start.ps1 folder C:\path\to\folder --resume

# Dry run (list files without processing)
.\start.ps1 folder C:\path\to\folder --dry-run
```

### Using AUTO-TAG with Other Applications

AUTO-TAG writes tags in a format compatible with many photo management applications:

- **digiKam**: Tags appear in the Tags panel
- **Adobe Lightroom**: Tags appear in the Keywords panel
- **Adobe Bridge**: Tags appear in the Keywords panel
- **XnView**: Tags appear in the Categories panel

You can integrate AUTO-TAG into your workflow by:

1. Processing images with AUTO-TAG first
2. Importing the tagged images into your photo management application
3. Using the generated tags to organize and search your photo collection