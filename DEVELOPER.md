# AUTO-TAG Developer Documentation

This document provides detailed information for developers working on the AUTO-TAG project. It covers the system architecture, implementation details, and guidelines for future maintenance and development.

## System Architecture

AUTO-TAG follows a modular architecture with the following key components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Interface │────▶│  Core Processing│────▶│  Tagging System │
│  (CLI/Scripts)  │     │  (AI Models)    │     │  (ExifTool)     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Configuration  │     │  Model Manager  │     │  Vendor System  │
│  (YAML)         │     │  (Downloads)    │     │  (ExifTool)     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Key Components

1. **User Interface**: PowerShell scripts (`start.ps1`, `install.ps1`) that provide a CLI interface for users.
2. **Core Processing**: Python modules that handle image analysis using AI models.
3. **Tagging System**: ExifTool wrapper that writes metadata to images.
4. **Configuration**: YAML-based configuration system.
5. **Model Manager**: Handles downloading and managing AI models.
6. **Vendor System**: Manages vendored dependencies like ExifTool.

## Implementation Details

### AI Models

AUTO-TAG uses three main AI models:

1. **CLIP** (`models/clip_model.py`): Used for scene classification and clothing detection.
   - Based on OpenAI's CLIP (Contrastive Language-Image Pre-training)
   - Provides zero-shot classification capabilities
   - Handles scene/indoor/outdoor and clothing status detection

2. **YOLOv8** (`models/yolo_model.py`): Used for object and person detection.
   - Based on Ultralytics' YOLOv8
   - Detects people and counts them
   - Categorizes images as solo or group based on person count

3. **FaceNet** (planned): Will be used for face analysis.
   - Will handle age, gender, and mood detection
   - Will support person identification

All models inherit from the `BaseModel` class (`models/base_model.py`), which provides a common interface for model operations.

### ExifTool Integration

The ExifTool integration (`tagging/exif_wrapper.py`) provides a robust system for writing metadata to images:

1. **Platform Detection**: Automatically detects the operating system and uses the appropriate ExifTool binary.
2. **Fallback Mechanism**: Tries multiple sources for ExifTool:
   - System-installed ExifTool
   - Vendored ExifTool included with the application
   - Custom path specified in configuration
3. **Tag Writing**: Writes tags to the XMP-digiKam:TagsList field in the image metadata.
4. **Error Handling**: Provides robust error handling and logging for ExifTool operations.

### Distribution System

The distribution system (`build_package.py`, `auto-tag.spec`) handles packaging the application for distribution:

1. **PyInstaller**: Uses PyInstaller to create standalone executables.
2. **Resource Bundling**: Bundles all necessary resources, including:
   - Python code
   - AI models
   - ExifTool binaries
   - Configuration files
3. **Platform Support**: Creates packages for Windows (primary), with potential for macOS and Linux support.

## Development Guidelines

### Adding New Features

When adding new features to AUTO-TAG:

1. **Follow the Modular Architecture**: Add new components as separate modules.
2. **Maintain Backward Compatibility**: Ensure existing functionality continues to work.
3. **Update Documentation**: Update both user and developer documentation.
4. **Add Tests**: Write tests for new functionality.

### Adding New Models

To add a new AI model:

1. Create a new model class that inherits from `BaseModel` in `models/base_model.py`.
2. Implement the required methods:
   - `load_model()`: Load the model from disk or download it.
   - `predict()`: Run inference on an image.
   - `cleanup()`: Release resources when done.
3. Add the model to the model catalog in `config/model_catalog.json`.
4. Update the tag generation logic in `process_single.py`.

Example:

```python
from models.base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, model_path=None):
        super().__init__(model_path)
        self.model = None
        
    def load_model(self):
        # Load the model
        self.model = ...
        return True
        
    def predict(self, image_path):
        # Run inference
        result = ...
        return result
        
    def cleanup(self):
        # Release resources
        self.model = None
```

### Extending Tag Schema

To add new tag categories:

1. Update the tag generation logic in `process_single.py`.
2. Add the new category to the documentation in `README.md`.
3. Test with various images to ensure the tags are generated correctly.

Example:

```python
def generate_tags(clip_result, people_result, new_model_result):
    tags = []
    
    # Existing tags
    # ...
    
    # New tags
    if new_model_result and "new_category" in new_model_result:
        new_tag, confidence = new_model_result["new_category"]
        tags.append(f"new_category/{new_tag}")
    
    return tags
```

### Packaging for Distribution

To create a distribution package:

1. Update the version information in:
   - `auto-tag.spec`
   - `file_version_info.txt`
   - `start.ps1`
   - `README.md`
2. Run the build script:
   ```powershell
   python build_package.py --clean
   ```
3. Test the package on a clean system.
4. Create a release with the distribution package.

## Troubleshooting Development Issues

### Common Issues

1. **Model Loading Failures**:
   - Check that the model files exist in the correct location.
   - Verify that the model versions match what's expected.
   - Check for CUDA/GPU compatibility issues.

2. **ExifTool Integration Issues**:
   - Verify that ExifTool is properly installed or vendored.
   - Check file permissions for writing metadata.
   - Test ExifTool directly from the command line.

3. **Packaging Issues**:
   - Ensure all dependencies are properly specified in `auto-tag.spec`.
   - Check for hidden imports that might be missing.
   - Test the package on a clean system without Python installed.

### Debugging

For detailed debugging:

1. Enable debug logging:
   ```powershell
   .\start.ps1 -Debug
   ```

2. Check the log files in the `logs` directory.

3. For PyInstaller issues, use the `--debug` flag:
   ```powershell
   pyinstaller --debug auto-tag.spec
   ```

## Future Development Roadmap

### Planned Features

1. **Face Analysis**: Implement age, gender, and mood detection.
2. **Person Identification**: Add support for identifying specific people.
3. **Batch Processing Improvements**: Enhance performance and reliability.
4. **GUI Interface**: Develop a graphical user interface.
5. **Cloud Integration**: Add support for cloud storage services.

### Technical Debt and Improvements

1. **Code Refactoring**: Improve code organization and reduce duplication.
2. **Test Coverage**: Increase test coverage for core functionality.
3. **Documentation**: Enhance inline documentation and developer guides.
4. **Error Handling**: Improve error handling and user feedback.
5. **Performance Optimization**: Optimize model loading and inference.

## Contributing

Contributions to AUTO-TAG are welcome! Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Make your changes, following the development guidelines.
3. Add or update tests as necessary.
4. Update documentation to reflect your changes.
5. Submit a pull request with a clear description of your changes.

## License

AUTO-TAG is licensed under the MIT License. See the `LICENSE` file for details.