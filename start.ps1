# start.ps1 - Startup script for AUTO-TAG
[CmdletBinding()]
param (
    [Parameter(Position=0)]
    [ValidateSet("image", "folder", "minio", "models", "settings", "version", "")]
    [string]$Mode = "",
    
    [Parameter(Position=1)]
    [string]$Path = "",
    
    [switch]$Recursive,
    [switch]$SkipModelCheck,
    [switch]$Help,
    [switch]$DevMode,
    [switch]$DebugMode,
    [string]$LogLevel = "INFO"
)

# Application version
$AppVersion = "1.0.0"

# Basic functions
# ----------------------

function Show-Help {
    Write-Host "`nAUTO-TAG System - Help" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host "`nUsage:`n"
    Write-Host "  .\start.ps1                         # Start interactive menu"
    Write-Host "  .\start.ps1 image [image_path]      # Process a single image"
    Write-Host "  .\start.ps1 folder [folder_path]    # Process a folder"
    Write-Host "  .\start.ps1 minio                   # Process MinIO buckets"
    Write-Host "  .\start.ps1 models                  # Check/download models"
    Write-Host "  .\start.ps1 settings                # View/edit settings"
    Write-Host "`nOptions:`n"
    Write-Host "  -Recursive                          # Include subfolders when processing a folder"
    Write-Host "  -SkipModelCheck                     # Skip model verification on startup"
    Write-Host "  -Help                               # Show this help"
    Write-Host "`nExamples:`n"
    Write-Host "  .\start.ps1 image C:\Pictures\photo.jpg"
    Write-Host "  .\start.ps1 folder C:\Pictures -Recursive"
}

function Initialize-Environment {
    $ScriptPath = $PSScriptRoot
    $CondaBasePath = "$env:USERPROFILE\miniconda3"
    $EnvName = "autotag"
    
    # Set up logging based on LogLevel parameter
    if ($DebugMode) {
        $LogLevel = "DEBUG"
    }
    
    $LogLevelMap = @{
        "DEBUG" = "Debug"
        "INFO" = "Cyan"
        "WARNING" = "Yellow"
        "ERROR" = "Red"
        "CRITICAL" = "Red"
    }
    
    function Write-Log {
        param (
            [string]$Message,
            [string]$Level = "INFO"
        )
        
        if ($LogLevelMap.ContainsKey($Level)) {
            $Color = $LogLevelMap[$Level]
            Write-Host "[$Level] $Message" -ForegroundColor $Color
        }
        else {
            Write-Host "[$Level] $Message"
        }
    }
    
    # Determine if we're running in development or end-user mode
    if (-not $DevMode) {
        # Check if we're in an installed location
        $InstallDir = "$env:LOCALAPPDATA\AUTO-TAG"
        if (Test-Path "$InstallDir\AUTO-TAG.exe") {
            Write-Log "Running in end-user mode from $InstallDir" -Level "DEBUG"
            $EndUserMode = $true
        }
        else {
            # Check if we have a Conda environment
            $condaExists = Get-Command "conda" -ErrorAction SilentlyContinue
            if ($condaExists -or (Test-Path $CondaBasePath)) {
                Write-Log "Running in development mode" -Level "DEBUG"
                $DevMode = $true
            }
            else {
                Write-Log "No installation detected. Assuming end-user mode." -Level "WARNING"
                $EndUserMode = $true
            }
        }
    }
    
    # Handle development mode initialization
    if ($DevMode) {
        # Check if Conda is installed
        $condaExists = Get-Command "conda" -ErrorAction SilentlyContinue
        if (-not $condaExists -and -not (Test-Path $CondaBasePath)) {
            Write-Log "Error: Conda not found. Please run install.ps1 first." -Level "ERROR"
            exit 1
        }
        
        # Add Conda to path for this session if needed
        if (-not $condaExists) {
            $env:PATH = "$CondaBasePath\Scripts;$CondaBasePath;$env:PATH"
        }
        
        # Check if environment exists
        $envExists = & conda env list | Select-String -Pattern "^$EnvName\s"
        if (-not $envExists) {
            Write-Log "Error: Conda environment '$EnvName' not found. Please run install.ps1 first." -Level "ERROR"
            exit 1
        }
        
        # Activate Conda environment
        & "$CondaBasePath\Scripts\activate.bat" $EnvName
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Error: Failed to activate Conda environment '$EnvName'." -Level "ERROR"
            exit 1
        }
    }
    
    # Check if configuration exists
    $configFile = Join-Path -Path $ScriptPath -ChildPath "config\settings.yaml"
    if (-not (Test-Path $configFile)) {
        Write-Log "Warning: Configuration file not found. Creating default configuration..." -Level "WARNING"
        
        # Create config directory if it doesn't exist
        $configDir = Join-Path -Path $ScriptPath -ChildPath "config"
        if (-not (Test-Path $configDir)) {
            New-Item -Path $configDir -ItemType Directory -Force | Out-Null
        }
        
        # Create basic settings file
        @"
# AUTO-TAG Configuration

# Paths
paths:
  input_folder: "./data/input"
  output_folder: "./data/output"
  models_dir: "./models"
  temp_dir: "./temp"

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
  exiftool_path: ""
"@ | Out-File -FilePath $configFile -Encoding utf8
    }
    
    # Check if required directories exist
    $requiredDirs = @(
        (Join-Path -Path $ScriptPath -ChildPath "models"),
        (Join-Path -Path $ScriptPath -ChildPath "data\input"),
        (Join-Path -Path $ScriptPath -ChildPath "data\output"),
        (Join-Path -Path $ScriptPath -ChildPath "logs")
    )
    
    foreach ($dir in $requiredDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            Write-Log "Created directory: $dir" -Level "INFO"
        }
    }
    
    # Check for vendor directory and create if needed
    $vendorDir = Join-Path -Path $ScriptPath -ChildPath "vendor"
    if (-not (Test-Path $vendorDir)) {
        New-Item -Path $vendorDir -ItemType Directory -Force | Out-Null
        New-Item -Path "$vendorDir\bin" -ItemType Directory -Force | Out-Null
        Write-Log "Created vendor directory: $vendorDir" -Level "INFO"
    }
    
    # Return environment info
    return @{
        ScriptPath = $ScriptPath
        DevMode = $DevMode
        EndUserMode = $EndUserMode
        ConfigFile = $configFile
    }
}

function Select-FilePath {
    param (
        [string]$Title = "Select File",
        [string]$Filter = "Images (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png"
    )
    
    Add-Type -AssemblyName System.Windows.Forms
    $openFileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $openFileDialog.Filter = $Filter
    $openFileDialog.Title = $Title
    $openFileDialog.Multiselect = $false
    
    if ($openFileDialog.ShowDialog() -eq "OK") {
        return $openFileDialog.FileName
    }
    
    return $null
}

function Select-FolderPath {
    param (
        [string]$Description = "Select Folder"
    )
    
    Add-Type -AssemblyName System.Windows.Forms
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderBrowser.Description = $Description
    
    if ($folderBrowser.ShowDialog() -eq "OK") {
        return $folderBrowser.SelectedPath
    }
    
    return $null
}

function Invoke-ProcessImage {
    param (
        [string]$ImagePath
    )
    
    if (-not $ImagePath) {
        $ImagePath = Select-FilePath -Title "Select image to tag"
        if (-not $ImagePath) {
            Write-Host "Cancelled." -ForegroundColor Yellow
            return
        }
    }
    
    if (-not (Test-Path $ImagePath)) {
        Write-Host "Error: Image file not found: $ImagePath" -ForegroundColor Red
        return
    }
    
    Write-Host "`nProcessing image: $ImagePath" -ForegroundColor Cyan
    
    # Run image processing
    python "$PSScriptRoot\process_single.py" --input "$ImagePath"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nImage successfully processed!" -ForegroundColor Green
    }
    else {
        Write-Host "`nError processing image." -ForegroundColor Red
    }
}

function Invoke-ProcessFolder {
    param (
        [string]$FolderPath,
        [bool]$ProcessRecursively
    )
    
    if (-not $FolderPath) {
        $FolderPath = Select-FolderPath -Description "Select folder with images"
        if (-not $FolderPath) {
            Write-Host "Cancelled." -ForegroundColor Yellow
            return
        }
        
        # If no path was provided, ask about recursion
        if (-not $ProcessRecursively) {
            $recursiveChoice = Read-Host "Include subfolders? (y/n)"
            $ProcessRecursively = $recursiveChoice -eq "y"
        }
    }
    
    if (-not (Test-Path $FolderPath)) {
        Write-Host "Error: Folder not found: $FolderPath" -ForegroundColor Red
        return
    }
    
    Write-Host "`nProcessing folder: $FolderPath" -ForegroundColor Cyan
    if ($ProcessRecursively) {
        Write-Host "Including subfolders." -ForegroundColor Cyan
    }
    
    # Run folder processing
    if ($ProcessRecursively) {
        python "$PSScriptRoot\batch_process.py" --input "$FolderPath" --recursive
    }
    else {
        python "$PSScriptRoot\batch_process.py" --input "$FolderPath"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nFolder successfully processed!" -ForegroundColor Green
    }
    else {
        Write-Host "`nError processing folder." -ForegroundColor Red
    }
}

function Invoke-ProcessMinio {
    Write-Host "`nStarting MinIO processing..." -ForegroundColor Cyan
    
    # Run MinIO processing
    python "$PSScriptRoot\process_minio.py"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nMinIO buckets successfully processed!" -ForegroundColor Green
    }
    else {
        Write-Host "`nError processing MinIO buckets." -ForegroundColor Red
    }
}

function Invoke-ManageModels {
    Write-Host "`nStarting model management..." -ForegroundColor Cyan
    
    # Run model manager
    python "$PSScriptRoot\models\model_manager.py" --check
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nModel check completed." -ForegroundColor Green
    }
    else {
        Write-Host "`nError checking models." -ForegroundColor Red
    }
}

function Invoke-ManageSettings {
    Write-Host "`nConfiguration..." -ForegroundColor Cyan
    
    # Show current configuration
    python "$PSScriptRoot\config_loader.py"
    
    # Ask if settings should be edited
    $editChoice = Read-Host "`nWould you like to edit the settings.yaml file? (y/n)"
    if ($editChoice -eq "y") {
        # Open settings.yaml in default editor
        $settingsFile = Join-Path -Path $PSScriptRoot -ChildPath "config\settings.yaml"
        Start-Process notepad $settingsFile -Wait
        Write-Host "Settings saved. Restart AUTO-TAG to apply changes." -ForegroundColor Green
    }
}

function Show-Menu {
    Clear-Host
    Write-Host "`n=============================" -ForegroundColor Cyan
    Write-Host "      AUTO-TAG SYSTEM      " -ForegroundColor Cyan
    Write-Host "      Version $AppVersion      " -ForegroundColor Cyan
    Write-Host "=============================" -ForegroundColor Cyan
    
    # Check GPU status
    try {
        $gpuAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
        if ($gpuAvailable -eq "True") {
            $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
            Write-Host "GPU: $gpuName" -ForegroundColor Green
        }
        else {
            Write-Host "GPU: Not available (using CPU mode)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "GPU: Status unknown" -ForegroundColor Yellow
    }
    
    # Show mode
    if ($DevMode) {
        Write-Host "Mode: Development" -ForegroundColor Magenta
    }
    else {
        Write-Host "Mode: End-User" -ForegroundColor Magenta
    }
    
    Write-Host "`n1. Tag single image"
    Write-Host "2. Process folder"
    Write-Host "3. Process MinIO buckets"
    Write-Host "4. Check/download models"
    Write-Host "5. View/edit settings"
    Write-Host "6. Show version info"
    Write-Host "7. Exit"
    
    $choice = Read-Host "`nSelect an option (1-7)"
    
    return $choice
}

function Show-VersionInfo {
    Write-Host "`n=============================" -ForegroundColor Cyan
    Write-Host "      AUTO-TAG VERSION      " -ForegroundColor Cyan
    Write-Host "=============================" -ForegroundColor Cyan
    Write-Host "Version: $AppVersion" -ForegroundColor White
    Write-Host "Build Date: April 17, 2025" -ForegroundColor White
    
    # Show environment info
    if ($DevMode) {
        Write-Host "`nRunning in Development Mode" -ForegroundColor Magenta
        
        # Show Python version
        try {
            $pythonVersion = python -c "import sys; print(f'Python {sys.version.split()[0]}')" 2>$null
            Write-Host "Python: $pythonVersion" -ForegroundColor White
        }
        catch {
            Write-Host "Python: Not detected" -ForegroundColor Yellow
        }
        
        # Show PyTorch version
        try {
            $torchVersion = python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>$null
            Write-Host "PyTorch: $torchVersion" -ForegroundColor White
        }
        catch {
            Write-Host "PyTorch: Not installed" -ForegroundColor Yellow
        }
        
        # Show CUDA version
        try {
            $cudaVersion = python -c "import torch; print(f'CUDA {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')" 2>$null
            Write-Host "CUDA: $cudaVersion" -ForegroundColor White
        }
        catch {
            Write-Host "CUDA: Not detected" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "`nRunning in End-User Mode" -ForegroundColor Magenta
        Write-Host "Installation Directory: $($env:LOCALAPPDATA)\AUTO-TAG" -ForegroundColor White
    }
    
    # Show ExifTool version
    try {
        $exiftoolVersion = python -c "from tagging.exif_wrapper import get_tag_writer; tw = get_tag_writer(); print('ExifTool: ' + ('Available' if tw.exiftool_path else 'Not available'))" 2>$null
        Write-Host $exiftoolVersion -ForegroundColor White
    }
    catch {
        Write-Host "ExifTool: Status unknown" -ForegroundColor Yellow
    }
    
    Write-Host "`nCopyright © 2025 AUTO-TAG Team" -ForegroundColor White
    Write-Host "All rights reserved." -ForegroundColor White
}

# Main program
# ------------

# Show help if requested
if ($Help) {
    Show-Help
    exit 0
}

# Initialize environment
Initialize-Environment

# Check models if not skipped
if (-not $SkipModelCheck -and $Mode -ne "models") {
    Write-Host "`nChecking models..." -ForegroundColor Yellow
    python "$PSScriptRoot\models\model_manager.py" --check
}

# If no mode specified, show menu
if (-not $Mode) {
    $running = $true
    
    while ($running) {
        $choice = Show-Menu
        
        switch ($choice) {
            "1" { Invoke-ProcessImage }
            "2" { Invoke-ProcessFolder -ProcessRecursively $Recursive }
            "3" { Invoke-ProcessMinio }
            "4" { Invoke-ManageModels }
            "5" { Invoke-ManageSettings }
            "6" { Show-VersionInfo }
            "7" { $running = $false }
            default {
                Write-Host "`nInvalid input. Please select 1-7." -ForegroundColor Yellow
                Start-Sleep -Seconds 1
            }
        }
        
        if ($running -and $choice -in 1..6) {
            Write-Host "`nPress any key to continue..."
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
    }
    
    Write-Host "`nExiting AUTO-TAG. Goodbye!" -ForegroundColor Cyan
    exit 0
}

# Process according to specified mode
switch ($Mode) {
    "image" {
        Invoke-ProcessImage -ImagePath $Path
    }
    "folder" {
        Invoke-ProcessFolder -FolderPath $Path -ProcessRecursively $Recursive
    }
    "minio" {
        Invoke-ProcessMinio
    }
    "models" {
        Invoke-ManageModels
    }
    "settings" {
        Invoke-ManageSettings
    }
    "version" {
        Show-VersionInfo
    }
    default {
        Write-Host "Unknown mode: $Mode" -ForegroundColor Red
        Show-Help
        exit 1
    }
}

exit 0