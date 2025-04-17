# install.ps1 - Installation script for AUTO-TAG
[CmdletBinding()]
param (
    [switch]$OfflineMode,
    [switch]$MinimalInstall,
    [switch]$ForceReinstall,
    [string]$ModelsPath,
    [switch]$SkipPrompts,
    [switch]$Uninstall,
    [switch]$DevMode,
    [switch]$EndUserMode,
    [string]$InstallDir,
    [string]$CustomCondaPath,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptPath = $PSScriptRoot
# Default Conda path - will be overridden if CustomCondaPath is provided
$CondaBasePath = if ($CustomCondaPath) { $CustomCondaPath } else { "C:\Miniconda3" }
$EnvName = "autotag"
$LogFile = "$ScriptPath\install.log"
$InstallDir = if ($InstallDir) { $InstallDir } else { "$env:LOCALAPPDATA\AUTO-TAG" }
$UninstallRegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\AUTO-TAG"
$AppVersion = "1.0.0"

# Color definitions
$colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Title = "Magenta"
}

# Create log file
if (-not (Test-Path $LogFile)) {
    New-Item -Path $LogFile -ItemType File -Force | Out-Null
}

function Write-LogMessage {
    param (
        [string]$Message,
        [string]$Level = "INFO",
        [string]$Color = $colors.Info
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Level : $Message" -ForegroundColor $Color
    
    # Also write to log file
    Add-Content -Path $LogFile -Value "[$timestamp] $Level : $Message"
}

function Show-Header {
    Clear-Host
    Write-Host "`n====================================" -ForegroundColor $colors.Title
    Write-Host "       AUTO-TAG INSTALLATION        " -ForegroundColor $colors.Title
    Write-Host "====================================`n" -ForegroundColor $colors.Title
}

function Check-Administrator {
    $currentUser = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    $isAdmin = $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        Write-LogMessage "This installation should be run as Administrator." -Level "WARNING" -Color $colors.Warning
        
        if (-not $SkipPrompts) {
            $continue = Read-Host "Continue without admin rights? (y/n)"
            if ($continue -ne "y") {
                Write-LogMessage "Installation aborted" -Level "INFO" -Color $colors.Info
                exit 1
            }
        }
    }
    else {
        Write-LogMessage "Running with Administrator rights" -Level "INFO" -Color $colors.Success
    }
}

function Test-CondaInstallation {
    Write-LogMessage "Checking Conda installation..." -Level "INFO" -Color $colors.Info
    
    # Check if conda command is available
    $condaExists = Get-Command "conda" -ErrorAction SilentlyContinue
    if ($condaExists) {
        Write-LogMessage "Conda is already available in PATH" -Level "SUCCESS" -Color $colors.Success
        
        # Get the actual path to conda command
        $condaPath = $condaExists.Source
        
        # Extract the base Conda path
        if ($condaPath -like "*Library\bin\conda.bat") {
            # If conda is a .bat file in Library\bin, go three directories up
            $condaBasePath = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $condaPath))
            Write-LogMessage "Conda command is a .bat file, adjusting base path calculation" -Level "INFO" -Color $colors.Info
        } else {
            # If conda is an .exe file in Scripts, go two directories up
            $condaBasePath = Split-Path -Parent (Split-Path -Parent $condaPath)
        }
        
        Write-LogMessage "Found Conda base path: $condaBasePath" -Level "SUCCESS" -Color $colors.Success
        $global:CondaBasePath = $condaBasePath
        return $condaBasePath
    }
    
    # Check standard Miniconda paths
    $standardPaths = @(
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\Miniconda3",
        "$env:LOCALAPPDATA\miniconda3",
        "$env:LOCALAPPDATA\Miniconda3",
        "C:\Miniconda3",
        "C:\miniconda3"
    )
    
    foreach ($path in $standardPaths) {
        $condaExe = Join-Path -Path $path -ChildPath "Scripts\conda.exe"
        if (Test-Path $condaExe) {
            Write-LogMessage "Found Conda at: $condaExe" -Level "SUCCESS" -Color $colors.Success
            $global:CondaBasePath = $path
            return $path
        }
    }
    
    # Check custom path if provided
    if ($CustomCondaPath) {
        $condaExe = Join-Path -Path $CustomCondaPath -ChildPath "Scripts\conda.exe"
        if (Test-Path $condaExe) {
            Write-LogMessage "Found Conda at custom path: $condaExe" -Level "SUCCESS" -Color $colors.Success
            return $CustomCondaPath
        } else {
            Write-LogMessage "Conda not found at custom path: $CustomCondaPath" -Level "WARNING" -Color $colors.Warning
        }
    }
    
    Write-LogMessage "Conda installation not found" -Level "INFO" -Color $colors.Info
    return $false
}

function Install-Conda {
    Write-LogMessage "Installing Miniconda..." -Level "INFO" -Color $colors.Info
    
    # Determine installation path
    $installPath = if ($CustomCondaPath) { $CustomCondaPath } else { "C:\Miniconda3" }
    
    # Check if path contains spaces or special characters
    if ($installPath -match " " -or $installPath -match "[^a-zA-Z0-9\\:_.-]") {
        Write-LogMessage "Warning: Installation path contains spaces or special characters" -Level "WARNING" -Color $colors.Warning
        Write-LogMessage "This may cause issues with Conda. Consider using a simple path like C:\Miniconda3" -Level "WARNING" -Color $colors.Warning
        
        if (-not $Force -and -not $SkipPrompts) {
            $continue = Read-Host "Continue with this path anyway? (y/n)"
            if ($continue -ne "y") {
                Write-LogMessage "Installation aborted" -Level "INFO" -Color $colors.Info
                return $false
            }
        }
    }
    
    # Download Miniconda installer
    $condaInstaller = "$env:TEMP\miniconda_installer.exe"
    try {
        Write-LogMessage "Downloading Miniconda installer..." -Level "INFO" -Color $colors.Info
        Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $condaInstaller
    }
    catch {
        Write-LogMessage "Error downloading Miniconda: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    # Install Miniconda
    Write-LogMessage "Installing Miniconda to $installPath..." -Level "INFO" -Color $colors.Info
    $installArgs = "/S /RegisterPython=1 /AddToPath=1 /D=$installPath"
    
    try {
        Start-Process -FilePath $condaInstaller -ArgumentList $installArgs -Wait
        Remove-Item $condaInstaller -Force
        
        # Verify installation
        $condaExe = Join-Path -Path $installPath -ChildPath "Scripts\conda.exe"
        if (Test-Path $condaExe) {
            Write-LogMessage "Miniconda successfully installed to $installPath" -Level "SUCCESS" -Color $colors.Success
            $global:CondaBasePath = $installPath
            return $installPath
        } else {
            Write-LogMessage "Miniconda installation failed: conda.exe not found" -Level "ERROR" -Color $colors.Error
            return $false
        }
    }
    catch {
        Write-LogMessage "Error installing Miniconda: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Fix-CondaPath {
    param (
        [string]$CondaPath
    )
    
    Write-LogMessage "Fixing Conda PATH..." -Level "INFO" -Color $colors.Info
    
    try {
        # Get current PATH
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        
        # Check if Conda paths are already in PATH
        $condaInPath = $currentPath -like "*$CondaPath*"
        $condaScriptsInPath = $currentPath -like "*$CondaPath\Scripts*"
        
        if (-not $condaInPath) {
            # Add Conda to PATH
            $newPath = "$CondaPath;$currentPath"
            [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
            Write-LogMessage "Added Conda to PATH" -Level "SUCCESS" -Color $colors.Success
        }
        
        if (-not $condaScriptsInPath) {
            # Add Conda Scripts to PATH
            $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
            $newPath = "$CondaPath\Scripts;$currentPath"
            [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
            Write-LogMessage "Added Conda Scripts to PATH" -Level "SUCCESS" -Color $colors.Success
        }
        
        # Update PATH for current session
        $env:PATH = "$CondaPath\Scripts;$CondaPath;$env:PATH"
        
        return $true
    } catch {
        Write-LogMessage "Error fixing Conda PATH: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Create-CondaProfile {
    Write-LogMessage "Creating PowerShell profile for Conda..." -Level "INFO" -Color $colors.Info
    
    try {
        # Check if $PROFILE is empty and use a default path if needed
        $profilePath = $PROFILE
        if ([string]::IsNullOrEmpty($profilePath)) {
            Write-LogMessage "$PROFILE is empty, using default profile path" -Level "WARNING" -Color $colors.Warning
            $profilePath = "$env:USERPROFILE\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1"
            Write-LogMessage "Using default profile path: $profilePath" -Level "INFO" -Color $colors.Info
        }
        
        # Check if profile exists
        if (-not (Test-Path $profilePath)) {
            # Create profile directory if it doesn't exist
            $profileDir = Split-Path -Parent $profilePath
            if (-not (Test-Path $profileDir)) {
                New-Item -Path $profileDir -ItemType Directory -Force | Out-Null
            }
            
            # Create empty profile
            New-Item -Path $profilePath -ItemType File -Force | Out-Null
            Write-LogMessage "Created PowerShell profile at: $profilePath" -Level "SUCCESS" -Color $colors.Success
        }
        
        # Check if profile already contains conda initialization
        $profileContent = Get-Content -Path $profilePath -Raw -ErrorAction SilentlyContinue
        if ($profileContent -and $profileContent -match "conda initialize") {
            Write-LogMessage "PowerShell profile already contains conda initialization" -Level "INFO" -Color $colors.Info
            return $true
        }
        
        # Add conda initialization to profile
        $condaInit = @"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
If (Test-Path "ENV:CONDA_EXE") {
    `$Env:CONDA_EXE | Out-Null
} else {
    # Find conda executable
    `$condaPaths = @(
        "`$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "`$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "`$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "`$env:LOCALAPPDATA\Miniconda3\Scripts\conda.exe",
        "C:\Miniconda3\Scripts\conda.exe",
        "C:\miniconda3\Scripts\conda.exe"
    )
    
    foreach (`$path in `$condaPaths) {
        if (Test-Path `$path) {
            `$Env:CONDA_EXE = `$path
            break
        }
    }
}

If (Test-Path `$Env:CONDA_EXE) {
    `$Env:_CE_M = ""
    `$Env:_CE_CONDA = ""
    
    # Run conda hook
    `$condaModuleScript = (& `$Env:CONDA_EXE "shell.powershell" "hook") | Out-String
    Invoke-Expression `$condaModuleScript
}
# <<< conda initialize <<<

"@
        
        Add-Content -Path $profilePath -Value $condaInit
        Write-LogMessage "Added conda initialization to PowerShell profile" -Level "SUCCESS" -Color $colors.Success
        return $true
    } catch {
        Write-LogMessage "Error creating PowerShell profile: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Initialize-Conda {
    param (
        [string]$CondaPath = $CondaBasePath
    )
    
    Write-LogMessage "Initializing Conda..." -Level "INFO" -Color $colors.Info
    
    try {
        # Initialize conda for PowerShell
        $condaExe = Join-Path -Path $CondaPath -ChildPath "Scripts\conda.exe"
        
        # Run conda init
        & $condaExe "init" "powershell"
        
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "Conda initialization successful" -Level "SUCCESS" -Color $colors.Success
            
            # Reload PATH
            $env:PATH = "$CondaPath\Scripts;$CondaPath;$env:PATH"
            
            return $true
        } else {
            Write-LogMessage "Conda initialization failed with exit code: $LASTEXITCODE" -Level "WARNING" -Color $colors.Warning
            
            # Try creating a manual profile as a fallback
            Write-LogMessage "Trying to create a manual Conda profile..." -Level "INFO" -Color $colors.Info
            $profileCreated = Create-CondaProfile
            if ($profileCreated) {
                Write-LogMessage "Manual Conda profile created successfully" -Level "SUCCESS" -Color $colors.Success
                return $true
            } else {
                Write-LogMessage "Failed to create Conda profile" -Level "ERROR" -Color $colors.Error
                return $false
            }
        }
    }
    catch {
        Write-LogMessage "Error initializing Conda: $_" -Level "ERROR" -Color $colors.Error
        
        # Try creating a manual profile as a fallback
        Write-LogMessage "Trying to create a manual Conda profile..." -Level "INFO" -Color $colors.Info
        $profileCreated = Create-CondaProfile
        if ($profileCreated) {
            Write-LogMessage "Manual Conda profile created successfully" -Level "SUCCESS" -Color $colors.Success
            return $true
        } else {
            Write-LogMessage "Failed to create Conda profile" -Level "ERROR" -Color $colors.Error
            return $false
        }
    }
}

function Create-CondaEnvironment {
    param (
        [string]$CondaPath = $CondaBasePath
    )
    
    Write-LogMessage "Creating Conda environment '$EnvName'..." -Level "INFO" -Color $colors.Info
    
    # Add Conda to PATH for current session
    $env:PATH = "$CondaPath\Scripts;$CondaPath;$env:PATH"
    
    # Check if environment already exists
    $condaExe = Join-Path -Path $CondaPath -ChildPath "Scripts\conda.exe"
    $envExists = & $condaExe env list | Select-String -Pattern "^$EnvName\s"
    
    if ($envExists -and $ForceReinstall) {
        Write-LogMessage "Removing existing environment..." -Level "INFO" -Color $colors.Info
        & $condaExe env remove -n $EnvName -y
        if ($LASTEXITCODE -ne 0) {
            Write-LogMessage "Error removing existing environment" -Level "ERROR" -Color $colors.Error
            return $false
        }
    }
    elseif ($envExists) {
        Write-LogMessage "Environment '$EnvName' already exists" -Level "INFO" -Color $colors.Info
        
        if (-not $SkipPrompts -and -not $Force) {
            $recreate = Read-Host "Recreate environment? (y/n)"
            if ($recreate -eq "y") {
                & $condaExe env remove -n $EnvName -y
                if ($LASTEXITCODE -ne 0) {
                    Write-LogMessage "Error removing existing environment" -Level "ERROR" -Color $colors.Error
                    return $false
                }
            }
            else {
                return $true
            }
        }
        else {
            return $true
        }
    }
    
    # Create environment from environment.yml
    $envFile = Join-Path -Path $ScriptPath -ChildPath "environment.yml"
    if (Test-Path $envFile) {
        Write-LogMessage "Creating environment from environment.yml..." -Level "INFO" -Color $colors.Info
        & $condaExe env create -f $envFile -y
    }
    else {
        Write-LogMessage "environment.yml not found, creating basic environment..." -Level "WARNING" -Color $colors.Warning
        & $condaExe create -n $EnvName python=3.10 -y
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-LogMessage "Error creating Conda environment" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    Write-LogMessage "Conda environment '$EnvName' successfully created" -Level "SUCCESS" -Color $colors.Success
    return $true
}

function Install-Dependencies {
    Write-LogMessage "Installing dependencies..." -Level "INFO" -Color $colors.Info
    
    # Activate the Conda environment
    & "$CondaBasePath\Scripts\activate.bat" $EnvName
    
    if ($MinimalInstall) {
        Write-LogMessage "Performing minimal installation (core dependencies only)" -Level "INFO" -Color $colors.Info
        & "$CondaBasePath\envs\$EnvName\python.exe" -m pip install -e .
    }
    else {
        Write-LogMessage "Installing all dependencies (this may take a while)..." -Level "INFO" -Color $colors.Info
        
        # Install PyTorch with CUDA support if environment.yml doesn't exist
        $envFile = Join-Path -Path $ScriptPath -ChildPath "environment.yml"
        if (-not (Test-Path $envFile)) {
            Write-LogMessage "Installing PyTorch with CUDA support..." -Level "INFO" -Color $colors.Info
            & "$CondaBasePath\Scripts\conda.exe" install -n $EnvName pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
            
            # Install other dependencies
            & "$CondaBasePath\envs\$EnvName\python.exe" -m pip install open-clip-torch ultralytics facenet-pytorch pillow tqdm pyyaml requests minio exiftool-vendored
        }
        
        # Install package in development mode
        & "$CondaBasePath\envs\$EnvName\python.exe" -m pip install -e .
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-LogMessage "Error installing dependencies" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    Write-LogMessage "Dependencies successfully installed" -Level "INFO" -Color $colors.Success
    return $true
}

function Setup-ProjectStructure {
    Write-LogMessage "Setting up project structure..." -Level "INFO" -Color $colors.Info
    
    # Create directories
    $dirs = @(
        "config",
        "models",
        "data/input",
        "data/output",
        "logs"
    )
    
    foreach ($dir in $dirs) {
        $path = Join-Path -Path $ScriptPath -ChildPath $dir
        if (-not (Test-Path $path)) {
            New-Item -Path $path -ItemType Directory -Force | Out-Null
            Write-LogMessage "Directory created: $dir" -Level "INFO" -Color $colors.Info
        }
    }
    
    # Create settings.yaml if not exists
    $settingsFile = Join-Path -Path $ScriptPath -ChildPath "config\settings.yaml"
    if (-not (Test-Path $settingsFile)) {
        $templateFile = Join-Path -Path $ScriptPath -ChildPath "config\settings.yaml.template"
        if (Test-Path $templateFile) {
            Copy-Item -Path $templateFile -Destination $settingsFile
            Write-LogMessage "Configuration created from template: settings.yaml" -Level "INFO" -Color $colors.Success
        }
        else {
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
  offline_mode: $($OfflineMode.ToString().ToLower())
"@ | Out-File -FilePath $settingsFile -Encoding utf8
            Write-LogMessage "Basic configuration created: settings.yaml" -Level "INFO" -Color $colors.Success
        }
    }
    
    # Ensure start.ps1 exists
    $startScript = Join-Path -Path $ScriptPath -ChildPath "start.ps1"
    if (-not (Test-Path $startScript)) {
        Write-LogMessage "Start script missing: start.ps1" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    Write-LogMessage "Project structure successfully set up" -Level "INFO" -Color $colors.Success
    return $true
}

function Copy-OfflineModels {
    if (-not $OfflineMode -or -not $ModelsPath) {
        return $true
    }
    
    Write-LogMessage "Copying offline models from $ModelsPath..." -Level "INFO" -Color $colors.Info
    
    if (-not (Test-Path $ModelsPath)) {
        Write-LogMessage "Offline models path not found: $ModelsPath" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    try {
        # Copy models from specified path
        $modelsDir = Join-Path -Path $ScriptPath -ChildPath "models"
        
        # Create directory structure
        $modelFolders = @("clip", "yolov8", "facenet")
        foreach ($folder in $modelFolders) {
            $targetDir = Join-Path -Path $modelsDir -ChildPath $folder
            if (-not (Test-Path $targetDir)) {
                New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
            }
        }
        
        # Copy CLIP model
        $clipSource = Join-Path -Path $ModelsPath -ChildPath "clip_vit_b32.pth"
        $clipTarget = Join-Path -Path $modelsDir -ChildPath "clip\clip_vit_b32.pth"
        if (Test-Path $clipSource) {
            Copy-Item -Path $clipSource -Destination $clipTarget -Force
            Write-LogMessage "CLIP model copied" -Level "INFO" -Color $colors.Success
        }
        
        # Copy YOLOv8 model
        $yoloSource = Join-Path -Path $ModelsPath -ChildPath "yolov8n.pt"
        $yoloTarget = Join-Path -Path $modelsDir -ChildPath "yolov8\yolov8n.pt"
        if (Test-Path $yoloSource) {
            Copy-Item -Path $yoloSource -Destination $yoloTarget -Force
            Write-LogMessage "YOLOv8 model copied" -Level "INFO" -Color $colors.Success
        }
        
        # Copy FaceNet model
        $faceSource = Join-Path -Path $ModelsPath -ChildPath "facenet_model.pth"
        $faceTarget = Join-Path -Path $modelsDir -ChildPath "facenet\facenet_model.pth"
        if (Test-Path $faceSource) {
            Copy-Item -Path $faceSource -Destination $faceTarget -Force
            Write-LogMessage "FaceNet model copied" -Level "INFO" -Color $colors.Success
        }
        
        Write-LogMessage "Offline models successfully copied" -Level "INFO" -Color $colors.Success
        
        # Set offline_mode in settings.yaml
        $settingsFile = Join-Path -Path $ScriptPath -ChildPath "config\settings.yaml"
        if (Test-Path $settingsFile) {
            $settings = Get-Content $settingsFile -Raw
            $settings = $settings -replace "offline_mode:\s*false", "offline_mode: true"
            $settings | Out-File -FilePath $settingsFile -Encoding utf8 -Force
            Write-LogMessage "Offline mode enabled in settings" -Level "INFO" -Color $colors.Info
        }
        
        return $true
    }
    catch {
        Write-LogMessage "Error copying offline models: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Create-Shortcut {
    param (
        [string]$TargetPath,
        [string]$ShortcutPath,
        [string]$Arguments = "",
        [string]$WorkingDirectory = "",
        [string]$IconPath = ""
    )
    
    try {
        $WScriptShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
        $Shortcut.TargetPath = $TargetPath
        
        if ($Arguments) {
            $Shortcut.Arguments = $Arguments
        }
        
        if ($WorkingDirectory) {
            $Shortcut.WorkingDirectory = $WorkingDirectory
        }
        
        if ($IconPath -and (Test-Path $IconPath)) {
            $Shortcut.IconLocation = "$IconPath,0"
        }
        
        $Shortcut.Save()
        Write-LogMessage "Shortcut created: $ShortcutPath" -Level "INFO" -Color $colors.Success
        return $true
    }
    catch {
        Write-LogMessage "Error creating shortcut: $_" -Level "WARNING" -Color $colors.Warning
        return $false
    }
}

function Create-DesktopShortcut {
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    $ShortcutFile = "$DesktopPath\AUTO-TAG.lnk"
    
    if ($DevMode) {
        # Development mode shortcut
        Create-Shortcut -TargetPath "powershell.exe" `
                       -ShortcutPath $ShortcutFile `
                       -Arguments "-ExecutionPolicy Bypass -File `"$ScriptPath\start.ps1`"" `
                       -WorkingDirectory $ScriptPath `
                       -IconPath (Join-Path -Path $ScriptPath -ChildPath "resources\icon.ico")
    }
    else {
        # End-user mode shortcut
        Create-Shortcut -TargetPath "$InstallDir\AUTO-TAG.exe" `
                       -ShortcutPath $ShortcutFile `
                       -WorkingDirectory $InstallDir `
                       -IconPath "$InstallDir\resources\icon.ico"
    }
}

function Create-StartMenuShortcut {
    $StartMenuPath = [Environment]::GetFolderPath("Programs")
    $StartMenuFolder = "$StartMenuPath\AUTO-TAG"
    
    # Create Start Menu folder
    if (-not (Test-Path $StartMenuFolder)) {
        New-Item -Path $StartMenuFolder -ItemType Directory -Force | Out-Null
    }
    
    $ShortcutFile = "$StartMenuFolder\AUTO-TAG.lnk"
    $UninstallShortcut = "$StartMenuFolder\Uninstall AUTO-TAG.lnk"
    
    if ($DevMode) {
        # Development mode shortcut
        Create-Shortcut -TargetPath "powershell.exe" `
                       -ShortcutPath $ShortcutFile `
                       -Arguments "-ExecutionPolicy Bypass -File `"$ScriptPath\start.ps1`"" `
                       -WorkingDirectory $ScriptPath `
                       -IconPath (Join-Path -Path $ScriptPath -ChildPath "resources\icon.ico")
    }
    else {
        # End-user mode shortcut
        Create-Shortcut -TargetPath "$InstallDir\AUTO-TAG.exe" `
                       -ShortcutPath $ShortcutFile `
                       -WorkingDirectory $InstallDir `
                       -IconPath "$InstallDir\resources\icon.ico"
        
        # Uninstall shortcut
        Create-Shortcut -TargetPath "powershell.exe" `
                       -ShortcutPath $UninstallShortcut `
                       -Arguments "-ExecutionPolicy Bypass -File `"$InstallDir\uninstall.ps1`"" `
                       -WorkingDirectory $InstallDir
    }
}

function Register-Uninstaller {
    # Create registry entries for uninstallation
    try {
        # Create the uninstall registry key
        if (-not (Test-Path $UninstallRegPath)) {
            New-Item -Path $UninstallRegPath -Force | Out-Null
        }
        
        # Set the registry values
        New-ItemProperty -Path $UninstallRegPath -Name "DisplayName" -Value "AUTO-TAG" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $UninstallRegPath -Name "DisplayVersion" -Value $AppVersion -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $UninstallRegPath -Name "Publisher" -Value "AUTO-TAG Team" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $UninstallRegPath -Name "InstallLocation" -Value $InstallDir -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $UninstallRegPath -Name "UninstallString" -Value "powershell.exe -ExecutionPolicy Bypass -File `"$InstallDir\uninstall.ps1`"" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $UninstallRegPath -Name "DisplayIcon" -Value "$InstallDir\resources\icon.ico" -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $UninstallRegPath -Name "NoModify" -Value 1 -PropertyType DWord -Force | Out-Null
        New-ItemProperty -Path $UninstallRegPath -Name "NoRepair" -Value 1 -PropertyType DWord -Force | Out-Null
        
        Write-LogMessage "Uninstaller registered in Windows registry" -Level "INFO" -Color $colors.Success
        return $true
    }
    catch {
        Write-LogMessage "Error registering uninstaller: $_" -Level "WARNING" -Color $colors.Warning
        return $false
    }
}

function Create-UninstallScript {
    # Create the uninstall script
    $UninstallScript = @"
# AUTO-TAG Uninstaller
[CmdletBinding()]
param (
    [switch]`$KeepUserData,
    [switch]`$KeepModels
)

`$ErrorActionPreference = "Stop"
`$InstallDir = "$InstallDir"
`$UninstallRegPath = "$UninstallRegPath"
`$EnvName = "$EnvName"
`$CondaBasePath = "$CondaBasePath"

# Function to write messages
function Write-Message {
    param (
        [string]`$Message,
        [string]`$Color = "White"
    )
    
    Write-Host `$Message -ForegroundColor `$Color
}

# Show header
Write-Host "`n====================================" -ForegroundColor Magenta
Write-Host "       AUTO-TAG UNINSTALLER        " -ForegroundColor Magenta
Write-Host "====================================`n" -ForegroundColor Magenta

# Remove desktop shortcut
`$DesktopShortcut = [Environment]::GetFolderPath("Desktop") + "\AUTO-TAG.lnk"
if (Test-Path `$DesktopShortcut) {
    Remove-Item -Path `$DesktopShortcut -Force
    Write-Message "Removed desktop shortcut" -Color "Cyan"
}

# Remove Start Menu shortcuts
`$StartMenuFolder = [Environment]::GetFolderPath("Programs") + "\AUTO-TAG"
if (Test-Path `$StartMenuFolder) {
    Remove-Item -Path `$StartMenuFolder -Recurse -Force
    Write-Message "Removed Start Menu shortcuts" -Color "Cyan"
}

# Remove Conda environment if it exists
if (Test-Path "`$CondaBasePath\envs\`$EnvName") {
    Write-Message "Removing Conda environment..." -Color "Yellow"
    & "`$CondaBasePath\Scripts\conda.exe" env remove -n `$EnvName -y
    Write-Message "Conda environment removed" -Color "Green"
}

# Remove registry entries
if (Test-Path `$UninstallRegPath) {
    Remove-Item -Path `$UninstallRegPath -Recurse -Force
    Write-Message "Removed registry entries" -Color "Cyan"
}

# Remove application files
if (Test-Path `$InstallDir) {
    # Keep user data if requested
    if (`$KeepUserData) {
        Write-Message "Keeping user data as requested" -Color "Yellow"
        # Copy user data to a backup location
        `$BackupDir = "`$env:TEMP\AUTO-TAG-Backup"
        New-Item -Path `$BackupDir -ItemType Directory -Force | Out-Null
        
        # Copy config and data folders
        if (Test-Path "`$InstallDir\config") {
            Copy-Item -Path "`$InstallDir\config" -Destination `$BackupDir -Recurse -Force
        }
        if (Test-Path "`$InstallDir\data") {
            Copy-Item -Path "`$InstallDir\data" -Destination `$BackupDir -Recurse -Force
        }
    }
    
    # Keep models if requested
    if (`$KeepModels) {
        Write-Message "Keeping model files as requested" -Color "Yellow"
        # Copy models to a backup location
        `$BackupDir = "`$env:TEMP\AUTO-TAG-Backup"
        New-Item -Path `$BackupDir -ItemType Directory -Force | Out-Null
        
        if (Test-Path "`$InstallDir\models") {
            Copy-Item -Path "`$InstallDir\models" -Destination `$BackupDir -Recurse -Force
        }
    }
    
    # Remove the installation directory
    Remove-Item -Path `$InstallDir -Recurse -Force
    Write-Message "Removed application files" -Color "Cyan"
    
    # Restore user data if kept
    if (`$KeepUserData -and (Test-Path "`$BackupDir\config" -or Test-Path "`$BackupDir\data")) {
        Write-Message "Saving user data to `$env:USERPROFILE\AUTO-TAG-Data" -Color "Yellow"
        New-Item -Path "`$env:USERPROFILE\AUTO-TAG-Data" -ItemType Directory -Force | Out-Null
        
        if (Test-Path "`$BackupDir\config") {
            Copy-Item -Path "`$BackupDir\config" -Destination "`$env:USERPROFILE\AUTO-TAG-Data" -Recurse -Force
        }
        if (Test-Path "`$BackupDir\data") {
            Copy-Item -Path "`$BackupDir\data" -Destination "`$env:USERPROFILE\AUTO-TAG-Data" -Recurse -Force
        }
    }
    
    # Restore models if kept
    if (`$KeepModels -and (Test-Path "`$BackupDir\models")) {
        Write-Message "Saving model files to `$env:USERPROFILE\AUTO-TAG-Data\models" -Color "Yellow"
        New-Item -Path "`$env:USERPROFILE\AUTO-TAG-Data" -ItemType Directory -Force | Out-Null
        Copy-Item -Path "`$BackupDir\models" -Destination "`$env:USERPROFILE\AUTO-TAG-Data" -Recurse -Force
    }
    
    # Clean up backup
    if (Test-Path `$BackupDir) {
        Remove-Item -Path `$BackupDir -Recurse -Force
    }
}

Write-Host "`nAUTO-TAG has been uninstalled successfully!" -ForegroundColor Green
Write-Host "Thank you for using AUTO-TAG." -ForegroundColor Cyan

# Keep the window open
Read-Host "`nPress Enter to close this window"
"@

    # Write the uninstall script to the installation directory
    $UninstallScriptPath = "$InstallDir\uninstall.ps1"
    New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null
    $UninstallScript | Out-File -FilePath $UninstallScriptPath -Encoding utf8 -Force
    
    Write-LogMessage "Uninstall script created: $UninstallScriptPath" -Level "INFO" -Color $colors.Success
    return $true
}

function Install-EndUserMode {
    Write-LogMessage "Installing in End-User mode to $InstallDir" -Level "INFO" -Color $colors.Info
    
    # Create installation directory
    if (-not (Test-Path $InstallDir)) {
        New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null
        Write-LogMessage "Created installation directory: $InstallDir" -Level "INFO" -Color $colors.Success
    }
    
    # Check if we have a pre-built package
    $DistDir = Join-Path -Path $ScriptPath -ChildPath "dist\AUTO-TAG"
    if (Test-Path $DistDir) {
        # Copy the pre-built package
        Write-LogMessage "Copying pre-built package from $DistDir" -Level "INFO" -Color $colors.Info
        Copy-Item -Path "$DistDir\*" -Destination $InstallDir -Recurse -Force
        Write-LogMessage "Copied pre-built package to $InstallDir" -Level "INFO" -Color $colors.Success
    }
    else {
        # We don't have a pre-built package, so we need to build it
        Write-LogMessage "No pre-built package found. Building from source..." -Level "INFO" -Color $colors.Warning
        
        # Check if PyInstaller is installed
        $PyInstallerInstalled = $false
        try {
            $PyInstallerVersion = python -c "import PyInstaller; print(PyInstaller.__version__)" 2>$null
            if ($PyInstallerVersion) {
                $PyInstallerInstalled = $true
                Write-LogMessage "PyInstaller version: $PyInstallerVersion" -Level "INFO" -Color $colors.Success
            }
        }
        catch {
            Write-LogMessage "PyInstaller not found" -Level "WARNING" -Color $colors.Warning
        }
        
        if (-not $PyInstallerInstalled) {
            Write-LogMessage "Installing PyInstaller..." -Level "INFO" -Color $colors.Info
            python -m pip install pyinstaller
            if ($LASTEXITCODE -ne 0) {
                Write-LogMessage "Failed to install PyInstaller" -Level "ERROR" -Color $colors.Error
                return $false
            }
        }
        
        # Run the build script
        Write-LogMessage "Running build script..." -Level "INFO" -Color $colors.Info
        python build_package.py
        if ($LASTEXITCODE -ne 0) {
            Write-LogMessage "Failed to build package" -Level "ERROR" -Color $colors.Error
            return $false
        }
        
        # Check if the build was successful
        if (Test-Path $DistDir) {
            # Copy the built package
            Write-LogMessage "Copying built package from $DistDir" -Level "INFO" -Color $colors.Info
            Copy-Item -Path "$DistDir\*" -Destination $InstallDir -Recurse -Force
            Write-LogMessage "Copied built package to $InstallDir" -Level "INFO" -Color $colors.Success
        }
        else {
            Write-LogMessage "Build failed: $DistDir not found" -Level "ERROR" -Color $colors.Error
            return $false
        }
    }
    
    # Create uninstall script
    Create-UninstallScript
    
    # Register uninstaller
    Register-Uninstaller
    
    # Create shortcuts
    Create-DesktopShortcut
    Create-StartMenuShortcut
    
    Write-LogMessage "End-User installation completed successfully" -Level "INFO" -Color $colors.Success
    return $true
}

function Uninstall-AutoTag {
    Write-LogMessage "Uninstalling AUTO-TAG..." -Level "INFO" -Color $colors.Info
    
    # Check if installed in end-user mode
    $EndUserInstalled = Test-Path $UninstallRegPath
    
    if ($EndUserInstalled) {
        # Get the installation directory from registry
        $InstallLocation = (Get-ItemProperty -Path $UninstallRegPath -Name "InstallLocation" -ErrorAction SilentlyContinue).InstallLocation
        
        if ($InstallLocation -and (Test-Path $InstallLocation)) {
            # Run the uninstall script
            $UninstallScript = Join-Path -Path $InstallLocation -ChildPath "uninstall.ps1"
            
            if (Test-Path $UninstallScript) {
                Write-LogMessage "Running uninstall script: $UninstallScript" -Level "INFO" -Color $colors.Info
                & powershell.exe -ExecutionPolicy Bypass -File $UninstallScript
                return $true
            }
        }
    }
    
    # If we get here, either it's not installed in end-user mode or the uninstall script wasn't found
    # Perform manual uninstallation
    
    # Remove desktop shortcut
    $DesktopShortcut = [Environment]::GetFolderPath("Desktop") + "\AUTO-TAG.lnk"
    if (Test-Path $DesktopShortcut) {
        Remove-Item -Path $DesktopShortcut -Force
        Write-LogMessage "Removed desktop shortcut" -Level "INFO" -Color $colors.Success
    }
    
    # Remove Start Menu shortcuts
    $StartMenuFolder = [Environment]::GetFolderPath("Programs") + "\AUTO-TAG"
    if (Test-Path $StartMenuFolder) {
        Remove-Item -Path $StartMenuFolder -Recurse -Force
        Write-LogMessage "Removed Start Menu shortcuts" -Level "INFO" -Color $colors.Success
    }
    
    # Remove Conda environment if it exists
    if (Test-Path "$CondaBasePath\envs\$EnvName") {
        Write-LogMessage "Removing Conda environment..." -Level "INFO" -Color $colors.Info
        & "$CondaBasePath\Scripts\conda.exe" env remove -n $EnvName -y
        Write-LogMessage "Conda environment removed" -Level "INFO" -Color $colors.Success
    }
    
    # Remove registry entries
    if (Test-Path $UninstallRegPath) {
        Remove-Item -Path $UninstallRegPath -Recurse -Force
        Write-LogMessage "Removed registry entries" -Level "INFO" -Color $colors.Success
    }
    
    # If installed in end-user mode, remove the installation directory
    if ($EndUserInstalled -and $InstallLocation -and (Test-Path $InstallLocation)) {
        Remove-Item -Path $InstallLocation -Recurse -Force
        Write-LogMessage "Removed installation directory: $InstallLocation" -Level "INFO" -Color $colors.Success
    }
    
    Write-LogMessage "Uninstallation completed successfully" -Level "INFO" -Color $colors.Success
    return $true
}

function Check-GPU {
    try {
        # Check if NVIDIA tools are installed
        $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
        
        if ($nvidiaSmi) {
            $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            Write-LogMessage "GPU detected: $gpuInfo" -Level "INFO" -Color $colors.Success
            return $true
        }
        else {
            # Alternatively check if CUDA is installed
            $cudaPath = "$env:CUDA_PATH"
            if ($cudaPath) {
                Write-LogMessage "CUDA installation found: $cudaPath" -Level "INFO" -Color $colors.Success
                return $true
            }
        }
    }
    catch {
        # Ignore errors
    }
    
    Write-LogMessage "No NVIDIA GPU detected. AUTO-TAG works better with a GPU." -Level "WARNING" -Color $colors.Warning
    return $false
}

function Main {
    Show-Header
    
    # Handle uninstallation if requested
    if ($Uninstall) {
        $uninstalled = Uninstall-AutoTag
        if ($uninstalled) {
            Write-LogMessage "AUTO-TAG has been uninstalled successfully!" -Level "INFO" -Color $colors.Success
        }
        else {
            Write-LogMessage "Uninstallation failed" -Level "ERROR" -Color $colors.Error
        }
        
        if (-not $SkipPrompts) {
            Read-Host "Press Enter to exit"
        }
        
        exit 0
    }
    
    # Determine installation mode
    if (-not $DevMode -and -not $EndUserMode) {
        if (-not $SkipPrompts) {
            Write-Host "`nPlease select installation mode:" -ForegroundColor $colors.Info
            Write-Host "1. Development Mode (for developers, uses Conda environment)" -ForegroundColor $colors.Info
            Write-Host "2. End-User Mode (for regular users, standalone installation)" -ForegroundColor $colors.Info
            
            $modeChoice = Read-Host "Enter choice (1 or 2)"
            if ($modeChoice -eq "1") {
                $DevMode = $true
            }
            elseif ($modeChoice -eq "2") {
                $EndUserMode = $true
            }
            else {
                Write-LogMessage "Invalid choice. Defaulting to Development Mode." -Level "WARNING" -Color $colors.Warning
                $DevMode = $true
            }
        }
        else {
            # Default to Development Mode if not specified
            $DevMode = $true
        }
    }
    
    # Check Administrator rights
    Check-Administrator
    
    # Check GPU availability
    Check-GPU
    
    # Handle End-User Mode installation
    if ($EndUserMode) {
        $installed = Install-EndUserMode
        if ($installed) {
            Write-LogMessage "AUTO-TAG has been installed successfully in End-User mode!" -Level "INFO" -Color $colors.Success
            Write-LogMessage "You can start AUTO-TAG using the desktop shortcut or from the Start Menu." -Level "INFO" -Color $colors.Info
        }
        else {
            Write-LogMessage "End-User installation failed" -Level "ERROR" -Color $colors.Error
        }
        
        if (-not $SkipPrompts) {
            Read-Host "Press Enter to finish installation"
        }
        
        exit 0
    }
    
    # Development Mode installation
    Write-LogMessage "Installing in Development Mode" -Level "INFO" -Color $colors.Info
    
    # Check if Conda is installed
    $condaPath = Test-CondaInstallation
    
    if (-not $condaPath) {
        Write-LogMessage "Miniconda not found, installing..." -Level "INFO" -Color $colors.Info
        
        if (-not $SkipPrompts -and -not $Force) {
            $installConda = Read-Host "Install Miniconda? (y/n)"
            if ($installConda -ne "y") {
                Write-LogMessage "Installation aborted" -Level "INFO" -Color $colors.Info
                exit 1
            }
        }
        
        $condaPath = Install-Conda
        if (-not $condaPath) {
            Write-LogMessage "Error installing Conda, installation aborted" -Level "ERROR" -Color $colors.Error
            exit 1
        }
        
        # Update CondaBasePath with the actual installation path
        $CondaBasePath = $condaPath
    }
    else {
        Write-LogMessage "Conda already installed at: $condaPath" -Level "INFO" -Color $colors.Success
        # Update CondaBasePath with the detected path
        $CondaBasePath = $condaPath
    }
    
    # Fix Conda PATH
    $pathFixed = Fix-CondaPath -CondaPath $CondaBasePath
    if (-not $pathFixed) {
        Write-LogMessage "Failed to fix Conda PATH" -Level "ERROR" -Color $colors.Error
        exit 1
    }
    
    # Initialize Conda
    $condaInitialized = Initialize-Conda -CondaPath $CondaBasePath
    if (-not $condaInitialized) {
        Write-LogMessage "Error initializing Conda, installation aborted" -Level "ERROR" -Color $colors.Error
        exit 1
    }
    
    # Set up project structure
    $structureSetup = Setup-ProjectStructure
    if (-not $structureSetup) {
        Write-LogMessage "Error setting up project structure" -Level "ERROR" -Color $colors.Error
        exit 1
    }
    
    # Create Conda environment
    $envCreated = Create-CondaEnvironment -CondaPath $CondaBasePath
    if (-not $envCreated) {
        Write-LogMessage "Error creating Conda environment, installation aborted" -Level "ERROR" -Color $colors.Error
        exit 1
    }
    
    # Install dependencies
    $depsInstalled = Install-Dependencies
    if (-not $depsInstalled) {
        Write-LogMessage "Error installing dependencies, installation aborted" -Level "ERROR" -Color $colors.Error
        exit 1
    }
    
    # Copy offline models if specified
    if ($OfflineMode) {
        $modelsCopied = Copy-OfflineModels
        if (-not $modelsCopied) {
            Write-LogMessage "Error copying offline models" -Level "WARNING" -Color $colors.Warning
            
            if (-not $SkipPrompts) {
                $continue = Read-Host "Continue anyway? (y/n)"
                if ($continue -ne "y") {
                    Write-LogMessage "Installation aborted" -Level "INFO" -Color $colors.Info
                    exit 1
                }
            }
        }
    }
    
    # Create desktop shortcut
    Create-DesktopShortcut
    
    # Create Start Menu shortcuts
    Create-StartMenuShortcut
    
    # Installation complete
    Write-LogMessage "AUTO-TAG installation completed in Development Mode!" -Level "INFO" -Color $colors.Success
    Write-LogMessage "Start AUTO-TAG using the desktop shortcut or with .\start.ps1" -Level "INFO" -Color $colors.Info
    
    if (-not $SkipPrompts) {
        Read-Host "Press Enter to finish installation"
    }
}

# Start main program
Main