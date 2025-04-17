# install.ps1 - Ein-Klick-Installer für AUTO-TAG
[CmdletBinding()]
param (
    [switch]$SkipPrompts,
    [switch]$OfflineInstall,
    [string]$ModelsPath,
    [switch]$SkipPythonCheck
)

# Konstanten
$ScriptPath = $PSScriptRoot
$RequiredPythonVersion = "3.10"
$CondaInstallURL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$CondaBasePath = "$env:USERPROFILE\miniconda3"
$EnvName = "autotag"
$LogFile = "$ScriptPath\install.log"

# Farbdefinitionen
$colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Title = "Magenta"
}

# Erstelle Logdatei
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
    
    # Auch ins Logfile schreiben
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
        Write-LogMessage "Diese Installation sollte als Administrator ausgeführt werden." -Level "WARNING" -Color $colors.Warning
        
        if (-not $SkipPrompts) {
            $continue = Read-Host "Fortfahren ohne Admin-Rechte? (j/n)"
            if ($continue -ne "j") {
                Write-LogMessage "Installation abgebrochen" -Level "INFO" -Color $colors.Info
                exit 1
            }
        }
    }
    else {
        Write-LogMessage "Ausführung mit Administrator-Rechten" -Level "INFO" -Color $colors.Success
    }
}

function Check-Python {
    if ($SkipPythonCheck) {
        Write-LogMessage "Python-Prüfung übersprungen" -Level "INFO" -Color $colors.Info
        return $true
    }
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+\.\d+)") {
            $version = $matches[1]
            if ([Version]$version -ge [Version]$RequiredPythonVersion) {
                Write-LogMessage "Python gefunden: $pythonVersion" -Level "INFO" -Color $colors.Success
                return $true
            }
            else {
                Write-LogMessage "Python-Version zu alt: $version (mindestens $RequiredPythonVersion erforderlich)" -Level "WARNING" -Color $colors.Warning
            }
        }
    }
    catch {
        Write-LogMessage "Python nicht gefunden." -Level "WARNING" -Color $colors.Warning
    }
    
    # Python nicht gefunden oder zu alt, versuche Conda zu installieren
    return $false
}

function Install-Conda {
    Write-LogMessage "Miniconda wird installiert..." -Level "INFO" -Color $colors.Info
    
    # Lade Miniconda-Installer herunter
    $condaInstaller = "$env:TEMP\miniconda_installer.exe"
    try {
        Invoke-WebRequest -Uri $CondaInstallURL -OutFile $condaInstaller
    }
    catch {
        Write-LogMessage "Fehler beim Herunterladen von Miniconda: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    # Installiere Miniconda
    $installArgs = "/S /D=$CondaBasePath"
    if (-not $SkipPrompts) {
        $installArgs += " /AddToPath=1"
    }
    
    try {
        Start-Process -FilePath $condaInstaller -ArgumentList $installArgs -Wait
        Remove-Item $condaInstaller -Force
        
        # Aktualisiere PATH für die aktuelle Sitzung
        $env:PATH = "$CondaBasePath\Scripts;$CondaBasePath;$env:PATH"
        
        Write-LogMessage "Miniconda erfolgreich installiert" -Level "INFO" -Color $colors.Success
        return $true
    }
    catch {
        Write-LogMessage "Fehler bei der Miniconda-Installation: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Create-CondaEnvironment {
    Write-LogMessage "Erstelle Conda-Umgebung '$EnvName'..." -Level "INFO" -Color $colors.Info
    
    # Prüfe, ob die Umgebung bereits existiert
    $envExists = conda env list | Select-String $EnvName
    
    if ($envExists) {
        Write-LogMessage "Umgebung '$EnvName' existiert bereits" -Level "INFO" -Color $colors.Info
        
        if (-not $SkipPrompts) {
            $recreate = Read-Host "Umgebung neu erstellen? (j/n)"
            if ($recreate -eq "j") {
                conda env remove -n $EnvName -y
            }
            else {
                return $true
            }
        }
    }
    
    # Erstelle die Umgebung
    try {
        conda create -n $EnvName python=3.10 -y
        if ($LASTEXITCODE -ne 0) {
            Write-LogMessage "Fehler beim Erstellen der Conda-Umgebung" -Level "ERROR" -Color $colors.Error
            return $false
        }
        
        Write-LogMessage "Conda-Umgebung '$EnvName' erfolgreich erstellt" -Level "INFO" -Color $colors.Success
        return $true
    }
    catch {
        Write-LogMessage "Fehler beim Erstellen der Conda-Umgebung: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Install-Dependencies {
    Write-LogMessage "Installiere Abhängigkeiten..." -Level "INFO" -Color $colors.Info
    
    # Aktiviere die Conda-Umgebung
    try {
        & "$CondaBasePath\Scripts\activate.bat" $EnvName
        
        # Installiere PyTorch
        Write-LogMessage "Installiere PyTorch (dies kann einige Minuten dauern)..." -Level "INFO" -Color $colors.Info
        conda install -n $EnvName pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        
        # Installiere restliche Abhängigkeiten mit pip
        Write-LogMessage "Installiere weitere Abhängigkeiten..." -Level "INFO" -Color $colors.Info
        & "$CondaBasePath\envs\$EnvName\python.exe" -m pip install open-clip-torch ultralytics facenet-pytorch pillow tqdm pyyaml requests minio
        
        Write-LogMessage "Abhängigkeiten erfolgreich installiert" -Level "INFO" -Color $colors.Success
        return $true
    }
    catch {
        Write-LogMessage "Fehler bei der Installation der Abhängigkeiten: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Check-GPU {
    try {
        # Prüfe, ob NVIDIA-Tools installiert sind
        $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
        
        if ($nvidiaSmi) {
            $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            Write-LogMessage "GPU erkannt: $gpuInfo" -Level "INFO" -Color $colors.Success
            return $true
        }
        else {
            # Alternativ prüfen, ob CUDA installiert ist
            $cudaPath = "$env:CUDA_PATH"
            if ($cudaPath) {
                Write-LogMessage "CUDA-Installation gefunden: $cudaPath" -Level "INFO" -Color $colors.Success
                return $true
            }
        }
    }
    catch {
        # Fehler ignorieren
    }
    
    Write-LogMessage "Keine NVIDIA GPU erkannt. AUTO-TAG läuft besser mit einer GPU." -Level "WARNING" -Color $colors.Warning
    return $false
}

function Setup-ProjectStructure {
    Write-LogMessage "Erstelle Projektstruktur..." -Level "INFO" -Color $colors.Info
    
    # Erstelle Verzeichnisse
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
            Write-LogMessage "Verzeichnis erstellt: $dir" -Level "INFO" -Color $colors.Info
        }
    }
    
    # Erstelle settings.yaml, falls nicht vorhanden
    $settingsFile = Join-Path -Path $ScriptPath -ChildPath "config\settings.yaml"
    if (-not (Test-Path $settingsFile)) {
        @"
# AUTO-TAG Konfiguration
input_folder: $ScriptPath\data\input
output_folder: $ScriptPath\data\output
tag_mode: append
min_confidence: 80
min_face_size: 40
use_gpu: true
auto_download: true
offline_mode: false

# MinIO-Konfiguration (optional)
minio:
  endpoint: localhost:9000
  access_key: minioadmin
  secret_key: minioadmin
  secure: false
  input_bucket: images
  output_bucket: tagged-images

# Online-Identitätserkennung (optional)
online_identity:
  enabled: false
  api_key: 
  service: open_face  # open_face oder pimeyes
"@ | Out-File -FilePath $settingsFile -Encoding utf8
        Write-LogMessage "Konfigurationsdatei erstellt: settings.yaml" -Level "INFO" -Color $colors.Success
    }
    
    # Stelle sicher, dass start.ps1 existiert
    $startScript = Join-Path -Path $ScriptPath -ChildPath "start.ps1"
    if (-not (Test-Path $startScript)) {
        Write-LogMessage "Startskript fehlt: start.ps1" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    Write-LogMessage "Projektstruktur erfolgreich eingerichtet" -Level "INFO" -Color $colors.Success
    return $true
}

function Create-Shortcut {
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    $ShortcutFile = "$DesktopPath\AUTO-TAG.lnk"
    
    try {
        $WScriptShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WScriptShell.CreateShortcut($ShortcutFile)
        $Shortcut.TargetPath = "powershell.exe"
        $Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$ScriptPath\start.ps1`""
        $Shortcut.WorkingDirectory = $ScriptPath
        
        # Füge Icon hinzu, falls vorhanden
        $iconPath = Join-Path -Path $ScriptPath -ChildPath "assets\icon.ico"
        if (Test-Path $iconPath) {
            $Shortcut.IconLocation = "$iconPath,0"
        }
        
        $Shortcut.Save()
        Write-LogMessage "Desktop-Verknüpfung erstellt" -Level "INFO" -Color $colors.Success
        return $true
    }
    catch {
        Write-LogMessage "Fehler beim Erstellen der Verknüpfung: $_" -Level "WARNING" -Color $colors.Warning
        return $false
    }
}

function Copy-OfflineModels {
    if (-not $OfflineInstall -or -not $ModelsPath) {
        return $true
    }
    
    Write-LogMessage "Kopiere Offline-Modelle von $ModelsPath..." -Level "INFO" -Color $colors.Info
    
    if (-not (Test-Path $ModelsPath)) {
        Write-LogMessage "Offline-Modell-Pfad nicht gefunden: $ModelsPath" -Level "ERROR" -Color $colors.Error
        return $false
    }
    
    try {
        # Kopiere Modelle aus dem angegebenen Pfad
        $modelsDir = Join-Path -Path $ScriptPath -ChildPath "models"
        
        # Erstelle Verzeichnisstruktur
        $modelFolders = @("clip", "yolov8n", "facenet")
        foreach ($folder in $modelFolders) {
            $targetDir = Join-Path -Path $modelsDir -ChildPath $folder
            if (-not (Test-Path $targetDir)) {
                New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
            }
        }
        
        # Kopiere CLIP-Modell
        $clipSource = Join-Path -Path $ModelsPath -ChildPath "clip_vit_b32.pth"
        $clipTarget = Join-Path -Path $modelsDir -ChildPath "clip\clip_vit_b32.pth"
        if (Test-Path $clipSource) {
            Copy-Item -Path $clipSource -Destination $clipTarget -Force
            Write-LogMessage "CLIP-Modell kopiert" -Level "INFO" -Color $colors.Success
        }
        
        # Kopiere YOLOv8-Modell
        $yoloSource = Join-Path -Path $ModelsPath -ChildPath "yolov8n.pt"
        $yoloTarget = Join-Path -Path $modelsDir -ChildPath "yolov8n\yolov8n.pt"
        if (Test-Path $yoloSource) {
            Copy-Item -Path $yoloSource -Destination $yoloTarget -Force
            Write-LogMessage "YOLOv8-Modell kopiert" -Level "INFO" -Color $colors.Success
        }
        
        # Kopiere FaceNet-Modell
        $faceSource = Join-Path -Path $ModelsPath -ChildPath "facenet_model.pth"
        $faceTarget = Join-Path -Path $modelsDir -ChildPath "facenet\facenet_model.pth"
        if (Test-Path $faceSource) {
            Copy-Item -Path $faceSource -Destination $faceTarget -Force
            Write-LogMessage "FaceNet-Modell kopiert" -Level "INFO" -Color $colors.Success
        }
        
        Write-LogMessage "Offline-Modelle erfolgreich kopiert" -Level "INFO" -Color $colors.Success
        
        # Setze offline_mode in settings.yaml
        $settingsFile = Join-Path -Path $ScriptPath -ChildPath "config\settings.yaml"
        if (Test-Path $settingsFile) {
            $settings = Get-Content $settingsFile -Raw | ForEach-Object { $_ -replace "offline_mode: false", "offline_mode: true" }
            $settings | Out-File -FilePath $settingsFile -Encoding utf8 -Force
            Write-LogMessage "Offline-Modus in Einstellungen aktiviert" -Level "INFO" -Color $colors.Info
        }
        
        return $true
    }
    catch {
        Write-LogMessage "Fehler beim Kopieren der Offline-Modelle: $_" -Level "ERROR" -Color $colors.Error
        return $false
    }
}

function Main {
    Show-Header
    
    # Prüfe Administrator-Rechte
    Check-Administrator
    
    # Prüfe GPU-Verfügbarkeit
    Check-GPU
    
    # Prüfe Python-Installation
    $pythonInstalled = Check-Python
    
    # Wenn Python nicht gefunden wird, installiere Conda
    if (-not $pythonInstalled) {
        Write-LogMessage "Python $RequiredPythonVersion+ nicht gefunden, installiere Miniconda..." -Level "INFO" -Color $colors.Info
        
        if (-not $SkipPrompts) {
            $installConda = Read-Host "Miniconda installieren? (j/n)"
            if ($installConda -ne "j") {
                Write-LogMessage "Installation abgebrochen" -Level "INFO" -Color $colors.Info
                exit 1
            }
        }
        
        $condaInstalled = Install-Conda
        if (-not $condaInstalled) {
            Write-LogMessage "Fehler bei der Conda-Installation, Installation abgebrochen" -Level "ERROR" -Color $colors.Error
            exit 1
        }
        
        # Erstelle Conda-Umgebung
        $envCreated = Create-CondaEnvironment
        if (-not $envCreated) {
            Write-LogMessage "Fehler beim Erstellen der Conda-Umgebung, Installation abgebrochen" -Level "ERROR" -Color $colors.Error
            exit 1
        }
        
        # Installiere Abhängigkeiten
        $depsInstalled = Install-Dependencies
        if (-not $depsInstalled) {
            Write-LogMessage "Fehler bei der Installation der Abhängigkeiten, Installation abgebrochen" -Level "ERROR" -Color $colors.Error
            exit 1
        }
    }
    else {
        # Python ist installiert, verwende die normale pip-Installation
        Write-LogMessage "Verwende existierende Python-Installation" -Level "INFO" -Color $colors.Info
        
        # Installiere Abhängigkeiten mit pip
        try {
            Write-LogMessage "Installiere Abhängigkeiten mit pip..." -Level "INFO" -Color $colors.Info
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            pip install open-clip-torch ultralytics facenet-pytorch pillow tqdm pyyaml requests minio
            Write-LogMessage "Abhängigkeiten erfolgreich installiert" -Level "INFO" -Color $colors.Success
        }
        catch {
            Write-LogMessage "Fehler bei der Installation der Abhängigkeiten: $_" -Level "ERROR" -Color $colors.Error
            exit 1
        }
    }
    
    # Projektstruktur einrichten
    $structureSetup = Setup-ProjectStructure
    if (-not $structureSetup) {
        Write-LogMessage "Fehler beim Einrichten der Projektstruktur" -Level "ERROR" -Color $colors.Error
        exit 1
    }
    
    # Offline-Modelle kopieren, falls angegeben
    if ($OfflineInstall) {
        $modelsCopied = Copy-OfflineModels
        if (-not $modelsCopied) {
            Write-LogMessage "Fehler beim Kopieren der Offline-Modelle" -Level "WARNING" -Color $colors.Warning
            
            if (-not $SkipPrompts) {
                $continue = Read-Host "Trotzdem fortfahren? (j/n)"
                if ($continue -ne "j") {
                    Write-LogMessage "Installation abgebrochen" -Level "INFO" -Color $colors.Info
                    exit 1
                }
            }
        }
    }
    
    # Verknüpfung erstellen
    Create-Shortcut
    
    # Installation abgeschlossen
    Write-LogMessage "AUTO-TAG-Installation abgeschlossen!" -Level "INFO" -Color $colors.Success
    Write-LogMessage "Starten Sie AUTO-TAG über die Desktop-Verknüpfung oder mit .\start.ps1" -Level "INFO" -Color $colors.Info
    
    if (-not $SkipPrompts) {
        Read-Host "Drücken Sie Enter, um die Installation abzuschließen"
    }
}

# Starte Hauptprogramm
Main