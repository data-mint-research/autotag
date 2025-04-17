# AUTO-TAG Refaktorisierungsplan für Windows 11 Pro

## Aktuelle Probleme

1. **Komplexe Abhängigkeiten**: Poetry und mehrere Bibliotheken mit komplizierten Abhängigkeiten
2. **Unzuverlässige Modell-Downloads**: Links zu Modellen können veralten oder sind nicht robust
3. **Installations- und Laufzeitprobleme** auf Windows 11 Pro

## Lösungsansatz

Komplette Refaktorisierung mit Fokus auf:
- Robuste standalone Installation
- Lokale Modellverfügbarkeit
- Vereinfachte Abhängigkeiten

## 1. Architekturänderungen

### 1.1 Abhängigkeitsmanagement

Ersetzen von Poetry durch eine vereinfachte Lösung:
- Einzelne `environment.yml` für Conda
- Standallone-Bundle mit allen notwendigen Abhängigkeiten
- Conda-Umgebung für isolierte Installation

### 1.2 Modellverwaltung

- Lokale Modell-Speicherung in einem Git LFS Repository
- Alternativ: Azure Blob Storage oder S3-kompatibler Speicher für Modelle
- Offline-Installer mit gebündelten Modellen (via Git LFS)

### 1.3 Vereinfachtes Framework

- Vereinheitlichung der Modell-Interfaces
- Konsistente Error-Handling-Strategie
- Zentrale Konfiguration ohne Umgebungsvariablen

## 2. Technische Implementation

### 2.1 Basis-Setup

```python
# Minimal setup.py anstelle von pyproject.toml
from setuptools import setup, find_packages

setup(
    name="auto-tag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0", 
        "torchvision>=0.15.0",
        "open-clip-torch>=2.20.0",
        "ultralytics>=8.0.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    python_requires='>=3.10',
)
```

### 2.2 Conda-Environment

```yaml
# environment.yml - für einfache Einrichtung
name: autotag
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip=23.0
  - pytorch=2.0.1
  - torchvision=0.15.2
  - pytorch-cuda=11.8
  - pip:
    - open-clip-torch==2.20.0
    - ultralytics==8.0.20
    - facenet-pytorch==2.5.3
    - exiftool-vendored==0.5.3  # Gebündelte ExifTool-Version
```

### 2.3 Modell-Downloader

```python
# Robuster Modell-Downloader mit mehreren Fallbacks
import os
import requests
import hashlib
from tqdm import tqdm
import json
import time

MODEL_SOURCES = {
    "primary": "https://your-fast-cdn.com/models/",
    "fallback1": "https://github.com/your-org/autotag-models/releases/download/v1.0/",
    "fallback2": "https://huggingface.co/your-org/autotag-models/resolve/main/",
    "local": "./offline_models/"
}

MODEL_REGISTRY = {
    "clip": {
        "filename": "clip_vit_b32.pth",
        "size": 354355280,
        "sha256": "a4ccb0c288dd8c53e8ef99417d08e3731ecf29c9e39297a45f37c56e5366ca6e"
    },
    "yolov8n": {
        "filename": "yolov8n.pt", 
        "size": 6246000,
        "sha256": "6dbb68b8a5d19992f5a5e3b99d1ba466893dcf618bd5e8c0fe551705eb1f6315"
    }
}

def download_with_retries(model_name, dest_folder, max_retries=5):
    """Robust model downloader with retries and multiple sources"""
    os.makedirs(dest_folder, exist_ok=True)
    
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        raise ValueError(f"Unknown model: {model_name}")
    
    filename = model_info["filename"]
    expected_hash = model_info["sha256"]
    dest_path = os.path.join(dest_folder, filename)
    
    # Check if valid model already exists
    if os.path.exists(dest_path) and verify_hash(dest_path, expected_hash):
        print(f"✓ Model {model_name} already exists and is valid")
        return dest_path
        
    # Try each source with retries
    for source_name, base_url in MODEL_SOURCES.items():
        attempts = 0
        while attempts < max_retries:
            try:
                url = f"{base_url}{filename}"
                if source_name == "local":
                    local_path = os.path.join(base_url, filename)
                    if os.path.exists(local_path):
                        with open(local_path, "rb") as src, open(dest_path, "wb") as dst:
                            dst.write(src.read())
                        if verify_hash(dest_path, expected_hash):
                            print(f"✓ Model {model_name} copied from local storage")
                            return dest_path
                    # Skip to next source if local file doesn't exist or is invalid
                    break
                
                print(f"Downloading {model_name} from {source_name}...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get("content-length", model_info["size"]))
                with open(dest_path, "wb") as f, tqdm(
                    total=total_size, unit="B", unit_scale=True, 
                    desc=f"Downloading {filename}"
                ) as progress:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
                            
                if verify_hash(dest_path, expected_hash):
                    print(f"✓ Model {model_name} downloaded successfully from {source_name}")
                    return dest_path
                else:
                    print(f"× Hash verification failed for {model_name} from {source_name}")
                    attempts += 1
            
            except (requests.RequestException, IOError) as e:
                print(f"× Download error ({source_name}, attempt {attempts+1}/{max_retries}): {e}")
                attempts += 1
                time.sleep(2)  # Backoff before retry
        
    raise RuntimeError(f"Failed to download model {model_name} from all sources")

def verify_hash(file_path, expected_hash):
    """Verify SHA-256 hash of downloaded file"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()
    return actual_hash == expected_hash
```

### 2.4 ExifTool-Integration

Ersetzen der subprocess-basierten ExifTool-Aufrufe durch eine gebündelte Version:

```python
# exiftool_wrapper.py
from exiftool_vendored import ExifTool

class TagWriter:
    def __init__(self):
        self.et = ExifTool()
        self.et.start()
    
    def __del__(self):
        self.et.terminate()
    
    def write_tags(self, image_path, tags, mode="append"):
        """Write tags to image metadata using vendored ExifTool"""
        tag_list = ",".join(tags)
        
        try:
            if mode == "overwrite":
                self.et.execute(f"-XMP-digiKam:TagsList={tag_list}", "-overwrite_original", image_path)
            elif mode == "append":
                self.et.execute(f"-XMP-digiKam:TagsList+={tag_list}", "-overwrite_original", image_path)
            else:
                raise ValueError("mode must be 'append' or 'overwrite'")
            
            return True
        except Exception as e:
            print(f"Error writing tags to {image_path}: {e}")
            return False
```

## 3. Installation & Ausführung

### 3.1 Vereinfachtes Installations-Skript (Windows)

```powershell
# install.ps1
[CmdletBinding()]
param (
    [switch]$OfflineMode,
    [switch]$MinimalInstall,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$CondaBasePath = "$env:USERPROFILE\miniconda3"
$EnvName = "autotag"

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "Diese Installation sollte als Administrator ausgeführt werden."
    $continue = Read-Host "Fortfahren? (j/n)"
    if ($continue -ne "j") { exit }
}

# Install Miniconda if not exists
if (-not (Test-Path $CondaBasePath)) {
    Write-Host "Miniconda wird installiert..." -ForegroundColor Yellow
    $condaInstaller = "$env:TEMP\miniconda_installer.exe"
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $condaInstaller
    Start-Process -FilePath $condaInstaller -ArgumentList "/S /D=$CondaBasePath" -Wait
    Remove-Item $condaInstaller
}

# Add conda to path for this session
$env:PATH = "$CondaBasePath\Scripts;$CondaBasePath;$env:PATH"

# Initialize conda
Write-Host "Conda wird initialisiert..." -ForegroundColor Yellow
& "$CondaBasePath\Scripts\conda.exe" init powershell

# Create or update environment
if ((& "$CondaBasePath\Scripts\conda.exe" env list | Select-String -Pattern "^$EnvName\s") -or $ForceReinstall) {
    if ($ForceReinstall) {
        Write-Host "Vorhandene Umgebung wird entfernt..." -ForegroundColor Yellow
        & "$CondaBasePath\Scripts\conda.exe" env remove -n $EnvName -y
    }
}

Write-Host "Python-Umgebung wird erstellt..." -ForegroundColor Yellow
& "$CondaBasePath\Scripts\conda.exe" env create -f environment.yml -y

# Activate environment and install local package
Write-Host "AUTO-TAG wird installiert..." -ForegroundColor Yellow
& "$CondaBasePath\Scripts\activate.bat" $EnvName
& "$CondaBasePath\envs\$EnvName\python.exe" -m pip install -e .

# Create desktop shortcut
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutFile = "$DesktopPath\AUTO-TAG.lnk"
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutFile)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$PWD\start.ps1`""
$Shortcut.WorkingDirectory = $PWD
$Shortcut.IconLocation = "$PWD\assets\icon.ico,0"
$Shortcut.Save()

Write-Host "`n✓ Installation abgeschlossen!" -ForegroundColor Green
Write-Host "AUTO-TAG kann jetzt über die Desktop-Verknüpfung oder mit `".\start.ps1`" gestartet werden."
```

### 3.2 Überarbeitetes Startmenü

```powershell
# start.ps1 (überarbeitet)
[CmdletBinding()]
param (
    [switch]$SkipModelCheck,
    [switch]$BatchMode,
    [string]$InputFolder
)

$CondaBasePath = "$env:USERPROFILE\miniconda3"
$EnvName = "autotag"
$ScriptPath = $PSScriptRoot

Function Activate-Conda {
    $env:PATH = "$CondaBasePath\Scripts;$CondaBasePath;$env:PATH"
    & "$CondaBasePath\Scripts\activate.bat" $EnvName
}

# Aktiviere Conda-Umgebung
Activate-Conda

if ($BatchMode) {
    if ($InputFolder) {
        & python "$ScriptPath\batch_process.py" --input $InputFolder
    }
    else {
        & python "$ScriptPath\batch_process.py"
    }
    exit
}

# Zeige GUI-Menü an
Write-Host "`n=== AUTO-TAG System ===" -ForegroundColor Cyan

if (-not $SkipModelCheck) {
    Write-Host "`nPrüfe Modelle..." -ForegroundColor Yellow
    & python "$ScriptPath\check_models.py"
}

$options = @(
    "1. Einzelbild taggen"
    "2. Verzeichnis verarbeiten"
    "3. Einstellungen"
    "4. Modelle verwalten"
    "5. Beenden"
)

Write-Host "`nWähle eine Option:" -ForegroundColor Green
$options | ForEach-Object { Write-Host $_ }
$choice = Read-Host "Eingabe (1-5)"

switch ($choice) {
    "1" {
        Add-Type -AssemblyName System.Windows.Forms
        $openFileDialog = New-Object System.Windows.Forms.OpenFileDialog
        $openFileDialog.Filter = "Bilder (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png"
        $openFileDialog.Title = "Bild zum Taggen auswählen"
        if ($openFileDialog.ShowDialog() -eq "OK") {
            & python "$ScriptPath\process_single.py" --input $openFileDialog.FileName
        }
    }
    "2" {
        Add-Type -AssemblyName System.Windows.Forms
        $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
        $folderBrowser.Description = "Verzeichnis mit Bildern auswählen"
        if ($folderBrowser.ShowDialog() -eq "OK") {
            & python "$ScriptPath\batch_process.py" --input $folderBrowser.SelectedPath
        }
    }
    "3" {
        & python "$ScriptPath\settings.py"
    }
    "4" {
        & python "$ScriptPath\model_manager.py" --interactive
    }
    "5" {
        Write-Host "Auf Wiedersehen!" -ForegroundColor Yellow
        exit
    }
    default {
        Write-Host "Ungültige Eingabe. Bitte 1-5 eingeben." -ForegroundColor Red
    }
}
```

## 4. Modell-Bundle & Lokale Modelle

### 4.1 Offline-Modellpaket

Ein Modellpaket für den Offline-Betrieb erstellen:

1. Git LFS-Repository zum Speichern großer Modelldateien
2. Azure Blob Storage oder S3 mit vorkonfigurierten SAS-Tokens für direkte Downloads
3. Optionales lokales Modellpaket (30-40 GB) zum manuellen Kopieren

### 4.2 Modellkatalog

```json
{
  "models": {
    "clip": {
      "filename": "clip_vit_b32.pth",
      "size": 354355280,
      "sha256": "a4ccb0c288dd8c53e8ef99417d08e3731ecf29c9e39297a45f37c56e5366ca6e",
      "sources": {
        "git-lfs": "https://github.com/your-org/autotag-models/raw/main/clip/clip_vit_b32.pth",
        "azure": "https://autotagmodels.blob.core.windows.net/models/clip/clip_vit_b32.pth",
        "huggingface": "https://huggingface.co/your-org/autotag-models/resolve/main/clip/clip_vit_b32.pth"
      },
      "required": true
    },
    "yolov8n": {
      "filename": "yolov8n.pt",
      "size": 6246000,
      "sha256": "6dbb68b8a5d19992f5a5e3b99d1ba466893dcf618bd5e8c0fe551705eb1f6315",
      "sources": {
        "git-lfs": "https://github.com/your-org/autotag-models/raw/main/yolo/yolov8n.pt",
        "azure": "https://autotagmodels.blob.core.windows.net/models/yolo/yolov8n.pt",
        "huggingface": "https://huggingface.co/your-org/autotag-models/resolve/main/yolo/yolov8n.pt"
      },
      "required": true
    }
  },
  "offline_package": {
    "url": "https://github.com/your-org/autotag-models/releases/download/v1.0/all_models.zip",
    "size": 1500000000,
    "sha256": "..."
  },
  "version": "1.0.0"
}
```

## 5. Optimierungen für RTX 5090

### 5.1 GPU-Optimierungen

```python
# gpu_utils.py
import torch
import gc

def optimize_for_gpu():
    """Configure PyTorch for optimal performance on RTX 5090"""
    if not torch.cuda.is_available():
        print("CUDA ist nicht verfügbar - verwende CPU")
        return False
    
    # Set cuda device
    torch.cuda.set_device(0)
    
    # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set cudnn benchmark mode
    torch.backends.cudnn.benchmark = True
    
    # Disable gradient calculation for inference
    torch.set_grad_enabled(False)
    
    # Pre-allocate some CUDA memory to avoid fragmentation
    dummy = torch.zeros(1, device='cuda')
    del dummy
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"GPU-Optimierung aktiviert: {torch.cuda.get_device_name(0)}")
    return True

def cleanup_gpu():
    """Free GPU memory after processing"""
    torch.cuda.empty_cache()
    gc.collect()
```

### 5.2 Batch-Verarbeitung mit GPU-Optimierung

```python
# Hochoptimierte Batch-Verarbeitung für RTX 5090
import os
import torch
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor
from .gpu_utils import optimize_for_gpu, cleanup_gpu

def process_batch(image_folder, output_folder=None, batch_size=32):
    """Process multiple images in batches for optimal GPU utilization"""
    optimize_for_gpu()
    
    # Find all images in folder
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        process_image_batch(batch, image_folder, output_folder)
        
        # Free memory after each batch
        cleanup_gpu()
    
    print(f"Processed {len(image_files)} images")

def process_image_batch(image_batch, input_folder, output_folder):
    """Process a batch of images for optimal GPU throughput"""
    # Load all images in batch
    images = []
    for img_file in image_batch:
        try:
            img_path = os.path.join(input_folder, img_file)
            image = Image.open(img_path).convert('RGB')
            images.append((img_file, image))
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    # Process with models if any images were loaded
    if not images:
        return
    
    # Run models efficiently on the batch
    with torch.no_grad():
        # Scene/clothing recognition (batch processing)
        scene_results = run_clip_batch([img for _, img in images])
        
        # Person counting (batch processing)
        person_results = run_yolo_batch([img for _, img in images])
        
        # Face analysis (batch processing for efficiency)
        face_results = run_face_analysis_batch([img for _, img in images])
    
    # Process and save results
    for idx, (img_file, _) in enumerate(images):
        try:
            tags = []
            
            # Add all detected tags from various models
            # [Implementation details omitted for brevity]
            
            # Write tags to file
            input_path = os.path.join(input_folder, img_file)
            # [Tag writing implementation omitted for brevity]
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
```

## 6. Packaging & Distribution

Für einfache Verteilung und Installation:

1. **Self-contained Executable:**
   - PyInstaller mit UPX zur Komprimierung
   - Einbetten von Conda Mini-Environment

2. **Offline-Installation:**
   - Vollständiges Paket mit Python, Abhängigkeiten und Modellen (ca. 10-15 GB)

3. **Windows-Installer:**
   - NSIS oder Inno Setup Installer mit Optionen für:
     - Volle Installation (mit allen Modellen)
     - Minimale Installation (Modelle nach Bedarf)
     - GPU-Optimierte Installation

## Zeitplan und Ressourcen

1. **Phase 1**: Core-Refaktorisierung & Conda-Setup (3 Tage)
2. **Phase 2**: Robuster Modell-Downloader & Offline-Bundle (2 Tage)
3. **Phase 3**: GPU-Optimierung & Parallele Verarbeitung (2 Tage)
4. **Phase 4**: Distribution-Package & Installer (3 Tage)

## Technische Voraussetzungen

- Windows 11 Pro
- 64 GB RAM
- RTX 5090 (CUDA 12.x)
- 50 GB freier Speicherplatz (mit Modellen)

---

Diese Refaktorisierung behält alle Funktionalitäten des ursprünglichen AUTO-TAG Systems bei, während sie die Robustheit deutlich erhöht und die Abhängigkeiten minimiert. Der Fokus liegt auf zuverlässiger lokaler Installation und Ausführung auf Windows 11 Pro mit optimaler Nutzung der vorhandenen Hardware (64 GB RAM, RTX 5090).