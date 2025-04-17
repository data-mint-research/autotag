# start.ps1 - Vereinfachtes Startskript für AUTO-TAG
[CmdletBinding()]
param (
    [Parameter(Position=0)]
    [ValidateSet("image", "folder", "minio", "models", "settings")]
    [string]$Mode = "",
    
    [Parameter(Position=1)]
    [string]$Path = "",
    
    [switch]$Recursive,
    
    [switch]$Help
)

# Grundlegende Funktionen
# ----------------------

function Show-Help {
    Write-Host "`nAUTO-TAG System - Hilfe" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host "`nVerwendung:`n"
    Write-Host "  .\start.ps1                         # Starte interaktives Menü"
    Write-Host "  .\start.ps1 image [Bildpfad]        # Verarbeite ein einzelnes Bild"
    Write-Host "  .\start.ps1 folder [Ordnerpfad]     # Verarbeite einen Ordner"
    Write-Host "  .\start.ps1 minio                   # Verarbeite MinIO-Buckets"
    Write-Host "  .\start.ps1 models                  # Modelle prüfen/herunterladen"
    Write-Host "  .\start.ps1 settings                # Einstellungen anzeigen/bearbeiten"
    Write-Host "`nOptionen:`n"
    Write-Host "  -Recursive                          # Bei Ordnerverarbeitung auch Unterordner einbeziehen"
    Write-Host "  -Help                               # Diese Hilfe anzeigen"
    Write-Host "`nBeispiele:`n"
    Write-Host "  .\start.ps1 image C:\Bilder\foto.jpg"
    Write-Host "  .\start.ps1 folder C:\Bilder -Recursive"
}

function Initialize-Environment {
    # Prüfe, ob Python verfügbar ist
    try {
        $pythonVersion = python --version 2>&1
    }
    catch {
        Write-Host "Fehler: Python konnte nicht gefunden werden." -ForegroundColor Red
        Write-Host "Bitte installiere Python 3.10+ und führe install.ps1 aus." -ForegroundColor Red
        exit 1
    }
    
    # Prüfe, ob die .env Datei existiert
    $envFile = Join-Path -Path $PSScriptRoot -ChildPath ".env"
    $envExampleFile = Join-Path -Path $PSScriptRoot -ChildPath ".env.example"
    
    if (-not (Test-Path $envFile)) {
        if (Test-Path $envExampleFile) {
            Write-Host "Information: .env Datei nicht gefunden, kopiere .env.example..." -ForegroundColor Yellow
            Copy-Item -Path $envExampleFile -Destination $envFile
            Write-Host "Kopiert! Bitte passe die .env Datei an deine Bedürfnisse an." -ForegroundColor Green
        }
        else {
            Write-Host "Warnung: Weder .env noch .env.example gefunden. Die Standardeinstellungen werden verwendet." -ForegroundColor Yellow
        }
    }
    
    # Stelle sicher, dass Konfigurations-Loader vorhanden ist
    $configLoaderPath = Join-Path -Path $PSScriptRoot -ChildPath "config_loader.py"
    if (-not (Test-Path $configLoaderPath)) {
        Write-Host "Fehler: config_loader.py nicht gefunden. Installation könnte beschädigt sein." -ForegroundColor Red
        exit 1
    }
}

function Select-FilePath {
    param (
        [string]$Title = "Datei auswählen",
        [string]$Filter = "Bilder (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png"
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
        [string]$Description = "Verzeichnis auswählen"
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
        $ImagePath = Select-FilePath -Title "Bild zum Taggen auswählen"
        if (-not $ImagePath) {
            Write-Host "Abgebrochen." -ForegroundColor Yellow
            return
        }
    }
    
    if (-not (Test-Path $ImagePath)) {
        Write-Host "Fehler: Bilddatei nicht gefunden: $ImagePath" -ForegroundColor Red
        return
    }
    
    Write-Host "`nVerarbeite Bild: $ImagePath" -ForegroundColor Cyan
    
    # Führe Bildverarbeitung aus
    python "$PSScriptRoot\process_single.py" --input "$ImagePath"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nBild erfolgreich verarbeitet!" -ForegroundColor Green
    }
    else {
        Write-Host "`nFehler bei der Bildverarbeitung." -ForegroundColor Red
    }
}

function Invoke-ProcessFolder {
    param (
        [string]$FolderPath,
        [bool]$ProcessRecursively
    )
    
    if (-not $FolderPath) {
        $FolderPath = Select-FolderPath -Description "Verzeichnis mit Bildern auswählen"
        if (-not $FolderPath) {
            Write-Host "Abgebrochen." -ForegroundColor Yellow
            return
        }
        
        # Wenn kein Pfad übergeben wurde, frage nach der Rekursion
        if (-not $ProcessRecursively) {
            $recursiveChoice = Read-Host "Unterordner einbeziehen? (j/n)"
            $ProcessRecursively = $recursiveChoice -eq "j"
        }
    }
    
    if (-not (Test-Path $FolderPath)) {
        Write-Host "Fehler: Verzeichnis nicht gefunden: $FolderPath" -ForegroundColor Red
        return
    }
    
    Write-Host "`nVerarbeite Verzeichnis: $FolderPath" -ForegroundColor Cyan
    if ($ProcessRecursively) {
        Write-Host "Unterordner werden einbezogen." -ForegroundColor Cyan
    }
    
    # Führe Verzeichnisverarbeitung aus
    if ($ProcessRecursively) {
        python "$PSScriptRoot\batch_process.py" --input "$FolderPath" --recursive
    }
    else {
        python "$PSScriptRoot\batch_process.py" --input "$FolderPath"
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nVerzeichnis erfolgreich verarbeitet!" -ForegroundColor Green
    }
    else {
        Write-Host "`nFehler bei der Verzeichnisverarbeitung." -ForegroundColor Red
    }
}

function Invoke-ProcessMinio {
    Write-Host "`nStarte MinIO-Verarbeitung..." -ForegroundColor Cyan
    
    # Führe MinIO-Verarbeitung aus
    python "$PSScriptRoot\process_minio.py"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nMinIO-Buckets erfolgreich verarbeitet!" -ForegroundColor Green
    }
    else {
        Write-Host "`nFehler bei der MinIO-Verarbeitung." -ForegroundColor Red
    }
}

function Invoke-ManageModels {
    Write-Host "`nModellverwaltung wird gestartet..." -ForegroundColor Cyan
    
    # Führe Modellprüfung aus
    python "$PSScriptRoot\check_models.py" --interactive
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nModellprüfung abgeschlossen." -ForegroundColor Green
    }
    else {
        Write-Host "`nFehler bei der Modellprüfung." -ForegroundColor Red
    }
}

function Invoke-ManageSettings {
    Write-Host "`nKonfiguration..." -ForegroundColor Cyan
    
    # Zeige aktuelle Konfiguration an
    python "$PSScriptRoot\config_loader.py"
    
    # Frage, ob Einstellungen bearbeitet werden sollen
    $editChoice = Read-Host "`nMöchtest du die .env Datei bearbeiten? (j/n)"
    if ($editChoice -eq "j") {
        # Öffne .env-Datei im Standard-Editor
        $envFile = Join-Path -Path $PSScriptRoot -ChildPath ".env"
        Start-Process notepad $envFile -Wait
        Write-Host "Einstellungen wurden gespeichert. Starte AUTO-TAG neu, um die Änderungen zu übernehmen." -ForegroundColor Green
    }
}

function Show-Menu {
    Clear-Host
    Write-Host "`n=============================" -ForegroundColor Cyan
    Write-Host "      AUTO-TAG SYSTEM      " -ForegroundColor Cyan
    Write-Host "=============================" -ForegroundColor Cyan
    
    Write-Host "`n1. Einzelbild taggen"
    Write-Host "2. Verzeichnis verarbeiten"
    Write-Host "3. MinIO-Buckets verarbeiten"
    Write-Host "4. Modelle prüfen/herunterladen"
    Write-Host "5. Einstellungen anzeigen/bearbeiten"
    Write-Host "6. Beenden"
    
    $choice = Read-Host "`nWähle eine Option (1-6)"
    
    return $choice
}

# Hauptprogramm
# ------------

# Zeige Hilfe, wenn angefordert
if ($Help) {
    Show-Help
    exit 0
}

# Initialisiere Umgebung
Initialize-Environment

# Wenn kein Modus angegeben wurde, zeige Menü
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
            "6" { $running = $false }
            default { 
                Write-Host "`nUngültige Eingabe. Bitte 1-6 wählen." -ForegroundColor Yellow
                Start-Sleep -Seconds 1
            }
        }
        
        if ($running -and $choice -in 1..5) {
            Write-Host "`nDrücke eine Taste, um fortzufahren..."
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
    }
    
    Write-Host "`nAUTO-TAG wird beendet. Auf Wiedersehen!" -ForegroundColor Cyan
    exit 0
}

# Verarbeite entsprechend des angegebenen Modus
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
    default {
        Write-Host "Unbekannter Modus: $Mode" -ForegroundColor Red
        Show-Help
        exit 1
    }
}

exit 0