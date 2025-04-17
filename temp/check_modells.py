# check_models.py - Robuster Modell-Manager
import os
import sys
import json
import hashlib
import requests
from tqdm import tqdm
import time
import logging
import yaml
from pathlib import Path

# Konfiguration
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
MODEL_CATALOG = os.path.join(CONFIG_DIR, "model_catalog.json")
SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.yaml")

# Logging einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto-tag.log"))
    ]
)
logger = logging.getLogger('auto-tag')

# Prüfe, ob CUDA verfügbar ist
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        CUDA_DEVICE = torch.cuda.get_device_name(0)
        logger.info(f"CUDA verfügbar: {CUDA_DEVICE}")
    else:
        logger.warning("CUDA nicht verfügbar. CPU-Modus wird verwendet (langsamer).")
except ImportError:
    HAS_CUDA = False
    logger.warning("PyTorch nicht installiert. CUDA-Check übersprungen.")

def load_settings():
    """Lade Benutzereinstellungen"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Standardeinstellungen
    default_settings = {
        'input_folder': os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input"),
        'output_folder': os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output"),
        'tag_mode': 'append',
        'min_confidence': 80,
        'min_face_size': 40,
        'use_gpu': HAS_CUDA,
        'auto_download': True,
        'offline_mode': False,
    }
    
    # Erstelle Standardkonfiguration, wenn nicht vorhanden
    if not os.path.exists(SETTINGS_FILE):
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            yaml.dump(default_settings, f, default_flow_style=False)
        return default_settings
    
    # Lade vorhandene Einstellungen
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = yaml.safe_load(f)
            # Füge fehlende Standardwerte hinzu
            for key, value in default_settings.items():
                if key not in settings:
                    settings[key] = value
        return settings
    except Exception as e:
        logger.error(f"Fehler beim Laden der Einstellungen: {e}")
        return default_settings

def load_model_catalog():
    """Lade oder erstelle den Modellkatalog"""
    if not os.path.exists(MODEL_CATALOG):
        # Fallback-Katalog, wenn keine Internetverbindung vorhanden ist
        catalog = {
            "models": {
                "clip": {
                    "filename": "clip_vit_b32.pth",
                    "size": 354355280,
                    "sha256": "a4ccb0c288dd8c53e8ef99417d08e3731ecf29c9e39297a45f37c56e5366ca6e",
                    "sources": {
                        "github": "https://github.com/openai/CLIP/releases/download/v1.0/clip_vit_b32.pth",
                        "huggingface": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin",
                        "local": "./offline_models/clip_vit_b32.pth"
                    },
                    "required": True
                },
                "yolov8n": {
                    "filename": "yolov8n.pt",
                    "size": 6246000,
                    "sha256": "6dbb68b8a5d19992f5a5e3b99d1ba466893dcf618bd5e8c0fe551705eb1f6315",
                    "sources": {
                        "github": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        "huggingface": "https://huggingface.co/Ultralytics/yolov8/resolve/main/yolov8n.pt",
                        "local": "./offline_models/yolov8n.pt"
                    },
                    "required": True
                },
                "facenet": {
                    "filename": "facenet_model.pth",
                    "size": 89456789,
                    "sha256": "5e4c2578ffeff9e1dde7d0d10e025c4319b13e4d058577cf430c8df5cf613c45",
                    "sources": {
                        "github": "https://github.com/timesler/facenet-pytorch/releases/download/v2.5.2/20180402-114759-vggface2.pt",
                        "local": "./offline_models/facenet_model.pth"
                    },
                    "required": True
                }
            },
            "version": "1.0.0"
        }
        
        os.makedirs(os.path.dirname(MODEL_CATALOG), exist_ok=True)
        with open(MODEL_CATALOG, 'w') as f:
            json.dump(catalog, f, indent=2)
        return catalog
    
    try:
        with open(MODEL_CATALOG, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modellkatalogs: {e}")
        return None

def verify_hash(file_path, expected_hash):
    """Verifiziere SHA-256 Hash einer Datei"""
    if not os.path.exists(file_path):
        return False
        
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        return actual_hash == expected_hash
    except Exception as e:
        logger.error(f"Hash-Prüfung fehlgeschlagen: {e}")
        return False

def download_with_progress(url, dest_path, expected_size=None):
    """Lade Datei mit Fortschrittsanzeige herunter"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        if total_size == 0 and expected_size:
            total_size = expected_size
            
        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True,
            desc=os.path.basename(dest_path)
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        return True
    except Exception as e:
        logger.error(f"Download fehlgeschlagen: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def check_and_download_model(model_name, model_info, settings, interactive=True):
    """Prüfe und lade ein Modell bei Bedarf herunter"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, model_info["filename"])
    expected_hash = model_info.get("sha256")
    
    # Prüfe, ob eine gültige Modelldatei existiert
    if os.path.exists(model_path) and (not expected_hash or verify_hash(model_path, expected_hash)):
        logger.info(f"✓ Modell {model_name} ist gültig")
        return True
    
    # Wenn wir im Offline-Modus sind, suche im lokalen Offline-Ordner
    if settings.get('offline_mode', False):
        offline_path = os.path.join("offline_models", model_info["filename"])
        if os.path.exists(offline_path):
            try:
                # Kopiere Datei aus Offline-Ordner
                with open(offline_path, "rb") as src, open(model_path, "wb") as dst:
                    dst.write(src.read())
                if not expected_hash or verify_hash(model_path, expected_hash):
                    logger.info(f"✓ Modell {model_name} aus Offline-Speicher kopiert")
                    return True
            except Exception as e:
                logger.error(f"Fehler beim Kopieren aus Offline-Speicher: {e}")
        
        if model_info.get("required", False):
            logger.error(f"× Erforderliches Modell {model_name} nicht gefunden im Offline-Modus")
        else:
            logger.warning(f"× Optionales Modell {model_name} nicht gefunden im Offline-Modus")
        return False
    
    # Frage nach Erlaubnis zum Herunterladen, wenn interaktiv
    if not settings.get('auto_download', True) and interactive:
        print(f"\nModell '{model_name}' fehlt oder ist ungültig.")
        choice = input("Jetzt herunterladen? (j/n): ").strip().lower()
        if choice != "j":
            logger.info(f"Download von {model_name} übersprungen")
            return False
    
    # Versuche Download von allen Quellen
    sources = model_info.get("sources", {})
    if not sources:
        logger.error(f"Keine Download-Quellen für {model_name} gefunden")
        return False
    
    for source_name, url in sources.items():
        if source_name == "local":
            continue  # Lokale Quelle wurde bereits geprüft
            
        logger.info(f"Lade {model_name} von {source_name}...")
        if download_with_progress(url, model_path, model_info.get("size")):
            if not expected_hash or verify_hash(model_path, expected_hash):
                logger.info(f"✓ Modell {model_name} erfolgreich heruntergeladen und verifiziert")
                return True
            else:
                logger.error(f"× Hash-Verifizierung fehlgeschlagen für {model_name} von {source_name}")
                os.remove(model_path)
    
    # Wenn wir hier ankommen, sind alle Download-Versuche fehlgeschlagen
    if model_info.get("required", False):
        logger.error(f"× Download des erforderlichen Modells {model_name} fehlgeschlagen")
    else:
        logger.warning(f"× Download des optionalen Modells {model_name} fehlgeschlagen")
    return False

def check_all_models(interactive=True):
    """Prüfe und lade alle Modelle"""
    settings = load_settings()
    catalog = load_model_catalog()
    
    if not catalog or "models" not in catalog:
        logger.error("Modellkatalog konnte nicht geladen werden")
        return False
    
    success = True
    missing_required = []
    
    for model_name, model_info in catalog["models"].items():
        result = check_and_download_model(model_name, model_info, settings, interactive)
        if not result and model_info.get("required", False):
            missing_required.append(model_name)
            success = False
    
    if success:
        logger.info("✓ Alle erforderlichen Modelle sind verfügbar")
    else:
        logger.error(f"× Folgende erforderliche Modelle fehlen: {', '.join(missing_required)}")
        
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AUTO-TAG Modellverwaltung")
    parser.add_argument("--interactive", action="store_true", help="Interaktiver Modus")
    parser.add_argument("--force-check", action="store_true", help="Erzwinge Neuprüfung aller Modelle")
    args = parser.parse_args()
    
    success = check_all_models(interactive=args.interactive)
    
    if success:
        print("\n✓ Alle erforderlichen Modelle sind bereit.")
    else:
        print("\n× Einige erforderliche Modelle fehlen. AUTO-TAG wird möglicherweise nicht korrekt funktionieren.")