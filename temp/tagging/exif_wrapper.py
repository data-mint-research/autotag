# tagging/exiftool_wrapper.py
import os
import subprocess
import logging
import platform
import tempfile
import shutil
from pathlib import Path

logger = logging.getLogger('auto-tag')

class TagWriter:
    def __init__(self):
        self.exiftool_path = self._find_exiftool()
        if not self.exiftool_path:
            logger.warning("ExifTool nicht gefunden. Versuche eingebettete Version zu verwenden.")
            self.exiftool_path = self._setup_embedded_exiftool()
            
        if not self.exiftool_path:
            logger.error("ExifTool konnte nicht gefunden oder eingerichtet werden!")
        else:
            logger.info(f"ExifTool gefunden: {self.exiftool_path}")
    
    def _find_exiftool(self):
        """Finde ExifTool im System-Pfad"""
        # Windows: Suche exiftool.exe
        if platform.system() == "Windows":
            # Versuche direkt im aktuellen Verzeichnis
            local_exiftool = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin', 'exiftool.exe')
            if os.path.exists(local_exiftool):
                return local_exiftool
                
            # Versuche im System-Pfad zu finden
            try:
                result = subprocess.run(["where", "exiftool"], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().split('\n')[0]
            except:
                pass
        else:
            # Linux/Mac: Suche exiftool im PATH
            try:
                result = subprocess.run(["which", "exiftool"], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except:
                pass
        
        return None
    
    def _setup_embedded_exiftool(self):
        """Extrahiere und richte die eingebettete ExifTool-Version ein"""
        bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
        os.makedirs(bin_dir, exist_ok=True)
        
        # Windows: Extrahiere die eingebettete exiftool.exe
        if platform.system() == "Windows":
            embedded_exiftool = os.path.join(bin_dir, 'exiftool.exe')
            
            # Wenn die Datei bereits existiert, verwende sie
            if os.path.exists(embedded_exiftool):
                return embedded_exiftool
                
            # Ansonsten müssten wir exiftool.exe aus einer Ressource extrahieren
            # (Dies würde in einem echten Projekt implementiert werden)
            logger.error("Eingebettete ExifTool-Version nicht verfügbar. Bitte ExifTool manuell installieren.")
            return None
        else:
            # Linux/Mac: Hier würden wir eine plattformspezifische Version extrahieren
            logger.error("Eingebettete ExifTool-Version für diese Plattform nicht verfügbar.")
            return None
    
    def test_exiftool(self):
        """Teste, ob ExifTool funktioniert"""
        if not self.exiftool_path:
            return False
            
        try:
            result = subprocess.run([self.exiftool_path, "-ver"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"ExifTool-Test fehlgeschlagen: {e}")
            return False
    
    def write_tags(self, image_path, tags, mode="append"):
        """Schreibe Tags in die Bilddatei"""
        if not self.exiftool_path:
            logger.error("ExifTool nicht verfügbar. Tags können nicht geschrieben werden.")
            return False
            
        if not os.path.exists(image_path):
            logger.error(f"Bilddatei nicht gefunden: {image_path}")
            return False
            
        # Formatiere Tags als kommagetrennte Liste
        tag_list = ",".join(tags)
        
        try:
            if mode == "overwrite":
                cmd = [
                    self.exiftool_path,
                    f"-XMP-digiKam:TagsList={tag_list}",
                    "-overwrite_original",
                    image_path
                ]
            elif mode == "append":
                # Lese vorhandene Tags
                existing_tags = self.read_tags(image_path)
                
                # Wenn bereits vorhanden, keine doppelten Tags
                combined_tags = list(set(existing_tags + tags))
                tag_list = ",".join(combined_tags)
                
                cmd = [
                    self.exiftool_path,
                    f"-XMP-digiKam:TagsList={tag_list}",
                    "-overwrite_original",
                    image_path
                ]
            else:
                logger.error(f"Ungültiger Modus: {mode}. Verwende 'append' oder 'overwrite'.")
                return False
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ExifTool-Fehler: {result.stderr}")
                return False
                
            logger.info(f"Tags geschrieben für {os.path.basename(image_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Schreiben der Tags: {e}")
            return False
    
    def read_tags(self, image_path):
        """Lese vorhandene Tags aus der Bilddatei"""
        if not self.exiftool_path or not os.path.exists(image_path):
            return []
            
        try:
            cmd = [
                self.exiftool_path,
                "-XMP-digiKam:TagsList",
                "-s",
                "-s",
                "-s",
                image_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0 or not result.stdout.strip():
                return []
                
            tags = result.stdout.strip().split(",")
            return [tag.strip() for tag in tags if tag.strip()]
            
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Tags: {e}")
            return []


# tagging/tag_generator.py
import os
import logging

logger = logging.getLogger('auto-tag')

# Definierte Reihenfolge für Tags
SORT_ORDER = [
    "person", "gender", "location", "age",
    "identity", "identity/source",
    "clothing", "mood", "people", "scene", "roomtype"
]

def generate_tags(clip_result, people_result, face_result, folder_name, identity_tags, min_confidence=80):
    """Generiere Tags aus Analyseergebnissen"""
    tags = []
    
    # CLIP-Ergebnisse
    if clip_result:
        # Szene (indoor/outdoor)
        if clip_result.get("scene") and clip_result["scene"][1]*100 >= min_confidence:
            tags.append(f"scene/{clip_result['scene'][0]}")
            
        # Raumtyp (kitchen, bedroom, etc.)
        if clip_result.get("roomtype") and clip_result["roomtype"][1]*100 >= min_confidence:
            tags.append(f"roomtype/{clip_result['roomtype'][0]}")
            
        # Kleidung (dressed/naked)
        if clip_result.get("clothing") and clip_result["clothing"][1]*100 >= min_confidence:
            tags.append(f"clothing/{clip_result['clothing'][0]}")
    
    # YOLO-Ergebnisse (Personenzählung)
    if people_result in ["solo", "group"]:
        tags.append(f"people/{people_result}")
    
    # Gesichtsanalyse-Ergebnisse
    if face_result:
        if face_result.get("gender"):
            tags.append(f"gender/{face_result['gender']}")
        if face_result.get("age"):
            tags.append(f"age/{face_result['age']}")
        if face_result.get("mood"):
            tags.append(f"mood/{face_result['mood']}")
    
    # Ordnernamen-Analyse für Person/Ort
    if folder_name:
        # Person-Tag aus Ordnername
        if folder_name and folder_name.replace("_", "").isalpha():
            tags.append(f"person/{folder_name}")
            
        # Orts-Tag aus Ordnername
        if folder_name.lower() in ["berlin", "rome", "paris"]:
            tags.append(f"location/{folder_name}")
    
    # Identitäts-Tags hinzufügen
    if identity_tags:
        tags.extend(identity_tags)
    
    # Sortiere Tags in definierter Reihenfolge
    return sort_tags(tags)

def sort_tags(tag_list):
    """Sortiere Tags nach definierter Reihenfolge"""
    def tag_key(tag):
        for i, prefix in enumerate(SORT_ORDER):
            if tag.startswith(prefix + "/"):
                return i
        return len(SORT_ORDER)
    
    return sorted(tag_list, key=tag_key)


# tagging/__init__.py
from .exiftool_wrapper import TagWriter
from .tag_generator import generate_tags, sort_tags

__all__ = ['TagWriter', 'generate_tags', 'sort_tags']


# process_single.py - Verarbeitung einzelner Bilder
import os
import argparse
import logging
from PIL import Image
import yaml

# Lokale Module importieren
from config import load_settings
from models.clip_model import analyze_scene_clothing
from models.yolo_model import count_people
from models.face_model import analyze_faces
from models.identity_model import determine_identity
from tagging import TagWriter, generate_tags

logger = logging.getLogger('auto-tag')

def process_image(image_path, settings):
    """Verarbeite ein einzelnes Bild"""
    if not os.path.exists(image_path):
        logger.error(f"Bilddatei nicht gefunden: {image_path}")
        return False

    logger.info(f"Verarbeite Bild: {os.path.basename(image_path)}")
    
    try:
        # Bild öffnen (als Test)
        img = Image.open(image_path)
        img.close()
        
        # Analysiere mit verfügbaren Modellen
        # 1. CLIP - Szene & Kleidung
        clip_result = analyze_scene_clothing(image_path)
        
        # 2. YOLO - Personenzählung
        people_result = count_people(image_path)
        
        # 3. Gesichtsanalyse
        face_result = analyze_faces(image_path)
        
        # 4. Identitätsprüfung
        folder_name = os.path.basename(os.path.dirname(image_path))
        identity_tags = determine_identity(image_path, folder_name)
        
        # Generiere Tags
        tags = generate_tags(
            clip_result, 
            people_result, 
            face_result, 
            folder_name, 
            identity_tags,
            min_confidence=settings.get('min_confidence', 80)
        )
        
        # Schreibe Tags
        tag_writer = TagWriter()
        success = tag_writer.write_tags(
            image_path, 
            tags, 
            mode=settings.get('tag_mode', 'append')
        )
        
        # Kopiere Bild in Ausgabeverzeichnis, wenn konfiguriert
        output_folder = settings.get('output_folder')
        if success and output_folder and os.path.exists(output_folder):
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            if image_path != output_path:  # Vermeide Überschreiben der Originaldatei
                img = Image.open(image_path)
                img.save(output_path)
                logger.info(f"Kopiert nach: {os.path.basename(output_path)}")
        
        logger.info(f"Verarbeitung abgeschlossen: {len(tags)} Tags hinzugefügt")
        logger.info(f"Tags: {tags}")
        return True
        
    except Exception as e:
        logger.error(f"Fehler bei der Bildverarbeitung: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AUTO-TAG Einzelbildverarbeitung")
    parser.add_argument("--input", required=True, help="Pfad zur Bilddatei")
    args = parser.parse_args()
    
    # Einstellungen laden
    settings = load_settings()
    
    # Bild verarbeiten
    success = process_image(args.input, settings)
    
    if success:
        print(f"\n✓ Bild erfolgreich verarbeitet: {os.path.basename(args.input)}")
    else:
        print(f"\n× Fehler bei der Verarbeitung von: {os.path.basename(args.input)}")