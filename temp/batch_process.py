# batch_process.py - Batch-Verarbeitung von Bildern mit Unterordner-Unterstützung
import os
import argparse
import logging
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Lokale Module importieren
from config_loader import get_config, get_input_folder, get_output_folder, should_process_subdirectories
from models.clip_model import analyze_scene_clothing
from models.yolo_model import count_people
from models.face_model import analyze_faces
from models.identity_model import determine_identity
from tagging import TagWriter, generate_tags

# Logger einrichten
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auto-tag.log")
    ]
)
logger = logging.getLogger('auto-tag')

def find_images(folder_path, recursive=None, max_depth=None, current_depth=0):
    """Finde alle Bilder im angegebenen Ordner, optional rekursiv"""
    # Wenn recursive nicht angegeben wurde, verwende den Wert aus der Konfiguration
    if recursive is None:
        recursive = should_process_subdirectories()
    
    # Wenn max_depth nicht angegeben wurde, verwende den Wert aus der Konfiguration
    if max_depth is None:
        max_depth = get_config().get('MAX_RECURSION_DEPTH', 0)
    
    # Max depth 0 bedeutet keine Begrenzung
    if max_depth == 0:
        max_depth = float('inf')
    
    if not os.path.exists(folder_path):
        logger.error(f"Verzeichnis nicht gefunden: {folder_path}")
        return []
    
    image_files = []
    
    # Erhalte alle Dateien im aktuellen Verzeichnis
    try:
        items = os.listdir(folder_path)
        
        # Finde Bilder im aktuellen Verzeichnis
        for item in items:
            item_path = os.path.join(folder_path, item)
            
            # Wenn es eine Datei ist und die Endung passt
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(item_path)
            
            # Wenn es ein Verzeichnis ist und wir rekursiv suchen sollen
            elif os.path.isdir(item_path) and recursive and current_depth < max_depth:
                # Rekursiv Bilder in Unterverzeichnissen finden
                sub_images = find_images(
                    item_path, 
                    recursive=recursive,
                    max_depth=max_depth,
                    current_depth=current_depth + 1
                )
                image_files.extend(sub_images)
    
    except Exception as e:
        logger.error(f"Fehler beim Durchsuchen von {folder_path}: {e}")
    
    return image_files

def process_image(image_path, config, tag_writer, online_identity_enabled=False):
    """Verarbeite ein einzelnes Bild und füge Tags hinzu"""
    try:
        logger.info(f"Verarbeite: {image_path}")
        
        # Analysiere mit allen Modellen
        clip_result = analyze_scene_clothing(image_path)
        people_result = count_people(image_path)
        face_result = analyze_faces(image_path)
        
        # Gesichter extrahieren für Online-Identifikation (falls aktiviert)
        faces = None
        if online_identity_enabled:
            # Hier würde der Code stehen, um Gesichter zu extrahieren
            # In einer vollständigen Implementierung würde dies von der face_model-Funktion bereitgestellt
            pass
        
        # Ordnernamen aus Pfad extrahieren
        folder_name = os.path.basename(os.path.dirname(image_path))
        
        # Identitäts-Tags ermitteln
        identity_tags = determine_identity(
            image_path, 
            folder_name, 
            faces,
            use_online=online_identity_enabled
        )
        
        # Tags generieren
        tags = generate_tags(
            clip_result, 
            people_result, 
            face_result, 
            folder_name, 
            identity_tags,
            min_confidence=config.get('MIN_CONFIDENCE_PERCENT', 80)
        )
        
        # Tags in Datei schreiben
        success = tag_writer.write_tags(
            image_path, 
            tags, 
            mode=config.get('TAG_MODE', 'append')
        )
        
        if success:
            # Wenn ein Ausgabeordner angegeben ist und sich vom Eingabeordner unterscheidet
            output_folder = config.get('OUTPUT_FOLDER')
            if output_folder and output_folder != os.path.dirname(image_path):
                # Erstelle relative Pfadstruktur
                input_folder = config.get('INPUT_FOLDER')
                if image_path.startswith(input_folder):
                    rel_path = os.path.relpath(image_path, input_folder)
                    output_path = os.path.join(output_folder, rel_path)
                    
                    # Stelle sicher, dass das Zielverzeichnis existiert
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Kopiere die Datei (mit Tags)
                    import shutil
                    shutil.copy2(image_path, output_path)
                    logger.info(f"Kopiert nach: {output_path}")
        
        return image_path, tags, success
    
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung von {image_path}: {e}")
        return image_path, [], False

def batch_process(input_folder=None, recursive=None, max_workers=None, dry_run=False):
    """Verarbeite alle Bilder im angegebenen Ordner"""
    # Lade Konfiguration
    config = get_config()
    
    # Verwende Standardwerte aus der Konfiguration, wenn nicht angegeben
    if input_folder is None:
        input_folder = get_input_folder()
    
    if recursive is None:
        recursive = should_process_subdirectories()
    
    if max_workers is None:
        max_workers = config.get('NUM_WORKERS', 4)
    
    # Finde alle Bilder im angegebenen Ordner
    logger.info(f"Suche Bilder in {input_folder} {'(rekursiv)' if recursive else ''}...")
    image_files = find_images(input_folder, recursive=recursive)
    
    if not image_files:
        logger.warning(f"Keine Bilder gefunden in {input_folder}")
        return []
    
    logger.info(f"Gefunden: {len(image_files)} Bilder")
    
    # Wenn Dry-Run, gib einfach die Liste der gefundenen Bilder zurück
    if dry_run:
        return image_files
    
    # Tag-Writer initialisieren
    tag_writer = TagWriter()
    
    # Online-Identitätserkennung aktiviert?
    online_identity_enabled = config.get('ONLINE_IDENTITY_ENABLED', False)
    
    # Initialisiere Ergebnisliste
    results = []
    start_time = time.time()
    
    # Verarbeite Bilder parallel mit ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Erstelle Future-Objekte für jedes Bild
        futures = {
            executor.submit(
                process_image, 
                image_path, 
                config, 
                tag_writer, 
                online_identity_enabled
            ): image_path for image_path in image_files
        }
        
        # Verarbeite die Ergebnisse mit Fortschrittsanzeige
        for future in tqdm(
            futures, 
            desc="Verarbeite Bilder", 
            unit="bild",
            ncols=100
        ):
            image_path, tags, success = future.result()
            results.append({
                "path": image_path,
                "tags": tags,
                "success": success
            })
    
    # Speichere Ergebnisse in JSON-Datei
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Verarbeitung abgeschlossen: {len(results)} Bilder in {elapsed_time:.2f} Sekunden")
    
    # Erstelle Ergebnisbericht
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Erfolgreich: {success_count}/{len(results)} Bilder")
    
    # Speichere detaillierte Ergebnisse
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"batch_results_{timestamp}.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "input_folder": input_folder,
            "recursive": recursive,
            "total_images": len(image_files),
            "success_count": success_count,
            "elapsed_time": elapsed_time,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Ergebnisse gespeichert in: {result_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="AUTO-TAG Batch-Verarbeitung")
    parser.add_argument("--input", help="Eingabeordner mit Bildern")
    parser.add_argument("--recursive", action="store_true", help="Unterordner einbeziehen")
    parser.add_argument("--workers", type=int, help="Anzahl paralleler Worker")
    parser.add_argument("--dry-run", action="store_true", help="Nur Bilder auflisten, keine Verarbeitung")
    args = parser.parse_args()
    
    # Führe Batch-Verarbeitung aus
    results = batch_process(
        input_folder=args.input,
        recursive=args.recursive, 
        max_workers=args.workers,
        dry_run=args.dry_run
    )
    
    if args.dry_run:
        print("\nGefundene Bilder:")
        for image_path in results:
            print(f"  {image_path}")
        print(f"\nGesamt: {len(results)} Bilder")
    
    return 0 if results is not None else 1

if __name__ == "__main__":
    exit(main())