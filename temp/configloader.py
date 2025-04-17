# config_loader.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Standardwerte für Konfiguration
DEFAULT_CONFIG = {
    # Pfade
    'INPUT_FOLDER': './data/input',
    'OUTPUT_FOLDER': './data/output',
    'MODELS_DIR': './models',
    'TEMP_DIR': './temp',
    
    # Tag-Einstellungen
    'TAG_MODE': 'append',
    'MIN_CONFIDENCE_PERCENT': 80,
    'MIN_FACE_SIZE': 40,
    'MIN_PERSON_HEIGHT': 40,
    
    # Hardware-Einstellungen
    'USE_GPU': True,
    'CUDA_DEVICE_ID': 0,
    'NUM_WORKERS': 4,
    'BATCH_SIZE': 8,
    
    # Modell-Einstellungen
    'AUTO_DOWNLOAD_MODELS': True,
    'FORCE_MODEL_UPDATE': False,
    'OFFLINE_MODE': False,
    
    # Rekursive Ordnerverarbeitung
    'PROCESS_SUBDIRECTORIES': True,
    'MAX_RECURSION_DEPTH': 0,
    
    # MinIO-Einstellungen
    'MINIO_ENDPOINT': 'localhost:9000',
    'MINIO_ACCESS_KEY': 'minioadmin',
    'MINIO_SECRET_KEY': 'minioadmin',
    'MINIO_SECURE': False,
    'MINIO_INPUT_BUCKET': 'images',
    'MINIO_OUTPUT_BUCKET': 'tagged-images',
    
    # Online-Identitätserkennung
    'ONLINE_IDENTITY_ENABLED': False,
    'ONLINE_IDENTITY_SERVICE': 'open_face',
    'ONLINE_IDENTITY_API_KEY': '',
    
    # Logging
    'LOG_LEVEL': 'INFO',
    'MAX_LOG_SIZE_MB': 10,
    'LOG_BACKUP_COUNT': 3
}

# Logger einrichten
logger = logging.getLogger("auto-tag")

class Config:
    """Lade und verwalte die Konfiguration aus .env-Datei"""
    
    def __init__(self, env_file=None):
        """Initialisiere die Konfiguration mit optionaler .env-Datei"""
        self.config = DEFAULT_CONFIG.copy()
        
        # Setze Basis-Verzeichnis
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Wenn keine .env-Datei angegeben wurde, suche nach einer
        if env_file is None:
            env_file = os.path.join(self.base_dir, '.env')
            
            # Falls .env nicht existiert, suche nach .env.example
            if not os.path.exists(env_file):
                example_env = os.path.join(self.base_dir, '.env.example')
                if os.path.exists(example_env):
                    logger.warning(f".env nicht gefunden, kopiere .env.example nach .env")
                    self._copy_example_env(example_env, env_file)
        
        # Lade Umgebungsvariablen aus .env
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Konfiguration aus {env_file} geladen")
        else:
            logger.warning(f"Keine .env-Datei gefunden unter {env_file}, verwende Standardwerte")
        
        # Lade Konfiguration aus Umgebungsvariablen
        self._load_from_env()
        
        # Konvertiere Pfade in absolute Pfade
        self._convert_paths()
        
        # Stelle sicher, dass wichtige Verzeichnisse existieren
        self._ensure_directories()
    
    def _copy_example_env(self, example_env, target_env):
        """Kopiere .env.example nach .env"""
        try:
            with open(example_env, 'r') as source:
                content = source.read()
            
            with open(target_env, 'w') as target:
                target.write(content)
                
            logger.info(f".env.example erfolgreich nach .env kopiert")
        except Exception as e:
            logger.error(f"Fehler beim Kopieren von .env.example: {e}")
    
    def _load_from_env(self):
        """Lade Konfigurationswerte aus Umgebungsvariablen"""
        for key in self.config:
            env_value = os.getenv(key)
            
            if env_value is not None:
                # Konvertiere Werte in richtigen Typ
                if isinstance(self.config[key], bool):
                    self.config[key] = env_value.lower() in ('true', 'yes', '1', 'y')
                elif isinstance(self.config[key], int):
                    try:
                        self.config[key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Ungültiger Wert für {key}: {env_value}, verwende Standard: {self.config[key]}")
                elif isinstance(self.config[key], float):
                    try:
                        self.config[key] = float(env_value)
                    except ValueError:
                        logger.warning(f"Ungültiger Wert für {key}: {env_value}, verwende Standard: {self.config[key]}")
                else:
                    self.config[key] = env_value
    
    def _convert_paths(self):
        """Konvertiere relative Pfade in absolute Pfade"""
        path_keys = ['INPUT_FOLDER', 'OUTPUT_FOLDER', 'MODELS_DIR', 'TEMP_DIR']
        
        for key in path_keys:
            path_value = self.config[key]
            
            # Wenn es ein relativer Pfad ist, mache ihn absolut
            if not os.path.isabs(path_value):
                self.config[key] = os.path.abspath(os.path.join(self.base_dir, path_value))
    
    def _ensure_directories(self):
        """Stelle sicher, dass wichtige Verzeichnisse existieren"""
        dirs_to_create = [
            self.config['INPUT_FOLDER'],
            self.config['OUTPUT_FOLDER'],
            self.config['MODELS_DIR'],
            self.config['TEMP_DIR']
        ]
        
        for directory in dirs_to_create:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Verzeichnis existiert oder wurde erstellt: {directory}")
            except Exception as e:
                logger.error(f"Fehler beim Erstellen des Verzeichnisses {directory}: {e}")
    
    def get(self, key, default=None):
        """Hole einen Konfigurationswert"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """Ermöglicht Zugriff über config['KEY']"""
        return self.config.get(key, None)
    
    def __contains__(self, key):
        """Ermöglicht 'in' Prüfung"""
        return key in self.config
    
    def to_dict(self):
        """Gib die komplette Konfiguration als Dictionary zurück"""
        return self.config.copy()

# Globale Konfigurationsinstanz für einfachen Import
config = Config()

# Für direkten Import von Werten
def get_config():
    """Hole die Konfiguration als Singleton"""
    return config

# Zugriffsmethoden für häufig verwendete Konfigurationswerte
def get_input_folder():
    return config['INPUT_FOLDER']

def get_output_folder():
    return config['OUTPUT_FOLDER']

def get_models_dir():
    return config['MODELS_DIR']

def get_tag_mode():
    return config['TAG_MODE']

def use_gpu():
    return config['USE_GPU']

def should_process_subdirectories():
    return config['PROCESS_SUBDIRECTORIES']

def get_max_recursion_depth():
    return config['MAX_RECURSION_DEPTH']

def is_online_identity_enabled():
    return config['ONLINE_IDENTITY_ENABLED']

# Wenn das Skript direkt ausgeführt wird, zeige die geladene Konfiguration an
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Lade Konfiguration
    config = Config()
    
    # Zeige die geladene Konfiguration
    print("\n=== AUTO-TAG Konfiguration ===\n")
    for key, value in sorted(config.to_dict().items()):
        if key.endswith('KEY') or key.endswith('SECRET'):
            # Verstecke Schlüssel und Geheimnisse
            print(f"{key}: {'*' * len(str(value))}")
        else:
            print(f"{key}: {value}")