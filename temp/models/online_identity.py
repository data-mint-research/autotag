# models/online_identity.py
import os
import requests
import logging
import json
import base64
from io import BytesIO
from PIL import Image
import time

logger = logging.getLogger('auto-tag')

class OnlineIdentityService:
    """Basisklasse für Online-Identifikationsdienste"""
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.initialized = False
        self._load_config()
        
    def _load_config(self):
        """Lade Konfiguration aus settings.yaml oder Umgebungsvariablen"""
        try:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
            settings_path = os.path.join(config_dir, "settings.yaml")
            
            if os.path.exists(settings_path):
                import yaml
                with open(settings_path, 'r') as f:
                    settings = yaml.safe_load(f)
                    
                if 'online_identity' in settings:
                    if not self.api_key and 'api_key' in settings['online_identity']:
                        self.api_key = settings['online_identity']['api_key']
        except Exception as e:
            logger.error(f"Fehler beim Laden der Online-Identitätskonfiguration: {e}")
    
    def is_available(self):
        """Prüft, ob der Dienst verfügbar ist"""
        return self.api_key is not None
    
    def identify(self, face_image):
        """Identifiziert ein Gesicht (von Unterklassen zu implementieren)"""
        raise NotImplementedError("Muss von Unterklassen implementiert werden")


class PimEyesService(OnlineIdentityService):
    """PimEyes-Implementierung für Gesichtsidentifikation"""
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.api_url = "https://api.pimeyes.com/face/recognition"
        self.initialized = self.api_key is not None
    
    def identify(self, face_image):
        """Identifiziert ein Gesicht mit PimEyes API"""
        if not self.initialized:
            logger.warning("PimEyes-Service nicht initialisiert (kein API-Key)")
            return None
        
        try:
            # Konvertiere Bild zu base64
            if isinstance(face_image, str) and os.path.exists(face_image):
                # Wenn Pfad angegeben
                image = Image.open(face_image)
            elif isinstance(face_image, Image.Image):
                # Wenn PIL-Image übergeben
                image = face_image
            else:
                raise ValueError("Ungültiges Bildformat")
            
            # Resize für API-Limits
            max_size = (512, 512)
            image.thumbnail(max_size, Image.LANCZOS)
            
            # Konvertiere zu JPEG und dann base64
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # API-Anfrage
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            data = {
                'image': img_str,
                'limit': 3,  # Top 3 Matches
                'threshold': 0.6  # 60% Konfidenz
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            results = response.json()
            
            # Verarbeite Ergebnisse
            if 'matches' in results and results['matches']:
                best_match = results['matches'][0]
                
                return {
                    'name': best_match.get('name', 'Unknown'),
                    'location': best_match.get('location', None),
                    'confidence': best_match.get('similarity', 0.0),
                    'source': 'pimeyes'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Fehler bei PimEyes-Anfrage: {e}")
            return None


class OpenFaceRecognitionAPI(OnlineIdentityService):
    """Offene API für Gesichtserkennung (kostenlos/mit niedrigem Limit)"""
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.api_url = "https://face-recognition26.p.rapidapi.com/recognition"
        self.headers = {
            "X-RapidAPI-Key": self.api_key or "",
            "X-RapidAPI-Host": "face-recognition26.p.rapidapi.com"
        }
        self.initialized = self.api_key is not None
    
    def identify(self, face_image):
        """Identifiziert ein Gesicht mit der RapidAPI Face Recognition API"""
        if not self.initialized:
            logger.warning("Open Face Recognition API nicht initialisiert (kein API-Key)")
            return None
        
        try:
            # Konvertiere Bild zu base64
            if isinstance(face_image, str) and os.path.exists(face_image):
                # Wenn Pfad angegeben
                with open(face_image, "rb") as f:
                    img_data = f.read()
            elif isinstance(face_image, Image.Image):
                # Wenn PIL-Image übergeben
                buffer = BytesIO()
                image = face_image
                # Resize für API-Limits
                max_size = (800, 800)
                image.thumbnail(max_size, Image.LANCZOS)
                image.save(buffer, format="JPEG")
                img_data = buffer.getvalue()
            else:
                raise ValueError("Ungültiges Bildformat")
                
            # Multipart-Anfrage
            files = {"image": img_data}
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                files=files,
                timeout=30
            )
            response.raise_for_status()
            results = response.json()
            
            if results.get('status') == 'success' and results.get('data', []):
                matches = results.get('data', [])
                if matches:
                    best_match = matches[0]
                    
                    # Extrahiere relevante Informationen
                    name = best_match.get('name', 'Unknown')
                    confidence = best_match.get('confidence', 0.0)
                    
                    # Daten zu Details und Herkunft könnten aus verschiedenen
                    # Feldern kommen - abhängig von der API-Implementierung
                    location = None
                    if 'details' in best_match and 'location' in best_match['details']:
                        location = best_match['details']['location']
                    
                    return {
                        'name': name,
                        'location': location,
                        'confidence': confidence,
                        'source': 'open_face_api'
                    }
            
            return None
                
        except Exception as e:
            logger.error(f"Fehler bei Open Face Recognition API-Anfrage: {e}")
            return None


# Factory für identitätserkennungsdienste
def get_identity_service(service_name=None):
    """Liefert einen passenden Identitätserkennungsdienst"""
    if service_name == "pimeyes":
        return PimEyesService()
    elif service_name == "open_face":
        return OpenFaceRecognitionAPI()
    else:
        # Versuche verfügbare Dienste zu finden
        services = [
            ("open_face", OpenFaceRecognitionAPI()),
            ("pimeyes", PimEyesService())
        ]
        
        for name, service in services:
            if service.is_available():
                logger.info(f"Verwende {name} für Online-Identitätserkennung")
                return service
        
        logger.warning("Kein Online-Identitätserkennungsdienst verfügbar")
        return None


# modules/identity_check/verify_identity.py (aktualisiert)
import os
import logging
from PIL import Image
from models.online_identity import get_identity_service

logger = logging.getLogger('auto-tag')

def is_plausible_name(name):
    """Grobe Namensprüfung: mind. 1 Wort, optional 2, ohne Zahlen"""
    if not name: 
        return False
    if any(char.isdigit() for char in name): 
        return False
    parts = name.replace("-", "_").split("_")
    return all(p.isalpha() and p[0].isupper() for p in parts)

def extract_name_from_path(image_path):
    """Extrahiert Namen aus dem Dateipfad"""
    folder = os.path.basename(os.path.dirname(image_path))
    if is_plausible_name(folder):
        return folder
    return None

def determine_identity(image_path, folder_name=None, faces=None, use_online=False):
    """Ermittelt Identitäts-Tags basierend auf Ordnernamen und optional Online-Suche"""
    tags = []
    
    # Extrahiere Namen aus Pfad (falls kein expliziter Ordnername übergeben)
    if folder_name is None:
        folder_name = extract_name_from_path(image_path)
    
    # Lokale Identifikation über Ordnername
    if folder_name and is_plausible_name(folder_name):
        tags.append("identity/verified")
        tags.append("identity/source/foldername/local")
        return tags
    
    # Online-Identifikation (falls aktiviert und kein lokaler Name gefunden)
    if use_online and faces:
        try:
            # Hole passenden Online-Identifikationsdienst
            identity_service = get_identity_service()
            
            if identity_service and identity_service.is_available():
                # Verwende das erste gefundene Gesicht für die Suche
                face_image = faces[0] if isinstance(faces, list) else faces
                
                # Online-Identifikation durchführen
                identity_result = identity_service.identify(face_image)
                
                if identity_result and identity_result.get('name') != 'Unknown':
                    # Wenn Person gefunden wurde
                    tags.append("identity/verified")
                    tags.append(f"identity/source/online/{identity_result['source']}")
                    
                    # Name hinzufügen
                    name = identity_result['name'].replace(' ', '_')
                    tags.append(f"person/{name}")
                    
                    # Ort hinzufügen, falls vorhanden
                    if identity_result.get('location'):
                        location = identity_result['location'].replace(' ', '_')
                        tags.append(f"location/{location}")
                        
                    logger.info(f"Online-Identifikation: {identity_result['name']} ({identity_result['confidence']:.2f})")
                    return tags
        except Exception as e:
            logger.error(f"Fehler bei Online-Identifikation: {e}")
    
    # Fallback: Unverifizierte Identität
    tags.append("identity/unverified")
    return tags