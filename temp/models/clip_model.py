# models/clip_model.py - Szene- & Kleidungsklassifikation mit optimiertem CLIP
import os
import torch
import logging
from PIL import Image
import numpy as np
from pathlib import Path

logger = logging.getLogger('auto-tag')

# Vorinitialisierte Klassen für Klassifikation
SCENE_CATEGORIES = ["indoor", "outdoor"]
ROOMTYPES = ["kitchen", "bathroom", "bedroom", "living room", "office"]
CLOTHING = ["dressed", "naked"]

class CLIPModel:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Status der Modellinitialisierung
        self.initialized = False
        
        # Modellpfad
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.model_path = os.path.join(models_dir, "clip", "clip_vit_b32.pth")
        
    def initialize(self):
        """Initialisiere das CLIP-Modell"""
        if self.initialized:
            return True
            
        try:
            # Prüfe, ob Modelldatei existiert
            if not os.path.exists(self.model_path):
                logger.error(f"CLIP-Modell nicht gefunden: {self.model_path}")
                return False
                
            # Importiere CLIP dynamisch (reduziert Abhängigkeiten)
            try:
                import open_clip
            except ImportError:
                logger.error("open_clip_torch ist nicht installiert")
                return False
                
            # Lade Modell und Transformer
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained=self.model_path
            )
            self.model = self.model.to(self.device).eval()
            
            # Lade Tokenizer
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            self.initialized = True
            logger.info("CLIP-Modell erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der CLIP-Initialisierung: {e}")
            self.initialized = False
            return False
    
    def load_image(self, image_path):
        """Lade und bereite Bild für CLIP vor"""
        try:
            image = Image.open(image_path).convert("RGB")
            return self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Fehler beim Laden des Bildes: {e}")
            return None
    
    def classify(self, image_tensor, label_list, topk=1):
        """Klassifiziere Bild gegen Textprompts"""
        if not self.initialized or image_tensor is None:
            return [("unknown", 0.0)]
            
        try:
            with torch.no_grad():
                text_inputs = self.tokenizer([f"a photo of {label}" for label in label_list]).to(self.device)
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalisiere Features für Kosinus-Ähnlichkeit
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Berechne Kosinus-Ähnlichkeit
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = similarity.squeeze().cpu().numpy()
                
                # Sortiere nach Wahrscheinlichkeit
                results = list(zip(label_list, probs))
                sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                
                return sorted_results[:topk]
                
        except Exception as e:
            logger.error(f"Klassifikationsfehler: {e}")
            return [("error", 0.0)]
    
    def __del__(self):
        """Bereinige Ressourcen"""
        if self.initialized and torch.cuda.is_available():
            torch.cuda.empty_cache()


# Globale Instanz (Singleton)
_clip_model = None

def get_clip_model():
    """Liefere oder initialisiere CLIP-Modell-Singleton"""
    global _clip_model
    
    if _clip_model is None:
        _clip_model = CLIPModel()
        _clip_model.initialize()
        
    return _clip_model

def analyze_scene_clothing(image_path):
    """Analysiere Szene und Kleidung mit CLIP"""
    model = get_clip_model()
    
    if not model.initialized:
        logger.error("CLIP-Modell konnte nicht initialisiert werden")
        return {
            "scene": ("unknown", 0.0),
            "roomtype": ("unknown", 0.0),
            "clothing": ("unknown", 0.0)
        }
    
    # Lade und bereite Bild vor
    image_tensor = model.load_image(image_path)
    if image_tensor is None:
        return {
            "scene": ("unknown", 0.0),
            "roomtype": ("unknown", 0.0),
            "clothing": ("unknown", 0.0)
        }
    
    # Klassifiziere Szene, Raumtyp und Kleidung
    scene = model.classify(image_tensor, SCENE_CATEGORIES)[0]
    room = model.classify(image_tensor, ROOMTYPES)[0]
    clothing = model.classify(image_tensor, CLOTHING)[0]
    
    return {
        "scene": scene,
        "roomtype": room,
        "clothing": clothing
    }


# models/yolo_model.py - Personenerkennung und -zählung mit YOLOv8
import os
import torch
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger('auto-tag')

# Mindestgröße eines "verwertbaren" Menschen in Pixel
MIN_PERSON_HEIGHT = 40

class YOLOModel:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Status der Modellinitialisierung
        self.initialized = False
        
        # Modellpfad
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.model_path = os.path.join(models_dir, "yolov8n", "yolov8n.pt")
        
    def initialize(self):
        """Initialisiere YOLOv8-Modell"""
        if self.initialized:
            return True
            
        try:
            # Prüfe, ob Modelldatei existiert
            if not os.path.exists(self.model_path):
                logger.error(f"YOLOv8-Modell nicht gefunden: {self.model_path}")
                return False
                
            # Importiere YOLO dynamisch (reduziert Abhängigkeiten)
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("ultralytics ist nicht installiert")
                return False
                
            # Lade Modell
            self.model = YOLO(self.model_path)
            
            self.initialized = True
            logger.info("YOLOv8-Modell erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der YOLO-Initialisierung: {e}")
            self.initialized = False
            return False
    
    def count_people(self, image_path, min_person_height=MIN_PERSON_HEIGHT):
        """Zähle Personen im Bild"""
        if not self.initialized:
            return "none"
            
        try:
            # Führe Inferenz durch
            results = self.model(image_path)
            
            # Zähle Personen (Klasse 0 in COCO)
            count = 0
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    cls = int(box.cls[0].item())
                    if cls == 0:  # Person
                        height = int(box.xywh[0][3].item())  # Höhe der Box
                        if height >= min_person_height:
                            count += 1
            
            # Kategorisiere das Ergebnis
            if count == 0:
                return "none"
            elif count == 1:
                return "solo"
            else:
                return "group"
                
        except Exception as e:
            logger.error(f"Fehler beim Zählen von Personen: {e}")
            return "none"
    
    def __del__(self):