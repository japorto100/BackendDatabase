import logging
import os
from PIL import Image
import numpy as np

from models_app.vision_processor import VisionProcessor
from models_app.colpali.processor import ColPaliProcessor
from models_app.ocr.selector import OCRModelSelector
from models_app.fusion.hybrid_fusion import HybridFusion

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_colpali_processor():
    """Test der ColPali-Processor-Komponente."""
    processor = ColPaliProcessor()
    
    # Testbild erstellen oder Pfad zu einem Testbild angeben
    test_image = Image.new('RGB', (100, 100), color='white')
    
    # Verarbeitung testen
    result = processor.process_image(test_image)
    
    logger.info(f"ColPali Verarbeitungsergebnis: {result.keys()}")
    
    if "error" in result:
        logger.error(f"Fehler: {result['error']}")
    else:
        logger.info(f"Erfolgreich! Embedding-Dimension: {result['embedding_dim']}")

def test_ocr_selector():
    """Test des OCR-Selektors."""
    selector = OCRModelSelector()
    
    # Testbild erstellen oder Pfad zu einem Testbild angeben
    test_image = Image.new('RGB', (100, 100), color='white')
    
    # Modellauswahl testen
    model_id = selector.select_model(test_image, {"language": "de"})
    logger.info(f"Ausgewähltes OCR-Modell: {model_id}")
    
    # Modellinstanz testen
    model = selector.get_model_instance(model_id)
    logger.info(f"Modellinstanz: {model}")

def test_full_vision_processor():
    """Test des gesamten Vision-Prozessors."""
    processor = VisionProcessor()
    
    # Testbild erstellen oder Pfad zu einem Testbild angeben
    test_image_path = "test_document.png"
    
    # Erstelle ein Testbild, falls es nicht existiert
    if not os.path.exists(test_image_path):
        # Erstelle ein einfaches Testbild
        img = Image.new('RGB', (800, 600), color='white')
        img.save(test_image_path)
    
    # Verarbeitung testen
    result = processor.process_document(test_image_path, query="Was ist der Inhalt?")
    
    logger.info(f"Vision Processor Ergebnis: {result.keys()}")
    
    if "error" in result:
        logger.error(f"Fehler: {result['error']}")
    else:
        logger.info(f"Erfolgreich! Fusion-Strategie: {result['fusion_strategy']}")

if __name__ == "__main__":
    logger.info("Teste ColPali Processor...")
    test_colpali_processor()
    
    logger.info("\nTeste OCR Selector...")
    test_ocr_selector()
    
    logger.info("\nTeste Vision Processor...")
    test_full_vision_processor()