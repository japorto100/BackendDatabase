"""
Spezialisierte Bildvorverarbeitungsfunktionen für verschiedene OCR-Adapter.
"""
from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

from .core import load_image, convert_to_array, convert_to_pil
from .enhancement import (
    denoise_image, binarize_image, deskew_image, 
    enhance_contrast, sharpen_image, normalize_image
)

def preprocess_for_tesseract(image: Union[str, np.ndarray, Image.Image], 
                            options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für Tesseract OCR."""
    options = options or {}
    
    # Standardoptionen überschreiben, falls custom Options angegeben wurden
    dpi = options.get("dpi", 300)
    denoise_method = options.get("denoise_method", "gaussian")
    threshold_method = options.get("threshold_method", "adaptive")
    
    # Bild laden
    img_array = convert_to_array(image)
    
    # Standardverarbeitungsschritte für Tesseract
    if options.get("denoise", True):
        img_array = denoise_image(img_array, method=denoise_method)
    
    if options.get("deskew", True):
        img_array = deskew_image(img_array)
    
    if options.get("binarize", True):
        img_array = binarize_image(img_array, method=threshold_method)
    
    return img_array

def preprocess_for_easyocr(image: Union[str, np.ndarray, Image.Image],
                          options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für EasyOCR."""
    options = options or {}
    
    # EasyOCR-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # EasyOCR arbeitet besser mit weniger Vorverarbeitung
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, method="clahe")
    
    if options.get("denoise", True):
        strength = options.get("denoise_strength", 5)  # Schwächeres Rauschentfernen für EasyOCR
        img_array = denoise_image(img_array, strength=strength)
    
    return img_array

def preprocess_for_paddleocr(image: Union[str, np.ndarray, Image.Image],
                           options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für PaddleOCR."""
    options = options or {}
    
    # PaddleOCR-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # PaddleOCR benötigt minimale Vorverarbeitung
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, method="clahe", 
                                 clip_limit=options.get("clip_limit", 2.0))
    
    if options.get("sharpen", True):
        img_array = sharpen_image(img_array, 
                              method=options.get("sharpen_method", "unsharp_mask"),
                              strength=options.get("sharpen_strength", 1.0))
    
    return img_array

def preprocess_for_layoutlmv3(image: Union[str, np.ndarray, Image.Image],
                             options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für LayoutLMv3."""
    options = options or {}
    
    # LayoutLMv3-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Layout-erhaltende Verarbeitung
    if options.get("normalize", True):
        img_array = normalize_image(img_array)
    
    if options.get("denoise", True):
        # Sanfte Rauschunterdrückung, um Layout-Informationen zu erhalten
        img_array = denoise_image(img_array, 
                              method=options.get("denoise_method", "bilateral"),
                              strength=options.get("denoise_strength", 7))
    
    return img_array

def preprocess_for_formula_recognition(image: Union[str, np.ndarray, Image.Image],
                                     options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für die Formelerkennung."""
    options = options or {}
    
    # Formel-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Starke Kontraststeigerung für mathematische Symbole
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 3.0))
    
    # Präzise Deskew ist für Formeln besonders wichtig
    if options.get("deskew", True):
        img_array = deskew_image(img_array, max_angle=options.get("max_angle", 10.0))
    
    # Schärfen für klare Kanten bei mathematischen Symbolen
    if options.get("sharpen", True):
        img_array = sharpen_image(img_array, 
                              method=options.get("sharpen_method", "unsharp_mask"),
                              strength=options.get("sharpen_strength", 2.0))
    
    return img_array

def preprocess_for_donut(image: Union[str, np.ndarray, Image.Image],
                       options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für Donut."""
    options = options or {}
    
    # Donut-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Donut benötigt weniger Vorverarbeitung, hauptsächlich Normalisierung
    if options.get("normalize", True):
        img_array = normalize_image(img_array)
    
    return img_array

def preprocess_for_nougat(image: Union[str, np.ndarray, Image.Image],
                        options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für Nougat."""
    options = options or {}
    
    # Nougat-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Nougat arbeitet gut mit normalisierten Bildern
    if options.get("normalize", True):
        img_array = normalize_image(img_array)
    
    if options.get("denoise", True):
        img_array = denoise_image(img_array, 
                              method=options.get("denoise_method", "gaussian"),
                              strength=options.get("denoise_strength", 5))
    
    return img_array

def preprocess_for_doctr(image: Union[str, np.ndarray, Image.Image],
                       options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für DocTR."""
    options = options or {}
    
    # DocTR-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # DocTR benötigt weniger Vorverarbeitung, hauptsächlich Normalisierung und moderate Verbesserung
    if options.get("normalize", True):
        img_array = normalize_image(img_array)
    
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 2.0))
    
    return img_array

def preprocess_for_microsoft(image: Union[str, np.ndarray, Image.Image],
                           options: Dict[str, Any] = None) -> Union[np.ndarray, Image.Image]:
    """Optimale Bildvorverarbeitung für Microsoft Read API."""
    options = options or {}
    
    # Microsoft-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Microsoft benötigt wenig Vorverarbeitung, hauptsächlich Bereinigung und Kontrast
    if options.get("denoise", True):
        img_array = denoise_image(img_array, 
                              method=options.get("denoise_method", "gaussian"),
                              strength=options.get("denoise_strength", 3))
    
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 1.5))
    
    # Microsoft API erwartet oft ein PIL-Bild
    if options.get("return_pil", True):
        return convert_to_pil(img_array)
    
    return img_array

def preprocess_for_table_extraction(image: Union[str, np.ndarray, Image.Image],
                                  options: Dict[str, Any] = None) -> np.ndarray:
    """Optimale Bildvorverarbeitung für Tabellenextraktion."""
    options = options or {}
    
    # Tabellen-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Für Tabellenerkennung ist Kantenerhaltung wichtig
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 2.5))
    
    if options.get("sharpen", True):
        img_array = sharpen_image(img_array, 
                              method=options.get("sharpen_method", "unsharp_mask"),
                              strength=options.get("sharpen_strength", 1.5))
    
    # Optional binarisieren für klare Tabellenlinien
    if options.get("binarize_for_lines", False):
        bin_img = binarize_image(img_array.copy(), 
                                method=options.get("threshold_method", "adaptive"))
        
        # Morphologische Operationen für Linienidentifikation
        kernel = np.ones((3, 3), np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        
        # Ergebnis nur zurückgeben, wenn explizit angefordert
        if options.get("return_binary", False):
            return bin_img
    
    return img_array

def detect_optimal_preprocessing(image: Union[str, np.ndarray, Image.Image]) -> str:
    """
    Erkennt die optimale Vorverarbeitungsstrategie basierend auf Bildinhalt.
    
    Args:
        image: Eingabebild
        
    Returns:
        str: Name der optimalen Vorverarbeitungsfunktion
    """
    img_array = convert_to_array(image)
    
    # Graustufenkonvertierung für Analyse
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    # Einfache Heuristiken für Dokumententyp
    
    # Testen auf Tabellen (viele horizontale und vertikale Linien)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None and len(lines) > 20:
        horizontal = 0
        vertical = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1) and abs(x2 - x1) > 100:
                horizontal += 1
            elif abs(y2 - y1) > abs(x2 - x1) and abs(y2 - y1) > 100:
                vertical += 1
        
        if horizontal > 5 and vertical > 5:
            return "preprocess_for_table_extraction"
    
    # Testen auf mathematische Formeln (komplexe Symbole, viele kleine zusammenhängende Komponenten)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    small_symbols = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10 < area < 200:  # Kleine mathematische Symbole
            small_symbols += 1
    
    if small_symbols > 30:
        return "preprocess_for_formula_recognition"
    
    # Testen auf komplexes Layout (viele Blöcke und verschiedene Bereiche)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    if len(regions) > 50:
        return "preprocess_for_layoutlmv3"
    
    # Standardfall: einfacher Text
    return "preprocess_for_tesseract"

def preprocess_for_document_image(image: Union[str, np.ndarray, Image.Image],
                              options: Dict[str, Any] = None) -> np.ndarray:
    """
    Optimierte Vorverarbeitung für Bilddokumente.
    
    Args:
        image: Eingabebild
        options: Optionen für die Vorverarbeitung
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    options = options or {}
    
    # Bilddokument-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Für allgemeine Dokumente ist Kantenerhaltung wichtig
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 2.0))
    
    if options.get("denoise", True):
        strength = options.get("denoise_strength", 5)
        img_array = denoise_image(img_array, 
                              method=options.get("denoise_method", "gaussian"),
                              strength=strength)
    
    if options.get("deskew", True):
        img_array = deskew_image(img_array)
    
    return img_array

def preprocess_for_scan_document(image: Union[str, np.ndarray, Image.Image],
                             options: Dict[str, Any] = None) -> np.ndarray:
    """
    Optimierte Vorverarbeitung für gescannte Dokumente.
    
    Args:
        image: Eingabebild
        options: Optionen für die Vorverarbeitung
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    options = options or {}
    
    # Scan-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Gescannte Dokumente benötigen meist stärkere Vorverarbeitung
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 2.5))
    
    if options.get("denoise", True):
        img_array = denoise_image(img_array, 
                              method=options.get("denoise_method", "nlm"),
                              strength=options.get("denoise_strength", 10))
    
    if options.get("deskew", True):
        img_array = deskew_image(img_array, max_angle=options.get("max_angle", 20.0))
    
    if options.get("binarize", False):
        img_array = binarize_image(img_array, 
                               method=options.get("threshold_method", "adaptive"))
    
    return img_array

def preprocess_for_hybrid_document(image: Union[str, np.ndarray, Image.Image],
                               options: Dict[str, Any] = None) -> np.ndarray:
    """
    Optimierte Vorverarbeitung für hybride Dokumente mit Text und Bildern.
    
    Args:
        image: Eingabebild
        options: Optionen für die Vorverarbeitung
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    options = options or {}
    
    # Hybrid-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Hybride Dokumente benötigen ausgewogene Vorverarbeitung
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 1.5))
    
    if options.get("denoise", True):
        img_array = denoise_image(img_array, 
                              method=options.get("denoise_method", "gaussian"),
                              strength=options.get("denoise_strength", 5))
    
    if options.get("deskew", True):
        img_array = deskew_image(img_array)
    
    return img_array

def preprocess_for_structured_document(image: Union[str, np.ndarray, Image.Image],
                                   options: Dict[str, Any] = None) -> np.ndarray:
    """
    Optimierte Vorverarbeitung für strukturierte Dokumente (Formulare, Rechnungen, etc.).
    
    Args:
        image: Eingabebild
        options: Optionen für die Vorverarbeitung
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    options = options or {}
    
    # Strukturierte Dokument-Vorverarbeitung
    img_array = convert_to_array(image)
    
    # Für strukturierte Dokumente ist Kantenerhaltung und Linienklarheit wichtig
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 2.5))
    
    if options.get("denoise", True):
        # Vorsichtiges Entrauschen, um Linien und Ränder zu erhalten
        img_array = denoise_image(img_array, 
                             method=options.get("denoise_method", "bilateral"),
                             strength=options.get("denoise_strength", 5))
    
    if options.get("sharpen", True):
        # Schärfen für deutlichere Linien und Kanten
        img_array = sharpen_image(img_array, 
                             method=options.get("sharpen_method", "unsharp_mask"),
                             strength=options.get("sharpen_strength", 1.5))
    
    if options.get("deskew", True):
        img_array = deskew_image(img_array)
    
    return img_array

def preprocess_for_colpali(image: Union[str, np.ndarray, Image.Image],
                         options: Dict[str, Any] = None) -> np.ndarray:
    """
    Optimierte Vorverarbeitung für ColPaLi (Columnar Page Layout).
    
    Args:
        image: Eingabebild
        options: Optionen für die Vorverarbeitung
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    options = options or {}
    
    # ColPaLi-spezifische Vorverarbeitung
    img_array = convert_to_array(image)
    
    # ColPaLi benötigt hohe Details und guten Kontrast für Spalten- und Layout-Erkennung
    if options.get("enhance_contrast", True):
        img_array = enhance_contrast(img_array, 
                                 method=options.get("contrast_method", "clahe"),
                                 clip_limit=options.get("clip_limit", 2.0))
    
    if options.get("denoise", True):
        # Vorsichtiges Entrauschen, um Details zu erhalten
        strength = options.get("denoise_strength", 3)
        img_array = denoise_image(img_array, 
                             method=options.get("denoise_method", "bilateral"),
                             strength=strength)
    
    if options.get("deskew", True):
        # Ausrichten ist wichtig für die Spalten-/Zeilen-Analyse
        img_array = deskew_image(img_array)
    
    return img_array 

def enhance_image_for_ocr(image: Union[str, np.ndarray, Image.Image], 
                       ocr_engine: str = "tesseract",
                       options: Dict[str, Any] = None) -> np.ndarray:
    """
    Verbessert ein Bild speziell für die angegebene OCR-Engine oder Dokumenttyp.
    
    Args:
        image: Eingabebild
        ocr_engine: Name der OCR-Engine oder Dokumenttyp
        options: Spezifische Optionen für die Vorverarbeitung
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    preprocess_funcs = {
        # OCR Engines
        "tesseract": preprocess_for_tesseract,
        "easyocr": preprocess_for_easyocr,
        "paddleocr": preprocess_for_paddleocr,
        "layoutlmv3": preprocess_for_layoutlmv3,
        "formula_recognition": preprocess_for_formula_recognition,
        "donut": preprocess_for_donut,
        "nougat": preprocess_for_nougat,
        "doctr": preprocess_for_doctr,
        "microsoft": preprocess_for_microsoft,
        "table_extraction": preprocess_for_table_extraction,
        
        # Dokumenttypen
        "document_image": preprocess_for_document_image,
        "scan_document": preprocess_for_scan_document,
        "hybrid_document": preprocess_for_hybrid_document,
        "structured_document": preprocess_for_structured_document,
        "colpali": preprocess_for_colpali
    }
    
    if ocr_engine in preprocess_funcs:
        return preprocess_funcs[ocr_engine](image, options)
    else:
        # Automatische Erkennung der optimalen Vorverarbeitung
        optimal_func = detect_optimal_preprocessing(image)
        return preprocess_funcs[optimal_func](image, options)

def apply_preprocessing_recommendations(image: Union[str, np.ndarray, Image.Image],
                                     recommendations: Dict[str, Any]) -> np.ndarray:
    """
    Wendet intelligente Vorverarbeitungsempfehlungen auf ein Bild an.
    
    Diese Funktion nimmt die Empfehlungen aus DocumentQualityAnalyzer.generate_preprocessing_recommendations()
    und wendet die empfohlenen Vorverarbeitungsschritte in der empfohlenen Reihenfolge an.
    
    Args:
        image: Eingabebild als Dateipfad, NumPy-Array oder PIL-Image
        recommendations: Vorverarbeitungsempfehlungen vom DocumentQualityAnalyzer
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    try:
        # Bild in NumPy-Array konvertieren
        img_array = convert_to_array(image)
        
        # Überprüfen, ob Vorverarbeitung erforderlich ist
        if not recommendations.get("preprocessing_required", False):
            return img_array
            
        # Vorverarbeitungsstrategie bestimmen
        strategy = recommendations.get("preprocessing_strategy", "standard")
        
        # Multi-stage Verarbeitung für stark beschädigte Dokumente
        if strategy == "aggressive" and "preprocessing_pipeline" in recommendations:
            return _apply_multi_stage_preprocessing(img_array, recommendations)
        
        # Priorisierte Liste der zu verwendenden Methoden abrufen
        methods = recommendations.get("recommended_methods", [])
        methods.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.get("priority", "low")])
        
        # Methoden der Reihe nach anwenden
        processed_img = img_array.copy()
        
        for method_info in methods:
            method_name = method_info.get("method", "")
            params = method_info.get("params", {})
            
            if method_name == "deblur":
                processed_img = _apply_deblur(processed_img, params)
                
            elif method_name == "enhance_contrast":
                processed_img = _apply_contrast_enhancement(processed_img, params)
                
            elif method_name == "denoise":
                processed_img = _apply_denoising(processed_img, params)
                
            elif method_name == "upscale":
                processed_img = _apply_upscaling(processed_img, params)
                
            elif method_name == "binarize":
                processed_img = _apply_binarization(processed_img, params)
                
        return processed_img
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Fehler bei der Anwendung der Vorverarbeitungsempfehlungen: {str(e)}")
        return convert_to_array(image)  # Im Fehlerfall Original zurückgeben

def _apply_multi_stage_preprocessing(img_array: np.ndarray, recommendations: Dict[str, Any]) -> np.ndarray:
    """
    Wendet eine mehrstufige Vorverarbeitungspipeline für stark beschädigte Dokumente an.
    
    Args:
        img_array: Eingabebild als NumPy-Array
        recommendations: Vorverarbeitungsempfehlungen
        
    Returns:
        np.ndarray: Vorverarbeitetes Bild
    """
    pipeline = recommendations.get("preprocessing_pipeline", {})
    stages = pipeline.get("stages", [])
    
    processed_img = img_array.copy()
    
    for stage in stages:
        stage_name = stage.get("stage", "")
        methods = stage.get("methods", [])
        
        for method_name in methods:
            # Finde die zugehörigen Parameter in den empfohlenen Methoden
            method_params = {}
            for recommended_method in recommendations.get("recommended_methods", []):
                if recommended_method.get("method") == method_name:
                    method_params = recommended_method.get("params", {})
                    break
            
            if method_name == "denoise":
                processed_img = _apply_denoising(processed_img, method_params)
                
            elif method_name == "enhance_contrast":
                processed_img = _apply_contrast_enhancement(processed_img, method_params)
                
            elif method_name == "deblur":
                processed_img = _apply_deblur(processed_img, method_params)
                
            elif method_name == "binarize":
                processed_img = _apply_binarization(processed_img, method_params)
                
            elif method_name == "upscale":
                processed_img = _apply_upscaling(processed_img, method_params)
    
    return processed_img

def _apply_deblur(img_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Entfernt Unschärfe aus einem Bild."""
    from .enhancement import sharpen_image
    
    method = params.get("method", "unsharp_mask")
    strength = params.get("strength", 1.0)
    
    if method == "wiener":
        # Wiener-Filter für stärkere Unschärfen
        from scipy import signal
        kernel_size = int(params.get("kernel_size", 5))
        psf = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
        deblurred = signal.wiener(img_gray, (kernel_size, kernel_size), noise=None)
        
        # Normalisieren und zurück zu ursprünglichem Format konvertieren
        deblurred = (deblurred - deblurred.min()) / (deblurred.max() - deblurred.min()) * 255
        deblurred = deblurred.astype(np.uint8)
        
        if len(img_array.shape) == 3:
            deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        
        return deblurred
    else:
        # Standardmethode: Unsharp Mask
        return sharpen_image(img_array, method="unsharp_mask", strength=strength)

def _apply_contrast_enhancement(img_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Verbessert den Kontrast eines Bildes."""
    from .enhancement import enhance_contrast
    
    algorithm = params.get("algorithm", "clahe")
    strength = params.get("strength", 1.0)
    local_adaptation = params.get("local_adaptation", False)
    
    if algorithm == "clahe":
        return enhance_contrast(img_array, method="clahe", clip_limit=strength*3.0)
    elif algorithm == "adaptive_histogram_equalization":
        return enhance_contrast(img_array, method="adaptive_histogram", tile_size=8 if local_adaptation else 16)
    else:
        return enhance_contrast(img_array)

def _apply_denoising(img_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Entfernt Rauschen aus einem Bild."""
    from .enhancement import denoise_image
    
    algorithm = params.get("algorithm", "gaussian")
    strength = params.get("strength", 0.5)
    preserve_edges = params.get("preserve_edges", False)
    
    if algorithm == "bilateral_filter":
        return denoise_image(img_array, method="bilateral", strength=int(strength*15))
    elif algorithm == "non_local_means":
        return denoise_image(img_array, method="fastNL", strength=int(strength*15))
    else:
        return denoise_image(img_array, method="gaussian", strength=int(strength*10))

def _apply_upscaling(img_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Skaliert ein Bild hoch."""
    from .transformation import resize_image
    
    factor = params.get("factor", 2.0)
    algorithm = params.get("algorithm", "bicubic")
    
    if algorithm == "sr_cnn":
        # Super-Resolution CNN für bessere Qualität
        try:
            from cv2 import dnn_superres
            sr = dnn_superres.DnnSuperResImpl_create()
            
            # Standardmodell verwenden, falls verfügbar
            import os
            model_path = os.environ.get("SUPER_RESOLUTION_MODEL_PATH", "/app/models/EDSR_x4.pb")
            
            if os.path.exists(model_path):
                sr.readModel(model_path)
                sr.setModel("edsr", 4)
                upscaled = sr.upsample(img_array)
                
                # Auf gewünschte Größe anpassen
                target_width = int(img_array.shape[1] * factor)
                target_height = int(img_array.shape[0] * factor)
                
                return resize_image(upscaled, target_width, target_height, method="bicubic")
            else:
                # Fallback, wenn Modell nicht verfügbar
                return resize_image(img_array, factor, method="bicubic")
        except:
            # Fallback bei Fehlern
            return resize_image(img_array, factor, method="bicubic")
    else:
        return resize_image(img_array, factor, method=algorithm)

def _apply_binarization(img_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Binarisiert ein Bild."""
    from .enhancement import binarize_image
    
    algorithm = params.get("algorithm", "otsu")
    block_size = params.get("block_size", 11)
    
    if algorithm == "adaptive_otsu":
        # Kombinierte Methode für stark degradierte Dokumente
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Globale Otsu-Binarisierung als Basis
        _, global_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive Binarisierung für lokale Details
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, block_size, 2)
        
        # Kombinieren beider Ergebnisse
        combined = cv2.bitwise_and(global_thresh, adaptive_thresh)
        
        if len(img_array.shape) == 3:
            combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            
        return combined
    elif algorithm == "sauvola":
        return binarize_image(img_array, method="sauvola", block_size=block_size)
    else:
        return binarize_image(img_array, method=algorithm) 