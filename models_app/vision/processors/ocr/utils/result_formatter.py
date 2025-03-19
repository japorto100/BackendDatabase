"""
Funktionen zur Standardisierung und Formatierung von OCR-Ergebnissen.
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Any, Union

logger = logging.getLogger(__name__)

def create_standard_result(
    text: str = "",
    blocks: List[Dict[str, Any]] = None,
    confidence: float = 0.0,
    model: str = "unknown",
    language: str = "unknown",
    metadata: Dict[str, Any] = None,
    raw_output: Any = None
) -> Dict[str, Any]:
    """
    Erstellt ein standardisiertes OCR-Ergebnis.
    
    Args:
        text: Erkannter Text
        blocks: Liste von Textblöcken mit Position und Konfidenz
        confidence: Gesamtkonfidenz des Ergebnisses
        model: Name des verwendeten Modells
        language: Erkannte oder verwendete Sprache
        metadata: Zusätzliche Metadaten
        raw_output: Rohausgabe des OCR-Modells
        
    Returns:
        Standardisiertes OCR-Ergebnis als Dictionary
    """
    if blocks is None:
        blocks = []
        
    if metadata is None:
        metadata = {}
        
    result = {
        "text": text,
        "blocks": blocks,
        "confidence": float(confidence),
        "model": model,
        "language": language,
        "metadata": metadata
    }
    
    # Rohausgabe nur hinzufügen, wenn explizit angefordert
    if raw_output is not None:
        result["raw_output"] = raw_output
        
    return result

def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Führt mehrere OCR-Ergebnisse zu einem zusammen.
    
    Args:
        results: Liste von OCR-Ergebnissen
        
    Returns:
        Zusammengeführtes OCR-Ergebnis
    """
    if not results:
        return create_standard_result()
        
    if len(results) == 1:
        return results[0]
        
    # Text zusammenführen
    all_text = "\n".join([r.get("text", "") for r in results if r.get("text")])
    
    # Blöcke zusammenführen
    all_blocks = []
    for result in results:
        blocks = result.get("blocks", [])
        if blocks:
            all_blocks.extend(blocks)
            
    # Konfidenz berechnen (gewichteter Durchschnitt)
    total_confidence = 0.0
    total_weight = 0.0
    
    for result in results:
        conf = result.get("confidence", 0.0)
        weight = len(result.get("text", "")) if result.get("text") else 1
        total_confidence += conf * weight
        total_weight += weight
        
    avg_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
    
    # Metadaten zusammenführen
    all_metadata = {}
    for i, result in enumerate(results):
        metadata = result.get("metadata", {})
        if metadata:
            all_metadata[f"source_{i}"] = metadata
            
    # Modelle und Sprachen auflisten
    models = [r.get("model", "unknown") for r in results]
    languages = [r.get("language", "unknown") for r in results]
    
    return create_standard_result(
        text=all_text,
        blocks=all_blocks,
        confidence=avg_confidence,
        model="+".join(set(models)),
        language="+".join(set(languages)),
        metadata=all_metadata
    )

def format_as_text(result: Dict[str, Any]) -> str:
    """
    Formatiert ein OCR-Ergebnis als einfachen Text.
    
    Args:
        result: OCR-Ergebnis
        
    Returns:
        Formatierter Text
    """
    return result.get("text", "")

def format_as_html(result: Dict[str, Any]) -> str:
    """
    Formatiert ein OCR-Ergebnis als HTML.
    
    Args:
        result: OCR-Ergebnis
        
    Returns:
        HTML-formatierter Text
    """
    text = result.get("text", "")
    blocks = result.get("blocks", [])
    
    if not blocks:
        # Einfache HTML-Formatierung ohne Positionsinformationen
        paragraphs = text.split("\n\n")
        html_parts = ["<div class='ocr-result'>"]
        
        for p in paragraphs:
            if p.strip():
                lines = p.split("\n")
                html_parts.append("<p>")
                html_parts.append("<br>".join(lines))
                html_parts.append("</p>")
                
        html_parts.append("</div>")
        return "".join(html_parts)
    else:
        # Erweiterte HTML-Formatierung mit Positionsinformationen
        html_parts = ["<div class='ocr-result'>"]
        
        for block in blocks:
            block_text = block.get("text", "")
            confidence = block.get("conf", 0.0)
            bbox = block.get("bbox", None)
            
            if bbox:
                style = f"position: absolute; left: {bbox[0]}px; top: {bbox[1]}px; width: {bbox[2]-bbox[0]}px; height: {bbox[3]-bbox[1]}px;"
                html_parts.append(f"<div class='ocr-block' style='{style}' data-confidence='{confidence:.2f}'>")
            else:
                html_parts.append(f"<div class='ocr-block' data-confidence='{confidence:.2f}'>")
                
            html_parts.append(block_text.replace("\n", "<br>"))
            html_parts.append("</div>")
            
        html_parts.append("</div>")
        return "".join(html_parts)

def format_as_json(result: Dict[str, Any]) -> str:
    """
    Formatiert ein OCR-Ergebnis als JSON-String.
    
    Args:
        result: OCR-Ergebnis
        
    Returns:
        JSON-formatierter String
    """
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Fehler bei der JSON-Formatierung: {str(e)}")
        return json.dumps({"error": "Fehler bei der JSON-Formatierung", "text": result.get("text", "")})

def format_as_markdown(result: Dict[str, Any]) -> str:
    """
    Formatiert ein OCR-Ergebnis als Markdown.
    
    Args:
        result: OCR-Ergebnis
        
    Returns:
        Markdown-formatierter Text
    """
    text = result.get("text", "")
    model = result.get("model", "unknown")
    confidence = result.get("confidence", 0.0)
    
    md_parts = [
        f"# OCR-Ergebnis\n",
        f"**Modell:** {model}\n",
        f"**Konfidenz:** {confidence:.2f}\n",
        f"\n---\n\n",
        text
    ]
    
    return "\n".join(md_parts)

def format_table_as_csv(table_data) -> str:
    """
    Formatiert Tabellendaten als CSV.
    
    Args:
        table_data: Tabellendaten als Liste von Listen oder DataFrame
        
    Returns:
        CSV-formatierter String
    """
    try:
        if isinstance(table_data, pd.DataFrame):
            return table_data.to_csv(index=False)
        else:
            df = pd.DataFrame(table_data)
            return df.to_csv(index=False)
    except Exception as e:
        logger.error(f"Fehler bei der CSV-Formatierung: {str(e)}")
        return ""

def format_table_as_html(table_data) -> str:
    """
    Formatiert Tabellendaten als HTML-Tabelle.
    
    Args:
        table_data: Tabellendaten als Liste von Listen oder DataFrame
        
    Returns:
        HTML-formatierte Tabelle
    """
    try:
        if isinstance(table_data, pd.DataFrame):
            return table_data.to_html(index=False)
        else:
            df = pd.DataFrame(table_data)
            return df.to_html(index=False)
    except Exception as e:
        logger.error(f"Fehler bei der HTML-Formatierung: {str(e)}")
        return "<table><tr><td>Fehler bei der Formatierung</td></tr></table>"

def format_table_as_json(table_data) -> str:
    """
    Formatiert Tabellendaten als JSON.
    
    Args:
        table_data: Tabellendaten als Liste von Listen oder DataFrame
        
    Returns:
        JSON-formatierter String
    """
    try:
        if isinstance(table_data, pd.DataFrame):
            return table_data.to_json(orient="records")
        else:
            df = pd.DataFrame(table_data)
            return df.to_json(orient="records")
    except Exception as e:
        logger.error(f"Fehler bei der JSON-Formatierung: {str(e)}")
        return "[]" 