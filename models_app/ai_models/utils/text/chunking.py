"""
Chunking-Strategien für Dokumente

Bietet Funktionen für verschiedene Chunking-Strategien:
- Fixed-Size-Chunking
- Semantisches Chunking
- Adaptives Chunking
"""

from typing import Dict, List, Optional, Any
import re

def fixed_size_chunking(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Teilt einen Text in Chunks fester Größe auf.
    
    Args:
        text: Der zu chunkende Text
        chunk_size: Größe jedes Chunks in Zeichen
        overlap: Überlappung zwischen Chunks in Zeichen
        
    Returns:
        List[Dict]: Liste von Chunks mit Text und Metadaten
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Berechne das Ende des aktuellen Chunks
        end = min(start + chunk_size, len(text))
        
        # Finde das nächste Zeilenende oder Leerzeichen nach dem Ende, wenn möglich
        if end < len(text):
            # Versuche, an einem Satzende zu trennen
            next_period = text.find(".", end)
            next_newline = text.find("\n", end)
            next_space = text.find(" ", end)
            
            candidates = [pos for pos in [next_period, next_newline, next_space] if pos != -1]
            if candidates:
                end = min(candidates) + 1  # +1, um den Trenner einzuschließen
        
        # Chunk extrahieren
        chunk = text[start:end]
        
        chunks.append({
            "text": chunk,
            "metadata": {
                "start": start,
                "end": end,
                "size": len(chunk),
                "strategy": "fixed"
            }
        })
        
        # Nächster Startpunkt mit Überlappung
        start = end - overlap
    
    return chunks

def semantic_chunking(text: str, max_chunk_size: int = 2000) -> List[Dict]:
    """
    Teilt einen Text basierend auf semantischen Grenzen (Abschnitte, Absätze).
    
    Args:
        text: Der zu chunkende Text
        max_chunk_size: Maximale Größe eines Chunks
        
    Returns:
        List[Dict]: Liste von Chunks mit Text und Metadaten
    """
    # Identifiziere Abschnittsüberschriften und Absätze
    
    # Abschnittsüberschriften erkennen
    heading_patterns = [
        r"^#+\s+.*$",                  # Markdown-Überschriften
        r"^[A-Z][\w\s]+:$",            # Überschriften mit Doppelpunkt
        r"^\d+\.\d+\s+[A-Z][\w\s]+$",  # Nummerierte Unterabschnitte
        r"^\d+\.\s+[A-Z][\w\s]+$",     # Nummerierte Abschnitte
        r"^[A-Z][A-Z\s]+$"             # GROSSGESCHRIEBENE ÜBERSCHRIFTEN
    ]
    
    # Text in Zeilen aufteilen
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for line in lines:
        # Prüfe, ob die Zeile eine Überschrift ist
        is_heading = any(re.match(pattern, line) for pattern in heading_patterns)
        
        # Wenn eine Überschrift gefunden wird oder der Chunk zu groß wird,
        # beginne einen neuen Chunk
        if (is_heading and current_chunk) or current_chunk_size > max_chunk_size:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "size": len(chunk_text),
                    "strategy": "semantic",
                    "has_heading": is_heading
                }
            })
            current_chunk = []
            current_chunk_size = 0
        
        # Zeile zum aktuellen Chunk hinzufügen
        current_chunk.append(line)
        current_chunk_size += len(line) + 1  # +1 für den Zeilenumbruch
    
    # Letzten Chunk hinzufügen, falls vorhanden
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "size": len(chunk_text),
                "strategy": "semantic"
            }
        })
    
    return chunks

def adaptive_chunking(text: str, metadata: Optional[Dict] = None) -> List[Dict]:
    """
    Passt die Chunking-Strategie dynamisch an die Dokumenteigenschaften an.
    
    Args:
        text: Der zu chunkende Text
        metadata: Metadaten über das Dokument für adaptive Entscheidungen
        
    Returns:
        List[Dict]: Liste von Chunks mit Text und Metadaten
    """
    if metadata is None:
        metadata = {}
    
    # Dokumenteigenschaften analysieren
    doc_length = len(text)
    has_tables = metadata.get("has_tables", False)
    has_images = metadata.get("has_images", False)
    has_formulas = metadata.get("has_formulas", False)
    complexity = metadata.get("complexity", 0.5)
    
    # Anpassung der Chunk-Parameter basierend auf Dokumenteigenschaften
    if complexity > 0.7 or has_formulas:
        # Komplexe Dokumente: Kleinere Chunks mit mehr Überlappung
        chunk_size = 800
        overlap = 250
    elif has_tables or has_images:
        # Dokumente mit Tabellen/Bildern: Mittlere Chunks mit mittlerer Überlappung
        chunk_size = 1200
        overlap = 200
    else:
        # Einfache Textdokumente: Größere Chunks mit weniger Überlappung
        chunk_size = 1500
        overlap = 150
    
    # Bei sehr kurzen Dokumenten: Ein Chunk für das ganze Dokument
    if doc_length < chunk_size:
        return [{
            "text": text,
            "metadata": {
                "size": doc_length,
                "strategy": "adaptive_single"
            }
        }]
    
    # Für längere Dokumente: Kombiniere semantisches und Fixed-Size-Chunking
    chunks = []
    
    # Versuche zuerst semantisches Chunking
    semantic_chunks = semantic_chunking(text, max_chunk_size=chunk_size*1.5)
    
    # Prüfe, ob semantisches Chunking zu großen Chunks führt
    large_chunks = [c for c in semantic_chunks if len(c["text"]) > chunk_size*1.2]
    
    if len(large_chunks) > len(semantic_chunks) / 3:  # Wenn zu viele große Chunks
        # Verwende Fixed-Size-Chunking für große Chunks
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk["text"]) > chunk_size*1.2:
                # Wende Fixed-Size-Chunking auf diesen großen Chunk an
                sub_chunks = fixed_size_chunking(chunk["text"], chunk_size, overlap)
                for sub_chunk in sub_chunks:
                    sub_chunk["metadata"]["strategy"] = "adaptive_fixed"
                final_chunks.extend(sub_chunks)
            else:
                # Behalte kleinere Chunks
                chunk["metadata"]["strategy"] = "adaptive_semantic"
                final_chunks.append(chunk)
        chunks = final_chunks
    else:
        # Semantisches Chunking hat gut funktioniert
        for chunk in semantic_chunks:
            chunk["metadata"]["strategy"] = "adaptive_semantic"
        chunks = semantic_chunks
    
    return chunks

def chunk_text_for_model(text: str, metadata: Optional[Dict] = None, strategy: str = "adaptive") -> List[Dict]:
    """
    Chunks Text basierend auf der angegebenen Strategie.
    
    Args:
        text: Der zu chunkende Text
        metadata: Metadaten über das Dokument (optional)
        strategy: Welche Chunking-Strategie verwendet werden soll (fixed, semantic, adaptive)
        
    Returns:
        List[Dict]: Liste von Chunks mit Text und Metadaten
    """
    if not text:
        return []
        
    if strategy == "fixed":
        return fixed_size_chunking(text)
    elif strategy == "semantic":
        return semantic_chunking(text)
    else:  # adaptive (Standard)
        return adaptive_chunking(text, metadata) 