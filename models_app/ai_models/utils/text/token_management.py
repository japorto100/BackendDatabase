"""
Token-Management-Strategien 

Gemeinsame Funktionen für das Management von Tokens bei langen Kontexten.
"""

from typing import Dict, List, Any, Tuple, Optional

def determine_token_strategy(text_length: int, max_context: int) -> str:
    """
    Bestimmt die optimale Token-Management-Strategie basierend auf Textlänge.
    
    Args:
        text_length: Anzahl der Tokens im Text
        max_context: Maximale Kontextgröße des Modells
        
    Returns:
        str: Die ausgewählte Strategie (direct, truncate, sliding_window, summarize)
    """
    if text_length < max_context * 0.8:
        return "direct"  # Text passt direkt in den Kontext
    elif text_length < max_context * 1.5:
        return "truncate"  # Text kürzen auf maximale Länge
    elif text_length < max_context * 3:
        return "sliding_window"  # Sliding Window für sehr lange Dokumente
    else:
        return "summarize"  # Zusammenfassung für extrem lange Dokumente

def select_relevant_chunks(chunks: List[Dict], query: str, limit: int = 10) -> List[Dict]:
    """
    Wählt die relevantesten Chunks basierend auf einer Abfrage aus.
    
    Args:
        chunks: Liste von Dokumentchunks
        query: Die Benutzerabfrage
        limit: Maximale Anzahl zurückzugebender Chunks
        
    Returns:
        List[Dict]: Die relevantesten Chunks
    """
    # Relevanz-Scoring für jeden Chunk
    scored_chunks = []
    query_words = set(query.lower().split())
    
    for chunk in chunks:
        text = chunk["text"].lower()
        score = 0
        
        # Zähle Übereinstimmungen mit Query-Wörtern
        for word in query_words:
            if len(word) > 3 and word in text:  # Ignoriere sehr kurze Wörter
                score += 1
        
        # Strategie-basierte Anpassung
        if chunk.get("metadata", {}).get("strategy") == "adaptive_semantic":
            score *= 1.2  # Bonus für semantisch bedeutsame Chunks
        
        scored_chunks.append((score, chunk))
    
    # Sortiere nach Relevanz und gib die Top-N zurück
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:limit]] 