"""
Utilities zum Kombinieren von Ergebnissen aus verschiedenen Chunks

Bietet Funktionen zum Zusammenführen von Antworten aus mehreren Verarbeitungsschritten.
"""

from typing import Dict, List, Any, Optional
import re

def combine_chunk_results(results: List[Dict[str, Any]], strategy: str = "hierarchical") -> Dict[str, Any]:
    """
    Kombiniert die Ergebnisse aus mehreren Chunk-Verarbeitungen.
    
    Args:
        results: Liste von Ergebnissen aus verschiedenen Chunks
        strategy: Kombinationsstrategie (hierarchical, weighted, simple)
        
    Returns:
        Dict: Kombiniertes Ergebnis
    """
    if not results:
        return {"response": "", "confidence": 0.0, "strategy": "empty_result"}
    
    if len(results) == 1:
        return results[0]
    
    if strategy == "simple":
        return _simple_combine(results)
    elif strategy == "weighted":
        return _weighted_combine(results)
    else:  # hierarchical (default)
        return _hierarchical_combine(results)

def _simple_combine(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Einfache Kombination: Konkateniert alle Antworten mit Absätzen.
    
    Args:
        results: Liste von Ergebnissen
        
    Returns:
        Dict: Kombiniertes Ergebnis
    """
    combined_response = "\n\n".join([r.get("response", "") for r in results])
    avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results)
    
    return {
        "response": combined_response,
        "confidence": avg_confidence,
        "strategy": "simple_combine",
        "sources": [r.get("model_used", "unknown") for r in results]
    }

def _weighted_combine(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gewichtete Kombination: Berücksichtigt Konfidenzwerte.
    
    Args:
        results: Liste von Ergebnissen
        
    Returns:
        Dict: Kombiniertes Ergebnis
    """
    # Extrahiere Antworten und Konfidenzwerte
    responses = [r.get("response", "") for r in results]
    confidences = [r.get("confidence", 0) for r in results]
    
    # Normalisiere Konfidenzwerte für die Gewichtung
    total_confidence = sum(confidences)
    if total_confidence == 0:
        # Bei Nullwerten gleiche Gewichtung
        weights = [1/len(results)] * len(results)
    else:
        weights = [c/total_confidence for c in confidences]
    
    # Sortiere nach Gewichtung (höchste zuerst)
    sorted_responses = [x for _, x in sorted(zip(weights, responses), reverse=True)]
    
    # Kombiniere mit Gewichtungsinformation
    combined_response = "\n\n".join(sorted_responses)
    
    return {
        "response": combined_response,
        "confidence": max(confidences),  # Verwende höchsten Konfidenzwert
        "strategy": "weighted_combine",
        "weights": weights,
        "sources": [r.get("model_used", "unknown") for r in results]
    }

def _hierarchical_combine(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hierarchische Kombination: Erstellt eine Zusammenfassung aus allen Ergebnissen.
    Identifiziert gemeinsame Konzepte und erstellt eine strukturierte Antwort.
    
    Args:
        results: Liste von Ergebnissen
        
    Returns:
        Dict: Kombiniertes Ergebnis
    """
    # Sammle alle Antworten
    responses = [r.get("response", "") for r in results]
    
    # Extrahiere Schlüsselkonzepte durch Identifikation gemeinsamer Satzanfänge
    concepts = _extract_common_concepts(responses)
    
    if not concepts:
        # Fallback auf gewichtete Kombination
        return _weighted_combine(results)
    
    # Strukturiere die Antwort basierend auf identifizierten Konzepten
    combined_response = "Zusammenfassung:\n\n"
    
    for concept in concepts:
        combined_response += f"- {concept}\n"
    
    combined_response += "\n\nDetails:\n\n"
    combined_response += "\n\n".join(responses)
    
    # Berechne einen kombinierten Konfidenzwert
    # Verwende den Durchschnitt der Top-3 Konfidenzwerte (oder weniger, wenn nicht genug vorhanden)
    confidences = sorted([r.get("confidence", 0) for r in results], reverse=True)
    top_confidences = confidences[:min(3, len(confidences))]
    avg_confidence = sum(top_confidences) / len(top_confidences)
    
    return {
        "response": combined_response,
        "confidence": avg_confidence,
        "strategy": "hierarchical_combine",
        "concepts": concepts,
        "sources": [r.get("model_used", "unknown") for r in results]
    }

def _extract_common_concepts(texts: List[str]) -> List[str]:
    """
    Extrahiert gemeinsame Konzepte aus mehreren Texten.
    
    Args:
        texts: Liste von Texten
        
    Returns:
        List[str]: Liste gemeinsamer Konzepte
    """
    if not texts:
        return []
    
    # Teile jeden Text in Sätze auf
    all_sentences = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        all_sentences.append(sentences)
    
    # Identifiziere häufige Satzanfänge (potenzielle Konzepte)
    sentence_starts = {}
    for sentences in all_sentences:
        for sentence in sentences:
            # Extrahiere die ersten 5-8 Wörter als potenzielle Konzeptidentifikation
            words = sentence.split()
            if len(words) >= 5:
                start = " ".join(words[:min(8, len(words))])
                sentence_starts[start] = sentence_starts.get(start, 0) + 1
    
    # Behalte nur Konzepte, die in mindestens 1/3 der Texte vorkommen
    threshold = max(2, len(texts) // 3)
    common_concepts = []
    
    for start, count in sentence_starts.items():
        if count >= threshold:
            # Suche den vollständigen Satz, der mit diesem Start beginnt
            for sentences in all_sentences:
                for sentence in sentences:
                    if sentence.startswith(start):
                        if sentence not in common_concepts:
                            common_concepts.append(sentence)
                        break
    
    return common_concepts[:5]  # Begrenze auf 5 Schlüsselkonzepte 