"""
Prompt-Templates für verschiedene LLM Provider

Standardisierte Templates zur einheitlichen Prompt-Erstellung.
"""

from typing import Dict, Optional

def create_document_prompt(text: str, query: str, format: str = "generic") -> str:
    """
    Erstellt einheitliche Prompts für Dokumentanfragen.
    
    Args:
        text: Der Dokumenttext
        query: Die Benutzerabfrage
        format: Das zu verwendende Format (generic, anthropic, openai, deepseek, qwq)
        
    Returns:
        str: Der formatierte Prompt
    """
    templates = {
        "generic": "Dokument: {text}\n\nFrage: {query}",
        "anthropic": "<document>\n{text}\n</document>\n\nBitte beantworte folgende Frage: {query}",
        "openai": "Du analysierst folgendes Dokument:\n{text}\n\nBeantworte diese Frage: {query}",
        "deepseek": "<|system|>\nDu bist ein hilfreicher Assistent, der Dokumente analysiert.\n<|/system|>\n<|user|>\nDokument:\n{text}\n\nFrage: {query}\n<|/user|>\n<|assistant|>",
        "qwq": "Dokument:\n{text}\n\nBitte beantworte die folgende Frage zum Dokument: {query}"
    }
    
    return templates.get(format, templates["generic"]).format(text=text, query=query)

def create_summary_prompt(text: str, format: str = "generic") -> str:
    """
    Erstellt einheitliche Prompts für Dokumentzusammenfassungen.
    
    Args:
        text: Der zu zusammenfassende Text
        format: Das zu verwendende Format (generic, anthropic, openai, deepseek, qwq)
        
    Returns:
        str: Der formatierte Prompt
    """
    templates = {
        "generic": "Bitte fasse folgendes Dokument zusammen:\n\n{text}",
        "anthropic": "<document>\n{text}\n</document>\n\nBitte erstelle eine prägnante Zusammenfassung dieses Dokuments.",
        "openai": "Du erstellst eine Zusammenfassung des folgenden Dokuments:\n{text}\n\nZusammenfassung:",
        "deepseek": "<|system|>\nDu bist ein hilfreicher Assistent, der Texte zusammenfasst.\n<|/system|>\n<|user|>\nBitte fasse diesen Text zusammen:\n{text}\n<|/user|>\n<|assistant|>",
        "qwq": "Bitte erstelle eine umfassende Zusammenfassung des folgenden Dokuments:\n\n{text}"
    }
    
    return templates.get(format, templates["generic"]).format(text=text) 