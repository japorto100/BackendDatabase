"""
Verarbeitung von @-Mentions als Evidenzquellen.
"""
import json
import uuid
import logging
from .models import Evidence
from .mention_providers import get_mention_provider
from .web_mention_provider import WebMentionProvider

logger = logging.getLogger(__name__)

class MentionProcessor:
    """
    Verarbeitet @-Mentions und lädt deren Inhalte als Kontext für Chats.
    """
    
    def __init__(self):
        self.mention_provider = get_mention_provider()
        
        # Register the WebMentionProvider
        self.register_provider('web', WebMentionProvider())
    
    def process_mentions(self, mentions, query_id=None):
        """
        Verarbeitet eine Liste von @-Mentions und lädt deren Inhalte.
        
        Args:
            mentions (list): Liste von Mention-Objekten, normalerweise im Format
                            {category: '...', id: '...', name: '...'}
            query_id (UUID, optional): ID der Anfrage, falls Evidenz gespeichert werden soll
        
        Returns:
            tuple: (context, sources, evidence) - Kontext für den Chat, Quellen und Evidenzobjekte
        """
        if not mentions:
            return [], [], []
            
        context = []
        sources = []
        evidence_objects = []
        
        for mention in mentions:
            try:
                category = mention.get('category')
                item_id = mention.get('id')
                name = mention.get('name', f"{category}:{item_id}")
                
                if not category or not item_id:
                    continue
                    
                # Details vom Provider abrufen
                details = self.mention_provider.get_item_details(category, item_id)
                
                # Fehler verarbeiten
                if details is None or 'error' in details:
                    logger.warning(f"Keine Details für Mention {category}:{item_id} gefunden")
                    continue
                    
                # Inhalt als Kontext hinzufügen
                if 'content' in details and details['content']:
                    context_entry = f"--- Inhalt von {name} ({category}) ---\n{details['content']}\n\n"
                    context.append(context_entry)
                    
                # Metadaten als Kontext
                metadata_context = []
                for key, value in details.items():
                    if key not in ['content', 'id'] and not isinstance(value, (dict, list)):
                        metadata_context.append(f"{key}: {value}")
                
                if metadata_context:
                    context.append(f"--- Metadaten von {name} ({category}) ---\n")
                    context.append("\n".join(metadata_context) + "\n\n")
                    
                # Quelle für Evidenz speichern
                source = {
                    'id': f"mention-{category}-{item_id}",
                    'type': category,
                    'title': name,
                    'content': details.get('content', ''),
                    'isMention': True,
                    'metadata': details
                }
                sources.append(source)
                
                # Evidence in der Datenbank speichern, wenn query_id vorhanden
                if query_id:
                    evidence = Evidence.objects.create(
                        query_id=query_id,
                        source_type=category,
                        content=details.get('content', ''),
                        highlights=json.dumps([]),
                        is_mention=True
                    )
                    evidence_objects.append(evidence)
                    
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung von Mention {mention}: {str(e)}")
                
        return context, sources, evidence_objects
    
    def format_context(self, context_list):
        """
        Formatiert die Kontextliste als String für die Verwendung in Prompts.
        
        Args:
            context_list (list): Liste von Kontexteinträgen
            
        Returns:
            str: Formatierter Kontext
        """
        if not context_list:
            return ""
            
        return "".join(context_list) 