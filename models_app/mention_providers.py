"""
MentionProvider Abstraktionsschicht.

Dieses Modul definiert Schnittstellen und Implementierungen für verschiedene Datenquellen,
die im @-Mention-System verwendet werden können. Es dient als Bridge zwischen dem
Django-Backend und externen Datenquellen (PostgreSQL, MongoDB, OneDrive, etc.).
"""

from abc import ABC, abstractmethod
import os
import json
from django.conf import settings
import requests
from typing import List, Dict, Any, Optional
from search_app.providers.universal import UniversalSearchProvider
from search_app.providers import get_search_provider
import logging
from django.core.cache import cache
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class MentionProvider(ABC):
    """Basisklasse für alle Mention-Provider."""
    
    @abstractmethod
    def get_categories(self) -> List[Dict[str, str]]:
        """
        Verfügbare Kategorien für @-Mentions abrufen.
        
        Returns:
            List[Dict[str, str]]: Liste von Kategorien mit id, name und icon.
        """
        pass
    
    @abstractmethod
    def search(self, category: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Suche nach Elementen in einer bestimmten Kategorie.
        
        Args:
            category: Die zu durchsuchende Kategorie (z.B. 'projekte', 'dokumente').
            query: Die Suchanfrage.
            limit: Maximale Anzahl der zurückzugebenden Ergebnisse.
            
        Returns:
            List[Dict[str, Any]]: Liste von Ergebnissen mit id, name, path, etc.
        """
        pass
    
    @abstractmethod
    def get_item_details(self, category: str, item_id: str) -> Dict[str, Any]:
        """
        Details zu einem bestimmten Element abrufen.
        
        Args:
            category: Die Kategorie des Elements.
            item_id: Die ID des Elements.
            
        Returns:
            Dict[str, Any]: Details des Elements.
        """
        pass

class LocalMentionProvider(MentionProvider):
    """
    Lokale Implementierung des MentionProviders für Entwicklungszwecke.
    Verwendet JSON-Dateien oder die Django-Datenbank als Datenquelle.
    """
    
    def __init__(self):
        """Initialisiere den LocalMentionProvider."""
        # Standardkategorien für die lokale Implementierung
        self.default_categories = [
            {'id': 'projekte', 'name': 'Projekte', 'icon': '🏗️'},
            {'id': 'dokumente', 'name': 'Dokumente', 'icon': '📄'},
            {'id': 'kontakte', 'name': 'Kontakte', 'icon': '👥'},
            {'id': 'web', 'name': 'Web-Suche', 'icon': '🌐'},
        ]
    
    def get_categories(self) -> List[Dict[str, str]]:
        """
        Verfügbare Kategorien für @-Mentions abrufen.
        In der lokalen Implementierung werden feste Kategorien zurückgegeben.
        
        Returns:
            List[Dict[str, str]]: Liste von Kategorien mit id, name und icon.
        """
        # In einer erweiterten Implementierung könnten Kategorien aus der Datenbank abgerufen werden
        return self.default_categories
    
    def search(self, category: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Suche nach Elementen in einer bestimmten Kategorie.
        
        Args:
            category: Die zu durchsuchende Kategorie.
            query: Die Suchanfrage.
            limit: Maximale Anzahl der zurückzugebenden Ergebnisse.
            
        Returns:
            List[Dict[str, Any]]: Liste von Ergebnissen.
        """
        results = []
        
        if category == 'projekte':
            # Beispielimplementierung für Projekte aus der Django-Datenbank
            from .models import Project  # Diese Modellklasse müsste existieren
            projects = Project.objects.filter(name__icontains=query)[:limit]
            for project in projects:
                results.append({
                    'id': str(project.id),
                    'name': project.name,
                    'description': project.description,
                    'type': 'project',
                    'icon': '🏗️'
                })
        
        elif category == 'dokumente':
            # Beispielimplementierung für Dokumente aus der Django-Datenbank
            from .models import UploadedFile  # Verwenden des existierenden Modells
            files = UploadedFile.objects.filter(file__icontains=query)[:limit]
            for file in files:
                file_path = file.file.url if file.file else ""
                file_name = file.file.name.split('/')[-1] if file.file else "Unbekannte Datei"
                results.append({
                    'id': str(file.id),
                    'name': file_name,
                    'path': file_path,
                    'project': getattr(file, 'project_name', None),
                    'type': self._get_file_type(file_name),
                    'icon': self._get_file_icon(file_name)
                })
        
        elif category == 'web':
            # Beispielimplementierung für gespeicherte Web-Suchen
            from .models import SearchQuery  # Verwenden des existierenden Modells
            searches = SearchQuery.objects.filter(query__icontains=query)[:limit]
            for search in searches:
                results.append({
                    'id': str(search.id),
                    'name': search.query,
                    'description': f"Suche vom {search.created_at.strftime('%d.%m.%Y')}",
                    'type': 'web',
                    'icon': '🌐'
                })
        
        return results
    
    def get_item_details(self, category: str, item_id: str) -> Dict[str, Any]:
        """
        Details zu einem bestimmten Element abrufen.
        
        Args:
            category: Die Kategorie des Elements.
            item_id: Die ID des Elements.
            
        Returns:
            Dict[str, Any]: Details des Elements.
        """
        if category == 'projekte':
            from .models import Project  # Diese Modellklasse müsste existieren
            try:
                project = Project.objects.get(id=item_id)
                return {
                    'id': str(project.id),
                    'name': project.name,
                    'description': project.description,
                    'type': 'project',
                    'created_at': project.created_at,
                    'updated_at': project.updated_at,
                    'files': self._get_project_files(project.id)
                }
            except Project.DoesNotExist:
                return {'error': 'Projekt nicht gefunden'}
        
        elif category == 'dokumente':
            from .models import UploadedFile
            try:
                file = UploadedFile.objects.get(id=item_id)
                file_path = file.file.url if file.file else ""
                file_name = file.file.name.split('/')[-1] if file.file else "Unbekannte Datei"
                return {
                    'id': str(file.id),
                    'name': file_name,
                    'path': file_path,
                    'type': self._get_file_type(file_name),
                    'uploaded_at': file.uploaded_at,
                    'content_type': getattr(file, 'content_type', None),
                    'content': self._get_file_content(file)
                }
            except UploadedFile.DoesNotExist:
                return {'error': 'Datei nicht gefunden'}
        
        return {'error': 'Kategorie oder Element nicht unterstützt'}
    
    def _get_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """Helper-Methode, um Dateien eines Projekts abzurufen."""
        # In einer erweiterten Implementierung würden hier die tatsächlichen Projektdateien abgerufen
        return []
    
    def _get_file_content(self, file_obj) -> str:
        """Helper-Methode, um den Inhalt einer Datei abzurufen."""
        # In einer erweiterten Implementierung würde hier der tatsächliche Dateiinhalt abgerufen
        return ""
    
    def _get_file_type(self, filename: str) -> str:
        """Helper-Methode, um den Dateityp aus dem Dateinamen zu extrahieren."""
        if not filename:
            return 'unknown'
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        return ext
    
    def _get_file_icon(self, filename: str) -> str:
        """Helper-Methode, um das passende Icon für den Dateityp zu bestimmen."""
        ext = self._get_file_type(filename)
        icons = {
            'pdf': '📕',
            'doc': '📘',
            'docx': '📘',
            'xls': '📗',
            'xlsx': '📗',
            'ppt': '📙',
            'pptx': '📙',
            'jpg': '🖼️',
            'jpeg': '🖼️',
            'png': '🖼️',
            'gif': '🖼️',
            'txt': '📄',
            'csv': '📊',
            'json': '📋',
            'js': '🔧',
            'py': '🐍',
            'html': '🌐',
            'css': '🎨',
        }
        return icons.get(ext, '📄')

class ExternalApiMentionProvider(MentionProvider):
    """
    Implementierung des MentionProviders für externe APIs.
    Diese Klasse kommuniziert mit dem anderen Backend über REST-APIs.
    """
    
    def __init__(self, api_url: str = None, api_key: str = None):
        """
        Initialisiere den ExternalApiMentionProvider.
        
        Args:
            api_url: Die Basis-URL der externen API.
            api_key: Der API-Schlüssel für die Authentifizierung.
        """
        self.api_url = api_url or settings.EXTERNAL_API_URL
        self.api_key = api_key or settings.EXTERNAL_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_categories(self) -> List[Dict[str, str]]:
        """
        Verfügbare Kategorien von der externen API abrufen.
        
        Returns:
            List[Dict[str, str]]: Liste von Kategorien mit id, name und icon.
        """
        try:
            response = requests.get(
                f"{self.api_url}/mentions/categories", 
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get('categories', [])
        except Exception as e:
            print(f"Fehler beim Abrufen der Kategorien: {e}")
            # Fallback zu Standardkategorien
            return [
                {'id': 'projekte', 'name': 'Projekte', 'icon': '🏗️'},
                {'id': 'dokumente', 'name': 'Dokumente', 'icon': '📄'},
                {'id': 'web', 'name': 'Web-Suche', 'icon': '🌐'},
            ]
    
    def search(self, category: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Suche nach Elementen in einer bestimmten Kategorie über die externe API.
        
        Args:
            category: Die zu durchsuchende Kategorie.
            query: Die Suchanfrage.
            limit: Maximale Anzahl der zurückzugebenden Ergebnisse.
            
        Returns:
            List[Dict[str, Any]]: Liste von Ergebnissen.
        """
        try:
            params = {
                'q': query,
                'limit': limit
            }
            response = requests.get(
                f"{self.api_url}/mentions/search/{category}", 
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            print(f"Fehler bei der Suche in der Kategorie {category}: {e}")
            return []
    
    def get_item_details(self, category: str, item_id: str) -> Dict[str, Any]:
        """
        Details zu einem bestimmten Element über die externe API abrufen.
        
        Args:
            category: Die Kategorie des Elements.
            item_id: Die ID des Elements.
            
        Returns:
            Dict[str, Any]: Details des Elements.
        """
        try:
            response = requests.get(
                f"{self.api_url}/mentions/{category}/{item_id}", 
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Fehler beim Abrufen der Details für {category}/{item_id}: {e}")
            return {'error': 'Fehler beim Abrufen der Details'}

class WebMentionProvider(MentionProvider):
    """
    Provider for @Web functionality.
    Allows automatic web search based on user input and adds results to chat context.
    Uses the UniversalSearchProvider to provide a similar experience to
    modern AI assistants like ChatGPT and Claude.
    """
    
    def __init__(self):
        """Initialize the WebMentionProvider."""
        self.cache_duration = getattr(settings, 'WEB_SEARCH_CACHE_DURATION', 3600)  # 1 hour default
    
    def get_categories(self) -> List[Dict[str, str]]:
        """
        Returns available categories for web mentions.
        This appears in the @mention dropdown.
        """
        # Simplified to just show Web Search as the option
        return [
            {"id": "web", "name": "Web Search", "icon": "🌐"}
        ]
    
    def search(self, category: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for results using the specified category and query.
        Always uses the universal search provider regardless of category.
        
        Args:
            category: Search category (ignored, always uses universal)
            query: Search query or user message
            limit: Maximum number of results
            
        Returns:
            list: List of search results
        """
        try:
            # Extract a search query if not explicitly provided
            search_query = self._extract_search_query(query)
            
            # Create cache key
            cache_key = f"web_mention:universal:{search_query}:{limit}"
            cached_results = cache.get(cache_key)
            if cached_results:
                return cached_results
            
            # Get the universal search provider specifically
            provider = get_search_provider("universal")
            if not provider:
                # Create a new instance if not found through the factory
                logger.info("Universal provider not found via factory, creating directly")
                provider = UniversalSearchProvider()
            
            # Perform the search
            search_results = self._perform_search(provider, search_query, limit)
            
            # Cache results
            cache.set(cache_key, search_results, self.cache_duration)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def _perform_search(self, provider, query, limit):
        """
        Performs the actual search using the provider.
        Handles the asynchronous nature of UniversalSearchProvider.
        """
        try:
            # Create basic filters
            filters = {
                'max_results': limit,
                'enhance': True  # Request enhanced results with AI processing
            }
            
            # For async providers (UniversalSearchProvider is async)
            if hasattr(provider, 'search') and asyncio.iscoroutinefunction(provider.search):
                # Run async search in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                search_results = loop.run_until_complete(provider.search(query, filters))
                loop.close()
            # For sync providers (fallback)
            else:
                search_results = provider.search(query, filters)
            
            # Format results for mention system
            return [
                {
                    "id": f"web_universal_{result.get('id', i)}",
                    "title": result.get("title", "Web search result"),
                    "description": self._format_description(result),
                    "url": result.get("url", ""),
                    "provider": "universal",
                    "relevance_score": result.get("relevance_score", 0.5),
                    # Include additional metadata that may be useful for context
                    "metadata": {
                        "source": result.get("source", "web"),
                        "timestamp": result.get("timestamp", ""),
                        "category": result.get("category", "")
                    }
                }
                for i, result in enumerate(search_results)
            ]
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def get_item_details(self, category, item_id):
        """
        Formats search results for the mention system.
        """
        formatted_results = []
        
        for i, result in enumerate(results):
            # Extract the URL if available
            url = result.get('url', '')
            if not url and 'link' in result:
                url = result['link']
            
            # Format the result
            formatted_results.append({
                'id': result.get('id', str(i)),
                'name': result.get('title', 'Search Result'),
                'description': self._format_description(result),
                'url': url,
                'type': 'web',
                'icon': '🌐',
                'relevance_score': result.get('relevance_score', 0)
            })
            
        return formatted_results
    
    def get_item_details(self, category: str, item_id: str) -> Dict[str, Any]:
        """
        Gets additional details about a specific search result.
        For web results, this typically means extracting the full content.
        
        Args:
            category: Item category (ignored, always uses universal)
            item_id: Item ID
            
        Returns:
            dict: Item details with content
        """
        try:
            # Extract the original ID from the item_id
            parts = item_id.split("_")
            if len(parts) < 3:
                return {"error": "Invalid item ID format"}
            
            original_id = "_".join(parts[2:])  # Combine remaining parts as the original ID
            
            # Try to get the cached result
            cache_key = f"web_mention:item:{category}:{item_id}"
            cached_item = cache.get(cache_key)
            if cached_item:
                return cached_item
                
            # Get the provider
            provider = get_search_provider("universal")
            if not provider:
                provider = UniversalSearchProvider()
                
            # Get the URL from the provider
            url = None
            result = {}
            
            # Handle both async and sync get_details methods
            if hasattr(provider, 'get_details'):
                if asyncio.iscoroutinefunction(provider.get_details):
                    # Run async get_details in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(provider.get_details(original_id))
                    loop.close()
                else:
                    result = provider.get_details(original_id)
                
                if result and 'url' in result:
                    url = result['url']
            
            # If we have a URL, fetch and extract content
            if url:
                content = self._extract_content_from_url(url)
            else:
                content = "Content could not be retrieved."
            
            # Format the complete result
            item_details = {
                "id": item_id,
                "title": result.get("title", "Web search result"),
                "content": content,
                "url": url,
                "metadata": {
                    "provider": "universal",
                    "source_type": "web_search",
                    "category": "web",
                    "timestamp": result.get("timestamp", ""),
                    "author": result.get("author", ""),
                    "publisher": result.get("publisher", "")
                }
            }
            
            # Cache the item details
            cache.set(cache_key, item_details, self.cache_duration)
            
            return item_details
                
        except Exception as e:
            logger.error(f"Error getting item details: {str(e)}")
            return {"error": str(e)}
    
    def extract_query_from_message(self, message_text):
        """
        Extracts a search query from a message containing @Web.
        
        Args:
            message_text: Full message text
            
        Returns:
            str: Extracted search query
        """
        # Remove @Web from the message
        clean_text = message_text.replace('@Web', '').strip()
        
        # Use simple keyword extraction for now
        # In production, this would use an LLM to generate a better query
        return clean_text
    
    def _extract_content_from_url(self, url):
        """
        Extracts readable content from a URL.
        """
        try:
            import requests
            from readability import Document
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                doc = Document(response.text)
                title = doc.title()
                content = doc.summary()
                
                # Clean HTML with BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n\n')
                
                # Truncate if too long
                if len(text) > 8000:
                    text = text[:8000] + "...\n[Content truncated due to length]"
                
                return f"## {title}\n\n{text}\n\nSource: {url}"
            
            return f"Could not extract content from {url}"
            
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return f"Error extracting content: {str(e)}"
    
    def _format_description(self, result):
        """Formats the description for a search result."""
        if "snippet" in result:
            return result["snippet"]
        elif "description" in result:
            return result["description"]
        else:
            return "No description available"
    
    def _extract_search_query(self, text):
        """
        Extracts or optimizes a search query from text.
        This would ideally use an LLM to generate an optimal search query.
        """
        # For now, use the text directly
        # In production, this would call an LLM to generate a better query
        return text

# Factory-Methode, um den richtigen MentionProvider zu erstellen
def get_mention_provider(provider_type: str = None) -> MentionProvider:
    """
    Factory-Methode zur Erstellung des richtigen MentionProviders.
    
    Args:
        provider_type: Der Typ des zu erstellenden Providers.
                      Mögliche Werte: 'local', 'external_api', 'web'.
                      
    Returns:
        MentionProvider: Eine Instanz einer MentionProvider-Unterklasse.
    """
    provider_type = provider_type or getattr(settings, 'MENTION_PROVIDER_TYPE', 'local')
    
    if provider_type == 'local':
        return LocalMentionProvider()
    elif provider_type == 'external_api':
        return ExternalApiMentionProvider()
    elif provider_type == 'web':
        return WebMentionProvider()
    else:
        raise ValueError(f"Unbekannter Provider-Typ: {provider_type}")

# Class to handle all mention providers
class MentionProcessor:
    """
    Central processor for all mention types.
    Loads and manages different mention providers.
    """
    
    def __init__(self):
        """Initialize with all available providers."""
        self.providers = {
            'local': get_mention_provider('local'),
            'external': get_mention_provider('external_api'),
            'web': get_mention_provider('web')
        }
    
    def get_provider(self, provider_type):
        """Get a specific provider by type."""
        return self.providers.get(provider_type) 