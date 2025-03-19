from typing import Dict, List
import aiohttp
from pathlib import Path
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class DocumentSearchProvider(BaseSearchProvider):
    """
    Provider für allgemeine Dokumentensuche (ohne OCR/ColPali).
    Für lokale Dokumente mit OCR/ColPali siehe LocalDocsSearchProvider.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_duration = settings.DOCUMENT_CACHE_DURATION
        self.max_file_size = settings.MAX_DOCUMENT_SIZE
        
    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Sucht in Remote-Dokumenten"""
        if not self._is_safe_remote_search(filters):
            raise ValueError("Remote document search not allowed or unsafe")
            
        cache_key = f"doc_remote:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results
            
        results = await self._search_remote_documents(query, filters)
        
        if results:
            cache.set(cache_key, results, self.cache_duration)
            
        return results

    async def _search_remote_documents(self, query: str, filters: Dict) -> List[Dict]:
        """Durchsucht Remote-Dokumente"""
        results = []
        allowed_domains = settings.ALLOWED_DOCUMENT_DOMAINS
        target_domain = filters.get('domain')
        
        if target_domain not in allowed_domains:
            return []
            
        # Implementiere sichere Remote-Dokumentensuche
        # TODO: Add implementation
        
        return results

    def _is_valid_document(self, path: Path) -> bool:
        """Prüft ob Dokument valid und sicher ist"""
        return (
            path.suffix.lower() in ['.pdf', '.doc', '.docx', '.txt'] and
            path.stat().st_size <= self.max_file_size and
            not self._contains_malware(path)
        )

    def _contains_malware(self, path: Path) -> bool:
        """Basic Malware Check"""
        # TODO: Implementiere bessere Malware-Erkennung
        return False

    def _is_safe_remote_search(self, filters: Dict) -> bool:
        """Prüft ob Remote-Suche sicher ist"""
        allowed_domains = settings.ALLOWED_DOCUMENT_DOMAINS
        target_domain = filters.get('domain')
        
        return target_domain in allowed_domains 