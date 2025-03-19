from typing import Dict, List
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class DocsSearchProvider(BaseSearchProvider):
    """
    Provider für technische Dokumentation mit Multi-Source Support:
    - ReadTheDocs
    - DevDocs
    - MDN Web Docs
    - Python Docs
    - Language-specific Docs
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_duration = settings.DOCS_CACHE_DURATION
        
        # API Endpoints
        self.readthedocs_url = "https://readthedocs.org/api/v3"
        self.devdocs_url = settings.DEVDOCS_API_URL
        self.mdn_url = "https://developer.mozilla.org/api/v1"
        
        # Source Weights
        self.source_weights = {
            'readthedocs': 1.0,
            'devdocs': 0.9,
            'mdn': 0.9,
            'python': 0.8,
            'custom': 0.7
        }
        
        # Custom Doc Sources aus Settings
        self.custom_sources = settings.CUSTOM_DOC_SOURCES

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Durchsucht alle konfigurierten Dokumentationsquellen"""
        cache_key = f"docs:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # Bestimme zu durchsuchende Quellen
        sources = filters.get('sources', ['readthedocs', 'devdocs', 'mdn', 'python'])
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Erstelle Tasks für jede Quelle
            if 'readthedocs' in sources:
                tasks.append(self._search_readthedocs(session, query, filters))
            if 'devdocs' in sources:
                tasks.append(self._search_devdocs(session, query, filters))
            if 'mdn' in sources:
                tasks.append(self._search_mdn(session, query, filters))
            if 'python' in sources:
                tasks.append(self._search_python_docs(session, query, filters))
            
            # Füge Custom Sources hinzu
            for source in self.custom_sources:
                if source['id'] in sources:
                    tasks.append(self._search_custom_source(session, source, query, filters))
            
            # Führe alle Suchen parallel aus
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und ranke Ergebnisse
            combined_results = self._combine_results(results)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _search_readthedocs(self, session: aiohttp.ClientSession, 
                                query: str, filters: Dict) -> List[Dict]:
        """Sucht in ReadTheDocs Projekten"""
        try:
            params = {
                'q': query,
                'page_size': filters.get('limit', 20),
                'format': 'json'
            }
            
            async with session.get(
                f"{self.readthedocs_url}/search",
                params=params
            ) as response:
                data = await response.json()
                
                results = []
                for item in data.get('results', []):
                    # Hole zusätzliche Kontext-Informationen
                    content = await self._fetch_page_content(session, item['link'])
                    context = self._extract_doc_context(content, query)
                    
                    results.append({
                        'type': 'readthedocs',
                        'title': item['title'],
                        'project': item['project'],
                        'version': item['version'],
                        'link': item['link'],
                        'content': content,
                        'context': context,
                        'relevance_score': self._calculate_doc_score(item, context)
                    })
                
                return results
                
        except Exception as e:
            print(f"ReadTheDocs search failed: {str(e)}")
            return []

    async def _fetch_page_content(self, session: aiohttp.ClientSession, url: str) -> str:
        """Holt den Inhalt einer Dokumentationsseite"""
        try:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Entferne Navigation, Sidebar etc.
                for elem in soup.select('nav, sidebar, footer'):
                    elem.decompose()
                    
                # Extrahiere Hauptinhalt
                main_content = soup.select_one('main, article, .document')
                return main_content.get_text() if main_content else ''
                
        except Exception:
            return ''

    def _extract_doc_context(self, content: str, query: str) -> Dict:
        """
        Extrahiert relevanten Dokumentationskontext:
        - Umgebende Abschnitte
        - Code-Beispiele
        - API-Definitionen
        """
        # TODO: Implement sophisticated documentation analysis
        return {
            'sections': [],
            'code_examples': [],
            'api_definitions': [],
            'relevance': 0.0
        }

    def _calculate_doc_score(self, item: Dict, context: Dict) -> float:
        """
        Berechnet Dokumentations-Score basierend auf:
        - Quellen-Gewichtung
        - Kontext-Relevanz
        - Aktualität
        - Qualität
        """
        base_score = self.source_weights.get(item.get('type', 'custom'), 0.5)
        
        # Kontext Score
        context_score = context.get('relevance', 0.0)
        
        # Aktualitäts-Score
        version = item.get('version', '')
        if version and version.lower() != 'latest':
            try:
                version_parts = version.split('.')
                version_score = float(version_parts[0]) / 10  # Normalisiere auf 0-1
            except (ValueError, IndexError):
                version_score = 0.5
        else:
            version_score = 1.0
            
        # Qualitäts-Score basierend auf Content
        quality_score = min(
            len(context.get('code_examples', [])) / 5,  # Normalisiere auf 0-1
            1.0
        )
        
        # Gewichteter Durchschnitt
        return (
            base_score * 0.3 +
            context_score * 0.3 +
            version_score * 0.2 +
            quality_score * 0.2
        )

    def _combine_results(self, results: List[List[Dict]]) -> List[Dict]:
        """
        Kombiniert und dedupliziert Ergebnisse aus verschiedenen Quellen
        """
        combined = []
        seen_urls = set()
        
        # Flatten und dedupliziere
        for source_results in results:
            for result in source_results:
                url = result.get('link')
                if url and url not in seen_urls:
                    combined.append(result)
                    seen_urls.add(url)
        
        # Sortiere nach Relevanz
        return sorted(combined, key=lambda x: x.get('relevance_score', 0), reverse=True) 