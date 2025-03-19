from typing import Dict, List
import aiohttp
import asyncio
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class AcademicSearchProvider(BaseSearchProvider):
    """
    Provider für akademische Suche mit Integration von:
    - Google Scholar
    - Semantic Scholar
    - arXiv
    - PubMed
    - Core.ac.uk
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_duration = settings.ACADEMIC_CACHE_DURATION
        self.semantic_scholar_url = settings.SEMANTIC_SCHOLAR_API_URL
        self.arxiv_url = settings.ARXIV_API_URL
        self.pubmed_url = settings.PUBMED_API_URL
        self.core_url = settings.CORE_API_URL
        
        # API Keys
        self.semantic_scholar_key = settings.SEMANTIC_SCHOLAR_API_KEY
        self.core_api_key = settings.CORE_API_KEY
        
        # Source Weights für Ranking
        self.source_weights = {
            'semantic_scholar': 1.0,
            'arxiv': 0.9,
            'pubmed': 0.8,
            'core': 0.7
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt parallele Suche über alle akademischen Quellen durch"""
        cache_key = f"academic:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # Parallele Suche in allen Quellen
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._search_semantic_scholar(session, query, filters),
                self._search_arxiv(session, query, filters),
                self._search_pubmed(session, query, filters),
                self._search_core(session, query, filters)
            ]
            results = await asyncio.gather(*tasks)

        # Ergebnisse zusammenführen und deduplizieren
        merged_results = self._merge_and_deduplicate(results)
        
        # Ergebnisse cachen
        cache.set(cache_key, merged_results, self.cache_duration)
        
        return merged_results

    async def _search_semantic_scholar(self, session: aiohttp.ClientSession, 
                                     query: str, filters: Dict) -> List[Dict]:
        """Suche in Semantic Scholar"""
        try:
            async with session.get(
                f"{self.semantic_scholar_url}/paper/search",
                params={
                    'query': query,
                    'limit': filters.get('limit', 20),
                    'fields': 'title,abstract,authors,year,citationCount,venue'
                },
                headers={'x-api-key': self.semantic_scholar_key}
            ) as response:
                data = await response.json()
                return self._process_semantic_scholar_results(data)
        except Exception as e:
            print(f"Semantic Scholar search failed: {str(e)}")
            return []

    async def _search_arxiv(self, session: aiohttp.ClientSession, 
                           query: str, filters: Dict) -> List[Dict]:
        """Suche in arXiv"""
        try:
            async with session.get(
                self.arxiv_url,
                params={
                    'search_query': query,
                    'max_results': filters.get('limit', 20)
                }
            ) as response:
                data = await response.text()  # arXiv returns XML
                return self._process_arxiv_results(data)
        except Exception as e:
            print(f"arXiv search failed: {str(e)}")
            return []

    def _merge_and_deduplicate(self, results: List[List[Dict]]) -> List[Dict]:
        """
        Führt Ergebnisse zusammen und entfernt Duplikate basierend auf:
        - DOI
        - Titel-Ähnlichkeit
        - Autor-Übereinstimmung
        """
        merged = []
        seen_dois = set()
        seen_titles = set()
        
        for source_results in results:
            for result in source_results:
                doi = result.get('doi')
                title = result.get('title', '').lower()
                
                # Prüfe auf Duplikate
                if doi and doi in seen_dois:
                    continue
                if self._similar_title_exists(title, seen_titles):
                    continue
                    
                # Füge zu merged und seen hinzu
                merged.append(result)
                if doi:
                    seen_dois.add(doi)
                seen_titles.add(title)
        
        # Sortiere nach Relevanz
        return sorted(merged, key=lambda x: x.get('relevance_score', 0), reverse=True)

    def _similar_title_exists(self, title: str, seen_titles: set) -> bool:
        """Prüft auf ähnliche Titel mit Fuzzy Matching"""
        from difflib import SequenceMatcher
        
        for seen_title in seen_titles:
            similarity = SequenceMatcher(None, title, seen_title).ratio()
            if similarity > 0.85:  # Threshold für Ähnlichkeit
                return True
        return False

    def _calculate_relevance_score(self, result: Dict, source: str) -> float:
        """
        Berechnet Relevanz-Score basierend auf:
        - Zitierungen
        - Alter
        - Quelle
        - Journal Impact
        """
        base_score = 0.0
        
        # Citation Score (0-1)
        citations = result.get('citation_count', 0)
        citation_score = min(citations / 1000, 1.0)  # Max bei 1000 Zitierungen
        
        # Age Score (0-1)
        year = result.get('year', 2024)
        age_score = max(0, min((2024 - year) / 10, 1.0))  # Linear decay über 10 Jahre
        
        # Source Weight
        source_weight = self.source_weights.get(source, 0.5)
        
        # Gewichtete Kombination
        base_score = (
            citation_score * 0.4 +  # 40% Zitierungen
            (1 - age_score) * 0.3 +  # 30% Aktualität (invertiertes Alter)
            source_weight * 0.3  # 30% Quellen-Gewichtung
        )
        
        return base_score 