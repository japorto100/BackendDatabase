from typing import Dict, List
import aiohttp
from .base import BaseSearchProvider
from django.conf import settings
from django.core.cache import cache

class WebSearchProvider(BaseSearchProvider):
    """
    Provider für Web-Suche via SearXNG mit AI-Enhancement.
    Nutzt LLM für Query-Reformulation und Result-Processing.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.searxng_url = settings.SEARXNG_URL
        self.llm_url = settings.LLM_API_URL
        self.cache_duration = settings.SEARCH_CACHE_DURATION
        self.max_results = settings.MAX_SEARCH_RESULTS
        
        # Default Search Engines
        self.default_engines = [
            'google',
            'bing',
            'duckduckgo',
            'brave',
            'stackoverflow',
            'github'
        ]
        
    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Web-Suche durch mit Query-Optimierung und Result-Processing"""
        cache_key = f"websearch:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results
            
        # 1. Query Analysis & Reformulation
        enhanced_query = await self._enhance_query(query)
        
        # 2. Fetch results from SearXNG
        raw_results = await self._fetch_searx_results(
            enhanced_query, 
            filters.get('engines', self.default_engines)
        )
        
        # 3. Process and enhance results
        processed_results = await self._process_results(query, raw_results)
        
        # 4. Cache results
        cache.set(cache_key, processed_results, self.cache_duration)
        
        return processed_results

    async def _enhance_query(self, query: str) -> str:
        """Nutzt LLM für Query-Verbesserung"""
        prompt = f"""
        Analyze and enhance this search query for better results.
        Original query: {query}
        
        Consider:
        1. Missing context
        2. Alternative terms
        3. Technical synonyms
        4. Common misspellings
        
        Return only the enhanced query without explanation.
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.llm_url,
                    json={'prompt': prompt}
                ) as response:
                    result = await response.json()
                    return result.get('response', query)
        except Exception as e:
            print(f"Query enhancement failed: {str(e)}")
            return query

    async def _fetch_searx_results(self, query: str, engines: List[str]) -> List[Dict]:
        """Fetcht Ergebnisse von SearXNG"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.searxng_url}/search",
                    params={
                        'q': query,
                        'format': 'json',
                        'engines': ','.join(engines),
                        'max_results': self.max_results
                    }
                ) as response:
                    results = await response.json()
                    return results.get('results', [])
        except Exception as e:
            print(f"SearXNG search failed: {str(e)}")
            return []

    async def _process_results(self, original_query: str, results: List[Dict]) -> List[Dict]:
        """
        Verarbeitet und verbessert Suchergebnisse mit LLM:
        - Relevanz-Scoring
        - Snippet-Generierung
        - Kategorisierung
        """
        if not results:
            return []
            
        prompt = f"""
        Analyze these search results for the query: {original_query}
        
        Results:
        {results[:5]}  # Process top 5 for efficiency
        
        For each result:
        1. Rate relevance (0-1)
        2. Generate better snippet
        3. Categorize (web, tech, news, etc.)
        4. Extract key points
        
        Return as JSON array.
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.llm_url,
                    json={'prompt': prompt}
                ) as response:
                    enhanced = await response.json()
                    
            # Merge enhanced data with original results
            for i, result in enumerate(results):
                if i < len(enhanced):
                    result.update(enhanced[i])
                    
            return sorted(results, key=lambda x: x.get('relevance', 0), reverse=True)
        except Exception as e:
            print(f"Result processing failed: {str(e)}")
            return results

    def _prepare_cache_key(self, query: str, filters: Dict) -> str:
        """Generiert eindeutigen Cache-Key"""
        filter_str = ','.join(f"{k}:{v}" for k, v in sorted(filters.items()))
        return f"websearch:{query}:{filter_str}" 