from typing import Dict, List
import aiohttp
from datetime import datetime
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class EUDataSearchProvider(BaseSearchProvider):
    """
    Provider für EU Open Data Portal mit:
    - Dataset Suche
    - Statistik APIs
    - Eurostat Integration
    - Multi-Language Support
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_duration = settings.EUDATA_CACHE_DURATION
        
        # API Endpoints
        self.data_europa_url = "https://data.europa.eu/api/hub/search/datasets"
        self.eurostat_url = "https://ec.europa.eu/eurostat/api/dissemination"
        self.eu_opendata_url = "https://open-data.europa.eu/api/v2"
        
        # API Keys
        self.eurostat_key = settings.EUROSTAT_API_KEY
        
        # Supported Languages
        self.languages = ['en', 'de', 'fr', 'it', 'es']
        
        # Dataset Type Weights
        self.type_weights = {
            'statistical': 1.0,
            'geospatial': 0.9,
            'financial': 0.9,
            'legislative': 0.8,
            'environmental': 0.8,
            'other': 0.7
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt EU Data Portal Suche durch"""
        cache_key = f"eudata:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # Bestimme Sprache
        lang = filters.get('language', 'en')
        if lang not in self.languages:
            lang = 'en'
            
        async with aiohttp.ClientSession() as session:
            # Parallel Suche in verschiedenen Quellen
            tasks = [
                self._search_datasets(session, query, lang, filters),
                self._search_eurostat(session, query, lang, filters),
                self._search_opendata(session, query, lang, filters)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und analysiere Ergebnisse
            combined_results = self._process_results(results)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _search_datasets(self, session: aiohttp.ClientSession, 
                             query: str, lang: str, filters: Dict) -> List[Dict]:
        """Sucht im EU Open Data Portal"""
        try:
            params = {
                'q': query,
                'limit': filters.get('limit', 20),
                'language': lang,
                'format': 'json'
            }
            
            # Füge Filter hinzu
            if 'theme' in filters:
                params['theme'] = filters['theme']
            if 'publisher' in filters:
                params['publisher'] = filters['publisher']
            if 'format' in filters:
                params['res_format'] = filters['format']
                
            async with session.get(
                self.data_europa_url,
                params=params
            ) as response:
                data = await response.json()
                
                # Verarbeite Datasets
                processed_datasets = []
                for dataset in data.get('results', []):
                    # Hole zusätzliche Metadaten
                    metadata = await self._get_dataset_metadata(
                        session, 
                        dataset['id']
                    )
                    
                    processed_datasets.append({
                        'type': 'dataset',
                        'id': dataset['id'],
                        'title': dataset.get('title', {}),
                        'description': dataset.get('description', {}),
                        'theme': dataset.get('theme', []),
                        'keywords': dataset.get('keywords', []),
                        'publisher': dataset.get('publisher', {}),
                        'issued': dataset.get('issued'),
                        'modified': dataset.get('modified'),
                        'language': dataset.get('language', []),
                        'format': dataset.get('format', []),
                        'metadata': metadata,
                        'relevance_score': self._calculate_dataset_score(dataset)
                    })
                    
                return processed_datasets
                
        except Exception as e:
            print(f"EU Dataset search failed: {str(e)}")
            return []

    async def _search_eurostat(self, session: aiohttp.ClientSession, 
                             query: str, lang: str, filters: Dict) -> List[Dict]:
        """Sucht in Eurostat Daten"""
        try:
            headers = {'Accept': 'application/json'}
            if self.eurostat_key:
                headers['Authorization'] = f'Bearer {self.eurostat_key}'
                
            params = {
                'query': query,
                'lang': lang,
                'format': 'json'
            }
            
            async with session.get(
                f"{self.eurostat_url}/search",
                params=params,
                headers=headers
            ) as response:
                data = await response.json()
                
                processed_stats = []
                for stat in data.get('items', []):
                    # Optional: Hole Vorschaudaten
                    preview = None
                    if filters.get('include_preview', False):
                        preview = await self._get_stat_preview(
                            session, 
                            stat['code']
                        )
                    
                    processed_stats.append({
                        'type': 'statistical',
                        'code': stat['code'],
                        'title': stat.get('title', {}),
                        'description': stat.get('description', {}),
                        'unit': stat.get('unit'),
                        'time_coverage': stat.get('time_coverage'),
                        'geo_coverage': stat.get('geo_coverage'),
                        'update_frequency': stat.get('update_frequency'),
                        'preview_data': preview,
                        'relevance_score': self._calculate_stat_score(stat)
                    })
                    
                return processed_stats
                
        except Exception as e:
            print(f"Eurostat search failed: {str(e)}")
            return []

    def _calculate_dataset_score(self, dataset: Dict) -> float:
        """
        Berechnet Dataset Score basierend auf:
        - Datentyp
        - Aktualität
        - Vollständigkeit
        - Sprachen
        """
        # Basis-Score nach Typ
        theme = dataset.get('theme', ['other'])[0].lower()
        base_score = self.type_weights.get(
            next((t for t in self.type_weights.keys() if t in theme), 'other')
        )
        
        # Aktualitäts-Score
        if dataset.get('modified'):
            modified = datetime.fromisoformat(dataset['modified'].replace('Z', '+00:00'))
            days_since_update = (datetime.now() - modified).days
            freshness_score = max(0, min(1, 1 - (days_since_update / 365)))
        else:
            freshness_score = 0.5
            
        # Vollständigkeits-Score
        completeness = sum([
            bool(dataset.get('title')),
            bool(dataset.get('description')),
            bool(dataset.get('keywords')),
            bool(dataset.get('publisher')),
            bool(dataset.get('format'))
        ]) / 5.0
        
        # Sprach-Score
        language_count = len(dataset.get('language', []))
        language_score = min(language_count / len(self.languages), 1.0)
        
        return (
            base_score * 0.4 +
            freshness_score * 0.3 +
            completeness * 0.2 +
            language_score * 0.1
        )

    def _calculate_stat_score(self, stat: Dict) -> float:
        """
        Berechnet Statistik Score basierend auf:
        - Zeitliche Abdeckung
        - Geografische Abdeckung
        - Update-Frequenz
        """
        base_score = self.type_weights['statistical']
        
        # Zeitliche Abdeckung
        time_coverage = stat.get('time_coverage', {})
        time_score = 0.0
        if time_coverage:
            try:
                start = datetime.strptime(time_coverage['start'], '%Y')
                end = datetime.strptime(time_coverage['end'], '%Y')
                years = (end - start).days / 365
                time_score = min(years / 20, 1.0)  # Normalisiere auf 20 Jahre
            except (ValueError, KeyError):
                time_score = 0.5
                
        # Geografische Abdeckung
        geo_coverage = len(stat.get('geo_coverage', []))
        geo_score = min(geo_coverage / 30, 1.0)  # Normalisiere auf 30 Länder
        
        # Update-Frequenz Score
        frequency_weights = {
            'daily': 1.0,
            'weekly': 0.9,
            'monthly': 0.8,
            'quarterly': 0.7,
            'yearly': 0.6,
            'default': 0.5
        }
        frequency_score = frequency_weights.get(
            stat.get('update_frequency', 'default').lower(),
            frequency_weights['default']
        )
        
        return (
            base_score * 0.4 +
            time_score * 0.2 +
            geo_score * 0.2 +
            frequency_score * 0.2
        ) 