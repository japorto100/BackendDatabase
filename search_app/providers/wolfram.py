from typing import Dict, List
import aiohttp
from django.conf import settings
from django.core.cache import cache
import asyncio
from .base import BaseSearchProvider

class WolframSearchProvider(BaseSearchProvider):
    """
    Provider für Wolfram Alpha Suche mit:
    - Short Answers
    - Full Results
    - Computational Results
    - Mathematical Notation
    - Data Analysis
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app_id = settings.WOLFRAM_APP_ID
        self.cache_duration = settings.WOLFRAM_CACHE_DURATION
        self.base_url = "https://api.wolframalpha.com/v2"
        
        # API Endpoints
        self.endpoints = {
            'short': f"{self.base_url}/result",
            'full': f"{self.base_url}/query",
            'simple': f"{self.base_url}/simple",
            'spoken': f"{self.base_url}/spoken"
        }
        
        # Result Type Weights
        self.type_weights = {
            'result': 1.0,      # Direkte Antworten
            'pod': 0.9,         # Informations-Pods
            'subpod': 0.8,      # Sub-Informationen
            'assumption': 0.7,   # Annahmen
            'warning': 0.6      # Warnungen/Hinweise
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Wolfram Alpha Suche durch"""
        cache_key = f"wolfram:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # Bestimme Suchtyp
        search_type = filters.get('type', 'full')
        
        async with aiohttp.ClientSession() as session:
            # Parallel Requests für verschiedene Antworttypen
            tasks = [
                self._get_short_answer(session, query),
                self._get_full_results(session, query),
                self._get_spoken_result(session, query)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und analysiere Ergebnisse
            combined_results = self._process_results(results, search_type)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _get_short_answer(self, session: aiohttp.ClientSession, query: str) -> Dict:
        """Holt kurze, direkte Antwort"""
        try:
            params = {
                'appid': self.app_id,
                'i': query,
                'output': 'json'
            }
            
            async with session.get(self.endpoints['short'], params=params) as response:
                return {
                    'type': 'short',
                    'result': await response.text(),
                    'success': response.status == 200
                }
                
        except Exception as e:
            print(f"Wolfram short answer failed: {str(e)}")
            return {'type': 'short', 'success': False}

    async def _get_full_results(self, session: aiohttp.ClientSession, query: str) -> Dict:
        """Holt vollständige Ergebnisse mit allen Pods"""
        try:
            params = {
                'appid': self.app_id,
                'input': query,
                'format': 'image,plaintext',
                'output': 'json'
            }
            
            async with session.get(self.endpoints['full'], params=params) as response:
                data = await response.json()
                return {
                    'type': 'full',
                    'pods': self._extract_pods(data),
                    'success': True
                }
                
        except Exception as e:
            print(f"Wolfram full results failed: {str(e)}")
            return {'type': 'full', 'success': False}

    def _extract_pods(self, data: Dict) -> List[Dict]:
        """Extrahiert und strukturiert Pods aus Wolfram Ergebnissen"""
        pods = []
        
        for pod in data.get('queryresult', {}).get('pods', []):
            processed_pod = {
                'title': pod.get('title', ''),
                'scanner': pod.get('scanner', ''),
                'position': pod.get('position', 0),
                'subpods': []
            }
            
            # Verarbeite Subpods
            for subpod in pod.get('subpods', []):
                processed_pod['subpods'].append({
                    'plaintext': subpod.get('plaintext', ''),
                    'image_url': subpod.get('img', {}).get('src', ''),
                    'title': subpod.get('title', '')
                })
            
            pods.append(processed_pod)
            
        return pods

    def _process_results(self, results: List[Dict], search_type: str) -> List[Dict]:
        """
        Verarbeitet und kombiniert verschiedene Ergebnistypen:
        - Extrahiert relevante Informationen
        - Strukturiert Daten
        - Berechnet Relevanz
        """
        processed = []
        
        # Short Answer
        short_answer = results[0]
        if short_answer['success']:
            processed.append({
                'type': 'result',
                'content': short_answer['result'],
                'format': 'text',
                'relevance': self.type_weights['result']
            })
        
        # Full Results
        full_results = results[1]
        if full_results['success']:
            for pod in full_results['pods']:
                # Verarbeite jeden Pod
                pod_result = {
                    'type': 'pod',
                    'title': pod['title'],
                    'content': [],
                    'format': 'mixed',
                    'relevance': self._calculate_pod_relevance(pod)
                }
                
                # Füge Subpod-Inhalte hinzu
                for subpod in pod['subpods']:
                    pod_result['content'].append({
                        'text': subpod['plaintext'],
                        'image': subpod['image_url'],
                        'title': subpod['title']
                    })
                
                processed.append(pod_result)
        
        # Spoken Result
        spoken_result = results[2]
        if spoken_result['success']:
            processed.append({
                'type': 'spoken',
                'content': spoken_result['result'],
                'format': 'text',
                'relevance': self.type_weights['result'] * 0.8
            })
        
        # Filtere basierend auf search_type
        if search_type != 'full':
            processed = [r for r in processed if r['type'] == search_type]
        
        # Sortiere nach Relevanz
        return sorted(processed, key=lambda x: x.get('relevance', 0), reverse=True)

    def _calculate_pod_relevance(self, pod: Dict) -> float:
        """
        Berechnet Relevanz eines Pods basierend auf:
        - Position
        - Scanner Type
        - Content Quality
        """
        base_score = self.type_weights['pod']
        
        # Position Score (frühere Pods sind relevanter)
        position_score = max(0, 1 - (pod['position'] / 10))
        
        # Scanner Type Score
        scanner_weights = {
            'Identity': 1.0,
            'Data': 0.9,
            'Math': 0.9,
            'Unit': 0.8,
            'Property': 0.8,
            'Default': 0.7
        }
        scanner_score = scanner_weights.get(pod['scanner'], scanner_weights['Default'])
        
        # Content Quality Score
        content_score = min(len(pod['subpods']) / 3, 1.0)
        
        return (
            base_score * 0.4 +
            position_score * 0.3 +
            scanner_score * 0.2 +
            content_score * 0.1
        ) 