from typing import Dict, List, Optional
import aiohttp
import jwt
from datetime import datetime, timedelta
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class MetabaseSearchProvider(BaseSearchProvider):
    """
    Provider für Metabase-Integration mit:
    - Dashboard Suche
    - Card/Visualization Suche
    - Query Ausführung
    - Ergebnis-Analyse
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = settings.METABASE_URL
        self.secret_key = settings.METABASE_SECRET_KEY
        self.cache_duration = settings.METABASE_CACHE_DURATION
        self.username = settings.METABASE_USERNAME
        self.password = settings.METABASE_PASSWORD
        
        # Session Token Cache
        self.SESSION_TOKEN_KEY = "metabase_session_token"
        
        # Result Type Weights
        self.type_weights = {
            'dashboard': 1.0,
            'card': 0.9,
            'table': 0.8,
            'segment': 0.7,
            'metric': 0.7
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Metabase-Suche durch"""
        cache_key = f"metabase:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # Stelle sicher, dass wir ein gültiges Session Token haben
        session_token = await self._get_session_token()
        
        async with aiohttp.ClientSession() as session:
            session.headers.update({
                'X-Metabase-Session': session_token
            })
            
            # Parallel Suche in verschiedenen Bereichen
            tasks = [
                self._search_dashboards(session, query, filters),
                self._search_cards(session, query, filters),
                self._search_tables(session, query, filters)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und analysiere Ergebnisse
            combined_results = await self._process_results(session, results, query)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _get_session_token(self) -> str:
        """Holt oder erneuert Metabase Session Token"""
        token = cache.get(self.SESSION_TOKEN_KEY)
        if token:
            return token
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/session",
                    json={
                        'username': self.username,
                        'password': self.password
                    }
                ) as response:
                    data = await response.json()
                    token = data.get('id')
                    if token:
                        # Cache Token für 23 Stunden (Token läuft nach 24h ab)
                        cache.set(self.SESSION_TOKEN_KEY, token, 60 * 60 * 23)
                        return token
                    raise Exception("Failed to get session token")
        except Exception as e:
            print(f"Metabase authentication failed: {str(e)}")
            raise

    async def _search_dashboards(self, session: aiohttp.ClientSession, 
                               query: str, filters: Dict) -> List[Dict]:
        """Sucht in Dashboards"""
        try:
            params = {'q': query}
            if 'archived' in filters:
                params['archived'] = filters['archived']
                
            async with session.get(
                f"{self.base_url}/api/dashboard",
                params=params
            ) as response:
                dashboards = await response.json()
                
                # Erweitere Dashboard-Informationen
                enhanced_dashboards = []
                for dashboard in dashboards:
                    # Hole zusätzliche Details
                    details = await self._get_dashboard_details(session, dashboard['id'])
                    
                    enhanced_dashboards.append({
                        'type': 'dashboard',
                        'id': dashboard['id'],
                        'name': dashboard['name'],
                        'description': dashboard.get('description', ''),
                        'creator': dashboard.get('creator', {}),
                        'updated_at': dashboard.get('updated_at'),
                        'collection_id': dashboard.get('collection_id'),
                        'cards': details.get('cards', []),
                        'parameters': details.get('parameters', []),
                        'relevance_score': self._calculate_dashboard_score(dashboard, details)
                    })
                    
                return enhanced_dashboards
                
        except Exception as e:
            print(f"Dashboard search failed: {str(e)}")
            return []

    async def _get_dashboard_details(self, session: aiohttp.ClientSession, 
                                   dashboard_id: int) -> Dict:
        """Holt detaillierte Dashboard-Informationen"""
        try:
            async with session.get(
                f"{self.base_url}/api/dashboard/{dashboard_id}"
            ) as response:
                return await response.json()
        except Exception:
            return {}

    async def _search_cards(self, session: aiohttp.ClientSession, 
                          query: str, filters: Dict) -> List[Dict]:
        """Sucht in Cards/Visualizations"""
        try:
            params = {'q': query}
            
            async with session.get(
                f"{self.base_url}/api/card",
                params=params
            ) as response:
                cards = await response.json()
                
                enhanced_cards = []
                for card in cards:
                    # Hole Query-Ergebnisse wenn gewünscht
                    results = None
                    if filters.get('include_results', False):
                        results = await self._execute_card_query(session, card['id'])
                    
                    enhanced_cards.append({
                        'type': 'card',
                        'id': card['id'],
                        'name': card['name'],
                        'description': card.get('description', ''),
                        'display': card.get('display'),
                        'visualization_settings': card.get('visualization_settings', {}),
                        'collection_id': card.get('collection_id'),
                        'database_id': card.get('database_id'),
                        'query_type': card.get('query_type'),
                        'results': results,
                        'relevance_score': self._calculate_card_score(card)
                    })
                    
                return enhanced_cards
                
        except Exception as e:
            print(f"Card search failed: {str(e)}")
            return []

    def _calculate_dashboard_score(self, dashboard: Dict, details: Dict) -> float:
        """
        Berechnet Dashboard Score basierend auf:
        - Anzahl der Cards
        - Aktualität
        - Parameter-Komplexität
        - Nutzung
        """
        base_score = self.type_weights['dashboard']
        
        # Cards Score
        cards_count = len(details.get('cards', []))
        cards_score = min(cards_count / 10, 1.0)
        
        # Aktualitäts-Score
        if dashboard.get('updated_at'):
            updated = datetime.fromisoformat(dashboard['updated_at'].replace('Z', '+00:00'))
            days_since_update = (datetime.now() - updated).days
            freshness_score = max(0, min(1, 1 - (days_since_update / 90)))  # Linear decay über 90 Tage
        else:
            freshness_score = 0.5
            
        # Parameter-Score
        params_count = len(details.get('parameters', []))
        params_score = min(params_count / 5, 1.0)
        
        return (
            base_score * 0.4 +
            cards_score * 0.3 +
            freshness_score * 0.2 +
            params_score * 0.1
        )

    def _calculate_card_score(self, card: Dict) -> float:
        """
        Berechnet Card Score basierend auf:
        - Visualisierungstyp
        - Query-Komplexität
        - Dokumentation
        """
        base_score = self.type_weights['card']
        
        # Visualization Score
        viz_weights = {
            'table': 0.7,
            'line': 0.8,
            'bar': 0.8,
            'pie': 0.7,
            'scatter': 0.9,
            'map': 0.9,
            'default': 0.6
        }
        viz_score = viz_weights.get(card.get('display'), viz_weights['default'])
        
        # Query Score (basierend auf Komplexität)
        query_type = card.get('query_type', '')
        query_weights = {
            'native': 1.0,    # Raw SQL
            'query': 0.8,     # Query Builder
            'default': 0.6
        }
        query_score = query_weights.get(query_type, query_weights['default'])
        
        # Dokumentations-Score
        doc_score = min(len(card.get('description', '')) / 200, 1.0)
        
        return (
            base_score * 0.4 +
            viz_score * 0.3 +
            query_score * 0.2 +
            doc_score * 0.1
        )

    async def _execute_card_query(self, session: aiohttp.ClientSession, 
                                card_id: int) -> Optional[Dict]:
        """Führt Query einer Card aus"""
        try:
            async with session.post(
                f"{self.base_url}/api/card/{card_id}/query"
            ) as response:
                return await response.json()
        except Exception:
            return None 