from typing import Dict, List
import aiohttp
from django.conf import settings
from django.core.cache import cache
import asyncio
from datetime import datetime
from .base import BaseSearchProvider

class ZefixSearchProvider(BaseSearchProvider):
    """
    Provider für Zefix (Zentraler Firmenindex) mit:
    - Firmensuche
    - Handelsregister-Einträge
    - Mutationen
    - Rechtsform-Filter
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://www.zefix.admin.ch/ZefixREST/api/v1"
        self.api_key = settings.ZEFIX_API_KEY
        self.cache_duration = settings.ZEFIX_CACHE_DURATION
        
        # Headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }
        
        # Rechtsform Mapping
        self.legal_forms = {
            'AG': 'Aktiengesellschaft',
            'GmbH': 'Gesellschaft mit beschränkter Haftung',
            'KG': 'Kommanditgesellschaft',
            'EG': 'Einzelfirma',
            'COOP': 'Genossenschaft'
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Zefix-Suche durch"""
        cache_key = f"zefix:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Basis-Suche
            companies = await self._search_companies(session, query, filters)
            
            # Hole Details für jede Firma
            tasks = [
                self._get_company_details(session, company['uid'])
                for company in companies
            ]
            
            details = await asyncio.gather(*tasks)
            
            # Kombiniere Basis-Daten mit Details
            results = self._combine_results(companies, details)
            
            # Cache Ergebnisse
            cache.set(cache_key, results, self.cache_duration)
            return results

    async def _search_companies(self, session: aiohttp.ClientSession, 
                              query: str, filters: Dict) -> List[Dict]:
        """Basis-Firmensuche"""
        try:
            params = {
                'name': query,
                'maxEntries': filters.get('limit', 20),
                'active': filters.get('active', True)
            }
            
            # Füge optionale Filter hinzu
            if 'canton' in filters:
                params['canton'] = filters['canton']
            if 'legal_form' in filters:
                params['legalForm'] = filters['legal_form']
                
            async with session.get(
                f"{self.base_url}/company/search",
                params=params
            ) as response:
                data = await response.json()
                return data.get('companies', [])
                
        except Exception as e:
            print(f"Zefix company search failed: {str(e)}")
            return []

    async def _get_company_details(self, session: aiohttp.ClientSession, 
                                 uid: str) -> Dict:
        """Holt detaillierte Firmeninformationen"""
        try:
            async with session.get(
                f"{self.base_url}/company/{uid}"
            ) as response:
                return await response.json()
        except Exception:
            return {}

    def _combine_results(self, companies: List[Dict], 
                        details: List[Dict]) -> List[Dict]:
        """Kombiniert Basis-Daten mit Details"""
        combined = []
        
        for company, detail in zip(companies, details):
            # Extrahiere relevante Informationen
            result = {
                'uid': company['uid'],
                'name': company['name'],
                'legal_form': {
                    'code': company.get('legalForm', ''),
                    'name': self.legal_forms.get(company.get('legalForm', ''), '')
                },
                'status': company.get('status', ''),
                'register': {
                    'canton': company.get('canton', ''),
                    'number': company.get('registryOfficeId', '')
                },
                'address': detail.get('address', {}),
                'purpose': detail.get('purpose', ''),
                'mutations': self._process_mutations(detail.get('mutations', [])),
                'relevance_score': self._calculate_relevance(company, detail)
            }
            
            combined.append(result)
            
        # Sortiere nach Relevanz
        return sorted(combined, key=lambda x: x['relevance_score'], reverse=True)

    def _process_mutations(self, mutations: List[Dict]) -> List[Dict]:
        """Verarbeitet Handelsregister-Mutationen"""
        processed = []
        
        for mutation in mutations:
            processed.append({
                'date': mutation.get('publicationDate'),
                'type': mutation.get('mutationType'),
                'description': mutation.get('message'),
                'publication': {
                    'date': mutation.get('publicationDate'),
                    'number': mutation.get('publicationNumber')
                }
            })
            
        return sorted(processed, key=lambda x: x['date'], reverse=True)

    def _calculate_relevance(self, company: Dict, detail: Dict) -> float:
        """
        Berechnet Relevanz-Score basierend auf:
        - Aktiv/Inaktiv Status
        - Vollständigkeit der Daten
        - Aktualität der letzten Mutation
        """
        # Status Score
        status_score = 1.0 if company.get('status') == 'ACTIVE' else 0.5
        
        # Vollständigkeits-Score
        completeness = sum([
            bool(company.get('name')),
            bool(company.get('legalForm')),
            bool(detail.get('purpose')),
            bool(detail.get('address')),
            bool(detail.get('mutations'))
        ]) / 5.0
        
        # Aktualitäts-Score
        mutations = detail.get('mutations', [])
        if mutations:
            try:
                latest = datetime.fromisoformat(mutations[0]['publicationDate'].replace('Z', '+00:00'))
                days_since_update = (datetime.now() - latest).days
                freshness_score = max(0, min(1, 1 - (days_since_update / 365)))
            except (IndexError, ValueError):
                freshness_score = 0.5
        else:
            freshness_score = 0.5
            
        return (
            status_score * 0.4 +
            completeness * 0.4 +
            freshness_score * 0.2
        ) 