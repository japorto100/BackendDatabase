from typing import Dict, List
import aiohttp
from bs4 import BeautifulSoup
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider
import asyncio

class SwissfirmsSearchProvider(BaseSearchProvider):
    """
    Provider für Swissfirms-Suche mit:
    - Firmensuche
    - Branchensuche
    - Standortsuche
    - Mitarbeitergrößen
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://www.swissfirms.ch/api/v1"
        self.api_key = settings.SWISSFIRMS_API_KEY
        self.cache_duration = settings.SWISSFIRMS_CACHE_DURATION
        
        # Headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }
        
        # Branchen-Kategorien
        self.industry_categories = {
            'manufacturing': 'Industrie & Produktion',
            'trade': 'Handel',
            'services': 'Dienstleistungen',
            'construction': 'Baugewerbe',
            'it': 'Informatik & Kommunikation'
        }
        
        # Mitarbeitergrößen-Kategorien
        self.employee_sizes = {
            'micro': '1-9',
            'small': '10-49',
            'medium': '50-249',
            'large': '250+'
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Swissfirms-Suche durch"""
        cache_key = f"swissfirms:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Parallel Suche in verschiedenen Kategorien
            tasks = [
                self._search_companies(session, query, filters),
                self._search_by_industry(session, query, filters),
                self._search_by_location(session, query, filters)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und dedupliziere Ergebnisse
            combined_results = self._process_results(results)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _search_companies(self, session: aiohttp.ClientSession, 
                              query: str, filters: Dict) -> List[Dict]:
        """Basis-Firmensuche"""
        try:
            params = {
                'q': query,
                'limit': filters.get('limit', 20),
                'lang': filters.get('language', 'de')
            }
            
            # Füge Filter hinzu
            if 'canton' in filters:
                params['canton'] = filters['canton']
            if 'industry' in filters:
                params['industry'] = filters['industry']
            if 'size' in filters:
                params['employee_size'] = filters['size']
                
            async with session.get(
                f"{self.base_url}/companies/search",
                params=params
            ) as response:
                data = await response.json()
                
                # Hole Details für jede Firma
                companies = []
                for company in data.get('results', []):
                    details = await self._get_company_details(
                        session, 
                        company['id']
                    )
                    companies.append(self._combine_company_data(company, details))
                    
                return companies
                
        except Exception as e:
            print(f"Swissfirms company search failed: {str(e)}")
            return []

    async def _get_company_details(self, session: aiohttp.ClientSession, 
                                 company_id: str) -> Dict:
        """Holt detaillierte Firmeninformationen"""
        try:
            async with session.get(
                f"{self.base_url}/companies/{company_id}"
            ) as response:
                return await response.json()
        except Exception:
            return {}

    def _combine_company_data(self, basic: Dict, details: Dict) -> Dict:
        """Kombiniert Basis- und Detail-Daten"""
        return {
            'id': basic['id'],
            'name': basic.get('name', ''),
            'legal_name': details.get('legal_name', ''),
            'description': details.get('description', ''),
            'industries': [
                {
                    'code': ind.get('code', ''),
                    'name': self.industry_categories.get(
                        ind.get('code', '').lower(),
                        ind.get('name', '')
                    )
                }
                for ind in details.get('industries', [])
            ],
            'address': details.get('address', {}),
            'contact': {
                'phone': details.get('phone'),
                'email': details.get('email'),
                'website': details.get('website')
            },
            'employees': {
                'range': self.employee_sizes.get(
                    details.get('employee_size', '').lower(),
                    details.get('employee_size', '')
                ),
                'count': details.get('employee_count')
            },
            'certifications': details.get('certifications', []),
            'memberships': details.get('memberships', []),
            'social_media': details.get('social_media', {}),
            'relevance_score': self._calculate_relevance(basic, details)
        }

    def _calculate_relevance(self, basic: Dict, details: Dict) -> float:
        """
        Berechnet Relevanz-Score basierend auf:
        - Datenvollständigkeit
        - Mitarbeitergröße
        - Online-Präsenz
        - Zertifizierungen
        """
        # Vollständigkeits-Score
        completeness = sum([
            bool(details.get('description')),
            bool(details.get('industries')),
            bool(details.get('address')),
            bool(details.get('phone')),
            bool(details.get('email')),
            bool(details.get('website'))
        ]) / 6.0
        
        # Größen-Score
        size_weights = {
            'large': 1.0,
            'medium': 0.8,
            'small': 0.6,
            'micro': 0.4
        }
        size_score = size_weights.get(
            details.get('employee_size', '').lower(),
            0.5
        )
        
        # Online-Präsenz-Score
        online_presence = sum([
            bool(details.get('website')),
            bool(details.get('social_media', {}).get('linkedin')),
            bool(details.get('social_media', {}).get('twitter')),
            bool(details.get('social_media', {}).get('facebook'))
        ]) / 4.0
        
        # Zertifizierungs-Score
        cert_score = min(len(details.get('certifications', [])) / 5, 1.0)
        
        return (
            completeness * 0.4 +
            size_score * 0.3 +
            online_presence * 0.2 +
            cert_score * 0.1
        )

    async def _extract_website_info(self, session: aiohttp.ClientSession, 
                                  url: str) -> Dict:
        """Extrahiert zusätzliche Informationen von der Firmenwebsite"""
        try:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                return {
                    'meta_description': soup.find('meta', {'name': 'description'})
                                          .get('content', '') if soup.find('meta', {'name': 'description'}) else '',
                    'keywords': soup.find('meta', {'name': 'keywords'})
                                    .get('content', '') if soup.find('meta', {'name': 'keywords'}) else '',
                    'title': soup.title.string if soup.title else ''
                }
        except Exception:
            return {} 