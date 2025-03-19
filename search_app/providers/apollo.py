from typing import Dict, List
import aiohttp
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider
from datetime import datetime

class ApolloSearchProvider(BaseSearchProvider):
    """
    Provider für Apollo.io B2B Sales Intelligence mit:
    - Kontakt & Firmensuche
    - Lead Scoring
    - Firmendaten-Anreicherung
    - Sales Intelligence
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = settings.APOLLO_API_KEY
        self.cache_duration = settings.APOLLO_CACHE_DURATION
        self.base_url = "https://api.apollo.io/v1"
        
        # Standard Headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Scoring Gewichte
        self.score_weights = {
            'contact_accuracy': 0.3,
            'company_data': 0.3,
            'engagement': 0.2,
            'freshness': 0.2
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Apollo.io Suche durch"""
        cache_key = f"apollo:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Parallel Suche nach Kontakten und Firmen
            tasks = [
                self._search_contacts(session, query, filters),
                self._search_organizations(session, query, filters)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und analysiere Ergebnisse
            combined_results = self._process_results(results)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _search_contacts(self, session: aiohttp.ClientSession, 
                             query: str, filters: Dict) -> List[Dict]:
        """Sucht nach Kontakten"""
        try:
            params = {
                'q': query,
                'page': filters.get('page', 1),
                'per_page': filters.get('limit', 20)
            }
            
            # Füge optionale Filter hinzu
            if 'title' in filters:
                params['person_titles'] = filters['title']
            if 'company' in filters:
                params['organizations'] = filters['company']
            if 'industry' in filters:
                params['industry'] = filters['industry']
                
            async with session.post(
                f"{self.base_url}/mixed_people/search",
                json=params
            ) as response:
                data = await response.json()
                return self._process_contacts(data.get('people', []))
                
        except Exception as e:
            print(f"Apollo contact search failed: {str(e)}")
            return []

    async def _search_organizations(self, session: aiohttp.ClientSession, 
                                  query: str, filters: Dict) -> List[Dict]:
        """Sucht nach Organisationen"""
        try:
            params = {
                'q': query,
                'page': filters.get('page', 1),
                'per_page': filters.get('limit', 20)
            }
            
            if 'size' in filters:
                params['employee_count'] = filters['size']
            if 'revenue' in filters:
                params['estimated_revenue'] = filters['revenue']
                
            async with session.post(
                f"{self.base_url}/organizations/search",
                json=params
            ) as response:
                data = await response.json()
                return self._process_organizations(data.get('organizations', []))
                
        except Exception as e:
            print(f"Apollo organization search failed: {str(e)}")
            return []

    def _process_contacts(self, contacts: List[Dict]) -> List[Dict]:
        """Verarbeitet Kontakt-Ergebnisse"""
        processed = []
        
        for contact in contacts:
            processed.append({
                'type': 'contact',
                'id': contact['id'],
                'name': {
                    'first': contact.get('first_name', ''),
                    'last': contact.get('last_name', '')
                },
                'title': contact.get('title', ''),
                'email': contact.get('email', ''),
                'phone': contact.get('phone_number', ''),
                'organization': {
                    'name': contact.get('organization_name', ''),
                    'website': contact.get('organization_website', '')
                },
                'social': {
                    'linkedin': contact.get('linkedin_url', ''),
                    'twitter': contact.get('twitter_url', '')
                },
                'location': contact.get('location', {}),
                'score': self._calculate_contact_score(contact)
            })
            
        return processed

    def _process_organizations(self, organizations: List[Dict]) -> List[Dict]:
        """Verarbeitet Organisations-Ergebnisse"""
        processed = []
        
        for org in organizations:
            processed.append({
                'type': 'organization',
                'id': org['id'],
                'name': org.get('name', ''),
                'website': org.get('website', ''),
                'industry': org.get('industry', ''),
                'size': {
                    'employees': org.get('employee_count', ''),
                    'range': org.get('employee_range', '')
                },
                'revenue': {
                    'estimated': org.get('estimated_revenue', ''),
                    'range': org.get('revenue_range', '')
                },
                'social': {
                    'linkedin': org.get('linkedin_url', ''),
                    'twitter': org.get('twitter_url', ''),
                    'facebook': org.get('facebook_url', '')
                },
                'technologies': org.get('technologies', []),
                'location': org.get('location', {}),
                'score': self._calculate_organization_score(org)
            })
            
        return processed

    def _calculate_contact_score(self, contact: Dict) -> float:
        """
        Berechnet Kontakt-Score basierend auf:
        - Daten-Vollständigkeit
        - Position/Seniority
        - Kontakt-Möglichkeiten
        - Aktualität
        """
        # Vollständigkeits-Score
        completeness = sum([
            bool(contact.get('first_name')),
            bool(contact.get('last_name')),
            bool(contact.get('title')),
            bool(contact.get('email')),
            bool(contact.get('phone_number'))
        ]) / 5.0
        
        # Positions-Score
        seniority_weights = {
            'c_level': 1.0,
            'vp': 0.9,
            'director': 0.8,
            'manager': 0.7,
            'senior': 0.6,
            'default': 0.5
        }
        title = contact.get('title', '').lower()
        seniority_score = next(
            (score for level, score in seniority_weights.items() if level in title),
            seniority_weights['default']
        )
        
        # Kontakt-Score
        contact_score = sum([
            bool(contact.get('email')),
            bool(contact.get('phone_number')),
            bool(contact.get('linkedin_url'))
        ]) / 3.0
        
        # Aktualitäts-Score
        last_updated = contact.get('updated_at')
        if last_updated:
            days_since_update = (datetime.now() - datetime.fromisoformat(last_updated.replace('Z', '+00:00'))).days
            freshness_score = max(0, min(1, 1 - (days_since_update / 90)))  # Linear decay über 90 Tage
        else:
            freshness_score = 0.5
            
        return (
            completeness * self.score_weights['contact_accuracy'] +
            seniority_score * self.score_weights['company_data'] +
            contact_score * self.score_weights['engagement'] +
            freshness_score * self.score_weights['freshness']
        )

    def _calculate_organization_score(self, org: Dict) -> float:
        """
        Berechnet Organisations-Score basierend auf:
        - Daten-Vollständigkeit
        - Unternehmensgröße
        - Technologie-Stack
        - Aktualität
        """
        # Vollständigkeits-Score
        completeness = sum([
            bool(org.get('name')),
            bool(org.get('website')),
            bool(org.get('industry')),
            bool(org.get('employee_count')),
            bool(org.get('estimated_revenue'))
        ]) / 5.0
        
        # Größen-Score
        size = org.get('employee_count', 0)
        size_score = min(size / 1000, 1.0) if size else 0.5
        
        # Technologie-Score
        tech_count = len(org.get('technologies', []))
        tech_score = min(tech_count / 20, 1.0)
        
        # Aktualitäts-Score
        last_updated = org.get('updated_at')
        if last_updated:
            days_since_update = (datetime.now() - datetime.fromisoformat(last_updated.replace('Z', '+00:00'))).days
            freshness_score = max(0, min(1, 1 - (days_since_update / 90)))
        else:
            freshness_score = 0.5
            
        return (
            completeness * self.score_weights['contact_accuracy'] +
            size_score * self.score_weights['company_data'] +
            tech_score * self.score_weights['engagement'] +
            freshness_score * self.score_weights['freshness']
        ) 