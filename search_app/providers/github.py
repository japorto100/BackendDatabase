from typing import Dict, List
import aiohttp
import base64
import asyncio
from datetime import datetime
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class GitHubSearchProvider(BaseSearchProvider):
    """
    Provider für GitHub-Suche mit Code-Analyse Features:
    - Repository Search
    - Code Search
    - Issue/PR Search
    - User/Org Search
    - Dependency Analysis
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = settings.GITHUB_API_KEY
        self.cache_duration = settings.GITHUB_CACHE_DURATION
        self.base_url = "https://api.github.com"
        
        # Search Type Weights
        self.type_weights = {
            'repository': 1.0,
            'code': 0.9,
            'issue': 0.8,
            'user': 0.7
        }
        
        # Default Headers
        self.headers = {
            'Authorization': f'token {self.api_key}',
            'Accept': 'application/vnd.github.v3+json'
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Multi-Type GitHub Suche durch"""
        cache_key = f"github:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        search_types = filters.get('types', ['repository', 'code', 'issue'])
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = []
            
            # Erstelle Tasks basierend auf gewünschten Suchtypen
            if 'repository' in search_types:
                tasks.append(self._search_repositories(session, query, filters))
            if 'code' in search_types:
                tasks.append(self._search_code(session, query, filters))
            if 'issue' in search_types:
                tasks.append(self._search_issues(session, query, filters))
            if 'user' in search_types:
                tasks.append(self._search_users(session, query, filters))
            
            # Führe Suchen parallel aus
            results = await asyncio.gather(*tasks)
            
            # Kombiniere und ranke Ergebnisse
            combined_results = self._combine_results(results, search_types)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _search_repositories(self, session: aiohttp.ClientSession, 
                                 query: str, filters: Dict) -> List[Dict]:
        """Sucht Repositories mit erweiterten Filtern"""
        try:
            # Baue Query mit GitHub-spezifischer Syntax
            q_parts = [query]
            
            if 'language' in filters:
                q_parts.append(f"language:{filters['language']}")
            if 'stars' in filters:
                q_parts.append(f"stars:>={filters['stars']}")
            if 'last_updated' in filters:
                q_parts.append(f"pushed:>={filters['last_updated']}")
                
            params = {
                'q': ' '.join(q_parts),
                'sort': filters.get('sort', 'stars'),
                'order': filters.get('order', 'desc'),
                'per_page': filters.get('limit', 20)
            }
            
            async with session.get(
                f"{self.base_url}/search/repositories",
                params=params
            ) as response:
                data = await response.json()
                
                # Erweitere Repository-Daten
                enhanced_repos = []
                for repo in data.get('items', []):
                    # Hole zusätzliche Repo-Details parallel
                    tasks = [
                        self._get_repo_stats(session, repo['full_name']),
                        self._get_repo_languages(session, repo['full_name']),
                        self._get_repo_topics(session, repo['full_name'])
                    ]
                    
                    repo_data = await asyncio.gather(*tasks)
                    
                    enhanced_repos.append({
                        'type': 'repository',
                        'id': repo['id'],
                        'name': repo['full_name'],
                        'url': repo['html_url'],
                        'description': repo['description'],
                        'stars': repo['stargazers_count'],
                        'forks': repo['forks_count'],
                        'last_updated': repo['updated_at'],
                        'language': repo['language'],
                        'stats': repo_data[0],
                        'languages': repo_data[1],
                        'topics': repo_data[2],
                        'relevance_score': self._calculate_repo_score(repo)
                    })
                    
                return enhanced_repos
                
        except Exception as e:
            print(f"GitHub repository search failed: {str(e)}")
            return []

    async def _search_code(self, session: aiohttp.ClientSession, 
                          query: str, filters: Dict) -> List[Dict]:
        """Sucht Code mit Kontext-Analyse"""
        try:
            params = {
                'q': query,
                'per_page': filters.get('limit', 20)
            }
            
            if 'language' in filters:
                params['q'] += f" language:{filters['language']}"
            if 'repo' in filters:
                params['q'] += f" repo:{filters['repo']}"
                
            async with session.get(
                f"{self.base_url}/search/code",
                params=params
            ) as response:
                data = await response.json()
                
                enhanced_code = []
                for item in data.get('items', []):
                    # Hole Dateiinhalt und analysiere
                    content = await self._get_file_content(session, item['url'])
                    context = self._extract_code_context(content, query)
                    
                    enhanced_code.append({
                        'type': 'code',
                        'file_name': item['name'],
                        'repo_name': item['repository']['full_name'],
                        'path': item['path'],
                        'url': item['html_url'],
                        'content': content,
                        'context': context,
                        'language': item['language'],
                        'relevance_score': self._calculate_code_score(item, context)
                    })
                    
                return enhanced_code
                
        except Exception as e:
            print(f"GitHub code search failed: {str(e)}")
            return []

    def _calculate_repo_score(self, repo: Dict) -> float:
        """
        Berechnet Repository Score basierend auf:
        - Stars
        - Forks
        - Recent Activity
        - Description Quality
        """
        # Stars Score (0-1)
        stars_score = min(repo['stargazers_count'] / 10000, 1.0)
        
        # Forks Score (0-1)
        forks_score = min(repo['forks_count'] / 1000, 1.0)
        
        # Activity Score (0-1)
        last_update = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
        days_since_update = (datetime.now() - last_update).days
        activity_score = max(0, min(1, 1 - (days_since_update / 365)))
        
        # Description Score (0-1)
        desc_score = min(len(repo.get('description', '')) / 200, 1.0)
        
        # Weighted Average
        return (
            stars_score * 0.4 +
            forks_score * 0.3 +
            activity_score * 0.2 +
            desc_score * 0.1
        )

    async def _get_file_content(self, session: aiohttp.ClientSession, url: str) -> str:
        """Holt und dekodiert Dateiinhalt"""
        try:
            async with session.get(url) as response:
                data = await response.json()
                if data.get('encoding') == 'base64':
                    return base64.b64decode(data['content']).decode('utf-8')
                return data.get('content', '')
        except Exception:
            return ''

    def _extract_code_context(self, content: str, query: str) -> Dict:
        """
        Extrahiert relevanten Code-Kontext:
        - Umgebende Funktionen/Klassen
        - Relevante Code-Blöcke
        - Dokumentation/Kommentare
        """
        # TODO: Implement sophisticated code analysis
        return {
            'surrounding_code': '',
            'documentation': '',
            'matched_lines': []
        } 