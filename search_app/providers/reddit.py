from typing import Dict, List
import aiohttp
import asyncio
from datetime import datetime
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class RedditSearchProvider(BaseSearchProvider):
    """
    Provider für Reddit-Suche mit erweiterten Features:
    - Subreddit-spezifische Suche
    - Sentiment Analysis
    - User Reputation Check
    - Content Quality Scoring
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_id = settings.REDDIT_CLIENT_ID
        self.client_secret = settings.REDDIT_CLIENT_SECRET
        self.user_agent = settings.REDDIT_USER_AGENT
        self.cache_duration = settings.REDDIT_CACHE_DURATION
        self.base_url = "https://oauth.reddit.com"
        
        # Content Type Weights
        self.type_weights = {
            'post': 1.0,
            'comment': 0.8,
            'wiki': 0.7
        }
        
        # Subreddit Quality Tiers
        self.subreddit_tiers = {
            'top': 1.0,      # Große, gut moderierte Subreddits
            'verified': 0.8,  # Verifizierte Communities
            'normal': 0.6,    # Standard Subreddits
            'new': 0.4       # Neue/kleine Subreddits
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt Reddit-Suche mit Content-Analysis durch"""
        cache_key = f"reddit:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # OAuth Token holen
        access_token = await self._get_access_token()
        
        async with aiohttp.ClientSession() as session:
            # Headers mit Token
            session.headers.update({
                'Authorization': f'Bearer {access_token}',
                'User-Agent': self.user_agent
            })
            
            # Parallel Suche in verschiedenen Bereichen
            tasks = [
                self._search_posts(session, query, filters),
                self._search_comments(session, query, filters),
                self._search_subreddits(session, query, filters)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Ergebnisse kombinieren und analysieren
            combined_results = await self._process_results(session, results)
            
            # Cache Ergebnisse
            cache.set(cache_key, combined_results, self.cache_duration)
            return combined_results

    async def _get_access_token(self) -> str:
        """Holt OAuth Access Token"""
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        
        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.post(
                'https://www.reddit.com/api/v1/access_token',
                data={'grant_type': 'client_credentials'}
            ) as response:
                data = await response.json()
                return data['access_token']

    async def _search_posts(self, session: aiohttp.ClientSession, 
                          query: str, filters: Dict) -> List[Dict]:
        """Sucht Reddit Posts"""
        try:
            params = {
                'q': query,
                'limit': filters.get('limit', 20),
                'sort': filters.get('sort', 'relevance'),
                'type': 'link'
            }
            
            if 'subreddit' in filters:
                params['restrict_sr'] = 'true'
                
            async with session.get(
                f"{self.base_url}/r/{filters.get('subreddit', 'all')}/search",
                params=params
            ) as response:
                data = await response.json()
                return data.get('data', {}).get('children', [])
                
        except Exception as e:
            print(f"Reddit post search failed: {str(e)}")
            return []

    async def _process_results(self, session: aiohttp.ClientSession, 
                             results: List[List[Dict]]) -> List[Dict]:
        """Verarbeitet und analysiert Suchergebnisse"""
        processed = []
        
        for result_type, items in zip(['post', 'comment', 'subreddit'], results):
            for item in items:
                data = item['data']
                
                # Basis-Metadaten
                processed_item = {
                    'type': result_type,
                    'id': data['id'],
                    'title': data.get('title', ''),
                    'text': data.get('selftext', data.get('body', '')),
                    'url': f"https://reddit.com{data['permalink']}",
                    'created_utc': datetime.fromtimestamp(data['created_utc']),
                    'score': data.get('score', 0),
                    'author': data.get('author', '[deleted]'),
                    'subreddit': data['subreddit'],
                }
                
                # Erweiterte Analyse
                processed_item.update({
                    'sentiment': await self._analyze_sentiment(processed_item['text']),
                    'quality_score': self._calculate_quality_score(processed_item),
                    'author_reputation': await self._get_author_reputation(
                        session, 
                        processed_item['author']
                    ),
                    'subreddit_tier': await self._get_subreddit_tier(
                        session, 
                        processed_item['subreddit']
                    )
                })
                
                processed.append(processed_item)
        
        # Sortiere nach Quality Score
        return sorted(processed, key=lambda x: x['quality_score'], reverse=True)

    def _calculate_quality_score(self, item: Dict) -> float:
        """
        Berechnet Quality Score basierend auf:
        - Content Type
        - Score/Karma
        - Alter
        - Subreddit Tier
        - Author Reputation
        """
        # Type Weight
        base_score = self.type_weights.get(item['type'], 0.5)
        
        # Score Weight (0-1)
        score = item.get('score', 0)
        score_weight = min(score / 1000, 1.0)  # Cap at 1000 score
        
        # Age Weight (0-1)
        age_hours = (datetime.now() - item['created_utc']).total_seconds() / 3600
        age_weight = max(0, min(1, 1 - (age_hours / (24 * 7))))  # Linear decay over 1 week
        
        # Subreddit Tier Weight
        subreddit_weight = self.subreddit_tiers.get(
            item.get('subreddit_tier', 'normal'),
            0.6
        )
        
        # Author Reputation Weight (0-1)
        reputation_weight = min(item.get('author_reputation', 0) / 10000, 1.0)
        
        # Weighted Average
        return (
            base_score * 0.2 +
            score_weight * 0.3 +
            age_weight * 0.1 +
            subreddit_weight * 0.2 +
            reputation_weight * 0.2
        )

    async def _analyze_sentiment(self, text: str) -> Dict:
        """Analysiert Sentiment des Textes"""
        # TODO: Implement with LLM/Sentiment Analysis Service
        return {
            'sentiment': 'neutral',
            'confidence': 0.0
        }

    async def _get_author_reputation(self, session: aiohttp.ClientSession, 
                                   author: str) -> float:
        """Berechnet Autor-Reputation"""
        if author == '[deleted]':
            return 0.0
            
        try:
            async with session.get(
                f"{self.base_url}/user/{author}/about"
            ) as response:
                data = await response.json()
                return data['data'].get('total_karma', 0)
        except Exception:
            return 0.0

    async def _get_subreddit_tier(self, session: aiohttp.ClientSession, 
                                 subreddit: str) -> str:
        """Bestimmt Subreddit Tier basierend auf Statistiken"""
        try:
            async with session.get(
                f"{self.base_url}/r/{subreddit}/about"
            ) as response:
                data = await response.json()
                subscribers = data['data'].get('subscribers', 0)
                
                if subscribers > 1000000:
                    return 'top'
                elif subscribers > 100000:
                    return 'verified'
                elif subscribers > 10000:
                    return 'normal'
                else:
                    return 'new'
        except Exception:
            return 'normal' 