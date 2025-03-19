from typing import Dict, List
import aiohttp
from datetime import datetime, timedelta
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

class YouTubeSearchProvider(BaseSearchProvider):
    """
    Provider für YouTube-Suche mit erweiterten Features:
    - Transcript Suche
    - Sentiment Analysis
    - Content Categorization
    - Quality Scoring
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = settings.YOUTUBE_API_KEY
        self.cache_duration = settings.YOUTUBE_CACHE_DURATION
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.transcript_api_url = settings.YOUTUBE_TRANSCRIPT_API_URL
        
        # Qualitäts-Metriken Gewichtung
        self.quality_weights = {
            'view_count': 0.3,
            'like_ratio': 0.2,
            'comment_count': 0.15,
            'channel_score': 0.2,
            'freshness': 0.15
        }

    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Führt YouTube-Suche mit Content-Analysis durch"""
        cache_key = f"youtube:{query}:{str(filters)}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # Basis-Suche
        async with aiohttp.ClientSession() as session:
            results = await self._search_videos(session, query, filters)
            
            if results:
                # Erweitere Ergebnisse mit zusätzlichen Daten
                enhanced_results = await self._enhance_results(session, results)
                
                # Cache Ergebnisse
                cache.set(cache_key, enhanced_results, self.cache_duration)
                return enhanced_results
            
            return []

    async def _search_videos(self, session: aiohttp.ClientSession, 
                            query: str, filters: Dict) -> List[Dict]:
        """Basis YouTube API Suche"""
        try:
            params = {
                'part': 'snippet,statistics',
                'q': query,
                'type': 'video',
                'maxResults': filters.get('limit', 20),
                'key': self.api_key,
                'relevanceLanguage': filters.get('language', 'en'),
                'videoDefinition': filters.get('quality', 'high'),
                'order': filters.get('sort', 'relevance')
            }
            
            # Duration Filter
            if 'duration' in filters:
                params['videoDuration'] = filters['duration']
            
            async with session.get(
                f"{self.base_url}/search",
                params=params
            ) as response:
                data = await response.json()
                return data.get('items', [])
                
        except Exception as e:
            print(f"YouTube search failed: {str(e)}")
            return []

    async def _enhance_results(self, session: aiohttp.ClientSession, 
                             results: List[Dict]) -> List[Dict]:
        """Erweitert Suchergebnisse mit zusätzlichen Daten"""
        enhanced = []
        
        for result in results:
            video_id = result['id']['videoId']
            
            # Parallel Requests für zusätzliche Daten
            tasks = [
                self._get_video_details(session, video_id),
                self._get_transcript(session, video_id),
                self._get_comments(session, video_id),
                self._get_channel_stats(session, result['snippet']['channelId'])
            ]
            
            video_data = await asyncio.gather(*tasks)
            
            # Kombiniere alle Daten
            enhanced_result = {
                'id': video_id,
                'title': result['snippet']['title'],
                'description': result['snippet']['description'],
                'thumbnail': result['snippet']['thumbnails']['high']['url'],
                'published_at': result['snippet']['publishedAt'],
                'channel': {
                    'id': result['snippet']['channelId'],
                    'title': result['snippet']['channelTitle'],
                    'stats': video_data[3]
                },
                'stats': video_data[0],
                'transcript': video_data[1],
                'top_comments': video_data[2],
                'quality_score': self._calculate_quality_score(result, video_data),
                'content_analysis': await self._analyze_content(
                    result['snippet']['title'],
                    result['snippet']['description'],
                    video_data[1]  # transcript
                )
            }
            
            enhanced.append(enhanced_result)
        
        # Sortiere nach Quality Score
        return sorted(enhanced, key=lambda x: x['quality_score'], reverse=True)

    def _calculate_quality_score(self, video: Dict, video_data: List) -> float:
        """
        Berechnet Quality Score basierend auf verschiedenen Metriken:
        - View Count
        - Like Ratio
        - Comment Engagement
        - Channel Authority
        - Content Freshness
        """
        stats = video_data[0]
        channel_stats = video_data[3]
        
        # View Score (0-1)
        views = int(stats.get('viewCount', 0))
        view_score = min(views / 1000000, 1.0)  # Cap at 1M views
        
        # Like Ratio (0-1)
        likes = int(stats.get('likeCount', 0))
        dislikes = int(stats.get('dislikeCount', 0))
        total_reactions = likes + dislikes
        like_ratio = likes / total_reactions if total_reactions > 0 else 0
        
        # Comment Engagement (0-1)
        comments = int(stats.get('commentCount', 0))
        comment_score = min(comments / views * 1000, 1.0) if views > 0 else 0
        
        # Channel Authority (0-1)
        subs = int(channel_stats.get('subscriberCount', 0))
        channel_score = min(subs / 1000000, 1.0)  # Cap at 1M subs
        
        # Content Freshness (0-1)
        published = datetime.fromisoformat(video['snippet']['publishedAt'].replace('Z', '+00:00'))
        age_days = (datetime.now() - published).days
        freshness = max(0, min(1, 1 - (age_days / 365)))  # Linear decay over 1 year
        
        # Weighted Average
        return sum(
            score * weight for score, weight in [
                (view_score, self.quality_weights['view_count']),
                (like_ratio, self.quality_weights['like_ratio']),
                (comment_score, self.quality_weights['comment_count']),
                (channel_score, self.quality_weights['channel_score']),
                (freshness, self.quality_weights['freshness'])
            ]
        )

    async def _analyze_content(self, title: str, description: str, 
                             transcript: str) -> Dict:
        """
        Analysiert Video-Content mit LLM für:
        - Topic Classification
        - Sentiment Analysis
        - Key Points Extraction
        - Quality Assessment
        """
        # TODO: Implement with LLM integration
        return {
            'topics': [],
            'sentiment': 'neutral',
            'key_points': [],
            'quality_metrics': {}
        } 