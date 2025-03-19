import asyncio
from typing import List, Dict, Any
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
from django.core.cache import cache
from .models import Provider, SearchQuery, SearchResult
from .providers import (
    UniversalSearchProvider,
    WebSearchProvider,
    AcademicSearchProvider,
    YouTubeSearchProvider,
    WolframSearchProvider,
    RedditSearchProvider,
    GitHubSearchProvider,
    DocsSearchProvider,
    LocalDocsSearchProvider,
    MetabaseSearchProvider,
    EUDataSearchProvider,
    ApolloSearchProvider,
    ZefixSearchProvider,
    SwissfirmsSearchProvider,
    CustomSearchProvider
)

class SearchService:
    """
    Hauptservice für die Suchfunktionalität.
    Koordiniert verschiedene Provider und handhabt Caching.
    """
    
    PROVIDER_MAP = {
        'universal': UniversalSearchProvider,
        'web': WebSearchProvider,
        'academic': AcademicSearchProvider,
        'youtube': YouTubeSearchProvider,
        'wolfram': WolframSearchProvider,
        'reddit': RedditSearchProvider,
        'github': GitHubSearchProvider,
        'docs': DocsSearchProvider,
        'local_docs': LocalDocsSearchProvider,
        'metabase': MetabaseSearchProvider,
        'eu_opendata': EUDataSearchProvider,
        'apollo': ApolloSearchProvider,
        'zefix': ZefixSearchProvider,
        'swissfirms': SwissfirmsSearchProvider,
        'custom': CustomSearchProvider,
        'api': CustomSearchProvider,
        'graphql': CustomSearchProvider,
        'database': CustomSearchProvider,
        'filesystem': CustomSearchProvider,
        'streaming': CustomSearchProvider,
        'enterprise': CustomSearchProvider
    }

    def __init__(self):
        self.cache_duration = getattr(settings, 'SEARCH_CACHE_DURATION', 3600)  # 1 hour default

    async def _search_provider(self, provider: Provider, query: str, filters: Dict) -> List[Dict]:
        """
        Führt asynchrone Suche für einen einzelnen Provider durch
        """
        provider_class = self.PROVIDER_MAP.get(provider.provider_type)
        if not provider_class:
            # Fallback auf CustomSearchProvider für unbekannte Provider-Typen
            provider_class = CustomSearchProvider
            
        provider_instance = provider_class(
            api_key=provider.api_key,
            base_url=provider.base_url,
            custom_headers=provider.custom_headers,
            config=provider.config
        )

        try:
            results = await provider_instance.search(query, filters)
            return results
        except Exception as e:
            # Log error and return empty results
            print(f"Error searching {provider.name}: {str(e)}")
            return []

    def search(self, query: str, user: Any, provider_ids: List[int], filters: Dict) -> List[Dict]:
        """
        Führt Suche über mehrere Provider durch
        """
        # Create SearchQuery record
        search_query = SearchQuery.objects.create(
            user=user,
            query=query,
            filters=filters
        )
        search_query.providers.set(provider_ids)

        # Get provider instances
        providers = Provider.objects.filter(id__in=provider_ids, is_active=True)
        
        # Check cache first
        cache_key = f"search:{query}:{'-'.join(map(str, provider_ids))}"
        cached_results = cache.get(cache_key)
        if cached_results:
            return cached_results

        # Run async searches
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [
            self._search_provider(provider, query, filters.get(provider.id, {}))
            for provider in providers
        ]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()

        # Flatten and process results
        all_results = []
        for provider, provider_results in zip(providers, results):
            for result in provider_results:
                search_result = SearchResult.objects.create(
                    query=search_query,
                    provider=provider,
                    title=result['title'],
                    snippet=result['snippet'],
                    url=result.get('url', ''),
                    metadata=result.get('metadata', {}),
                    rank=result.get('rank', 0),
                    relevance_score=result.get('relevance_score', 0),
                    cache_until=timezone.now() + timedelta(seconds=self.cache_duration)
                )
                all_results.append(search_result)

        # Update SearchQuery with results
        search_query.result_count = len(all_results)
        search_query.save()

        # Cache results
        cache.set(cache_key, all_results, self.cache_duration)

        return all_results

    def universal_search(self, query: str, user: Any) -> List[Dict]:
        """
        Führt universelle Suche mit AI-gesteuerter Provider-Auswahl durch
        """
        universal_provider = UniversalSearchProvider()
        selected_providers = universal_provider.analyze_query(query)
        
        return self.search(
            query=query,
            user=user,
            provider_ids=[p.id for p in selected_providers],
            filters={}
        ) 