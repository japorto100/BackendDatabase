from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from django.db.models import Q
import json
import re
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_http_methods
import uuid
import logging
from .search_engine import SearchEngine
from models_app.rag_manager import RAGModelManager

from .models import Provider, SearchQuery, SearchResult, UserProviderPreference
from .serializers import (
    ProviderSerializer, 
    SearchQuerySerializer, 
    SearchResultSerializer,
    UserProviderPreferenceSerializer
)
from .services import SearchService
from models_app.models import UploadedFile
from chat_app.models import Message
from django.shortcuts import render

# Set up logging
logger = logging.getLogger(__name__)

# In-memory storage for deep research sessions
deep_research_sessions = {}

class ProviderViewSet(viewsets.ModelViewSet):
    """
    ViewSet für Provider-Management.
    Erlaubt CRUD-Operationen und spezielle Aktionen für Provider.
    """
    queryset = Provider.objects.all()
    serializer_class = ProviderSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['get'])
    def active(self):
        """Gibt nur aktive Provider zurück"""
        active_providers = self.queryset.filter(is_active=True)
        serializer = self.get_serializer(active_providers, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def user_preferences(self, request):
        """Gibt Provider mit Benutzereinstellungen zurück"""
        providers = Provider.objects.filter(
            Q(is_default=True) | 
            Q(user_preferences__user=request.user)
        ).distinct()
        serializer = self.get_serializer(providers, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def validate(self, request, pk=None):
        """
        Validiert die Provider-Konfiguration
        """
        provider = self.get_object()
        
        # Asynchrone Validierung starten
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(validate_provider(provider))
        loop.close()
        
        if result['valid']:
            return Response({'message': result['message']})
        else:
            return Response({'error': result['message']}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def test_search(self, request, pk=None):
        """
        Führt eine Testsuche mit dem Provider durch
        """
        provider = self.get_object()
        query = request.data.get('query', 'test')
        
        # Asynchrone Testsuche starten
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(test_provider_search(provider, query))
        loop.close()
        
        return Response({
            'query': query,
            'results_count': len(results),
            'results': results[:5]  # Nur die ersten 5 Ergebnisse zurückgeben
        })

class SearchViewSet(viewsets.ViewSet):
    """
    ViewSet für Suchoperationen.
    Handhabt verschiedene Suchtypen und Provider-Kombinationen.
    """
    permission_classes = [IsAuthenticated]

    def create(self, request):
        """
        Führt eine Suche über alle ausgewählten Provider durch.
        """
        query = request.data.get('query')
        provider_ids = request.data.get('providers', [])
        filters = request.data.get('filters', {})
        
        # Suche in Cache
        cached_results = SearchResult.objects.filter(
            query__query=query,
            provider_id__in=provider_ids,
            cache_until__gt=timezone.now()
        )
        
        if cached_results.exists():
            serializer = SearchResultSerializer(cached_results, many=True)
            return Response(serializer.data)

        # Neue Suche durchführen
        search_service = SearchService()
        results = search_service.search(
            query=query,
            user=request.user,
            provider_ids=provider_ids,
            filters=filters
        )
        
        return Response(results)

    @action(detail=False, methods=['post'])
    def universal(self, request):
        """
        Führt eine universelle Suche durch (AI-gesteuerte Provider-Auswahl)
        """
        query = request.data.get('query')
        
        search_service = SearchService()
        results = search_service.universal_search(
            query=query,
            user=request.user
        )
        
        return Response(results)

class UserProviderPreferenceViewSet(viewsets.ModelViewSet):
    """
    ViewSet für Benutzer-Provider-Einstellungen.
    """
    serializer_class = UserProviderPreferenceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserProviderPreference.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def toggle(self, request, pk=None):
        """
        Aktiviert/Deaktiviert einen Provider für den Benutzer
        """
        preference = self.get_object()
        preference.is_enabled = not preference.is_enabled
        preference.save()
        
        serializer = self.get_serializer(preference)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def set_priority(self, request, pk=None):
        """
        Setzt die Priorität eines Providers für den Benutzer
        """
        preference = self.get_object()
        preference.priority = request.data.get('priority', 0)
        preference.save()
        
        serializer = self.get_serializer(preference)
        return Response(serializer.data)

def search_view(request):
    """Render the search interface"""
    return render(request, 'search_app/search.html')

@require_POST
@csrf_exempt
def detect_provider_type(request):
    """
    Erkennt automatisch den Provider-Typ basierend auf der URL
    """
    try:
        data = json.loads(request.body)
        url = data.get('url')
        
        if not url:
            return JsonResponse({'error': 'URL is required'}, status=400)
            
        # Asynchrone Erkennung starten
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(analyze_url(url))
        loop.close()
        
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

async def analyze_url(url):
    """
    Analysiert eine URL und versucht, den Provider-Typ zu erkennen
    """
    async with aiohttp.ClientSession() as session:
        try:
            # Zuerst die Hauptseite abrufen
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return {'providerType': 'web'}
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # 1. GraphQL-Endpoint erkennen
                graphql_patterns = [
                    r'graphql',
                    r'apollo',
                    r'gql',
                    r'hasura',
                ]
                
                for pattern in graphql_patterns:
                    if re.search(pattern, html, re.IGNORECASE):
                        # GraphQL-Endpoint gefunden
                        graphql_endpoint = find_graphql_endpoint(soup, url)
                        if graphql_endpoint:
                            return {
                                'providerType': 'graphql',
                                'config': {
                                    'graphql_endpoint': graphql_endpoint
                                },
                                'apiKeyRequired': True
                            }
                
                # 2. REST API erkennen
                api_patterns = [
                    r'api',
                    r'rest',
                    r'swagger',
                    r'openapi',
                    r'documentation'
                ]
                
                for pattern in api_patterns:
                    if re.search(pattern, html, re.IGNORECASE):
                        # API-Dokumentation gefunden
                        api_base_url = find_api_base_url(soup, url)
                        if api_base_url:
                            return {
                                'providerType': 'api',
                                'config': {
                                    'api_base_url': api_base_url
                                },
                                'apiKeyRequired': True
                            }
                
                # 3. Datenbank-Verbindung erkennen
                db_patterns = [
                    r'database',
                    r'db',
                    r'sql',
                    r'nosql',
                    r'mongodb',
                    r'postgresql'
                ]
                
                for pattern in db_patterns:
                    if re.search(pattern, html, re.IGNORECASE):
                        # Datenbank-Verbindung gefunden
                        return {
                            'providerType': 'database',
                            'config': {
                                'database_type': guess_database_type(html)
                            },
                            'apiKeyRequired': True
                        }
                
                # 4. Streaming-API erkennen
                streaming_patterns = [
                    r'streaming',
                    r'websocket',
                    r'sse',
                    r'events'
                ]
                
                for pattern in streaming_patterns:
                    if re.search(pattern, html, re.IGNORECASE):
                        # Streaming-API gefunden
                        return {
                            'providerType': 'streaming',
                            'config': {
                                'streaming_type': 'websocket' if 'websocket' in html.lower() else 'sse'
                            },
                            'apiKeyRequired': True
                        }
                
                # 5. Enterprise-Systeme erkennen
                enterprise_patterns = [
                    r'ldap',
                    r'active directory',
                    r'ftp',
                    r'sftp'
                ]
                
                for pattern in enterprise_patterns:
                    if re.search(pattern, html, re.IGNORECASE):
                        # Enterprise-System gefunden
                        return {
                            'providerType': 'enterprise',
                            'config': {
                                'enterprise_type': guess_enterprise_type(html)
                            },
                            'apiKeyRequired': True
                        }
                
                # Fallback: Web Scraping
                return {
                    'providerType': 'web',
                    'config': {
                        'selectors': guess_important_selectors(soup)
                    },
                    'apiKeyRequired': False
                }
                
        except Exception as e:
            print(f"Error analyzing URL: {str(e)}")
            return {'providerType': 'web'}

def find_graphql_endpoint(soup, base_url):
    """Findet GraphQL-Endpoint in der HTML-Seite"""
    # Suche nach GraphQL-Endpoint in Scripts
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'graphql' in script.string.lower():
            endpoint_match = re.search(r'[\'"]([^\'"]*/graphql[^\'"]*)[\'"]', script.string)
            if endpoint_match:
                endpoint = endpoint_match.group(1)
                if endpoint.startswith('/'):
                    return f"{base_url.rstrip('/')}{endpoint}"
                return endpoint
    
    # Suche nach GraphQL-Endpoint in Meta-Tags
    meta_tags = soup.find_all('meta')
    for tag in meta_tags:
        if tag.get('content') and 'graphql' in tag.get('content').lower():
            return tag.get('content')
    
    return None

def find_api_base_url(soup, base_url):
    """Findet API Base URL in der HTML-Seite"""
    # Suche nach API-Links
    api_links = soup.find_all('a', href=re.compile(r'api|swagger|openapi|docs'))
    if api_links:
        href = api_links[0].get('href')
        if href.startswith('/'):
            return f"{base_url.rstrip('/')}{href}"
        return href
    
    # Suche nach API-Erwähnungen in Scripts
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'api' in script.string.lower():
            api_match = re.search(r'[\'"]([^\'"]*api[^\'"/]*)[\'"]', script.string)
            if api_match:
                api_url = api_match.group(1)
                if api_url.startswith('/'):
                    return f"{base_url.rstrip('/')}{api_url}"
                return api_url
    
    # Fallback: Standard API-Pfad
    return f"{base_url.rstrip('/')}/api"

def guess_database_type(html):
    """Rät den Datenbanktyp basierend auf HTML-Inhalt"""
    if 'mongodb' in html.lower():
        return 'mongodb'
    elif 'postgresql' in html.lower():
        return 'postgresql'
    elif 'mysql' in html.lower():
        return 'mysql'
    elif 'sql' in html.lower():
        return 'sql'
    else:
        return 'unknown'

def guess_enterprise_type(html):
    """Rät den Enterprise-System-Typ basierend auf HTML-Inhalt"""
    if 'ldap' in html.lower():
        return 'ldap'
    elif 'active directory' in html.lower():
        return 'active_directory'
    elif 'ftp' in html.lower():
        return 'ftp'
    else:
        return 'unknown'

def guess_important_selectors(soup):
    """Rät wichtige CSS-Selektoren für Web Scraping"""
    selectors = {}
    
    # Suche nach Suchergebnissen
    search_results = soup.find_all('div', class_=re.compile(r'result|search|item|card'))
    if search_results:
        selectors['results'] = f"div.{search_results[0].get('class')[0]}"
    
    # Suche nach Titeln
    titles = soup.find_all(['h1', 'h2', 'h3'], class_=re.compile(r'title|heading'))
    if titles:
        selectors['title'] = f"{titles[0].name}.{titles[0].get('class')[0]}"
    
    # Suche nach Beschreibungen
    descriptions = soup.find_all(['p', 'div'], class_=re.compile(r'desc|summary|content'))
    if descriptions:
        selectors['description'] = f"{descriptions[0].name}.{descriptions[0].get('class')[0]}"
    
    return selectors

@require_POST
@csrf_exempt
def validate_api_key(request):
    """
    Validiert einen API-Key für einen bestimmten Provider-Typ
    """
    try:
        data = json.loads(request.body)
        provider_type = data.get('provider_type')
        base_url = data.get('base_url')
        api_key = data.get('api_key')
        
        if not all([provider_type, api_key]):
            return JsonResponse({'error': 'Provider type and API key are required'}, status=400)
            
        # Asynchrone Validierung starten
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(validate_key(provider_type, base_url, api_key))
        loop.close()
        
        if result['valid']:
            return JsonResponse({'message': result['message']})
        else:
            return JsonResponse({'error': result['message']}, status=400)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

async def validate_key(provider_type, base_url, api_key):
    """
    Validiert einen API-Key für verschiedene Provider-Typen
    """
    async with aiohttp.ClientSession() as session:
        try:
            # Apollo.io API-Key validieren
            if provider_type == 'apollo':
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                
                # Einfache Anfrage an Apollo API
                async with session.get(
                    'https://api.apollo.io/v1/auth/health', 
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return {
                            'valid': True,
                            'message': 'Apollo.io API-Key ist gültig'
                        }
                    else:
                        return {
                            'valid': False,
                            'message': f'Ungültiger Apollo.io API-Key: {response.status}'
                        }
            
            # GitHub API-Key validieren
            elif provider_type == 'github':
                headers = {
                    'Authorization': f'token {api_key}',
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                async with session.get(
                    'https://api.github.com/user', 
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return {
                            'valid': True,
                            'message': 'GitHub API-Key ist gültig'
                        }
                    else:
                        return {
                            'valid': False,
                            'message': f'Ungültiger GitHub API-Key: {response.status}'
                        }
            
            # Generische REST API-Key validieren
            elif provider_type == 'api':
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Versuche, einen Endpunkt zu erreichen
                test_url = f"{base_url.rstrip('/')}/health"
                async with session.get(test_url, headers=headers) as response:
                    if response.status < 400:  # Alles unter 400 gilt als Erfolg
                        return {
                            'valid': True,
                            'message': 'API-Key scheint gültig zu sein'
                        }
                    else:
                        return {
                            'valid': False,
                            'message': f'API-Key konnte nicht validiert werden: {response.status}'
                        }
            
            # GraphQL API-Key validieren
            elif provider_type == 'graphql':
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Einfache Introspection-Anfrage
                query = """
                {
                    __schema {
                        queryType {
                            name
                        }
                    }
                }
                """
                
                async with session.post(
                    base_url,
                    json={'query': query},
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return {
                            'valid': True,
                            'message': 'GraphQL API-Key ist gültig'
                        }
                    else:
                        return {
                            'valid': False,
                            'message': f'Ungültiger GraphQL API-Key: {response.status}'
                        }
            
            # Fallback für andere Provider-Typen
            else:
                return {
                    'valid': True,
                    'message': 'API-Key wurde gespeichert (Validierung nicht verfügbar)'
                }
                
        except Exception as e:
            return {
                'valid': False,
                'message': f'Fehler bei der Validierung: {str(e)}'
            }

async def validate_provider(provider):
    """
    Validiert einen Provider
    """
    from .services import SearchService
    
    service = SearchService()
    
    try:
        # Provider-Instanz erstellen
        provider_class = service.PROVIDER_MAP.get(provider.provider_type)
        if not provider_class:
            return {
                'valid': False,
                'message': f"Unbekannter Provider-Typ: {provider.provider_type}"
            }
            
        provider_instance = provider_class(
            api_key=provider.api_key,
            base_url=provider.base_url,
            custom_headers=provider.custom_headers,
            config=provider.config
        )
        
        # Provider-spezifische Validierung
        if provider.provider_type == 'api':
            # Teste API-Verbindung
            async with aiohttp.ClientSession() as session:
                headers = dict(provider.custom_headers or {})
                if provider.api_key:
                    headers['Authorization'] = f"Bearer {provider.api_key}"
                
                test_url = f"{provider.base_url.rstrip('/')}/health"
                async with session.get(test_url, headers=headers) as response:
                    if response.status >= 400:
                        return {
                            'valid': False,
                            'message': f"API-Verbindungsfehler: {response.status}"
                        }
        
        elif provider.provider_type == 'graphql':
            # Teste GraphQL-Verbindung
            if not hasattr(provider_instance, 'gql_client') or not provider_instance.gql_client:
                return {
                    'valid': False,
                    'message': "GraphQL-Client konnte nicht initialisiert werden"
                }
        
        elif provider.provider_type == 'database':
            # Teste Datenbankverbindung
            if not hasattr(provider_instance, 'db_engine') or not provider_instance.db_engine:
                return {
                    'valid': False,
                    'message': "Datenbankverbindungsfehler: {str(e)}"
                }
            
            # Teste Verbindung
            try:
                connection = provider_instance.db_engine.connect()
                connection.close()
            except Exception as e:
                return {
                    'valid': False,
                    'message': f"Datenbankverbindungsfehler: {str(e)}"
                }
        
        # Allgemeine Validierung bestanden
        return {
            'valid': True,
            'message': f"{provider.name} wurde erfolgreich validiert"
        }
    
    except Exception as e:
        return {
            'valid': False,
            'message': f"Validierungsfehler: {str(e)}"
        }

async def test_provider_search(provider, query):
    """
    Führt eine Testsuche mit einem Provider durch
    """
    from .services import SearchService
    
    service = SearchService()
    
    try:
        # Provider-Instanz erstellen
        provider_class = service.PROVIDER_MAP.get(provider.provider_type)
        if not provider_class:
            return []
            
        provider_instance = provider_class(
            api_key=provider.api_key,
            base_url=provider.base_url,
            custom_headers=provider.custom_headers,
            config=provider.config
        )
        
        # Suche durchführen
        results = await provider_instance.search(query, {})
        return results
    
    except Exception as e:
        logger.error(f"Error in test search: {str(e)}")
        return []

@csrf_exempt
@require_http_methods(["POST"])
def search(request):
    """
    Main search endpoint
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        provider = data.get('provider', 'universal')
        filters = data.get('filters', {})
        
        if not query:
            return JsonResponse({'error': 'Query is required'}, status=400)
        
        search_engine = SearchEngine()
        results = search_engine.search(query, provider=provider, filters=filters)
        
        return JsonResponse({'results': results})
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def deep_research(request):
    """
    Initialize a deep research session
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        max_iterations = int(data.get('max_iterations', 3))
        depth = data.get('depth', 'medium')
        breadth = data.get('breadth', 'balanced')
        
        if not query:
            return JsonResponse({'error': 'Query is required'}, status=400)
        
        # Create a unique ID for this research session
        research_id = str(uuid.uuid4())
        
        # Initialize the research session
        deep_research_sessions[research_id] = {
            'query': query,
            'max_iterations': max_iterations,
            'depth': depth,
            'breadth': breadth,
            'status': 'initialized',
            'progress': 0,
            'findings': [],
            'results': []
        }
        
        # Start the research process asynchronously
        # In a production environment, this would be handled by a task queue like Celery
        # For simplicity, we'll simulate it with a background thread
        import threading
        thread = threading.Thread(target=perform_deep_research, args=(research_id,))
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'research_id': research_id,
            'status': 'initialized'
        })
    except Exception as e:
        logger.error(f"Deep research initialization error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def deep_research_status(request, research_id):
    """
    Get the status of a deep research session
    """
    try:
        if research_id not in deep_research_sessions:
            return JsonResponse({'error': 'Research session not found'}, status=404)
        
        session = deep_research_sessions[research_id]
        
        return JsonResponse({
            'status': session['status'],
            'progress': session['progress']
        })
    except Exception as e:
        logger.error(f"Deep research status error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def deep_research_results(request, research_id):
    """
    Get the results of a deep research session
    """
    try:
        if research_id not in deep_research_sessions:
            return JsonResponse({'error': 'Research session not found'}, status=404)
        
        session = deep_research_sessions[research_id]
        
        if session['status'] != 'completed':
            return JsonResponse({
                'status': session['status'],
                'progress': session['progress'],
                'message': 'Research is still in progress'
            })
        
        return JsonResponse({
            'findings': session['findings'],
            'results': session['results']
        })
    except Exception as e:
        logger.error(f"Deep research results error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def perform_deep_research(research_id):
    """
    Background task to perform deep research
    """
    session = deep_research_sessions[research_id]
    search_engine = SearchEngine()
    rag_manager = RAGModelManager()
    
    try:
        session['status'] = 'in_progress'
        
        # Get parameters
        query = session['query']
        max_iterations = session['max_iterations']
        depth = session['depth']
        breadth = session['breadth']
        
        # Convert string parameters to numeric values for the algorithm
        depth_value = {'shallow': 1, 'medium': 2, 'deep': 3}.get(depth, 2)
        breadth_value = {'focused': 1, 'balanced': 2, 'broad': 3}.get(breadth, 2)
        
        # Start deep research
        logger.info(f"Starting deep research for query: {query}")
        
        # Initial search
        session['progress'] = 10
        initial_results = search_engine.search(query)
        
        # Extract key concepts for exploration
        session['progress'] = 20
        key_concepts = search_engine.extract_key_concepts(query, initial_results)
        
        # Initialize findings
        findings = []
        total_concepts = len(key_concepts)
        
        # Explore each concept
        for i, concept in enumerate(key_concepts[:breadth_value * 2]):
            # Update progress
            concept_progress = (i / total_concepts) * 70  # 70% of progress is concept exploration
            session['progress'] = 20 + int(concept_progress)
            session['status'] = f"Exploring concept: {concept['name']}"
            
            # Perform iterative exploration of this concept
            concept_findings = search_engine.deep_research_search(
                concept['query'],
                max_iterations=depth_value,
                depth=depth_value,
                breadth=breadth_value
            )
            
            findings.append({
                'concept': concept['name'],
                'summary': concept_findings['summary'],
                'sources': concept_findings['sources']
            })
        
        # Generate comprehensive research summary
        session['progress'] = 90
        session['status'] = "Synthesizing research findings"
        
        # Format findings for the LLM
        formatted_findings = "\n\n".join([
            f"CONCEPT: {finding['concept']}\n{finding['summary']}"
            for finding in findings
        ])
        
        # Generate final summary
        research_summary = rag_manager.kg_llm_interface.generate_research_summary(
            query, formatted_findings
        )
        
        # Add the overall summary as the first finding
        findings.insert(0, {
            'concept': 'Research Summary',
            'summary': research_summary,
            'sources': []
        })
        
        # Update session with results
        session['findings'] = findings
        session['results'] = initial_results
        session['status'] = 'completed'
        session['progress'] = 100
        
        logger.info(f"Deep research completed for query: {query}")
    except Exception as e:
        logger.error(f"Deep research error: {str(e)}")
        session['status'] = 'failed'
        session['error'] = str(e)
