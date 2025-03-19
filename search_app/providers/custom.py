from typing import Dict, List, Any, Optional
import aiohttp
import asyncio
import json
import re
import os
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import sqlalchemy
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from django.conf import settings
from django.core.cache import cache
from .base import BaseSearchProvider

logger = logging.getLogger(__name__)

class CustomSearchProvider(BaseSearchProvider):
    """
    Flexibler Provider für benutzerdefinierte Datenquellen.
    Unterstützt verschiedene Typen:
    
    1. Web Scraping
       - BeautifulSoup/Selenium für HTML Parsing
       - Rate Limiting & robots.txt
       - JavaScript Rendering
    
    2. REST API
       - Standard HTTP Methoden
       - API Key Authentication
       - Rate Limiting
    
    3. GraphQL API
       - Query Construction
       - Schema Validation
       - Batching/Caching
    
    4. Datenbank
       - SQL/NoSQL Queries
       - Connection Pooling
       - Query Optimization
    
    5. Lokales Dateisystem
       - File Operations
       - Indexing
       - Caching
    
    6. Streaming
       - Chunked Transfer
       - WebSocket
       - Server-Sent Events
    
    7. Enterprise Systeme
       - LDAP/Active Directory
       - FTP/SFTP
       - Custom Protokolle
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, 
                 custom_headers: Dict = None, config: Dict = None):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        self.config = config or {}
        
        # Cache-Konfiguration
        self.cache_duration = self.config.get('cache_duration', 
                                             getattr(settings, 'CUSTOM_CACHE_DURATION', 3600))
        
        # Standard Headers
        self.headers = {
            'User-Agent': 'PerplexicaSearchBot/1.0',
            'Accept': 'application/json, text/html',
            'Content-Type': 'application/json'
        }
        
        # API Key hinzufügen, falls vorhanden
        if self.api_key:
            auth_format = self.config.get('auth_format', 'bearer').lower()
            if auth_format == 'bearer':
                self.headers['Authorization'] = f'Bearer {self.api_key}'
            elif auth_format == 'token':
                self.headers['Authorization'] = f'Token {self.api_key}'
            elif auth_format == 'apikey':
                self.headers['X-API-Key'] = self.api_key
            
        # Custom Headers hinzufügen
        if self.custom_headers:
            self.headers.update(self.custom_headers)
        
        # Provider-Typ bestimmen
        self.provider_type = self._determine_provider_type()
        
        # Scoring-Gewichte
        self.weights = self.config.get('weights', {
            'relevance': 0.4,
            'freshness': 0.3,
            'quality': 0.3
        })
        
        # Initialisiere Provider-spezifische Clients
        self._init_clients()
    
    def _determine_provider_type(self) -> str:
        """Bestimmt den Provider-Typ basierend auf Konfiguration"""
        if self.config.get('provider_type'):
            return self.config.get('provider_type')
            
        # Automatische Erkennung
        if self.config.get('database_url') or self.config.get('connection_string'):
            return 'database'
        elif self.config.get('graphql_endpoint') or (self.base_url and 'graphql' in self.base_url):
            return 'graphql'
        elif self.config.get('filesystem_path'):
            return 'filesystem'
        elif self.config.get('streaming_endpoint'):
            return 'streaming'
        elif any(key in self.config for key in ['ldap_server', 'ftp_server']):
            return 'enterprise'
        elif self.api_key:
            return 'api'
        else:
            return 'web'
    
    def _init_clients(self):
        """Initialisiert Provider-spezifische Clients"""
        if self.provider_type == 'database':
            self._init_database_client()
        elif self.provider_type == 'graphql':
            self._init_graphql_client()
        elif self.provider_type == 'enterprise':
            self._init_enterprise_client()
    
    def _init_database_client(self):
        """Initialisiert Datenbankverbindung"""
        try:
            connection_string = self.config.get('connection_string') or self.config.get('database_url')
            if connection_string:
                self.db_engine = sqlalchemy.create_engine(connection_string)
                logger.info(f"Database connection initialized: {self.db_engine.name}")
            else:
                logger.warning("No database connection string provided")
                self.db_engine = None
        except Exception as e:
            logger.error(f"Error initializing database connection: {str(e)}")
            self.db_engine = None
    
    def _init_graphql_client(self):
        """Initialisiert GraphQL-Client"""
        try:
            endpoint = self.config.get('graphql_endpoint') or self.base_url
            
            # Headers für GraphQL-Anfragen
            headers = dict(self.custom_headers or {})
            if self.api_key:
                # Verschiedene Auth-Formate unterstützen
                if 'auth_format' in self.config:
                    if self.config['auth_format'] == 'bearer':
                        headers['Authorization'] = f"Bearer {self.api_key}"
                    elif self.config['auth_format'] == 'token':
                        headers['Authorization'] = f"Token {self.api_key}"
                    elif self.config['auth_format'] == 'apikey':
                        headers['X-API-Key'] = self.api_key
                else:
                    # Standard: Bearer-Token
                    headers['Authorization'] = f"Bearer {self.api_key}"
            
            transport = AIOHTTPTransport(url=endpoint, headers=headers)
            self.gql_client = Client(transport=transport, fetch_schema_from_transport=True)
            logger.info(f"GraphQL client initialized for endpoint: {endpoint}")
        except Exception as e:
            logger.error(f"Error initializing GraphQL client: {str(e)}")
            self.gql_client = None
    
    def _init_enterprise_client(self):
        """Initialisiert Enterprise-System-Clients (LDAP, FTP, etc.)"""
        enterprise_type = self.config.get('enterprise_type', '').lower()
        
        if enterprise_type == 'ldap':
            try:
                import ldap
                
                ldap_server = self.config.get('ldap_server')
                if ldap_server:
                    self.ldap_conn = ldap.initialize(ldap_server)
                    
                    # LDAP-Authentifizierung
                    if self.api_key and 'bind_dn' in self.config:
                        self.ldap_conn.simple_bind_s(self.config['bind_dn'], self.api_key)
                        logger.info(f"LDAP connection initialized: {ldap_server}")
                    else:
                        logger.warning("LDAP credentials incomplete")
            except ImportError:
                logger.error("python-ldap package not installed")
                self.ldap_conn = None
            except Exception as e:
                logger.error(f"Error initializing LDAP connection: {str(e)}")
                self.ldap_conn = None
        
        elif enterprise_type == 'ftp':
            try:
                from ftplib import FTP
                
                ftp_server = self.config.get('ftp_server')
                if ftp_server:
                    self.ftp_conn = FTP(ftp_server)
                    
                    # FTP-Authentifizierung
                    username = self.config.get('ftp_user', 'anonymous')
                    password = self.api_key or self.config.get('ftp_password', '')
                    
                    self.ftp_conn.login(username, password)
                    logger.info(f"FTP connection initialized: {ftp_server}")
            except ImportError:
                logger.error("ftplib module not available")
                self.ftp_conn = None
            except Exception as e:
                logger.error(f"Error initializing FTP connection: {str(e)}")
                self.ftp_conn = None
    
    async def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """
        Führt eine Suche basierend auf dem Provider-Typ durch
        """
        if not filters:
            filters = {}
            
        # Cache-Key generieren
        cache_key = f"custom_search:{self.provider_type}:{query}:{json.dumps(filters, sort_keys=True)}"
        
        # Cache prüfen
        cached_results = cache.get(cache_key)
        if cached_results and not filters.get('no_cache'):
            logger.info(f"Cache hit for query: {query}")
            return cached_results
            
        # Provider-spezifische Suche durchführen
        try:
            if self.provider_type == 'web':
                results = await self._web_search(query, filters)
            elif self.provider_type == 'api':
                results = await self._api_search(query, filters)
            elif self.provider_type == 'graphql':
                results = await self._graphql_search(query, filters)
            elif self.provider_type == 'database':
                results = await self._database_search(query, filters)
            elif self.provider_type == 'filesystem':
                results = await self._filesystem_search(query, filters)
            elif self.provider_type == 'streaming':
                results = await self._streaming_search(query, filters)
            elif self.provider_type == 'enterprise':
                results = await self._enterprise_search(query, filters)
            else:
                logger.error(f"Unsupported provider type: {self.provider_type}")
                return []
                
            # Ergebnisse im Cache speichern
            if results and not filters.get('no_cache'):
                cache.set(cache_key, results, self.cache_duration)
                
            return results
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []
    
    async def _web_search(self, query: str, filters: Dict) -> List[Dict]:
        """Web-Scraping-Suche"""
        if not self.base_url:
            logger.error("Base URL is required for web search")
            return []
            
        try:
            # Suchparameter
            search_url = self.config.get('search_url', f"{self.base_url}/search")
            search_params = {
                'q': query,
                'limit': filters.get('limit', 10)
            }
            
            # Zusätzliche Parameter aus Filtern
            for key, value in filters.items():
                if key not in ['limit', 'no_cache']:
                    search_params[key] = value
                    
            # Selektoren für Parsing
            selectors = self.config.get('selectors', {
                'results': '.search-results .result',
                'title': '.result-title',
                'snippet': '.result-snippet',
                'url': '.result-url'
            })
            
            # HTTP-Anfrage
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params, headers=self.headers) as response:
                    if response.status != 200:
                        logger.error(f"Web search failed with status {response.status}")
                        return []
                        
                    html = await response.text()
                    
            # HTML parsen
            soup = BeautifulSoup(html, 'html.parser')
            result_elements = soup.select(selectors['results'])
            
            results = []
            for idx, element in enumerate(result_elements):
                # Extrahiere Daten mit Selektoren
                title_element = element.select_one(selectors['title'])
                snippet_element = element.select_one(selectors['snippet'])
                url_element = element.select_one(selectors['url'])
                
                title = title_element.get_text(strip=True) if title_element else f"Result {idx+1}"
                snippet = snippet_element.get_text(strip=True) if snippet_element else ""
                
                # URL extrahieren (entweder aus Text oder href-Attribut)
                if url_element:
                    url = url_element.get('href') if url_element.get('href') else url_element.get_text(strip=True)
                else:
                    url = ""
                    
                # Vollständige URL erstellen, falls relativ
                if url and not (url.startswith('http://') or url.startswith('https://')):
                    url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
                
                # Relevanz-Score berechnen
                score = self._calculate_web_score(element)
                
                results.append({
                    'title': title,
                    'snippet': snippet,
                    'url': url,
                    'source': 'web',
                    'relevance_score': score,
                    'metadata': {
                        'position': idx + 1
                    }
                })
            
            return results
                
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []
    
    async def _api_search(self, query: str, filters: Dict) -> List[Dict]:
        """REST-API-Suche"""
        if not self.base_url:
            logger.error("Base URL is required for API search")
            return []
            
        try:
            # API-Endpunkte
            endpoints = self.config.get('endpoints', {
                'search': '/search'
            })
            
            search_endpoint = endpoints.get('search')
            if not search_endpoint:
                logger.error("Search endpoint not configured")
                return []
                
            # Vollständige URL
            search_url = f"{self.base_url.rstrip('/')}/{search_endpoint.lstrip('/')}"
            
            # Suchparameter
            search_params = {
                'query': query,
                'limit': filters.get('limit', 10)
            }
            
            # Zusätzliche Parameter aus Filtern
            for key, value in filters.items():
                if key not in ['limit', 'no_cache']:
                    search_params[key] = value
            
            # HTTP-Anfrage
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    search_url, 
                    params=search_params, 
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"API search failed with status {response.status}")
                        return []
                        
                    data = await response.json()
            
            # Ergebnisse transformieren
            results = []
            
            # Ergebnispfad in der JSON-Antwort
            results_path = self.config.get('results_path', 'results')
            
            # Extrahiere Ergebnisse aus dem JSON
            items = data
            for part in results_path.split('.'):
                if part in items:
                    items = items[part]
                else:
                    logger.error(f"Results path '{results_path}' not found in API response")
                    return []
            
            # Mapping für Felder
            field_mapping = self.config.get('field_mapping', {
                'title': 'title',
                'snippet': 'description',
                'url': 'url'
            })
            
            # Transformiere Ergebnisse
            for idx, item in enumerate(items):
                result = {
                    'title': self._extract_field(item, field_mapping.get('title', 'title')),
                    'snippet': self._extract_field(item, field_mapping.get('snippet', 'description')),
                    'url': self._extract_field(item, field_mapping.get('url', 'url')),
                    'source': 'api',
                    'relevance_score': self._calculate_api_score(item),
                    'metadata': {
                        'position': idx + 1,
                        'raw_data': item
                    }
                }
                
                results.append(result)
            
            return results
                
        except Exception as e:
            logger.error(f"API search failed: {str(e)}")
            return []
    
    def _extract_field(self, item: Dict, field_path: str) -> str:
        """Extrahiert ein Feld aus einem verschachtelten Dictionary"""
        if not field_path:
            return ""
            
        parts = field_path.split('.')
        value = item
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return ""
                
        return str(value) if value is not None else ""
    
    async def _graphql_search(self, query: str, filters: Dict) -> List[Dict]:
        """GraphQL-API-Suche"""
        if not hasattr(self, 'gql_client') or not self.gql_client:
            logger.error("GraphQL client not initialized")
            return []
            
        try:
            # GraphQL-Queries
            queries = self.config.get('queries', {})
            
            # Verwende die Suchquery oder eine Standard-Query
            search_query_name = filters.get('query_name', 'search')
            search_query_str = queries.get(search_query_name)
            
            if not search_query_str:
                logger.error(f"GraphQL query '{search_query_name}' not found in configuration")
                return []
                
            # Parse GraphQL-Query
            search_query = gql(search_query_str)
            
            # Query-Variablen
            variables = {
                'query': query,
                'limit': filters.get('limit', 10)
            }
            
            # Zusätzliche Variablen aus Filtern
            for key, value in filters.items():
                if key not in ['limit', 'no_cache', 'query_name']:
                    variables[key] = value
            
            # GraphQL-Anfrage ausführen
            result = await self.gql_client.execute_async(search_query, variable_values=variables)
            
            # Ergebnisse transformieren
            results = []
            
            # Ergebnispfad in der GraphQL-Antwort
            results_path = self.config.get('results_path', 'search.nodes')
            
            # Extrahiere Ergebnisse aus der Antwort
            items = result
            for part in results_path.split('.'):
                if part in items:
                    items = items[part]
                else:
                    logger.error(f"Results path '{results_path}' not found in GraphQL response")
                    return []
            
            # Mapping für Felder
            field_mapping = self.config.get('field_mapping', {
                'title': 'name',
                'snippet': 'description',
                'url': 'url'
            })
            
            # Transformiere Ergebnisse
            for idx, item in enumerate(items):
                result = {
                    'title': self._extract_field(item, field_mapping.get('title', 'name')),
                    'snippet': self._extract_field(item, field_mapping.get('snippet', 'description')),
                    'url': self._extract_field(item, field_mapping.get('url', 'url')),
                    'source': 'graphql',
                    'relevance_score': self._calculate_api_score(item),
                    'metadata': {
                        'position': idx + 1,
                        'raw_data': item
                    }
                }
                
                results.append(result)
            
            return results
                
        except Exception as e:
            logger.error(f"GraphQL search failed: {str(e)}")
            return []
    
    async def _database_search(self, query: str, filters: Dict) -> List[Dict]:
        """Datenbank-Suche"""
        if not hasattr(self, 'db_engine') or not self.db_engine:
            logger.error("Database engine not initialized")
            return []
            
        try:
            # SQL-Query
            sql_template = self.config.get('sql_template')
            if not sql_template:
                # Standard-SQL-Template
                table = self.config.get('table', 'items')
                search_fields = self.config.get('search_fields', ['title', 'description'])
                
                # Erstelle WHERE-Klausel für Volltextsuche
                where_clauses = []
                for field in search_fields:
                    where_clauses.append(f"{field} LIKE :search_term")
                
                where_clause = " OR ".join(where_clauses)
                
                sql_template = f"""
                SELECT * FROM {table}
                WHERE {where_clause}
                ORDER BY id DESC
                LIMIT :limit
                """
            
            # Parameter für SQL-Query
            params = {
                'search_term': f"%{query}%",
                'limit': filters.get('limit', 10)
            }
            
            # Zusätzliche Parameter aus Filtern
            for key, value in filters.items():
                if key not in ['limit', 'no_cache']:
                    params[key] = value
            
            # SQL-Query asynchron ausführen
            loop = asyncio.get_event_loop()
            connection = await loop.run_in_executor(None, self.db_engine.connect)
            
            result_proxy = await loop.run_in_executor(
                None,
                lambda: connection.execute(sql_template, params)
            )
            
            rows = await loop.run_in_executor(None, result_proxy.fetchall)
            
            # Verbindung schließen
            await loop.run_in_executor(None, connection.close)
            
            # Ergebnisse transformieren
            results = []
            
            # Mapping für Felder
            field_mapping = self.config.get('field_mapping', {
                'title': 'title',
                'snippet': 'description',
                'url': 'url'
            })
            
            # Transformiere Ergebnisse
            for idx, row in enumerate(rows):
                # Konvertiere Row zu Dictionary
                item = dict(row)
                
                result = {
                    'title': item.get(field_mapping.get('title', 'title'), f"Result {idx+1}"),
                    'snippet': item.get(field_mapping.get('snippet', 'description'), ""),
                    'url': item.get(field_mapping.get('url', 'url'), ""),
                    'source': 'database',
                    'relevance_score': 0.5,  # Standard-Score
                    'metadata': {
                        'position': idx + 1,
                        'raw_data': item
                    }
                }
                
                results.append(result)
            
            return results
                
        except Exception as e:
            logger.error(f"Database search failed: {str(e)}")
            return []
    
    async def _filesystem_search(self, query: str, filters: Dict) -> List[Dict]:
        """Dateisystem-Suche"""
        root_path = self.config.get('filesystem_path')
        if not root_path or not os.path.exists(root_path):
            logger.error(f"Invalid filesystem path: {root_path}")
            return []
            
        try:
            # Suchparameter
            max_depth = filters.get('max_depth', self.config.get('max_depth', 3))
            extensions = filters.get('extensions', self.config.get('extensions', None))
            limit = filters.get('limit', 10)
            
            # Dateisystem asynchron durchsuchen
            loop = asyncio.get_event_loop()
            matches = await loop.run_in_executor(
                None,
                lambda: self._search_files(root_path, query, max_depth, extensions, limit)
            )
            
            # Ergebnisse transformieren
            results = []
            for idx, (file_path, content, score) in enumerate(matches):
                # Relativen Pfad berechnen
                rel_path = os.path.relpath(file_path, root_path)
                
                # Dateiname extrahieren
                filename = os.path.basename(file_path)
                
                # Snippet aus Inhalt extrahieren
                snippet = self._extract_snippet(content, query)
                
                results.append({
                    'title': filename,
                    'snippet': snippet,
                    'url': f"file://{file_path}",
                    'source': 'filesystem',
                    'relevance_score': score,
                    'metadata': {
                        'path': rel_path,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    }
                })
            
            return results
                
        except Exception as e:
            logger.error(f"Filesystem search failed: {str(e)}")
            return []
    
    def _extract_snippet(self, content: str, query: str, context_chars: int = 100) -> str:
        """Extrahiert einen Snippet aus dem Inhalt um den Suchbegriff herum"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Finde Position des Suchbegriffs
        pos = content_lower.find(query_lower)
        if pos == -1:
            # Suchbegriff nicht gefunden, gib Anfang des Inhalts zurück
            return content[:200] + "..." if len(content) > 200 else content
            
        # Berechne Start- und Endposition für Snippet
        start = max(0, pos - context_chars)
        end = min(len(content), pos + len(query) + context_chars)
        
        # Extrahiere Snippet
        snippet = content[start:end]
        
        # Füge Ellipsen hinzu, wenn nötig
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
            
        return snippet
    
    def _search_files(self, root_path: str, query: str, max_depth: int, 
                      extensions: List[str] = None, limit: int = 10) -> List[tuple]:
        """Durchsucht Dateien nach einem Suchbegriff"""
        matches = []
        query_lower = query.lower()
        
        for root, dirs, files in os.walk(root_path):
            # Prüfe Suchtiefe
            depth = root[len(root_path):].count(os.sep)
            if depth > max_depth:
                continue
            
            for file in files:
                # Prüfe Dateiendung
                if extensions and not any(file.endswith(ext) for ext in extensions):
                    continue
                
                file_path = os.path.join(root, file)
                
                try:
                    # Lese Dateiinhalt
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Prüfe, ob Query im Inhalt oder Dateinamen vorkommt
                    content_lower = content.lower()
                    if query_lower in content_lower or query_lower in file.lower():
                        # Berechne Score basierend auf Häufigkeit und Position
                        score = self._calculate_file_score(file, content, query_lower)
                        matches.append((file_path, content, score))
                        
                        # Sortiere nach Score und begrenze Ergebnisse
                        matches.sort(key=lambda x: x[2], reverse=True)
                        if len(matches) > limit:
                            matches = matches[:limit]
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
        
        return matches
    
    def _calculate_file_score(self, filename: str, content: str, query: str) -> float:
        """Berechnet Relevanz-Score für eine Datei"""
        # Faktoren für Scoring
        filename_match = 3.0 if query in filename.lower() else 0.0
        content_count = content.lower().count(query)
        first_pos = content.lower().find(query) / max(1, len(content))
        
        # Gewichtete Summe
        return filename_match + min(1.0, content_count / 10) + (1.0 - first_pos)
    
    async def _streaming_search(self, query: str, filters: Dict) -> List[Dict]:
        """Streaming-API-Suche"""
        if not self.base_url:
            logger.error("Base URL is required for streaming search")
            return []
            
        try:
            # Streaming-Endpunkte
            streaming_endpoint = self.config.get('streaming_endpoint')
            if not streaming_endpoint:
                logger.error("Streaming endpoint not configured")
                return []
                
            # Vollständige URL
            stream_url = f"{self.base_url.rstrip('/')}/{streaming_endpoint.lstrip('/')}"
            
            # Suchparameter
            search_params = {
                'query': query,
                'limit': filters.get('limit', 10)
            }
            
            # Zusätzliche Parameter aus Filtern
            for key, value in filters.items():
                if key not in ['limit', 'no_cache']:
                    search_params[key] = value
            
            # Timeout für Streaming
            timeout = filters.get('timeout', self.config.get('timeout', 30))
            
            # Ergebnisse
            results = []
            
            # Streaming-Anfrage mit aiohttp
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        stream_url,
                        params=search_params,
                        headers=self.headers,
                        timeout=timeout
                    ) as response:
                        if response.status != 200:
                            logger.error(f"Streaming request failed: {response.status}")
                            return []
                        
                        # Verarbeite Streaming-Antwort
                        async for line in response.content:
                            line = line.strip()
                            if not line:
                                continue
                                
                            try:
                                # Versuche, Zeile als JSON zu parsen
                                data = json.loads(line)
                                
                                # Transformiere Ergebnis
                                result = self._transform_streaming_result(data, query)
                                if result:
                                    results.append(result)
                                    
                                    # Prüfe Limit
                                    if len(results) >= filters.get('limit', 10):
                                        break
                            except json.JSONDecodeError:
                                # Keine gültige JSON-Zeile
                                continue
                except asyncio.TimeoutError:
                    logger.warning(f"Streaming request timed out after {timeout} seconds")
                except Exception as e:
                    logger.error(f"Error in streaming search: {str(e)}")
            
            return results
                
        except Exception as e:
            logger.error(f"Streaming search failed: {str(e)}")
            return []
    
    def _transform_streaming_result(self, data: Dict, query: str) -> Dict:
        """Transformiert ein Streaming-Ergebnis in das Standardformat"""
        try:
            # Extrahiere Felder basierend auf Konfiguration
            title_field = self.config.get('title_field', 'title')
            content_field = self.config.get('content_field', 'content')
            url_field = self.config.get('url_field', 'url')
            score_field = self.config.get('score_field', 'score')
            
            # Extrahiere Werte
            title = self._get_nested_value(data, title_field) or "Untitled"
            content = self._get_nested_value(data, content_field) or ""
            url = self._get_nested_value(data, url_field) or ""
            score = float(self._get_nested_value(data, score_field) or 0.5)
            
            # Erstelle Ergebnis
            return {
                'title': title,
                'snippet': content[:200] + "..." if len(content) > 200 else content,
                'url': url,
                'source': 'streaming',
                'relevance_score': score,
                'metadata': data
            }
        except Exception as e:
            logger.error(f"Error transforming streaming result: {str(e)}")
            return None
    
    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Holt einen verschachtelten Wert aus einem Dictionary"""
        if not field_path:
            return None
            
        parts = field_path.split('.')
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
                
        return value
    
    async def _enterprise_search(self, query: str, filters: Dict) -> List[Dict]:
        """Enterprise-System-Suche"""
        enterprise_type = self.config.get('enterprise_type', '').lower()
        
        if enterprise_type == 'ldap':
            return await self._ldap_search(query, filters)
        elif enterprise_type == 'ftp':
            return await self._ftp_search(query, filters)
        else:
            logger.error(f"Unsupported enterprise type: {enterprise_type}")
            return []
    
    async def _ldap_search(self, query: str, filters: Dict) -> List[Dict]:
        """LDAP-Verzeichnissuche"""
        if not hasattr(self, 'ldap_conn') or not self.ldap_conn:
            logger.error("LDAP connection not initialized")
            return []
        
        try:
            import ldap
            
            # Hole LDAP-Suchparameter
            base_dn = self.config.get('base_dn', '')
            search_filter = self.config.get('search_filter', '(cn=*{query}*)')
            attributes = self.config.get('attributes', ['cn', 'mail', 'description'])
            
            # Ersetze Query-Platzhalter
            search_filter = search_filter.replace('{query}', query)
            
            # Führe LDAP-Suche asynchron aus
            loop = asyncio.get_event_loop()
            ldap_results = await loop.run_in_executor(
                None,
                lambda: self.ldap_conn.search_s(base_dn, ldap.SCOPE_SUBTREE, search_filter, attributes)
            )
            
            # Transformiere Ergebnisse
            results = []
            for idx, (dn, attrs) in enumerate(ldap_results):
                # Konvertiere binäre Werte zu Strings
                decoded_attrs = {}
                for key, values in attrs.items():
                    decoded_values = []
                    for value in values:
                        if isinstance(value, bytes):
                            try:
                                decoded_values.append(value.decode('utf-8'))
                            except UnicodeDecodeError:
                                decoded_values.append(f"<binary data: {len(value)} bytes>")
                        else:
                            decoded_values.append(value)
                    decoded_attrs[key] = decoded_values
                
                # Extrahiere Titel und Beschreibung
                cn = decoded_attrs.get('cn', ['Unknown'])[0]
                mail = decoded_attrs.get('mail', [''])[0]
                description = decoded_attrs.get('description', [''])[0]
                
                results.append({
                    'title': cn,
                    'snippet': description,
                    'url': f"ldap://{dn}",
                    'source': 'ldap',
                    'relevance_score': 0.5,  # Standard-Score
                    'metadata': {
                        'dn': dn,
                        'mail': mail,
                        'attributes': decoded_attrs
                    }
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in LDAP search: {str(e)}")
            return []
    
    async def _ftp_search(self, query: str, filters: Dict) -> List[Dict]:
        """FTP-Dateisuche"""
        if not hasattr(self, 'ftp_conn') or not self.ftp_conn:
            logger.error("FTP connection not initialized")
            return []
        
        try:
            # Hole FTP-Suchparameter
            directory = filters.get('directory', self.config.get('directory', '/'))
            max_depth = filters.get('max_depth', self.config.get('max_depth', 2))
            
            # Führe FTP-Suche asynchron aus
            loop = asyncio.get_event_loop()
            matches = await loop.run_in_executor(
                None,
                lambda: self._search_ftp_files(directory, query, max_depth)
            )
            
            # Transformiere Ergebnisse
            results = []
            for idx, (path, info) in enumerate(matches):
                # Extrahiere Dateinamen
                filename = path.split('/')[-1]
                
                results.append({
                    'title': filename,
                    'snippet': f"FTP file: {path}",
                    'url': f"ftp://{self.config.get('ftp_server')}{path}",
                    'source': 'ftp',
                    'relevance_score': 0.5,  # Standard-Score
                    'metadata': {
                        'path': path,
                        'size': info.get('size', 0),
                        'modified': info.get('modify', '')
                    }
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in FTP search: {str(e)}")
            return []
    
    def _search_ftp_files(self, directory: str, query: str, max_depth: int) -> List[tuple]:
        """Durchsucht FTP-Verzeichnis nach Dateien, die dem Suchbegriff entsprechen"""
        matches = []
        query_lower = query.lower()
        
        def search_dir(dir_path, current_depth):
            if current_depth > max_depth:
                return
            
            try:
                # Wechsle ins Verzeichnis
                self.ftp_conn.cwd(dir_path)
                
                # Liste Dateien und Verzeichnisse
                file_list = []
                self.ftp_conn.dir(lambda line: file_list.append(line))
                
                for line in file_list:
                    # Parse FTP-Listeneintrag
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    
                    permissions = parts[0]
                    size = int(parts[4])
                    month, day, year_or_time = parts[5:8]
                    filename = ' '.join(parts[8:])
                    
                    # Prüfe, ob es sich um ein Verzeichnis handelt
                    is_dir = permissions.startswith('d')
                    
                    # Vollständiger Pfad
                    full_path = f"{dir_path}/{filename}"
                    
                    if is_dir:
                        # Rekursiv ins Unterverzeichnis
                        search_dir(full_path, current_depth + 1)
                    else:
                        # Prüfe, ob Dateiname dem Suchbegriff entspricht
                        if query_lower in filename.lower():
                            matches.append((full_path, {
                                'size': size,
                                'modify': f"{month} {day} {year_or_time}"
                            }))
            except Exception as e:
                logger.error(f"Error searching FTP directory {dir_path}: {str(e)}")
        
        # Starte Suche im Hauptverzeichnis
        search_dir(directory, 0)
        return matches
    
    def _calculate_web_score(self, item_soup: BeautifulSoup) -> float:
        """
        Scoring für Web-Ergebnisse.
        
        Beispiel-Metriken:
        1. Text-Länge
        2. Keyword-Dichte
        3. HTML-Qualität
        """
        # Beispiel-Implementation
        relevance = len(item_soup.find_all('mark', class_='highlight'))
        freshness = 'new' in item_soup.get('class', [])
        quality = len(item_soup.text.strip()) / 1000  # Normalisiert auf 1000 Zeichen
        
        return (
            relevance * self.weights['relevance'] +
            freshness * self.weights['freshness'] +
            quality * self.weights['quality']
        )

    def _calculate_api_score(self, item: Dict) -> float:
        """
        Scoring für API-Ergebnisse.
        
        Beispiel-Metriken:
        1. API-spezifische Scores
        2. Daten-Qualität
        3. Business-Relevanz
        """
        # Beispiel-Implementation
        relevance = item.get('relevance_score', 0.5)
        freshness = self._calculate_freshness(item.get('updated_at'))
        quality = len(item.get('metadata', {})) / 10  # Normalisiert auf 10 Metadaten-Felder
        
        return (
            relevance * self.weights['relevance'] +
            freshness * self.weights['freshness'] +
            quality * self.weights['quality']
        )

    def _calculate_freshness(self, timestamp: str) -> float:
        """Berechnet Aktualitäts-Score"""
        if not timestamp:
            return 0.5
            
        try:
            updated = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            days_old = (datetime.now() - updated).days
            return max(0, min(1, 1 - (days_old / 365)))  # Linear decay über 1 Jahr
        except (ValueError, TypeError):
            return 0.5