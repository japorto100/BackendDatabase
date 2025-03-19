from django.db import models
from django.contrib.auth import get_user_model
from django.contrib.postgres.fields import JSONField
from django.utils.translation import gettext_lazy as _
import json

User = get_user_model()

class Provider(models.Model):
    """Provider model for search sources configuration"""
    PROVIDER_TYPES = [
        ('universal', 'Universal Search'),
        ('web', 'Web Search'),
        ('academic', 'Academic'),
        ('youtube', 'YouTube'),
        ('wolfram', 'Wolfram Alpha'),
        ('reddit', 'Reddit'),
        ('github', 'GitHub'),
        ('docs', 'Documentation'),
        ('local_docs', 'Local Documents'),
        ('metabase', 'Analytics'),
        ('eu_opendata', 'EU Data Portal'),
        ('apollo', 'Apollo.io'),
        ('zefix', 'Zefix'),
        ('swissfirms', 'Swissfirms'),
        ('custom', 'Custom Provider')
    ]

    name = models.CharField(max_length=100)
    provider_type = models.CharField(max_length=20, choices=PROVIDER_TYPES, default='web')
    icon = models.CharField(max_length=10, default='🔍')
    description = models.TextField(blank=True)
    is_default = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    
    # API Configuration
    base_url = models.URLField(blank=True, null=True)
    api_key = models.CharField(max_length=255, blank=True, null=True)
    custom_headers = JSONField(default=dict, blank=True)
    config = JSONField(default=dict, blank=True)
    
    # Provider specific settings
    filters = JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = _('Provider')
        verbose_name_plural = _('Providers')

    def __str__(self):
        return self.name

    @property
    def requires_api_key(self):
        """Gibt zurück, ob dieser Provider einen API-Key benötigt"""
        if self.provider_type == 'web':
            return False
        elif self.provider_type in ['api', 'graphql']:
            return True
        elif self.provider_type == 'database':
            # Prüfe, ob Datenbank-Authentifizierung benötigt wird
            conn_string = self.config.get('connection_string', '')
            return ':@' in conn_string
        elif self.provider_type == 'enterprise':
            # Enterprise-Systeme benötigen in der Regel Authentifizierung
            return True
        else:
            return False
    
    @property
    def is_properly_configured(self):
        """Prüft, ob der Provider korrekt konfiguriert ist"""
        if self.requires_api_key and not self.api_key:
            return False
            
        if self.provider_type == 'web' and not self.base_url:
            return False
            
        if self.provider_type == 'api' and not self.base_url:
            return False
            
        if self.provider_type == 'graphql' and not (self.base_url or self.config.get('graphql_endpoint')):
            return False
            
        if self.provider_type == 'database' and not (self.config.get('connection_string') or self.config.get('database_url')):
            return False
            
        if self.provider_type == 'filesystem' and not self.config.get('filesystem_path'):
            return False
            
        if self.provider_type == 'enterprise':
            enterprise_type = self.config.get('enterprise_type', '')
            if enterprise_type == 'ldap' and not self.config.get('ldap_server'):
                return False
            elif enterprise_type == 'ftp' and not self.config.get('ftp_server'):
                return False
                
        return True

class SearchQuery(models.Model):
    """Model to store search queries and their context"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='search_queries')
    query = models.TextField()
    
    # Search context
    providers = models.ManyToManyField(Provider, related_name='searches')
    filters = JSONField(default=dict)
    language = models.CharField(max_length=10, default='en')
    
    # Query metadata
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Performance tracking
    response_time = models.FloatField(null=True)
    result_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['-created_at']
        verbose_name = _('Search Query')
        verbose_name_plural = _('Search Queries')

    def __str__(self):
        return f"{self.query[:50]}... ({self.created_at})"

class SearchResult(models.Model):
    """Model to store and cache search results"""
    query = models.ForeignKey(SearchQuery, on_delete=models.CASCADE, related_name='results')
    provider = models.ForeignKey(Provider, on_delete=models.CASCADE, related_name='results')
    
    # Result data
    title = models.CharField(max_length=255)
    snippet = models.TextField()
    url = models.URLField(blank=True)
    metadata = JSONField(default=dict)
    
    # Result ranking
    rank = models.FloatField(default=0.0)
    relevance_score = models.FloatField(default=0.0)
    
    # Cache management
    created_at = models.DateTimeField(auto_now_add=True)
    cache_until = models.DateTimeField()
    
    class Meta:
        ordering = ['-rank', '-relevance_score']
        verbose_name = _('Search Result')
        verbose_name_plural = _('Search Results')
        indexes = [
            models.Index(fields=['query', 'provider', '-rank']),
            models.Index(fields=['cache_until']),
        ]

    def __str__(self):
        return f"{self.title} ({self.provider.name})"

class UserProviderPreference(models.Model):
    """Model to store user-specific provider preferences"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='provider_preferences')
    provider = models.ForeignKey(Provider, on_delete=models.CASCADE, related_name='user_preferences')
    
    is_enabled = models.BooleanField(default=True)
    priority = models.IntegerField(default=0)
    custom_filters = JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-priority']
        unique_together = ['user', 'provider']
        verbose_name = _('User Provider Preference')
        verbose_name_plural = _('User Provider Preferences')

    def __str__(self):
        return f"{self.user.username} - {self.provider.name}"
