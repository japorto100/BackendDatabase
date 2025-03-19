from rest_framework import serializers
from .models import Provider, SearchQuery, SearchResult, UserProviderPreference

class ProviderSerializer(serializers.ModelSerializer):
    """
    Serializer für Provider-Model
    """
    is_user_enabled = serializers.SerializerMethodField()
    user_priority = serializers.SerializerMethodField()
    user_filters = serializers.SerializerMethodField()
    is_properly_configured = serializers.BooleanField(read_only=True)
    requires_api_key = serializers.BooleanField(read_only=True)

    class Meta:
        model = Provider
        fields = [
            'id', 'name', 'provider_type', 'icon', 'description',
            'is_default', 'is_active', 'filters', 'config',
            'is_user_enabled', 'user_priority', 'user_filters',
            'is_properly_configured', 'requires_api_key'
        ]
        read_only_fields = ['created_at', 'updated_at']
        extra_kwargs = {
            'api_key': {'write_only': True}  # API-Key nicht in Responses anzeigen
        }

    def get_is_user_enabled(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            pref = UserProviderPreference.objects.filter(
                user=request.user,
                provider=obj
            ).first()
            return pref.is_enabled if pref else obj.is_default
        return obj.is_default

    def get_user_priority(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            pref = UserProviderPreference.objects.filter(
                user=request.user,
                provider=obj
            ).first()
            return pref.priority if pref else 0
        return 0

    def get_user_filters(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            pref = UserProviderPreference.objects.filter(
                user=request.user,
                provider=obj
            ).first()
            return pref.custom_filters if pref else {}
        return {}

class SearchQuerySerializer(serializers.ModelSerializer):
    """
    Serializer für Suchanfragen mit Provider-Details
    """
    providers = ProviderSerializer(many=True, read_only=True)
    result_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = SearchQuery
        fields = [
            'id', 'query', 'providers', 'filters', 'language',
            'created_at', 'response_time', 'result_count'
        ]
        read_only_fields = ['created_at', 'response_time', 'result_count']

class SearchResultSerializer(serializers.ModelSerializer):
    """
    Serializer für Suchergebnisse mit Provider-Info
    """
    provider = ProviderSerializer(read_only=True)
    
    class Meta:
        model = SearchResult
        fields = [
            'id', 'title', 'snippet', 'url', 'metadata',
            'rank', 'relevance_score', 'provider',
            'created_at', 'cache_until'
        ]
        read_only_fields = ['created_at', 'cache_until']

class UserProviderPreferenceSerializer(serializers.ModelSerializer):
    """
    Serializer für Benutzer-Provider-Einstellungen
    """
    provider_details = ProviderSerializer(source='provider', read_only=True)
    
    class Meta:
        model = UserProviderPreference
        fields = [
            'id', 'provider', 'provider_details', 'is_enabled',
            'priority', 'custom_filters', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
        
    def validate_priority(self, value):
        if value < 0:
            raise serializers.ValidationError("Priority cannot be negative")
        return value

    def validate_custom_filters(self, value):
        provider = self.instance.provider if self.instance else self.initial_data.get('provider')
        if provider and not isinstance(provider, Provider):
            provider = Provider.objects.get(id=provider)
            
        available_filters = provider.filters
        invalid_filters = set(value.keys()) - set(available_filters.keys())
        
        if invalid_filters:
            raise serializers.ValidationError(
                f"Invalid filters: {', '.join(invalid_filters)}"
            )
        return value 