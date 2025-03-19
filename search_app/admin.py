from django.contrib import admin
from django.utils.html import format_html
from .models import Provider, SearchQuery, SearchResult, UserProviderPreference

@admin.register(Provider)
class ProviderAdmin(admin.ModelAdmin):
    list_display = ('icon_display', 'name', 'provider_type', 'is_active', 'is_default')
    list_filter = ('provider_type', 'is_active', 'is_default')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        (None, {
            'fields': ('name', 'provider_type', 'icon', 'description')
        }),
        ('Status', {
            'fields': ('is_active', 'is_default')
        }),
        ('API Configuration', {
            'fields': ('base_url', 'api_key', 'custom_headers'),
            'classes': ('collapse',)
        }),
        ('Settings', {
            'fields': ('filters', 'config'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def icon_display(self, obj):
        return format_html('<span style="font-size: 1.5em;">{}</span>', obj.icon)
    icon_display.short_description = 'Icon'

@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ('truncated_query', 'user', 'created_at', 'result_count', 'response_time')
    list_filter = ('created_at', 'language')
    search_fields = ('query', 'user__username')
    readonly_fields = ('created_at', 'response_time', 'result_count')
    
    fieldsets = (
        (None, {
            'fields': ('user', 'query', 'providers')
        }),
        ('Search Context', {
            'fields': ('filters', 'language')
        }),
        ('Request Info', {
            'fields': ('ip_address', 'user_agent'),
            'classes': ('collapse',)
        }),
        ('Performance', {
            'fields': ('response_time', 'result_count'),
            'classes': ('collapse',)
        })
    )

    def truncated_query(self, obj):
        return f"{obj.query[:50]}..." if len(obj.query) > 50 else obj.query
    truncated_query.short_description = 'Query'

@admin.register(SearchResult)
class SearchResultAdmin(admin.ModelAdmin):
    list_display = ('title', 'provider', 'rank', 'relevance_score', 'created_at', 'cache_until')
    list_filter = ('provider', 'created_at')
    search_fields = ('title', 'snippet', 'url')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        (None, {
            'fields': ('query', 'provider', 'title', 'snippet', 'url')
        }),
        ('Ranking', {
            'fields': ('rank', 'relevance_score')
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
        ('Cache', {
            'fields': ('created_at', 'cache_until'),
            'classes': ('collapse',)
        })
    )

@admin.register(UserProviderPreference)
class UserProviderPreferenceAdmin(admin.ModelAdmin):
    list_display = ('user', 'provider', 'is_enabled', 'priority')
    list_filter = ('is_enabled', 'provider')
    search_fields = ('user__username', 'provider__name')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        (None, {
            'fields': ('user', 'provider', 'is_enabled', 'priority')
        }),
        ('Custom Settings', {
            'fields': ('custom_filters',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )