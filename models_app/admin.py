from django.contrib import admin
from django.utils.html import format_html
from django.urls import path
from django.shortcuts import render, redirect
from django import forms
from django.contrib import messages

from .models import UploadedFile, ModelConfig, Evidence, AIModel, ModelUsageStats, ElectricitySettings

# Updated imports for components in new locations
from analytics_app.electricity_cost import ElectricityCostTracker
from .ai_models.provider_factory import ProviderFactory
from .knowledge.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager

@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ('file', 'file_type', 'uploaded_at', 'processed')
    list_filter = ('file_type', 'processed', 'uploaded_at')
    search_fields = ('file',)

@admin.register(ModelConfig)
class ModelConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'is_active', 'updated_at')
    list_filter = ('model_type', 'is_active')
    search_fields = ('name',)

@admin.register(Evidence)
class EvidenceAdmin(admin.ModelAdmin):
    list_display = ('id', 'source_type', 'query_id', 'created_at')
    list_filter = ('source_type', 'created_at')
    search_fields = ('content',)

class AIModelForm(forms.ModelForm):
    class Meta:
        model = AIModel
        fields = '__all__'
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'config': forms.Textarea(attrs={'rows': 5}),
        }

@admin.register(AIModel)
class AIModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'provider', 'model_type', 'is_active', 'vision_capable', 'display_usage_count')
    list_filter = ('provider', 'model_type', 'is_active', 'vision_capable')
    search_fields = ('name', 'model_id', 'description')
    form = AIModelForm
    actions = ['activate_models', 'deactivate_models', 'test_model']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'model_id', 'provider', 'model_type', 'is_active')
        }),
        ('Capabilities', {
            'fields': ('vision_capable', 'max_tokens', 'description')
        }),
        ('Configuration', {
            'fields': ('config',),
            'classes': ('collapse',)
        }),
    )
    
    def display_usage_count(self, obj):
        try:
            stats = ModelUsageStats.objects.get(model_id=obj.model_id)
            return format_html(
                '<span title="Success: {}, Errors: {}">{}</span>',
                stats.success_count,
                stats.error_count,
                stats.total_count
            )
        except ModelUsageStats.DoesNotExist:
            return "0"
    display_usage_count.short_description = 'Usage'
    
    def activate_models(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f"{updated} models activated successfully.")
    activate_models.short_description = "Activate selected models"
    
    def deactivate_models(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f"{updated} models deactivated successfully.")
    deactivate_models.short_description = "Deactivate selected models"
    
    def test_model(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Please select exactly one model to test.", level=messages.ERROR)
            return
        
        model = queryset.first()
        
        # Updated to use ProviderFactory
        try:
            provider_config = {
                'provider': model.provider.lower(),
                'model': model.model_id,
                'temperature': 0.7,
                'max_tokens': 100
            }
            
            # Use the provider factory instead of direct imports
            provider = ProviderFactory.create_provider(provider_config)
            
            # Test the model with a simple prompt
            response, confidence = provider.generate_text("Hello, please respond with a short greeting.")
            
            # Update usage stats
            stats, created = ModelUsageStats.objects.get_or_create(model_id=model.model_id)
            stats.success_count += 1
            stats.total_count += 1
            stats.save()
            
            self.message_user(
                request, 
                f"Model test successful! Response: '{response}' (Confidence: {confidence:.2f})",
                level=messages.SUCCESS
            )
        except Exception as e:
            # Update error stats
            stats, created = ModelUsageStats.objects.get_or_create(model_id=model.model_id)
            stats.error_count += 1
            stats.total_count += 1
            stats.save()
            
            self.message_user(
                request,
                f"Model test failed: {str(e)}",
                level=messages.ERROR
            )
    test_model.short_description = "Test selected model"
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('model-dashboard/', self.admin_site.admin_view(self.model_dashboard_view), name='model-dashboard'),
        ]
        return custom_urls + urls
    
    def model_dashboard_view(self, request):
        # Get usage statistics
        stats = ModelUsageStats.objects.all().order_by('-total_count')[:10]
        
        # Get model counts by provider
        from django.db.models import Count
        provider_counts = AIModel.objects.values('provider').annotate(count=Count('id')).order_by('provider')
        
        # Get active vs. inactive counts
        active_count = AIModel.objects.filter(is_active=True).count()
        inactive_count = AIModel.objects.filter(is_active=False).count()
        
        context = {
            'title': 'AI Model Dashboard',
            'stats': stats,
            'provider_counts': provider_counts,
            'active_count': active_count,
            'inactive_count': inactive_count,
            'total_models': AIModel.objects.count(),
            'vision_capable_count': AIModel.objects.filter(vision_capable=True).count(),
        }
        
        return render(request, 'admin/model_dashboard.html', context)

@admin.register(ModelUsageStats)
class ModelUsageStatsAdmin(admin.ModelAdmin):
    list_display = ('model_id', 'total_count', 'success_count', 'error_count', 'success_rate')
    search_fields = ('model_id',)
    readonly_fields = ('model_id', 'total_count', 'success_count', 'error_count')
    
    def success_rate(self, obj):
        if obj.total_count == 0:
            return "N/A"
        rate = (obj.success_count / obj.total_count) * 100
        return f"{rate:.1f}%"
    success_rate.short_description = 'Success Rate'
    
    def has_add_permission(self, request):
        return False

class ElectricitySettingsForm(forms.ModelForm):
    class Meta:
        model = ElectricitySettings
        fields = '__all__'
    
    def clean_postal_code(self):
        postal_code = self.cleaned_data.get('postal_code')
        if postal_code and not postal_code.isdigit():
            raise forms.ValidationError("Please enter a valid Swiss postal code (numbers only)")
        return postal_code
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            # Add a button to update rate from canton defaults
            self.fields['canton'].help_text += ' <button type="button" id="update-from-canton">Use Canton Default</button>'

@admin.register(ElectricitySettings)
class ElectricitySettingsAdmin(admin.ModelAdmin):
    form = ElectricitySettingsForm
    list_display = ('user', 'kwh_rate', 'canton', 'updated_at', 'check_current_rates')
    search_fields = ('user__username', 'postal_code', 'canton')
    
    fieldsets = (
        (None, {
            'fields': ('user',)
        }),
        ('Electricity Rate', {
            'fields': ('kwh_rate', 'postal_code', 'canton'),
            'description': 'Set your electricity rate in CHF per kWh. Use the "Check Current Rates" button to find current rates.'
        }),
        ('Power Consumption Settings', {
            'fields': ('gpu_idle_watts', 'gpu_load_watts', 'cpu_idle_watts', 'cpu_load_watts'),
            'description': 'These values are used to estimate power consumption of your hardware.',
            'classes': ('collapse',),
        }),
    )
    
    def check_current_rates(self, obj):
        """Generate a link to ElCom with postal code pre-filled if available"""
        base_url = "https://strompreis.elcom.admin.ch/locality-search"
        if obj.postal_code:
            link_text = f"Check rates for {obj.postal_code}"
        else:
            link_text = "Check current rates"
        
        return format_html('<a href="{}" target="_blank">{}</a>', base_url, link_text)
    
    check_current_rates.short_description = "ElCom Rates"
    
    class Media:
        js = ('js/settings/electricity_settings.js',)
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('get-canton-rate/<str:canton>/',
                 self.admin_site.admin_view(self.get_canton_rate_view),
                 name='get-canton-rate'),
            path('electricity-dashboard/',
                 self.admin_site.admin_view(self.electricity_dashboard_view),
                 name='electricity-dashboard'),
        ]
        return custom_urls + urls
    
    def get_canton_rate_view(self, request, canton):
        """API endpoint to get the default rate for a canton"""
        """
        API endpoint to get the default electricity rate for a specified canton.
        
        Args:
            request: The HTTP request object
            canton: The canton code/name to look up the rate for
            
        Returns:
            JsonResponse with the default rate for the specified canton or an error message
        """
        from django.http import JsonResponse
        
        try:
            dummy_settings = ElectricitySettings(canton=canton)
            rate = dummy_settings.get_canton_default_rate()
            return JsonResponse({'success': True, 'rate': rate})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    def electricity_dashboard_view(self, request):
        """View for the electricity cost dashboard"""
        # Updated to use ElectricityCostTracker from analytics_app
        from analytics_app.electricity_cost import ElectricityCostTracker
        
        # Get tracker with current user's settings if available
        tracker = ElectricityCostTracker(request.user)
        
        # Get local models
        local_models = AIModel.objects.filter(provider='Local', is_active=True)
        
        # Get stats for each model
        model_stats = []
        for model in local_models:
            stats = tracker.get_model_stats(model.model_id)
            if stats:
                # Calculate forecast for typical usage
                forecast = tracker.forecast_cost(model.model_id, 1000, 100)
                model_stats.append({
                    'model': model,
                    'stats': stats,
                    'forecast': forecast
                })
        
        # Get user settings
        try:
            user_settings = ElectricitySettings.objects.get(user=request.user)
        except ElectricitySettings.DoesNotExist:
            user_settings = None
        
        context = {
            'title': 'Electricity Cost Dashboard',
            'model_stats': model_stats,
            'user_settings': user_settings,
            'kwh_rate': tracker.kwh_rate,
        }
        
        return render(request, 'admin/electricity_dashboard.html', context)
