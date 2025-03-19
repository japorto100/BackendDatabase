from django.db import models
import uuid
import json
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator

class UploadedFile(models.Model):
    FILE_TYPES = [
        ('document', 'Document'),
        ('image', 'Image'),
        ('video', 'Video'),
        ('audio', 'Audio'),
    ]

    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    file_type = models.CharField(max_length=10, choices=FILE_TYPES)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processing_results = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.file_type}: {self.file.name}"

class ModelConfig(models.Model):
    name = models.CharField(max_length=255)
    model_type = models.CharField(max_length=255)
    config = models.JSONField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Evidence(models.Model):
    """
    Stores evidence used by AI to generate responses, with source attribution.
    """
    class Meta:
        app_label = 'models_app'
        
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query_id = models.UUIDField(help_text="ID of the chat message or search query")
    source_type = models.CharField(max_length=50, help_text="Type of source (file, generated, processing, etc.)")
    content = models.TextField(help_text="The content of the evidence")
    highlights = models.JSONField(help_text="JSON array of highlighted sections with position and confidence")
    created_at = models.DateTimeField(auto_now_add=True)
    is_mention = models.BooleanField(default=False, help_text="Whether this evidence came from an @-mention")
    
    def __str__(self):
        return f"Evidence {self.id} for {self.source_type}"

class AIModel(models.Model):
    """
    Represents an AI model configuration
    """
    PROVIDER_CHOICES = [
        ('OpenAI', 'OpenAI'),
        ('Anthropic', 'Anthropic'),
        ('DeepSeek', 'DeepSeek'),
        ('Local', 'Local'),
        ('Other', 'Other'),
    ]
    
    MODEL_TYPE_CHOICES = [
        ('chat', 'Chat'),
        ('vision', 'Vision'),
        ('embedding', 'Embedding'),
        ('code', 'Code'),
    ]
    
    name = models.CharField(max_length=100)
    model_id = models.CharField(max_length=100, unique=True)
    provider = models.CharField(max_length=50, choices=PROVIDER_CHOICES)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES, default='chat')
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    vision_capable = models.BooleanField(default=False)
    max_tokens = models.IntegerField(default=2048)
    config = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.provider})"
    
    class Meta:
        ordering = ['provider', 'name']
        verbose_name = 'AI Model'
        verbose_name_plural = 'AI Models'


class ModelUsageStats(models.Model):
    """
    Tracks usage statistics for AI models
    """
    model_id = models.CharField(max_length=100, unique=True)
    total_count = models.IntegerField(default=0)
    success_count = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)
    
    last_used = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Stats for {self.model_id}"
    
    class Meta:
        ordering = ['-total_count']
        verbose_name = 'Model Usage Stats'
        verbose_name_plural = 'Model Usage Stats'

class ElectricitySettings(models.Model):
    """Settings for electricity cost calculations"""
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='electricity_settings')
    
    # Basic rate settings
    kwh_rate = models.DecimalField(
        max_digits=6, 
        decimal_places=4,
        default=0.2400,
        validators=[MinValueValidator(0.05), MaxValueValidator(1.0)],
        help_text="Price per kWh in CHF"
    )
    
    # Location for default suggestions
    postal_code = models.CharField(max_length=10, blank=True, help_text="Swiss postal code for rate suggestions")
    canton = models.CharField(
        max_length=2, 
        blank=True, 
        choices=[
            ('AG', 'Aargau'),
            ('AI', 'Appenzell Innerrhoden'),
            ('AR', 'Appenzell Ausserrhoden'),
            ('BE', 'Bern'),
            ('BL', 'Basel-Landschaft'),
            ('BS', 'Basel-Stadt'),
            ('FR', 'Fribourg'),
            ('GE', 'Geneva'),
            ('GL', 'Glarus'),
            ('GR', 'Graubünden'),
            ('JU', 'Jura'),
            ('LU', 'Lucerne'),
            ('NE', 'Neuchâtel'),
            ('NW', 'Nidwalden'),
            ('OW', 'Obwalden'),
            ('SG', 'St. Gallen'),
            ('SH', 'Schaffhausen'),
            ('SO', 'Solothurn'),
            ('SZ', 'Schwyz'),
            ('TG', 'Thurgau'),
            ('TI', 'Ticino'),
            ('UR', 'Uri'),
            ('VD', 'Vaud'),
            ('VS', 'Valais'),
            ('ZG', 'Zug'),
            ('ZH', 'Zurich'),
        ],
        help_text="Canton code (e.g., ZH, BE)"
    )
    
    # Hardware power consumption estimates
    gpu_idle_watts = models.IntegerField(default=15, help_text="GPU power consumption when idle (watts)")
    gpu_load_watts = models.IntegerField(default=200, help_text="GPU power consumption at full load (watts)")
    cpu_idle_watts = models.IntegerField(default=10, help_text="CPU power per core when idle (watts)")
    cpu_load_watts = models.IntegerField(default=35, help_text="CPU power per core at full load (watts)")
    
    # Last updated timestamp
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Electricity Settings for {self.user.username}"
    
    def get_canton_default_rate(self):
        """Get the default electricity rate for the selected canton"""
        canton_rates = {
            'ZH': 0.2547,  # Zurich
            'BE': 0.2312,  # Bern
            'LU': 0.2219,  # Lucerne
            'UR': 0.1998,  # Uri
            'SZ': 0.2076,  # Schwyz
            'OW': 0.2115,  # Obwalden
            'NW': 0.2087,  # Nidwalden
            'GL': 0.2265,  # Glarus
            'ZG': 0.1847,  # Zug
            'FR': 0.2108,  # Fribourg
            'SO': 0.2298,  # Solothurn
            'BS': 0.2387,  # Basel-Stadt
            'BL': 0.2312,  # Basel-Landschaft
            'SH': 0.2454,  # Schaffhausen
            'AR': 0.2287,  # Appenzell Ausserrhoden
            'AI': 0.2189,  # Appenzell Innerrhoden
            'SG': 0.2356,  # St. Gallen
            'GR': 0.2245,  # Graubünden
            'AG': 0.2267,  # Aargau
            'TG': 0.2198,  # Thurgau
            'TI': 0.2376,  # Ticino
            'VD': 0.2298,  # Vaud
            'VS': 0.2098,  # Valais
            'NE': 0.2387,  # Neuchâtel
            'GE': 0.2543,  # Geneva
            'JU': 0.2265,  # Jura
        }
        
        return canton_rates.get(self.canton, 0.2400)  # Default to average if canton not found
    
    class Meta:
        verbose_name = "Electricity Settings"
        verbose_name_plural = "Electricity Settings"
