from django.db import models
from django.contrib.auth.models import User

class AnalyticsEvent(models.Model):
    EVENT_TYPES = [
        ('api_request', 'API Request'),
        ('chat', 'Chat'),
        ('search', 'Search'),
        ('file_upload', 'File Upload'),
        ('model_inference', 'Model Inference'),
        ('model_load', 'Model Load'),
        ('model_error', 'Model Error'),
        ('model_update', 'Model Update'),
    ]

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    endpoint = models.CharField(max_length=255, null=True, blank=True)
    method = models.CharField(max_length=10, null=True, blank=True)
    status_code = models.IntegerField(null=True, blank=True)
    response_time = models.FloatField(null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    data = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # New fields for model management
    model_id = models.CharField(max_length=255, null=True, blank=True)
    model_version = models.CharField(max_length=50, null=True, blank=True)
    model_provider = models.CharField(max_length=100, null=True, blank=True)
    resource_usage = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['event_type']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['user']),
            models.Index(fields=['endpoint']),
            models.Index(fields=['model_id']),
            models.Index(fields=['model_provider']),
        ]

    def __str__(self):
        return f"{self.event_type} - {self.timestamp}"
