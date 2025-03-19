from rest_framework import serializers
from .models import AnalyticsEvent

class AnalyticsEventSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = AnalyticsEvent
        fields = ['id', 'event_type', 'event_data', 'timestamp', 'username']
        read_only_fields = ['id', 'timestamp']