from rest_framework import serializers
from .models import SearchQuery

class SearchQuerySerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = SearchQuery
        fields = ['id', 'query', 'results', 'timestamp', 'username']
        read_only_fields = ['id', 'timestamp']