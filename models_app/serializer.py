from rest_framework import serializers
from .models import UploadedFile, ModelConfig, Evidence

class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ['id', 'file', 'file_type', 'uploaded_at', 'processed', 'processing_results']
        read_only_fields = ['id', 'uploaded_at', 'processed', 'processing_results']

class ModelConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelConfig
        fields = ['id', 'name', 'model_type', 'config', 'is_active', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class EvidenceSerializer(serializers.ModelSerializer):
    """
    Serializer for Evidence model to track source attribution
    """
    class Meta:
        model = Evidence
        fields = [
            'id', 'query_id', 'source_type', 
            'content', 'highlights', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']