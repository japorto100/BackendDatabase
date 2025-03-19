from rest_framework import serializers
from .models import UploadedFile

class UploadedFileSerializer(serializers.ModelSerializer):
    """Serializer for UploadedFile model"""
    file_size = serializers.SerializerMethodField()
    file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = UploadedFile
        fields = [
            'id', 'file', 'file_type', 'original_filename', 
            'upload_date', 'processed', 'processing_results',
            'file_size', 'file_url'
        ]
        read_only_fields = ['id', 'upload_date', 'processed', 'processing_results']
    
    def get_file_size(self, obj):
        """Get the file size in a human-readable format"""
        return obj.get_file_size()
    
    def get_file_url(self, obj):
        """Get the URL for the file"""
        request = self.context.get('request')
        if request and obj.file:
            return request.build_absolute_uri(obj.file.url)
        return None 