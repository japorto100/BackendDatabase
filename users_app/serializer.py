from rest_framework import serializers
from .models import UserProfile, UserSettings
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id', 'username']

class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for UserProfile model"""
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = [
            'user', 'theme', 'default_model', 'default_provider',
            'temperature', 'max_tokens', 'top_p',
            'use_gpu', 'quantization_level',
            'openai_api_key', 'anthropic_api_key', 'deepseek_api_key',
            'daily_message_limit', 'daily_file_upload_limit',
            'messages_sent_today', 'files_uploaded_today'
        ]
        extra_kwargs = {
            'openai_api_key': {'write_only': True},
            'anthropic_api_key': {'write_only': True},
            'deepseek_api_key': {'write_only': True},
        }

class UserSettingsSerializer(serializers.ModelSerializer):
    """Serializer for UserSettings model"""
    
    class Meta:
        model = UserSettings
        fields = [
            'indexer_model', 'generation_model', 
            'resized_height', 'resized_width',
            'local_models_path', 'use_hyde', 'hyde_model',
            'batch_size', 'cache_models'
        ]