from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from .models import UserProfile, UserSettings
from .serializers import UserSerializer, UserProfileSerializer, UserSettingsSerializer
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, UserProfileForm, UserSettingsForm
import uuid
from django.http import JsonResponse
from models_app.ai_models import AIModelManager
from rest_framework import viewsets
from rest_framework.decorators import action

class UserListView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get all users (admin only)"""
        if not request.user.is_staff:
            return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
            
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get current user's profile"""
        try:
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def patch(self, request):
        """Update user profile preferences"""
        try:
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            serializer = UserProfileSerializer(profile, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def profile_view(request):
    """Display and update user profile"""
    if request.method == 'POST':
        profile_form = UserProfileForm(request.POST, instance=request.user.profile)
        user_form = UserSettingsForm(request.POST, instance=request.user)
        
        if profile_form.is_valid() and user_form.is_valid():
            profile_form.save()
            user_form.save()
            messages.success(request, 'Your profile has been updated.')
            return redirect('users:profile')
    else:
        profile_form = UserProfileForm(instance=request.user.profile)
        user_form = UserSettingsForm(instance=request.user)
    
    # Get available AI models
    ai_model_manager = AIModelManager()
    available_models = ai_model_manager.get_available_models()
    
    # Get usage statistics
    profile = request.user.profile
    usage_stats = {
        'messages_sent_today': profile.messages_sent_today,
        'files_uploaded_today': profile.files_uploaded_today,
        'daily_message_limit': profile.daily_message_limit,
        'daily_file_upload_limit': profile.daily_file_upload_limit,
    }
    
    return render(request, 'users_app/profile.html', {
        'profile_form': profile_form,
        'user_form': user_form,
        'available_models': available_models,
        'usage_stats': usage_stats
    })

class UserProfileAPIView(APIView):
    """API endpoint for user profile"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get user profile"""
        serializer = UserProfileSerializer(request.user.profile)
        return Response(serializer.data)
    
    def patch(self, request):
        """Update user profile"""
        serializer = UserProfileSerializer(request.user.profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@login_required
def theme_toggle(request):
    """Toggle between light and dark theme"""
    profile = request.user.profile
    
    # Toggle theme
    if profile.theme == 'light':
        profile.theme = 'dark'
    else:
        profile.theme = 'light'
    
    profile.save()
    
    return JsonResponse({'theme': profile.theme})

class GenerateAPIKeyView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Generate a new API key for the user"""
        try:
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            profile.api_key = str(uuid.uuid4())
            profile.save()
            return Response({"api_key": profile.api_key})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserSettingsViewSet(viewsets.ModelViewSet):
    serializer_class = UserSettingsSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserSettings.objects.filter(user=self.request.user)
    
    def get_object(self):
        settings, created = UserSettings.objects.get_or_create(user=self.request.user)
        return settings
    
    @action(detail=False, methods=['get'])
    def current(self, request):
        settings = self.get_object()
        serializer = self.get_serializer(settings)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'])
    def update_settings(self, request):
        settings = self.get_object()
        serializer = self.get_serializer(settings, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@login_required
def settings_view(request):
    """View für Benutzereinstellungen"""
    if request.method == 'POST':
        # Bestehender Code
        
        # Verarbeite benutzerdefiniertes Modell
        user_settings = request.user.settings
        user_settings.use_custom_model = 'use_custom_model' in request.POST
        user_settings.custom_model_url = request.POST.get('custom_model_url', '')
        user_settings.save()
        
        messages.success(request, 'Einstellungen erfolgreich aktualisiert.')
        return redirect('settings')
    
    # Bestehender Code
