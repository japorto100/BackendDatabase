import os
import uuid
import mimetypes
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse, Http404
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from .models import UploadedFile
from .serializers import UploadedFileSerializer
from models_app.ai_models import AIModelManager
from .api_docs import (
    file_upload_schema, file_detail_schema, 
    file_delete_schema, file_process_schema
)

# Create your views here.

@login_required
def file_upload_view(request):
    """Render the file upload interface"""
    # Get user's uploaded files
    user_files = UploadedFile.objects.filter(user=request.user).order_by('-upload_date')
    
    return render(request, 'files_app/upload.html', {
        'user_files': user_files
    })

class FileUploadAPIView(APIView):
    """API view for uploading files"""
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def get(self, request):
        """Get all files uploaded by the authenticated user"""
        files = UploadedFile.objects.filter(user=request.user).order_by('-upload_date')
        serializer = UploadedFileSerializer(files, many=True, context={'request': request})
        return Response(serializer.data)
    
    @file_upload_schema
    def post(self, request):
        """Upload a new file"""
        # Check if user has reached their daily file upload limit
        profile = request.user.profile
        if not profile.can_upload_file():
            return Response(
                {"error": "You have reached your daily file upload limit."},
                status=status.HTTP_429_TOO_MANY_REQUESTS
            )
        
        # Get the uploaded file
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return Response(
                {"error": "No file provided."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Determine file type
        file_type = self._determine_file_type(uploaded_file)
        
        # Create file record
        file_obj = UploadedFile.objects.create(
            user=request.user,
            file=uploaded_file,
            file_type=file_type,
            original_filename=uploaded_file.name
        )
        
        # Process file if needed
        if file_type in ['document', 'pdf']:
            # Process document with AI
            ai_model_manager = AIModelManager()
            processing_results = ai_model_manager.process_document(
                file_path=file_obj.file.path,
                file_type=file_type
            )
            
            # Update processing results
            file_obj.processed = True
            file_obj.processing_results = processing_results
            file_obj.save()
        
        # Increment user's file upload count
        profile.increment_file_upload_count()
        
        # Return file data
        serializer = UploadedFileSerializer(file_obj, context={'request': request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    def _determine_file_type(self, file_obj):
        """Determine the type of the uploaded file"""
        content_type = file_obj.content_type
        
        # Image types
        if content_type.startswith('image/'):
            return 'image'
        
        # Document types
        if content_type in ['application/pdf', 'application/msword', 
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                           'application/vnd.ms-excel',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           'text/plain', 'text/csv']:
            return 'document'
        
        # Audio types
        if content_type.startswith('audio/'):
            return 'audio'
        
        # Video types
        if content_type.startswith('video/'):
            return 'video'
        
        # Default
        return 'other'

class FileDetailAPIView(APIView):
    """API view for retrieving, updating, and deleting a file"""
    permission_classes = [IsAuthenticated]
    
    @file_detail_schema
    def get(self, request, file_id):
        """Get a specific file"""
        file_obj = get_object_or_404(UploadedFile, id=file_id, user=request.user)
        serializer = UploadedFileSerializer(file_obj, context={'request': request})
        return Response(serializer.data)
    
    @file_delete_schema
    def delete(self, request, file_id):
        """Delete a file"""
        file_obj = get_object_or_404(UploadedFile, id=file_id, user=request.user)
        file_obj.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

@login_required
def download_file(request, file_id):
    """Download a file"""
    file_obj = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    
    # Check if file exists
    if not file_obj.file or not os.path.exists(file_obj.file.path):
        raise Http404("File not found")
    
    # Open the file
    file_path = file_obj.file.path
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    # Determine content type
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = 'application/octet-stream'
    
    # Create response
    response = HttpResponse(file_data, content_type=content_type)
    response['Content-Disposition'] = f'attachment; filename="{file_obj.original_filename}"'
    
    return response

@login_required
@require_http_methods(["POST"])
def process_file(request, file_id):
    """Process a file with AI"""
    file_obj = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    
    # Check if file is already processed
    if file_obj.processed:
        return JsonResponse({
            'success': True,
            'message': 'File already processed',
            'results': file_obj.processing_results
        })
    
    # Process file with AI
    ai_model_manager = AIModelManager()
    processing_results = ai_model_manager.process_document(
        file_path=file_obj.file.path,
        file_type=file_obj.file_type
    )
    
    # Update processing results
    file_obj.processed = True
    file_obj.processing_results = processing_results
    file_obj.save()
    
    return JsonResponse({
        'success': True,
        'message': 'File processed successfully',
        'results': processing_results
    })
