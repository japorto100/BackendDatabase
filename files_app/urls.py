from django.urls import path
from . import views

app_name = 'files'

urlpatterns = [
    # Webansicht
    path('upload/', views.file_upload_view, name='upload'),
    
    # API-Endpunkte
    path('api/models/files/', views.FileUploadAPIView.as_view(), name='file-upload-api'),
    path('api/models/files/<uuid:file_id>/', views.FileDetailAPIView.as_view(), name='file-detail-api'),
    
    # Funktionale Endpunkte
    path('download/<uuid:file_id>/', views.download_file, name='download-file'),
    path('process/<uuid:file_id>/', views.process_file, name='process-file'),
] 