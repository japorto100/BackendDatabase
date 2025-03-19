import os
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

def user_directory_path(instance, filename):
    """Generate file path for uploaded files"""
    # File will be uploaded to MEDIA_ROOT/user_<id>/<date>/<uuid>_<filename>
    date_str = timezone.now().strftime('%Y-%m-%d')
    unique_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(filename)[1]
    new_filename = f"{unique_id}{ext}"
    return f"user_{instance.user.id}/{date_str}/{new_filename}"

class UploadedFile(models.Model):
    """Model for uploaded files"""
    FILE_TYPES = (
        ('image', 'Image'),
        ('document', 'Document'),
        ('audio', 'Audio'),
        ('video', 'Video'),
        ('other', 'Other'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_files')
    file = models.FileField(upload_to=user_directory_path)
    file_type = models.CharField(max_length=20, choices=FILE_TYPES, default='other')
    original_filename = models.CharField(max_length=255)
    upload_date = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processing_results = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.original_filename} ({self.file_type})"
    
    def delete(self, *args, **kwargs):
        """Delete the file when the model instance is deleted"""
        # Delete the file from storage
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        # Call the parent delete method
        super().delete(*args, **kwargs)
    
    def get_file_size(self):
        """Get the file size in a human-readable format"""
        if self.file and os.path.exists(self.file.path):
            size_bytes = os.path.getsize(self.file.path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} TB"
        return "Unknown"
