from django.contrib import admin
from .models import UploadedFile

@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'file_type', 'user', 'upload_date', 'processed')
    list_filter = ('file_type', 'processed', 'upload_date')
    search_fields = ('original_filename', 'user__username')
    readonly_fields = ('id', 'upload_date')
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(user=request.user)
