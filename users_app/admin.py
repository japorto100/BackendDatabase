from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'api_key', 'max_tokens', 'updated_at')
    search_fields = ('user__username', 'api_key')