from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

class UserProfile(models.Model):
    """Extended user profile information"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # Preferences
    theme = models.CharField(max_length=20, default='light', choices=[
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('system', 'System Default')
    ])
    
    # Model selection and configuration
    default_model = models.CharField(max_length=50, default='gpt-3.5-turbo')
    default_provider = models.CharField(max_length=50, default='openai', choices=[
        ('openai', 'OpenAI'),
        ('anthropic', 'Anthropic'),
        ('deepseek', 'DeepSeek'),
        ('local', 'Local Models')
    ])
    
    # Model parameters
    temperature = models.FloatField(default=0.7)
    max_tokens = models.IntegerField(default=1000)
    top_p = models.FloatField(default=0.9)
    
    # Resource management for local models
    use_gpu = models.BooleanField(default=True)
    quantization_level = models.CharField(max_length=10, default='4bit', choices=[
        ('4bit', '4-bit Quantization'),
        ('8bit', '8-bit Quantization'),
        ('16bit', '16-bit Precision'),
        ('32bit', 'Full Precision')
    ])
    
    # API keys (encrypted in a real implementation)
    openai_api_key = models.CharField(max_length=255, blank=True, null=True)
    anthropic_api_key = models.CharField(max_length=255, blank=True, null=True)
    deepseek_api_key = models.CharField(max_length=255, blank=True, null=True)
    
    # Usage limits
    daily_message_limit = models.IntegerField(default=100)
    daily_file_upload_limit = models.IntegerField(default=10)
    
    # Usage statistics
    messages_sent_today = models.IntegerField(default=0)
    files_uploaded_today = models.IntegerField(default=0)
    last_usage_reset = models.DateField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}'s profile"
    
    def reset_daily_usage(self):
        """Reset daily usage counters"""
        self.messages_sent_today = 0
        self.files_uploaded_today = 0
        self.last_usage_reset = timezone.now().date()
        self.save()
    
    def can_send_message(self):
        """Check if user can send more messages today"""
        # Reset counters if it's a new day
        today = timezone.now().date()
        if self.last_usage_reset != today:
            self.reset_daily_usage()
        
        return self.messages_sent_today < self.daily_message_limit
    
    def can_upload_file(self):
        """Check if user can upload more files today"""
        # Reset counters if it's a new day
        today = timezone.now().date()
        if self.last_usage_reset != today:
            self.reset_daily_usage()
        
        return self.files_uploaded_today < self.daily_file_upload_limit
    
    def increment_message_count(self):
        """Increment the messages sent counter"""
        # Reset counters if it's a new day
        today = timezone.now().date()
        if self.last_usage_reset != today:
            self.reset_daily_usage()
        
        self.messages_sent_today += 1
        self.save()
    
    def increment_file_upload_count(self):
        """Increment the files uploaded counter"""
        # Reset counters if it's a new day
        today = timezone.now().date()
        if self.last_usage_reset != today:
            self.reset_daily_usage()
        
        self.files_uploaded_today += 1
        self.save()

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create a UserProfile for every new User"""
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save the UserProfile when the User is saved"""
    instance.profile.save()

class UserSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='settings')
    
    # Existing fields
    indexer_model = models.CharField(max_length=100, default='vidore/colpali')
    generation_model = models.CharField(max_length=100, default='qwen')
    resized_height = models.IntegerField(default=280)
    resized_width = models.IntegerField(default=280)
    
    # New fields for model configuration
    local_models_path = models.CharField(max_length=255, default='models/', blank=True)
    
    # HyDE settings
    use_hyde = models.BooleanField(default=True)
    hyde_model = models.CharField(max_length=100, default='deepseek-coder-1.3b-instruct')
    
    # Performance settings
    batch_size = models.IntegerField(default=1)
    cache_models = models.BooleanField(default=True)
    
    # Custom model settings
    custom_model_url = models.CharField(max_length=500, blank=True, null=True, 
                                       help_text="URL zu einem benutzerdefinierten Modell (z.B. Hugging Face, GitHub)")
    use_custom_model = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Settings for {self.user.username}"