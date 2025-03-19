from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, UserSettings

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class UserSettingsForm(forms.ModelForm):
    """Form for updating user settings"""
    class Meta:
        model = UserSettings
        fields = [
            'first_name', 'last_name', 'email'
            'indexer_model', 'generation_model', 
            'resized_height', 'resized_width',
            'local_models_path', 'use_hyde', 'hyde_model',
            'batch_size', 'cache_models'
        ]
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['first_name'].widget.attrs.update({'class': 'form-control'})
        self.fields['last_name'].widget.attrs.update({'class': 'form-control'})
        self.fields['email'].widget.attrs.update({'class': 'form-control'})
        self.fields['indexer_model'].widget.attrs.update({'class': 'form-control'})
        self.fields['generation_model'].widget.attrs.update({'class': 'form-control'})
        self.fields['resized_height'].widget.attrs.update({'class': 'form-control'})
        self.fields['resized_width'].widget.attrs.update({'class': 'form-control'})
        self.fields['local_models_path'].widget.attrs.update({'class': 'form-control'})
        self.fields['use_hyde'].widget.attrs.update({'class': 'form-check-input'})
        self.fields['hyde_model'].widget.attrs.update({'class': 'form-control'})
        self.fields['batch_size'].widget.attrs.update({'class': 'form-control'})
        self.fields['cache_models'].widget.attrs.update({'class': 'form-check-input'})

class UserProfileForm(forms.ModelForm):
    """Form for updating user profile"""
    class Meta:
        model = UserProfile
        fields = [
            'theme', 'default_model', 'default_provider',
            'temperature', 'max_tokens', 'top_p',
            'use_gpu', 'quantization_level',
            'openai_api_key', 'anthropic_api_key', 'deepseek_api_key'
        ]
        widgets = {
            'openai_api_key': forms.PasswordInput(render_value=True),
            'anthropic_api_key': forms.PasswordInput(render_value=True),
            'deepseek_api_key': forms.PasswordInput(render_value=True),
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['theme'].widget.attrs.update({'class': 'form-select'})
        self.fields['default_model'].widget.attrs.update({'class': 'form-select'})
        self.fields['default_provider'].widget.attrs.update({'class': 'form-select'})
        self.fields['quantization_level'].widget.attrs.update({'class': 'form-select'})
        self.fields['temperature'].widget.attrs.update({'class': 'form-range', 'min': '0', 'max': '1', 'step': '0.1'})
        self.fields['top_p'].widget.attrs.update({'class': 'form-range', 'min': '0', 'max': '1', 'step': '0.1'})
        self.fields['max_tokens'].widget.attrs.update({'class': 'form-control'})
        self.fields['openai_api_key'].widget.attrs.update({'class': 'form-control'})
        self.fields['anthropic_api_key'].widget.attrs.update({'class': 'form-control'})
        self.fields['deepseek_api_key'].widget.attrs.update({'class': 'form-control'}) 