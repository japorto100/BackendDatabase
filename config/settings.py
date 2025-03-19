import os
from pathlib import Path
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'False') == 'True'

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Für TOML-Konfiguration
try:
    from utils.config_handler import config
    # Beispiel für Verwendung der TOML-Konfiguration
    DEFAULT_AI_MODEL = config.get('MODELS', 'DEFAULT_CHAT_MODEL', 'gpt-3.5-turbo')
    ENABLE_LOCAL_MODELS = config.get('MODELS', 'ENABLE_LOCAL_MODELS', False)
except ImportError:
    # Fallback, wenn TOML-Handler nicht verfügbar ist
    DEFAULT_AI_MODEL = 'gpt-3.5-turbo'
    ENABLE_LOCAL_MODELS = False

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'rest_framework',
    'corsheaders',
    
    # Local apps
    'chat_app',
    'models_app',
    'users_app',
    'analytics_app',
    'search_app',
    'drf_yasg',
    'files_app',
    'benchmark',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'analytics_app.middleware.RequestLoggingMiddleware',
    'analytics_app.middleware.SecurityHeadersMiddleware',
    'analytics_app.middleware.PerformanceMonitoringMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}

# CORS settings
CORS_ALLOW_ALL_ORIGINS = DEBUG  # Im Debug-Modus alle Origins erlauben
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Authentication
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'
LOGIN_URL = '/accounts/login/'

# Security settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Session settings
SESSION_COOKIE_SECURE = not DEBUG  # Use secure cookies in production
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
SESSION_COOKIE_AGE = 1209600  # 2 weeks

# CSRF settings
CSRF_COOKIE_SECURE = not DEBUG  # Use secure cookies in production
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Lax'
CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
]

# Email settings (for password reset)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'  # For development
# For production, use:
# EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
# EMAIL_HOST = 'smtp.example.com'
# EMAIL_PORT = 587
# EMAIL_USE_TLS = True
# EMAIL_HOST_USER = 'your-email@example.com'
# EMAIL_HOST_PASSWORD = 'your-password'

# Configure upload folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploaded_documents')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
SESSION_FOLDER = os.path.join(BASE_DIR, 'sessions')
INDEX_FOLDER = os.path.join(BASE_DIR, '.byaldi')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MentionProvider Konfiguration
MENTION_PROVIDER_TYPE = os.environ.get('MENTION_PROVIDER_TYPE', 'local')
EXTERNAL_API_URL = os.environ.get('EXTERNAL_API_URL', 'http://other-backend/api') # Wird eventuell nicht benötig da wir die andere backend hier integrieren werden
EXTERNAL_API_KEY = os.environ.get('EXTERNAL_API_KEY', '') # Wird eventuell nicht benötig da wir die andere backend hier integrieren werden

# Fallback für lokale Entwicklung, wenn das andere Backend nicht verfügbar ist
MENTION_PROVIDER_FALLBACK = True

# WebMention settings
WEBMENTION_CACHE_DURATION = 3600  # 1 hour
WEBMENTION_ENDPOINT = 'https://webmention.io/api/mentions'
WEBMENTION_DISCOVER_ENDPOINT = 'https://webmention.io/api/links'