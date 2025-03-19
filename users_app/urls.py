from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'users'

urlpatterns = [
    path('', views.UserListView.as_view(), name='user-list'),
    path('profile/', views.profile_view, name='profile'),
    path('api/profile/', views.UserProfileAPIView.as_view(), name='profile-api'),
    path('theme-toggle/', views.theme_toggle, name='theme-toggle'),
    path('generate-key/', views.GenerateAPIKeyView.as_view(), name='generate-key'),
    # Frontend URLs
    path('profile-page/', views.profile_view, name='profile-page'),
    path('register/', views.register, name='register'),
]