from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'providers', views.ProviderViewSet)
router.register(r'search', views.SearchViewSet, basename='search')
router.register(r'preferences', views.UserProviderPreferenceViewSet, basename='preferences')
router.register(r'search-queries', views.SearchQueryViewSet)
router.register(r'search-results', views.SearchResultViewSet)
router.register(r'user-provider-preferences', views.UserProviderPreferenceViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/detect-provider-type/', views.detect_provider_type, name='detect_provider_type'),
    path('api/validate-api-key/', views.validate_api_key, name='validate_api_key'),
    path('', include(router.urls)),
    path('api/search/', views.search, name='search'),
    path('api/search/deep-research/', views.deep_research, name='deep_research'),
    path('api/search/deep-research/<str:research_id>/status', views.deep_research_status, name='deep_research_status'),
    path('api/search/deep-research/<str:research_id>/results', views.deep_research_results, name='deep_research_results'),
]