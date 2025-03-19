from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.benchmark_dashboard, name='benchmark_dashboard'),
    path('create/', views.create_benchmark, name='create_benchmark'),
    path('results/<int:run_id>/', views.benchmark_results, name='benchmark_results'),
    path('run/<int:run_id>/', views.run_benchmark, name='run_benchmark'),
    
    # Vision benchmark URLs
    path('vision/', views.vision_benchmark_dashboard, name='vision_benchmark_dashboard'),
    path('vision/create/', views.create_vision_benchmark, name='create_vision_benchmark'),
    path('vision/run/<int:run_id>/', views.run_vision_benchmark, name='run_vision_benchmark'),
    path('vision/results/<int:run_id>/', views.vision_benchmark_results, name='vision_benchmark_results'),
    path('vision/compare/', views.compare_vision_providers, name='compare_vision_providers'),
] 