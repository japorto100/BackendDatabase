from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.AnalyticsListView.as_view(), name='analytics-list'),
    path('dashboard/', views.DashboardDataView.as_view(), name='dashboard-data'),
    path('dashboard-view/', views.dashboard_view, name='dashboard-view'),
    path('knowledge-graph/', views.knowledge_graph_view, name='knowledge_graph'),
    path('knowledge-graph/<str:graph_id>/', views.knowledge_graph_view, name='knowledge_graph_view'),
    path('api/graph-data/<str:graph_id>/', views.graph_data_api, name='graph_data_api'),
    path('vision/', views.vision_analytics_view, name='vision_analytics'),
]