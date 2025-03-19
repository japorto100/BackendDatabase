from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.ChatListView.as_view(), name='chat-list'),
    path('<uuid:chat_id>/', views.ChatDetailView.as_view(), name='chat-detail'),
    path('interface/', views.chat_view, name='chat-interface'),
]