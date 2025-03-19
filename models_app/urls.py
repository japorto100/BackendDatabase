from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import UploadedFileViewSet, ModelConfigViewSet, EvidenceViewSet
from .knowledge.knowledge_graph import views as kg_views
from .vision.multimodal import views as multimodal_views
from .ai_models import views as ai_models_views
from .ai_models.audio import views as tts_views
from .ai_models.vision import views as vision_views

app_name = 'models'

# Create a router and register our viewsets with it
router = DefaultRouter()
router.register(r'files', UploadedFileViewSet)
router.register(r'configurations', ModelConfigViewSet)
router.register(r'evidence', EvidenceViewSet)

urlpatterns = [
    path('', views.ModelListView.as_view(), name='model-list'),
    path('upload/', views.FileUploadView.as_view(), name='file-upload'),
    path('upload-interface/', views.upload_view, name='upload-interface'),
    path('api/files/preview', views.file_preview, name='file_preview'),
    path('api/', include(router.urls)),  # Include the router URLs
    path('api/mentions/categories', views.mention_categories, name='mention_categories'),
    path('api/mentions/search', views.mention_search, name='mention_search'),
    path('api/mentions/<str:category>/<str:item_id>', views.mention_item_details, name='mention_item_details'),
    path('mentions/web/categories/', views.get_web_search_categories, name='web_search_categories'),
    path('mentions/web/search/', views.search_web_mentions, name='web_search_mentions'),
    path('mentions/web/auto-search/', views.auto_web_search, name='auto_web_search'),
    path('api/search/hyde/', views.hyde_enhanced_search, name='hyde_enhanced_search'),
    path('api/models/available', views.available_models, name='available_models'),
    path('api/models/electricity-forecast', views.electricity_forecast, name='electricity_forecast'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/upload/', views.upload_file, name='upload_file'),
    path('api/files/', views.file_list, name='file_list'),
    path('api/evidence/', views.evidence_api, name='evidence_api'),
    path('api/mentions/', views.mention_api, name='mention_api'),
    path('api/kg-connector/', kg_views.kg_connector, name='kg_connector'),
    path('api/kg/enhance-query/', kg_views.enhance_query, name='kg_enhance_query'),
    path('api/kg/store-search-results/', kg_views.store_search_results, name='kg_store_search_results'),
    path('api/kg/entities/', kg_views.get_entities, name='kg_get_entities'),
    path('api/chat/multimodal/', multimodal_views.multimodal_chat, name='multimodal_chat'),
    path('api/search/multimodal/', multimodal_views.multimodal_search, name='multimodal_search'),
    
    # Vision API endpoints
    path('api/vision/analyze-image/', views.process_image, name='process_image'),
    path('api/vision/analyze-document/', views.analyze_document, name='analyze_document'),
    path('api/vision/models/', views.get_vision_models, name='get_vision_models'),
    
    # New vision-specific endpoints for our refactored services
    path('api/vision/services/gpt4v/', vision_views.gpt4v_service, name='gpt4v_service'),
    path('api/vision/services/gemini/', vision_views.gemini_service, name='gemini_service'),
    path('api/vision/services/qwen/', vision_views.qwen_service, name='qwen_service'),
    path('api/vision/services/lightweight/', vision_views.lightweight_service, name='lightweight_service'),
    path('api/vision/services/auto-select/', vision_views.auto_select_service, name='auto_select_service'),
    path('api/vision/services/', vision_views.list_services, name='list_vision_services'),
    
    path('api/document-feedback/', views.document_feedback, name='document_feedback'),
    
    # TTS API Endpoints - now pointing to the main views.py
    path('api/tts/engines/', views.tts_engines, name='tts_engines'),
    path('api/tts/voice-presets/', views.tts_voice_presets, name='tts_voice_presets'),
    path('api/tts/requirements/', views.tts_requirements, name='tts_requirements'),
    path('api/tts/synthesize/', views.tts_synthesize, name='tts_synthesize'),
    path('api/tts/voice-cloning/', views.tts_voice_cloning, name='tts_voice_cloning'),
    path('api/tts/custom-voice/', views.tts_custom_voice, name='tts_custom_voice'),
    path('api/tts/preset-voice/', views.tts_preset_voice, name='tts_preset_voice'),
    path('api/tts/download-model/', views.tts_download_model, name='tts_download_model'),
    path('tts/', views.tts_web_ui, name='tts_web_ui'),
    
    # TTS views textToSpeech folder
    path('api/tts/engines/', tts_views.list_engines, name='tts_engines'),
    path('api/tts/engines/<str:engine_id>/', tts_views.engine_details, name='tts_engine_details'),
    path('api/tts/synthesize/', tts_views.synthesize_speech, name='tts_synthesize'),
    path('api/tts/voice-cloning/', tts_views.voice_cloning, name='tts_voice_cloning'),
]