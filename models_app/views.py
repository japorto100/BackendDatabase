from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from .models import UploadedFile, ModelConfig, Evidence
from .serializers import UploadedFileSerializer, ModelConfigSerializer, EvidenceSerializer
from .ai_models import AIModelManager
from .ai_models.audio.textToSpeech import TTSFactory, TTSEngine
from django.shortcuts import render
import logging
import tempfile
import os
from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.conf import settings
import os
import mimetypes
import json
from rest_framework import viewsets
from .mention_providers import get_mention_provider
from .mention_processor import MentionProcessor
from django.contrib.auth.decorators import login_required
from utils.config_handler import config
from error_handlers import LocalGPTError
from django.views.decorators.csrf import csrf_exempt
from .knowledge.rag_manager import RAGModelManager
from .knowledge.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
from .knowledge.knowledge_graph.external_kb_connector import PerplexicaKGConnector
from .knowledge.entity_extractor import EntityExtractor
from models_app.multimodal.multimodal_responder import MultimodalResponder
from models_app.vision.document.factory.document_adapter_registry import registry as document_registry
from rest_framework.decorators import api_view, parser_classes
from .knowledge.hyde_processor import HyDEProcessor
from .ai_models.provider_factory import ProviderFactory
from .ai_models.vision import VisionProviderFactory, GPT4VisionService, GeminiVisionService, QwenVisionService, LightweightVisionService, DocumentVisionAdapter
from analytics_app.electricity_cost import ElectricityCostTracker

# Logger für diese Datei konfigurieren
logger = logging.getLogger(__name__)

# Initialize the TTS factory instead of manager
tts_factory = TTSFactory(default_engine=TTSEngine.SPARK)

class ModelListView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get all available models"""
        ai_model_manager = AIModelManager()
        available_models = ai_model_manager.get_available_models()
        return Response(available_models)

class FileUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Upload a file for processing"""
        file_serializer = UploadedFileSerializer(data=request.data)
        if file_serializer.is_valid():
            try:
                # Get the uploaded file
                uploaded_file = request.FILES['file']
                
                # Create a new uploaded file instance
                file_instance = file_serializer.save()
                
                # Process the file with appropriate model
                model = request.data.get('model', 'default')
                ai_model_manager = AIModelManager()
                try:
                    result = ai_model_manager.process_file(file_instance.file.path, model)
                    return Response(result, status=status.HTTP_200_OK)
                except LocalGPTError as e:
                    logger.error(f"LocalGPT error processing file: {str(e)}")
                    return Response(e.to_dict(), status=status.HTTP_422_UNPROCESSABLE_ENTITY)
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")
                    return Response({
                        "error": f"Error processing file: {str(e)}",
                        "error_code": "processing_error",
                        "details": {}
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                logger.error(f"Error handling file upload: {str(e)}")
                return Response({
                    "error": f"Error handling file upload: {str(e)}",
                    "error_code": "upload_error",
                    "details": {}
                }, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def upload_view(request):
    """Render the file upload interface"""
    return render(request, 'models_app/upload.html')

@require_GET
def file_preview(request):
    """
    API endpoint to get file content for preview
    """
    file_path = request.GET.get('path')
    
    if not file_path:
        return JsonResponse({'error': 'No file path provided'}, status=400)
    
    # Ensure the path is within the project directory for security
    absolute_path = os.path.abspath(os.path.join(settings.BASE_DIR, file_path))
    if not absolute_path.startswith(str(settings.BASE_DIR)):
        return JsonResponse({'error': 'Invalid file path'}, status=403)
    
    try:
        if not os.path.exists(absolute_path):
            return JsonResponse({'error': 'File not found'}, status=404)
        
        # Get file type
        file_type, _ = mimetypes.guess_type(absolute_path)
        
        # Determine file type for syntax highlighting
        extension = os.path.splitext(absolute_path)[1].lower()
        syntax_type = {
            '.md': 'markdown',
            '.js': 'javascript',
            '.py': 'python',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.txt': 'text'
        }.get(extension, 'text')
        
        # Read file content
        with open(absolute_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return JsonResponse({
            'name': os.path.basename(absolute_path),
            'path': file_path,
            'type': syntax_type,
            'mime_type': file_type,
            'content': content,
            'size': os.path.getsize(absolute_path)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

class EvidenceViewSet(viewsets.ModelViewSet):
    """
    API endpoint for Evidence retrieval and management
    """
    queryset = Evidence.objects.all()
    serializer_class = EvidenceSerializer
    
    def get_queryset(self):
        """
        Filter evidence by query_id if provided
        """
        queryset = Evidence.objects.all().order_by('-created_at')
        query_id = self.request.query_params.get('query_id', None)
        if query_id:
            queryset = queryset.filter(query_id=query_id)
        return queryset

@require_http_methods(["GET"])
def mention_categories(request):
    """
    API-Endpunkt zum Abrufen der verfügbaren Kategorien für @-Mentions.
    
    Beispiel-Antwort:
    {
        "categories": [
            {"id": "projekte", "name": "Projekte", "icon": "🏗️"},
            {"id": "dokumente", "name": "Dokumente", "icon": "📄"},
            ...
        ]
    }
    """
    try:
        provider = get_mention_provider()
        categories = provider.get_categories()
        
        return JsonResponse({
            "categories": categories,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Mention-Kategorien: {e}")
        return JsonResponse({
            "status": "error",
            "message": f"Fehler beim Abrufen der Kategorien: {str(e)}",
            "categories": []
        }, status=500)

@require_http_methods(["GET"])
def mention_search(request):
    """
    API-Endpunkt für die Suche nach @-Mentions in einer bestimmten Kategorie.
    
    Query parameters:
        category: Die zu durchsuchende Kategorie (z.B. 'projekte', 'dokumente')
        q: Die Suchanfrage
        limit: Maximale Anzahl der zurückzugebenden Ergebnisse (optional)
    
    Beispiel-URL: /api/mentions/search?category=projekte&q=hergiswill&limit=5
    
    Beispiel-Antwort:
    {
        "results": [
            {
                "id": "123",
                "name": "Hergiswill 782",
                "description": "Renovierungsprojekt",
                "type": "project",
                "icon": "🏗️"
            },
            ...
        ],
        "status": "success"
    }
    """
    try:
        category = request.GET.get('category', 'projekte')
        query = request.GET.get('q', '')
        limit = int(request.GET.get('limit', 10))
        
        provider = get_mention_provider()
        results = provider.search(category, query, limit)
        
        return JsonResponse({
            "results": results,
            "status": "success",
            "query": query,
            "category": category
        })
    except Exception as e:
        logger.error(f"Fehler bei der Mention-Suche: {e}")
        return JsonResponse({
            "status": "error",
            "message": f"Fehler bei der Suche: {str(e)}",
            "results": []
        }, status=500)

@require_http_methods(["GET"])
def mention_item_details(request, category, item_id):
    """
    API-Endpunkt zum Abrufen der Details eines bestimmten @-Mention-Elements.
    
    URL-Parameter:
        category: Die Kategorie des Elements.
        item_id: Die ID des Elements.
    
    Beispiel-URL: /api/mentions/projekte/123
    
    Beispiel-Antwort (für ein Projekt):
    {
        "id": "123",
        "name": "Hergiswill 782",
        "description": "Renovierungsprojekt",
        "client": "Muster AG",
        "address": "Hergiswillerstr. 782, 6052 Hergiswill",
        "start_date": "2023-03-15",
        "status": "active",
        "documents": [
            {
                "id": "456",
                "name": "Bauplan Erdgeschoss",
                "type": "pdf"
            },
            ...
        ]
    }
    """
    try:
        provider = get_mention_provider()
        details = provider.get_item_details(category, item_id)
        
        # Hier könnten wir auch die lokalen Daten mit den externen Daten kombinieren,
        # falls erforderlich.
        
        return JsonResponse(details)
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Mention-Details: {e}")
        return JsonResponse({
            "status": "error",
            "message": f"Fehler beim Abrufen der Details: {str(e)}"
        }, status=500)

@require_http_methods(["GET"])
def get_web_search_categories(request):
    """
    Returns available categories for web mentions.
    """
    try:
        mention_processor = MentionProcessor()
        web_provider = mention_processor.providers.get('web')
        
        if not web_provider:
            return JsonResponse({"error": "Web provider not found"}, status=404)
            
        categories = web_provider.get_categories()
        return JsonResponse({"categories": categories})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def search_web_mentions(request):
    """
    Searches for web content via the mention system.
    """
    try:
        query = request.GET.get('query', '')
        category = request.GET.get('category', 'web')
        limit = int(request.GET.get('limit', 5))
        
        if not query:
            return JsonResponse({"error": "Query parameter is required"}, status=400)
            
        mention_processor = MentionProcessor()
        web_provider = mention_processor.providers.get('web')
        
        if not web_provider:
            return JsonResponse({"error": "Web provider not found"}, status=404)
            
        results = web_provider.search(category, query, limit)
        return JsonResponse({"results": results})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["POST"])
def auto_web_search(request):
    """
    Automatically extracts search queries from messages with @Web mentions,
    performs searches, and adds results to context.
    """
    try:
        data = json.loads(request.body)
        message_text = data.get('query', '')
        category = data.get('category', 'web')
        message_id = data.get('message_id')
        
        if not message_text:
            return JsonResponse({"error": "Empty message"}, status=400)
            
        # Use LLM to extract the most relevant search query
        search_query = extract_search_query(message_text)
        
        # Perform the search
        mention_processor = MentionProcessor()
        web_provider = mention_processor.providers.get('web')
        
        if not web_provider:
            return JsonResponse({"error": "Web provider not found"}, status=404)
            
        search_results = web_provider.search(category, search_query, limit=3)
        
        # Get details for each result and add to context
        for result in search_results:
            item_id = result.get('id')
            details = web_provider.get_item_details(category, item_id)
            
            # Add to evidence/context for this message
            add_to_message_context(message_id, details)
        
        return JsonResponse({
            "success": True,
            "query": search_query,
            "results_count": len(search_results)
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def extract_search_query(message_text):
    """
    Uses LLM to extract the most relevant search query from the message.
    """
    # In production, this would use an LLM call to extract a relevant query
    # For now, we'll use a simple approach - remove @Web and use the rest
    query = message_text.replace('@Web', '').strip()
    return query

def add_to_message_context(message_id, content):
    """
    Adds search results to the message context/evidence.
    """
    # In production, this would update the message's context
    # and add the content to the evidence database
    # This is a placeholder
    pass

@require_http_methods(["POST"])
def hyde_enhanced_search(request):
    """
    Perform a search enhanced with HyDE (Hypothetical Document Embeddings).
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '')
        category = data.get('category', 'web')
        limit = int(data.get('limit', 5))
        
        if not query:
            return JsonResponse({"error": "Empty query"}, status=400)
        
        # Create HyDE processor
        processor = HyDEProcessor(request.user)
        
        # Perform enhanced search
        results = processor.enhance_search(query, category, limit)
        
        return JsonResponse({
            "success": True,
            "query": query,
            "results": results,
            "results_count": len(results)
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required
@require_http_methods(["GET"])
def available_models(request):
    """
    Get available AI models
    """
    try:
        model_type = request.GET.get('type', 'chat')
        
        # Get model manager
        model_manager = AIModelManager()
        
        # Get all models
        all_models = model_manager.models
        
        # Filter models based on type
        if model_type == 'chat':
            # For chat, include all models
            models = [
                {
                    'id': model_id,
                    'name': model_info['name'],
                    'provider': model_info['provider'],
                    'vision_capable': model_info.get('vision_capable', False),
                    'max_tokens': model_info.get('max_tokens', 2048)
                }
                for model_id, model_info in all_models.items()
            ]
            default_model = model_manager.default_model
        elif model_type == 'hyde':
            # For HyDE, prefer smaller, faster models
            models = [
                {
                    'id': model_id,
                    'name': model_info['name'],
                    'provider': model_info['provider'],
                    'max_tokens': model_info.get('max_tokens', 2048)
                }
                for model_id, model_info in all_models.items()
                if not model_info.get('vision_capable', False)  # Vision models are overkill for HyDE
            ]
            # Add local models that are good for HyDE
            local_models = [
                {
                    'id': 'deepseek-coder-1.3b-instruct',
                    'name': 'DeepSeek Coder 1.3B',
                    'provider': 'Local',
                    'max_tokens': 2048
                },
                {
                    'id': 'mistral-7b-instruct',
                    'name': 'Mistral 7B Instruct',
                    'provider': 'Local',
                    'max_tokens': 4096
                }
            ]
            models.extend(local_models)
            default_model = config.get("MODELS", "DEFAULT_HYDE_MODEL", "deepseek-coder-1.3b-instruct")
        else:
            # Unknown type
            return JsonResponse({
                'error': f'Unknown model type: {model_type}'
            }, status=400)
        
        return JsonResponse({
            'models': models,
            'default_model': default_model
        })
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        return JsonResponse({
            'error': 'Failed to get available models'
        }, status=500)

@login_required
@require_http_methods(["GET"])
def electricity_forecast(request):
    """
    Get electricity cost forecasts for local models
    """
    try:
        tracker = ElectricityCostTracker(request.user)
        
        # Get parameters
        model_id = request.GET.get('model_id', '')
        token_count = int(request.GET.get('token_count', 1000))
        daily_queries = int(request.GET.get('daily_queries', 100))
        
        # Get model stats
        if model_id:
            stats = tracker.get_model_stats(model_id)
            forecast = tracker.forecast_cost(model_id, token_count, daily_queries)
            
            if not stats:
                return JsonResponse({
                    'error': f'No usage data available for model {model_id}'
                }, status=404)
            
            return JsonResponse({
                'model_id': model_id,
                'stats': stats,
                'forecast': forecast
            })
        else:
            # Get all models with stats
            all_models = []
            from .models import AIModel
            
            models = AIModel.objects.filter(provider='Local', is_active=True)
            for model in models:
                stats = tracker.get_model_stats(model.model_id)
                if stats:
                    all_models.append({
                        'model_id': model.model_id,
                        'name': model.name,
                        'stats': stats,
                        'forecast': tracker.forecast_cost(model.model_id, token_count, daily_queries)
                    })
            
            return JsonResponse({
                'models': all_models,
                'electricity_rate': tracker.kwh_rate
            })
    except Exception as e:
        logger.error(f"Error getting electricity forecast: {str(e)}")
        return JsonResponse({
            'error': 'Failed to get electricity forecast'
        }, status=500)

@require_http_methods(["GET"])
def kg_connector(request):
    """
    Get KG connector information - simple endpoint to confirm KG connectivity
    """
    return JsonResponse({
        "status": "connected",
        "capabilities": ["query_enhancement", "search_storage", "entity_retrieval"]
    })

@csrf_exempt
@require_http_methods(["POST"])
def enhance_query(request):
    """
    Enhance a search query with KG knowledge
    """
    try:
        data = json.loads(request.body)
        query = data.get("query", "")
        
        if not query:
            return JsonResponse({"error": "Query is required"}, status=400)
        
        # Get KG manager and RAG manager
        kg_manager = KnowledgeGraphManager()
        rag_manager = RAGModelManager()
        
        # Initialize KG connector 
        vector_db = None
        model = rag_manager.get_model("default")
        if model:
            vector_db = model.vector_store
        
        # Create connector
        kg_connector = PerplexicaKGConnector(kg_manager, vector_db)
        
        # Enhance query
        enhanced_query = kg_connector.enhance_search_query(query)
        
        return JsonResponse({
            "original_query": query,
            "enhanced_query": enhanced_query
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Error enhancing query: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def store_search_results(request):
    """
    Store valuable search results in the knowledge graph
    """
    try:
        data = json.loads(request.body)
        query = data.get("query", "")
        results = data.get("results", [])
        
        if not query or not results:
            return JsonResponse({"error": "Query and results are required"}, status=400)
        
        # Get KG manager and RAG manager
        kg_manager = KnowledgeGraphManager()
        rag_manager = RAGModelManager()
        
        # Initialize KG connector
        vector_db = None
        model = rag_manager.get_model("default")
        if model:
            vector_db = model.vector_store
        
        # Create connector
        kg_connector = PerplexicaKGConnector(kg_manager, vector_db)
        
        # Store results
        success = kg_connector.store_search_results(query, results)
        
        return JsonResponse({
            "status": "success" if success else "partial_success",
            "stored_count": len(results)
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Error storing search results: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def get_entities(request):
    """
    Get entities from the knowledge graph related to a query
    """
    try:
        query = request.GET.get("query", "")
        
        if not query:
            return JsonResponse({"error": "Query parameter is required"}, status=400)
        
        # Get KG manager
        kg_manager = KnowledgeGraphManager()
        
        # Extract entities from query
        extractor = EntityExtractor()
        entities = extractor.extract_entities(query)
        
        # Get related entities from KG
        related_entities = []
        for entity in entities:
            # Find similar entities in KG
            similar = kg_manager.graph_storage.find_similar_entities(entity.get("label", ""), limit=5)
            related_entities.extend(similar)
        
        return JsonResponse({
            "query": query,
            "entities": related_entities
        })
        
    except Exception as e:
        logger.error(f"Error getting related entities: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500) 

@csrf_exempt
@require_http_methods(["POST"])
def process_image(request):
    """
    Process an image with multimodal capabilities
    """
    try:
        data = json.loads(request.body)
        image_url = data.get("image_url", "")
        prompt = data.get("prompt", "Describe this image in detail")
        model_id = data.get("model_id", "default_vision_model")
        
        if not image_url:
            return JsonResponse({"error": "Image URL is required"}, status=400)
        
        # Create vision provider factory
        factory = VisionProviderFactory()
        
        # Determine appropriate vision provider based on model_id
        if "gpt" in model_id.lower():
            config = {'model': model_id, 'vision_type': 'gpt4v'}
            service = GPT4VisionService(config)
        elif "qwen" in model_id.lower():
            config = {'model': model_id, 'vision_type': 'qwen'}
            service = QwenVisionService(config)
        elif "gemini" in model_id.lower():
            config = {'model': model_id, 'vision_type': 'gemini'}
            service = GeminiVisionService(config)
        else:
            # For other models, get a recommended provider or use lightweight
            config = {'model': model_id, 'vision_type': 'lightweight'}
            service = LightweightVisionService(config)
        
        # Process the image
        response_text, confidence = service.process_image(image_url, prompt)
        
        return JsonResponse({
            "response": response_text,
            "confidence": confidence,
            "image_url": image_url,
            "model_used": model_id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def analyze_document(request):
    """
    Analyze a document with multimodal capabilities
    """
    try:
        # Check if request has a file
        if 'document' not in request.FILES:
            return JsonResponse({"error": "Document file is required"}, status=400)
        
        document = request.FILES['document']
        prompt = request.POST.get("prompt", "Extract key information from this document")
        model_id = request.POST.get("model_id", "default_vision_model")
        
        # Save the document temporarily
        file_path = os.path.join(settings.MEDIA_ROOT, 'temp', document.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb+') as destination:
            for chunk in document.chunks():
                destination.write(chunk)
        
        # Use our new DocumentVisionAdapter
        from models_app.ai_models.vision import DocumentVisionAdapter
        
        # Determine the best vision service based on model_id
        if "gpt" in model_id.lower():
            config = {'model': model_id, 'vision_type': 'gpt4v'}
        elif "qwen" in model_id.lower():
            config = {'model': model_id, 'vision_type': 'qwen'}
        elif "gemini" in model_id.lower():
            config = {'model': model_id, 'vision_type': 'gemini'}
        else:
            # Default to a lightweight model
            config = {'model': model_id, 'vision_type': 'lightweight'}
        
        # Create DocumentVisionAdapter with the appropriate config
        document_adapter = DocumentVisionAdapter(config)
        
        # Process the document
        result = document_adapter.process_document(file_path, prompt)
        
        # Extract the response and confidence
        response_text = result.get('text', '')
        confidence = result.get('confidence', 0.0)
        metadata = result.get('metadata', {})
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return JsonResponse({
            "response": response_text,
            "confidence": confidence,
            "document_name": document.name,
            "model_used": model_id,
            "metadata": metadata
        })
        
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def get_vision_models(request):
    """
    Get available vision-capable models
    """
    try:
        # Create vision provider factory
        factory = VisionProviderFactory()
        
        # Define available vision models with their capabilities
        vision_models = [
            # GPT-4 Vision models
            {
                'id': 'gpt-4o',
                'name': 'GPT-4o',
                'provider': 'OpenAI',
                'max_tokens': 4096,
                'capabilities': ['image_description', 'visual_qa', 'document_analysis']
            },
            {
                'id': 'gpt-4-vision',
                'name': 'GPT-4 Vision',
                'provider': 'OpenAI',
                'max_tokens': 4096,
                'capabilities': ['image_description', 'visual_qa', 'document_analysis']
            },
            # Qwen models
            {
                'id': 'Qwen/Qwen2-VL-7B-Instruct',
                'name': 'Qwen2-VL-7B-Instruct',
                'provider': 'Qwen',
                'max_tokens': 2048,
                'capabilities': ['image_description', 'visual_qa', 'multimodal_chat']
            },
            {
                'id': 'Qwen/Qwen-VL-Chat',
                'name': 'Qwen-VL-Chat',
                'provider': 'Qwen',
                'max_tokens': 2048,
                'capabilities': ['image_description', 'visual_qa']
            },
            # Gemini models
            {
                'id': 'gemini-pro-vision',
                'name': 'Gemini Pro Vision',
                'provider': 'Google',
                'max_tokens': 2048,
                'capabilities': ['image_description', 'visual_qa', 'document_analysis']
            },
            # Lightweight models
            {
                'id': 'clip-phi',
                'name': 'CLIP-Phi (Lightweight)',
                'provider': 'Local',
                'max_tokens': 1024,
                'capabilities': ['image_description']
            },
            {
                'id': 'blip2-flan-t5',
                'name': 'BLIP2-Flan-T5 (Lightweight)',
                'provider': 'Local',
                'max_tokens': 1024,
                'capabilities': ['image_description', 'visual_qa']
            }
        ]
        
        # Get default vision model from config
        default_model = config.get("MODELS", "DEFAULT_VISION_MODEL", "gpt-4o")
        
        return JsonResponse({
            'models': vision_models,
            'default_model': default_model
        })
    except Exception as e:
        logger.error(f"Error getting vision models: {str(e)}")
        return JsonResponse({
            'error': 'Failed to get vision models'
        }, status=500)

@api_view(['POST'])
def document_feedback(request):
    """
    API endpoint to collect user feedback on document processing.
    This connects to the DocumentAdapterRegistry's feedback collection mechanism.
    
    Expected payload:
    {
        "document_id": "string",
        "adapter_name": "string",
        "rating": float,
        "feedback_text": "string",
        "improvement_aspects": ["string", "string", ...]
    }
    """
    try:
        # Extract data from request
        document_id = request.data.get('document_id')
        adapter_name = request.data.get('adapter_name')
        rating = request.data.get('rating')
        feedback_text = request.data.get('feedback_text', '')
        improvement_aspects = request.data.get('improvement_aspects', [])
        
        # Validation
        if not all([document_id, adapter_name, rating]):
            return Response({
                'status': 'error',
                'message': 'Missing required fields: document_id, adapter_name, and rating are required'
            }, status=400)
        
        # Get document path from database (in a real implementation)
        # For simplicity, we'll mock this part
        document_path = f"/path/to/documents/{document_id}"
        
        # Format feedback text with improvement aspects for better context
        if improvement_aspects:
            aspects_str = ", ".join(improvement_aspects)
            enhanced_feedback = f"{feedback_text}\n\nImprovement areas: {aspects_str}"
        else:
            enhanced_feedback = feedback_text
        
        # Submit feedback to the registry
        result = document_registry.collect_user_feedback(
            adapter_name=adapter_name,
            document_path=document_path,
            rating=float(rating),
            feedback_text=enhanced_feedback
        )
        
        # Trigger adapter priority adjustment if rating is particularly low or high
        if float(rating) <= 2.0 or float(rating) >= 4.5:
            adjustment_result = document_registry.adjust_adapter_priorities()
            result['priority_adjustment'] = adjustment_result
        
        return Response({
            'status': 'success',
            'message': 'Feedback recorded successfully',
            'data': result
        })
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Error processing feedback: {str(e)}'
        }, status=500)

@api_view(['GET'])
def tts_engines(request):
    """
    Get available TTS engines.
    """
    try:
        # Use the factory to list available engines
        engines = []
        for engine_name in tts_factory.list_available_engines():
            # Create more detailed engine info
            engine_info = {
                "id": engine_name,
                "name": engine_name.capitalize(),
                "description": f"{engine_name.capitalize()} Text-to-Speech engine",
                "features": []
            }
            
            # Add specific features based on engine
            if engine_name == TTSEngine.SPARK:
                engine_info["features"] = ["voice_cloning", "custom_voices", "bilingual"]
                engine_info["name"] = "Spark-TTS"
                engine_info["description"] = "LLM-based Text-to-Speech with voice cloning capabilities"
            elif engine_name == TTSEngine.COQUI:
                engine_info["features"] = ["multiple_languages", "custom_voices"]
                engine_info["name"] = "Coqui-TTS" 
                engine_info["description"] = "Open-source TTS with multilingual support"
            elif engine_name == TTSEngine.MOZILLA:
                engine_info["features"] = ["multiple_languages", "multispeaker"]
                engine_info["name"] = "Mozilla-TTS"
                engine_info["description"] = "Mozilla's Text-to-Speech framework with multispeaker support"
                
            engines.append(engine_info)
        
        return Response({
            'engines': engines,
            'default_engine': tts_factory.default_engine.value
        })
    except Exception as e:
        logger.error(f"Error getting TTS engines: {str(e)}")
        return Response({
            'error': f"Failed to get TTS engines: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def tts_voice_presets(request):
    """
    Get available voice presets for a TTS engine.
    """
    try:
        engine = request.query_params.get('engine', tts_factory.default_engine.value)
        
        # Get appropriate voices/presets based on engine
        presets = []
        service = tts_factory.get_service(engine)
        
        if engine == TTSEngine.SPARK:
            presets = service.get_available_voice_presets() if hasattr(service, 'get_available_voice_presets') else []
        elif engine == TTSEngine.COQUI:
            # For Coqui, transform the available voices into presets format
            voices = tts_factory.list_available_voices(engine)
            presets = [{"id": voice, "name": voice.replace("_", " ").title()} for voice in voices]
            
            # Add language info from any available method
            if hasattr(service, 'get_available_voice_presets'):
                presets = service.get_available_voice_presets()
        elif engine == TTSEngine.MOZILLA:
            # For Mozilla, get speakers if it's a multispeaker model
            voices = tts_factory.list_available_voices(engine)
            if voices:
                presets = [{"id": voice, "name": f"Speaker {voice}"} for voice in voices]
            else:
                # If no speakers available, create a default preset
                presets = [{"id": "default", "name": "Default Voice"}]
        
        return Response({
            'engine': engine,
            'presets': presets
        })
    except Exception as e:
        logger.error(f"Error getting voice presets: {str(e)}")
        return Response({
            'error': f"Failed to get voice presets: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def tts_requirements(request):
    """
    Check if TTS requirements are met.
    """
    try:
        # Check requirements for all engines
        requirements = {}
        
        # Check Spark TTS
        spark_service = tts_factory.get_service(TTSEngine.SPARK)
        if hasattr(spark_service.model_manager, 'check_dependencies'):
            spark_success, _ = spark_service.model_manager.check_dependencies()
            requirements["spark_dependencies"] = spark_success
        
        # Check Coqui TTS
        coqui_service = tts_factory.get_service(TTSEngine.COQUI)
        if hasattr(coqui_service.model_manager, 'check_dependencies'):
            coqui_success, _ = coqui_service.model_manager.check_dependencies()
            requirements["coqui_dependencies"] = coqui_success
            
        # Check Mozilla TTS
        mozilla_service = tts_factory.get_service(TTSEngine.MOZILLA)
        if hasattr(mozilla_service.model_manager, 'check_dependencies'):
            mozilla_success, _ = mozilla_service.model_manager.check_dependencies()
            requirements["mozilla_dependencies"] = mozilla_success
        
        # Check for GPU
        try:
            import torch
            requirements["gpu_available"] = torch.cuda.is_available()
        except ImportError:
            requirements["gpu_available"] = False
        
        # Determine overall status
        status_ok = all(requirements.values())
        
        return Response({
            'status': 'ok' if status_ok else 'missing_requirements',
            'requirements': requirements
        })
    except Exception as e:
        logger.error(f"Error checking TTS requirements: {str(e)}")
        return Response({
            'error': f"Failed to check TTS requirements: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@parser_classes([JSONParser])
def tts_synthesize(request):
    """
    Synthesize speech from text.
    """
    try:
        # Get parameters
        text = request.data.get('text')
        engine = request.data.get('engine', tts_factory.default_engine.value)
        options = request.data.get('options', {})
        
        if not text:
            return Response({
                'error': 'Text is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Synthesize speech with the factory
        output_path = tts_factory.synthesize_speech(text, engine=engine, **options)
        
        if output_path:
            # Get the file URL
            file_url = f"/media/tts_output/{os.path.basename(output_path)}"
            
            return Response({
                'success': True,
                'file_url': file_url,
                'file_path': output_path,
                'engine': engine
            })
        else:
            return Response({
                'success': False,
                'error': 'Failed to synthesize speech'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        return Response({
            'error': f"Failed to synthesize speech: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def tts_voice_cloning(request):
    """
    Clone a voice from reference audio and synthesize speech.
    """
    try:
        # Get parameters
        text = request.data.get('text')
        engine = request.data.get('engine', tts_factory.default_engine.value)
        
        if not text:
            return Response({
                'error': 'Text is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if 'reference_audio' not in request.FILES:
            return Response({
                'error': 'Reference audio file is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save reference audio to temporary file
        reference_audio = request.FILES['reference_audio']
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            for chunk in reference_audio.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        # Optional prompt text
        prompt_text = request.data.get('prompt_text')
        
        # Determine the correct options format based on engine
        options = {}
        if engine == TTSEngine.SPARK:
            options = {
                'reference_audio': tmp_path
            }
            if prompt_text:
                options['prompt_text'] = prompt_text
        else:
            # Other engines might handle voice cloning differently
            # This is a simplified example
            options = {
                'reference_audio': tmp_path
            }
        
        # Synthesize speech with the factory
        try:
            output_path = tts_factory.synthesize_speech(text, engine=engine, **options)
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        if output_path:
            # Get the file URL
            file_url = f"/media/tts_output/{os.path.basename(output_path)}"
            
            return Response({
                'success': True,
                'file_url': file_url,
                'file_path': output_path,
                'engine': engine
            })
        else:
            return Response({
                'success': False,
                'error': 'Failed to clone voice'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        logger.error(f"Error in voice cloning: {str(e)}")
        return Response({
            'error': f"Failed to clone voice: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@parser_classes([JSONParser])
def tts_custom_voice(request):
    """
    Create a custom voice and synthesize speech.
    """
    try:
        # Get parameters
        text = request.data.get('text')
        engine = request.data.get('engine', tts_factory.default_engine.value)
        voice_params = request.data.get('voice_params', {})
        
        if not text:
            return Response({
                'error': 'Text is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not voice_params:
            return Response({
                'error': 'Voice parameters are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Synthesize speech with the factory
        # Each engine might handle voice parameters differently
        if engine == TTSEngine.SPARK:
            output_path = tts_factory.synthesize_speech(text, engine=engine, voice_params=voice_params)
        elif engine in [TTSEngine.COQUI, TTSEngine.MOZILLA]:
            # These engines might handle params differently, like 'speaker_id', 'language', etc.
            output_path = tts_factory.synthesize_speech(text, engine=engine, **voice_params)
        else:
            output_path = None
            
        if output_path:
            # Get the file URL
            file_url = f"/media/tts_output/{os.path.basename(output_path)}"
            
            return Response({
                'success': True,
                'file_url': file_url,
                'file_path': output_path,
                'engine': engine,
                'voice_params': voice_params
            })
        else:
            return Response({
                'success': False,
                'error': 'Failed to create custom voice'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        logger.error(f"Error creating custom voice: {str(e)}")
        return Response({
            'error': f"Failed to create custom voice: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@parser_classes([JSONParser])
def tts_preset_voice(request):
    """
    Use a voice preset to synthesize speech.
    """
    try:
        # Get parameters
        text = request.data.get('text')
        engine = request.data.get('engine', tts_factory.default_engine.value)
        preset_id = request.data.get('preset_id')
        
        if not text:
            return Response({
                'error': 'Text is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not preset_id:
            return Response({
                'error': 'Preset ID is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Convert preset to appropriate parameters based on engine
        if engine == TTSEngine.SPARK:
            # For Spark, preset might be a voice preset
            output_path = tts_factory.synthesize_speech(text, engine=engine, voice_preset=preset_id)
        elif engine == TTSEngine.COQUI:
            # For Coqui, preset might correspond to a specific model
            output_path = tts_factory.synthesize_speech(text, engine=engine, model_key=preset_id)
        elif engine == TTSEngine.MOZILLA:
            # For Mozilla, preset might be a speaker ID
            output_path = tts_factory.synthesize_speech(text, engine=engine, speaker_id=preset_id)
        else:
            output_path = None
        
        if output_path:
            # Get the file URL
            file_url = f"/media/tts_output/{os.path.basename(output_path)}"
            
            return Response({
                'success': True,
                'file_url': file_url,
                'file_path': output_path,
                'engine': engine,
                'preset_id': preset_id
            })
        else:
            return Response({
                'success': False,
                'error': 'Failed to use voice preset'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        logger.error(f"Error using voice preset: {str(e)}")
        return Response({
            'error': f"Failed to use voice preset: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def tts_download_model(request):
    """
    Download a TTS model.
    """
    try:
        # Get parameters
        model_name = request.data.get('model_name')
        engine = request.data.get('engine', tts_factory.default_engine.value)
        
        if not model_name:
            return Response({
                'error': 'Model name is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Download model through the appropriate service
        service = tts_factory.get_service(engine)
        success = False
        
        if engine == TTSEngine.SPARK:
            success = service.model_manager.download_model(model_name)
        elif engine == TTSEngine.COQUI:
            success = service.model_manager.download_model(model_name)
        elif engine == TTSEngine.MOZILLA:
            success = service.model_manager.download_model(model_name)
        
        if success:
            return Response({
                'success': True,
                'message': f"Successfully downloaded model {model_name}"
            })
        else:
            return Response({
                'success': False,
                'error': f"Failed to download model {model_name}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return Response({
            'error': f"Failed to download model: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@login_required
def tts_web_ui(request):
    """
    Web UI for TTS functionality.
    """
    try:
        # Get TTS engines
        engines = []
        for engine_name in tts_factory.list_available_engines():
            engines.append({
                "id": engine_name,
                "name": engine_name.capitalize(),
            })
        
        # Get voice presets for the default engine
        default_engine = tts_factory.default_engine.value
        service = tts_factory.get_service(default_engine)
        
        presets = []
        if default_engine == TTSEngine.SPARK and hasattr(service, 'get_available_voice_presets'):
            presets = service.get_available_voice_presets()
        elif default_engine == TTSEngine.COQUI:
            presets = service.get_available_voice_presets() if hasattr(service, 'get_available_voice_presets') else []
        elif default_engine == TTSEngine.MOZILLA:
            voices = service.list_speakers() if hasattr(service, 'list_speakers') else []
            presets = [{"id": voice, "name": f"Speaker {voice}"} for voice in voices]
        
        # Check requirements for all engines
        requirements = {}
        
        # Check requirements for each engine
        for engine in TTSEngine:
            service = tts_factory.get_service(engine)
            if hasattr(service.model_manager, 'check_dependencies'):
                success, _ = service.model_manager.check_dependencies()
                requirements[f"{engine.value}_dependencies"] = success
        
        # Check for GPU
        try:
            import torch
            requirements["gpu_available"] = torch.cuda.is_available()
        except ImportError:
            requirements["gpu_available"] = False
        
        requirements_met = all(requirements.values())
        
        context = {
            'engines': engines,
            'default_engine': default_engine,
            'presets': presets,
            'requirements': requirements,
            'requirements_met': requirements_met
        }
        
        return render(request, 'models_app/tts_ui.html', context)
    except Exception as e:
        logger.error(f"Error rendering TTS web UI: {str(e)}")
        return HttpResponse(f"Error: {str(e)}", status=500)
