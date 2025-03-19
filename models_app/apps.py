from django.apps import AppConfig


class ModelsAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'models_app'

    def ready(self):
        # Import and initialize the RAG model manager when Django starts
        from .knowledge.rag_manager import RAGModelManager
        RAGModelManager()
        
        # Initialize knowledge graph manager for entity extraction
        from .knowledge.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
        KnowledgeGraphManager()
        
        # Register AI model providers
        from .ai_models.provider_factory import ProviderFactory
        ProviderFactory()
