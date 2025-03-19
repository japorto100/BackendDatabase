import os
import logging
from django.conf import settings
import torch
from models_app.indexing.indexer import index_processed_documents
from .rag_manager import RAGModelManager
from langchain.vectorstores import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from error_handlers.models_app_errors import handle_embedding_error
from models_app.vision.document.factory.document_adapter_registry import DocumentAdapterRegistry

logger = logging.getLogger(__name__)

class DocumentIndexer:
    """
    Handles document indexing for chat sessions.
    """
    
    @staticmethod
    def _get_optimal_models():
        """
        Determine optimal indexer and embedding models based on available resources
        
        Returns:
            tuple: (indexer_model, embedding_model)
        """
        try:
            # Check available resources
            has_gpu = torch.cuda.is_available()
            gpu_memory = 0
            
            if has_gpu:
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # Convert to MB
                except Exception:
                    gpu_memory = 8000  # Assume 8GB if we can't determine
            
            # Select models based on resources
            if has_gpu and gpu_memory >= 16000:  # High-end system
                return (
                    'barlacy/docling-multimodal-embedder-large',  # Advanced multimodal document indexer
                    'intfloat/e5-mistral-7b-instruct'  # Powerful text embeddings
                )
            elif has_gpu and gpu_memory >= 8000:  # Mid-range system
                return (
                    'barlacy/docling-multimodal-embedder-base',  # Smaller multimodal document indexer
                    'jinaai/jina-embeddings-v2-small-en',  # Better model than colpali with vision compatibility
                    'intfloat/multilingual-e5-large'  # Good multilingual embeddings
                )
            elif has_gpu:  # Low-end GPU system
                return (
                    'Alibaba-NLP/StructVBERT',  # Structure-aware visual-textual model (lighter than docling)
                    'sentence-transformers/all-mpnet-base-v2'  # Good balance of quality and efficiency
                )
            else:  # CPU-only system
                return (
                    'vidore/colpali',  # Standard model
                    'intfloat/multilingual-e5-base'  # Basic multilingual embeddings
                )
        except Exception as e:
            logger.warning(f"Error determining optimal models: {str(e)}. Using default models.")
            return ('vidore/colpali', 'intfloat/multilingual-e5-base')
    
    @staticmethod
    def index_documents_for_session(session_id, file_paths, indexer_model=None, embedding_model=None):
        """
        Indexes documents for a chat session.
        
        Args:
            session_id: The ID of the chat session
            file_paths: List of file paths to index
            indexer_model: The model to use for indexing
            embedding_model: The model to use for embeddings (optional)
            
        Returns:
            tuple: (success, message, indexed_files)
        """
        try:
            # Use optimal models if not specified
            if not indexer_model or not embedding_model:
                default_indexer, default_embedding = DocumentIndexer._get_optimal_models()
                indexer_model = indexer_model or default_indexer
                embedding_model = embedding_model or default_embedding
            
            logger.info(f"Using indexer model: {indexer_model}, embedding model: {embedding_model}")
            
            # Create session folder if it doesn't exist
            upload_folder = getattr(settings, 'UPLOAD_FOLDER', 'uploaded_documents')
            session_folder = os.path.join(upload_folder, str(session_id))
            os.makedirs(session_folder, exist_ok=True)
            
            # Set index path
            index_folder = getattr(settings, 'INDEX_FOLDER', os.path.join(os.getcwd(), '.byaldi'))
            index_path = os.path.join(index_folder, str(session_id))
            
            # Initialisieren des Dokument-Registrys als globale Instanz
            registry = DocumentAdapterRegistry()
            
            # Process documents first using vision adapters
            processed_results = []
            for file_path in file_paths:
                # Get appropriate adapter for file type
                adapter = registry.get_adapter_for_document(file_path)
                
                # Process document
                processed_doc = adapter.process_document(file_path)
                processed_results.append(processed_doc)
            
            # Setup embeddings
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model
                )
                logger.info(f"Initialized embeddings with model: {embedding_model}")
            except Exception as e:
                handle_embedding_error(embedding_model, str(e))
                # Fallback to default embedding model
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L12-v2"
                )
                logger.warning(f"Failed to initialize {embedding_model}, fell back to all-MiniLM-L12-v2")
            
            # Index processed documents
            rag = index_processed_documents(
                processed_results,
                session_id=str(session_id),
                index_path=index_path,
                indexer_model=indexer_model,
                embeddings=embeddings
            )
            
            if rag is None:
                raise ValueError("Indexing failed: RAG model is None")
            
            # Store the RAG model
            rag_manager = RAGModelManager()
            rag_manager.set_model(session_id, rag)
            
            # Get filenames from paths
            indexed_files = [os.path.basename(path) for path in file_paths]
            
            logger.info(f"Documents indexed successfully for session {session_id}")
            return True, "Files indexed successfully.", indexed_files
            
        except Exception as e:
            logger.error(f"Error indexing documents for session {session_id}: {str(e)}")
            return False, f"Error indexing files: {str(e)}", []

class BidirectionalIndexer:
    """
    Creates and maintains bidirectional links between knowledge graph entities 
    and their source documents/locations.
    
    This class is responsible for:
    1. Creating vector embeddings for entities to link to documents
    2. Establishing bidirectional mapping between entities and document chunks
    3. Allowing retrieval of documents from entity references and vice versa
    4. Enabling efficient entity resolution through embedding similarity
    """
    
    def __init__(self, graph_storage, vector_db):
        self.graph_storage = graph_storage
        self.vector_db = vector_db
        
        # Cache for frequently accessed entities/documents
        self._entity_cache = {}
        self._document_cache = {}
        
    def index_entity_sources(self, entity, source_info):
        """
        Index an entity with its source information
        
        Args:
            entity: KG entity (with id, type, label, properties)
            source_info: Document source information (path, page, coordinates, etc.)
        """
        # 1. Store the source reference in the entity
        if "source_info" not in entity:
            entity["source_info"] = []
        entity["source_info"].append(source_info)
        
        # 2. Create a vector embedding for the entity
        entity_text = self._get_entity_text(entity)
        
        # 3. Check if we already have document embedding for this location
        document_embedding = self._get_document_embedding(source_info)
        entity_embedding = None
        
        if document_embedding and self._is_entity_exact_chunk_match(entity, source_info):
            # Reuse document embedding if entity exactly matches a document chunk
            entity_embedding = document_embedding
        else:
            # Generate new embedding
            entity_embedding = self._create_embedding(entity_text)
        
        # 4. Store in vector DB with entity ID reference
        self.vector_db.add_texts(
            texts=[entity_text],
            metadata={
                "entity_id": entity["id"],
                "entity_type": entity.get("type", "unknown"),
                "source_doc": source_info["document_path"],
                "source_location": source_info.get("location", ""),
                "is_entity": True  # Flag to distinguish from document chunks
            },
            embeddings=[entity_embedding]
        )
        
    def _get_document_embedding(self, source_info):
        """Get embedding for document chunk if available"""
        if not source_info.get("document_path") or not source_info.get("location"):
            return None
            
        # Check if we have the document chunk in the vector store
        results = self.vector_db.search(
            query="",
            filter={
                "source_doc": source_info["document_path"],
                "location": source_info.get("location", ""),
                "is_entity": {"$exists": False}  # Only document chunks
            },
            search_type="metadata"
        )
        
        if results and len(results) > 0:
            return results[0].get("embedding")
        return None
        
    def _is_entity_exact_chunk_match(self, entity, source_info):
        """Check if entity exactly matches a document chunk"""
        # Implement logic to determine if the entity corresponds exactly to a document chunk
        # This could involve comparing text spans, locations, etc.
        # For simplicity, return False if we're not sure
        return False
        
    def _get_entity_text(self, entity):
        """Extract searchable text representation from entity"""
        parts = [
            f"Type: {entity.get('type', 'unknown')}",
            f"Label: {entity.get('label', '')}"
        ]
        
        # Add properties
        for key, value in entity.get("properties", {}).items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
        
        return " ".join(parts)
        
    def _create_embedding(self, text):
        """Create embedding for text using SentenceTransformers"""
        if not text:
            return None
        
        try:
            # Use consistent model across the application
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
            return model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            handle_embedding_error(text, "sentence-transformers/all-MiniLM-L12-v2", e)
            return None
        
    def find_entities_for_document(self, document_path, filter_criteria=None):
        """
        Find all entities extracted from a specific document
        
        Args:
            document_path: Path to the document
            filter_criteria: Optional criteria to filter entities (e.g., by type)
            
        Returns:
            List of entities associated with the document
        """
        query = {"source_doc": document_path, "is_entity": True}
        
        # Add filter criteria if provided
        if filter_criteria:
            query.update(filter_criteria)
            
        results = self.vector_db.search(
            query="",
            filter=query,
            search_type="metadata"
        )
        
        # Get the full entity objects from the graph storage
        entities = []
        for result in results:
            entity_id = result.metadata.get("entity_id")
            if entity_id:
                # Check cache first
                if entity_id in self._entity_cache:
                    entities.append(self._entity_cache[entity_id])
                else:
                    # Query graph storage
                    entity = self.graph_storage.get_entity(entity_id)
                    if entity:
                        self._entity_cache[entity_id] = entity
                        entities.append(entity)
        
        return entities
        
    def find_documents_for_entity(self, entity_id):
        """
        Find all documents that an entity appears in
        
        Args:
            entity_id: ID of the entity to search for
            
        Returns:
            List of document references where the entity appears
        """
        results = self.vector_db.search(
            query="",
            filter={"entity_id": entity_id, "is_entity": True},
            search_type="metadata"
        )
        
        documents = []
        for result in results:
            doc_path = result.metadata.get("source_doc")
            if doc_path:
                documents.append({
                    "document_path": doc_path,
                    "location": result.metadata.get("source_location", ""),
                    "score": 1.0  # Default score for exact matches
                })
        
        return documents
    
    def search_similar_entities(self, query_text, top_k=10):
        """
        Find entities similar to a text query
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            
        Returns:
            List of entities similar to the query text
        """
        # Create query embedding
        query_embedding = self._create_embedding(query_text)
        
        # Search for similar entity embeddings
        results = self.vector_db.search(
            query_embedding=query_embedding,
            filter={"is_entity": True},
            k=top_k
        )
        
        # Get the full entity objects
        entities = []
        for result in results:
            entity_id = result.metadata.get("entity_id")
            if entity_id:
                # Get entity from graph storage
                entity = self.graph_storage.get_entity(entity_id)
                if entity:
                    # Add similarity score
                    entity["similarity_score"] = result.score
                    entities.append(entity)
        
        return entities 