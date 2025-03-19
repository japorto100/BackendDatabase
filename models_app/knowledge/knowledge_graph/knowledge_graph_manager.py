"""
KnowledgeGraphManager: Central orchestrator for knowledge graph construction.
Coordinates document processing, entity extraction, and graph building.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import time
import threading
import concurrent.futures
from queue import Queue

from models_app.vision.document.factory.document_adapter_registry import DocumentAdapterRegistry
from models_app.knowledge_graph.entity_extractor import EntityExtractor
from models_app.knowledge_graph.relationship_detector import RelationshipDetector
from models_app.knowledge_graph.graph_builder import GraphBuilder
from models_app.knowledge_graph.graph_storage import GraphStorage
from models_app.vision.knowledge_graph.document_entity_extractor import DocumentEntityExtractor
from models_app.vision.knowledge_graph.visual_entity_extractor import VisualEntityExtractor
from models_app.vision.knowledge_graph.hybrid_entity_extractor import HybridEntityExtractor
from models_app.knowledge_graph.graph_visualization import GraphVisualization
from models_app.document_indexer import BidirectionalIndexer
from analytics_app.utils import monitor_kg_performance
from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
from models_app.next_layer_interface import NextLayerInterface


logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """
    Central manager for knowledge graph operations.
    Orchestrates the entire process from document processing to graph construction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.document_registry = DocumentAdapterRegistry()
        
        # Entity extractors
        self.document_entity_extractor = DocumentEntityExtractor()
        self.visual_entity_extractor = VisualEntityExtractor()
        self.hybrid_entity_extractor = HybridEntityExtractor()
        
        # Core knowledge graph components
        self.relationship_detector = RelationshipDetector()
        self.graph_builder = GraphBuilder()
        self.graph_storage = GraphStorage()
        
        # Initialize graph visualization
        self.visualization = GraphVisualization()
        
        # Document tracking
        self.document_graph_map = {}  # Maps document IDs to graph IDs
        
        # Add caching
        self._cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    @monitor_kg_performance
    def process_document_to_graph(self, document_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a document to create a knowledge graph
        """
        logger.info(f"Processing document to graph: {document_path}")
        
        # Get appropriate document adapter
        document_adapter = self.document_registry.get_adapter_for_document(document_path)
        if not document_adapter:
            raise ValueError(f"No suitable document adapter found for {document_path}")
            
        # Prepare document for extraction
        logger.debug(f"Preparing document for extraction: {document_path}")
        extraction_data = document_adapter.prepare_for_extraction(document_path, **kwargs)
        
        # Select appropriate entity extractor based on document characteristics
        if extraction_data.get("has_visual_elements") and extraction_data.get("has_text"):
            logger.debug(f"Using hybrid entity extractor for {document_path}")
            entity_extractor = self.hybrid_entity_extractor
        elif extraction_data.get("has_visual_elements"):
            logger.debug(f"Using visual entity extractor for {document_path}")
            entity_extractor = self.visual_entity_extractor
        else:
            logger.debug(f"Using document entity extractor for {document_path}")
            entity_extractor = self.document_entity_extractor
            
        # Extract entities
        logger.debug(f"Extracting entities from {document_path}")
        entities = entity_extractor.extract_from_document(extraction_data)
        
        # Detect relationships
        logger.debug(f"Detecting relationships for {document_path}")
        relationships = self.relationship_detector.detect_relationships(
            entities, 
            context=extraction_data.get("metadata")
        )
        
        # Build graph
        logger.debug(f"Building knowledge graph for {document_path}")
        graph = self.graph_builder.build_subgraph(
            entities, 
            relationships,
            context={
                "document_id": extraction_data.get("document_id"),
                "document_path": document_path,
                "document_type": extraction_data.get("document_type"),
                "metadata": extraction_data.get("metadata")
            }
        )
        
        # Store graph
        logger.debug(f"Storing knowledge graph for {document_path}")
        graph_id = self.graph_storage.store_graph(graph)
        
        # Track document-graph mapping
        self.document_graph_map[extraction_data.get("document_id")] = graph_id
        
        # Add bidirectional indexing
        bidirectional_indexer = BidirectionalIndexer(self.graph_storage, self._get_vector_db())
        
        # Index each entity with its source information
        for entity in entities:
            source_info = {
                "document_id": extraction_data.get("document_id"),
                "document_path": document_path,
                "location": entity.get("location", {}),
                "extraction_time": datetime.now().isoformat()
            }
            bidirectional_indexer.index_entity_sources(entity, source_info)
        
        logger.info(f"Document processed to graph: {document_path} -> Graph ID: {graph_id}")
        
        return {
            "graph_id": graph_id,
            "document_id": extraction_data.get("document_id"),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "graph_summary": self._generate_graph_summary(graph)
        }
        
    def _generate_graph_summary(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the graph"""
        entity_types = {}
        for entity in graph.get("entities", []):
            entity_type = entity.get("type")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
        relationship_types = {}
        for rel in graph.get("relationships", []):
            rel_type = rel.get("type")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
        return {
            "entity_count": len(graph.get("entities", [])),
            "relationship_count": len(graph.get("relationships", [])),
            "entity_types": entity_types,
            "relationship_types": relationship_types
        }
    
    def batch_process_documents(self, document_paths: List[str], **kwargs) -> Dict[str, Any]:
        """Process multiple documents and build a combined knowledge graph."""
        # Process each document to get individual graphs
        graphs = []
        for doc_path in document_paths:
            result = self.process_document_to_graph(doc_path, **kwargs)
            graphs.append(result["graph"])
        
        # Merge the individual graphs
        merged_graph = self.graph_builder.merge_subgraphs(graphs)
        
        # Store merged graph if requested
        graph_id = None
        if kwargs.get("store_graph", False):
            graph_id = self.graph_storage.store_graph(merged_graph)
        
        return {
            "graph": merged_graph,
            "graph_id": graph_id,
            "metadata": {
                "document_count": len(document_paths),
                "processing_timestamp": datetime.now().isoformat()
            }
        }
    
    def query_graph(self, query: str, graph_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Query the knowledge graph.
        
        Args:
            query: Query string or pattern
            graph_id: ID of the graph to query (if None, queries the default graph)
            **kwargs: Additional query parameters
            
        Returns:
            Query results
        """
        if graph_id:
            # Retrieve specific graph
            graph = self.graph_storage.retrieve_graph(graph_id)
        else:
            # Use default graph or create query parameters
            query_params = {"query": query, **kwargs}
            return self.graph_storage.query_graph(query_params)
        
        # Implement more sophisticated query processing here
        
        return {
            "results": [],  # Placeholder for actual results
            "metadata": {
                "query": query,
                "graph_id": graph_id,
                "timestamp": datetime.now().isoformat()
            }
        }

    def retrieve_graph(self, graph_id):
        """
        Retrieve a graph with caching
        """
        # Check cache first
        if graph_id in self._cache:
            cache_entry = self._cache[graph_id]
            if time.time() - cache_entry['timestamp'] < self.cache_timeout:
                return cache_entry['data']
        
        # Get from storage
        graph = self.graph_storage.retrieve_graph(graph_id)
        
        # Cache result
        if graph:
            self._cache[graph_id] = {
                'data': graph,
                'timestamp': time.time()
            }
        
        return graph
    
    def retrieve_paginated_graph(self, graph_id, page=1, page_size=100):
        """
        Retrieve a paginated view of a large graph
        
        Args:
            graph_id: The graph ID
            page: Page number (1-indexed)
            page_size: Number of entities per page
            
        Returns:
            Dict with paginated graph data
        """
        # Get complete graph
        full_graph = self.retrieve_graph(graph_id)
        if not full_graph:
            return None
            
        # Extract entities for this page
        all_entities = full_graph.get("entities", [])
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_entities = all_entities[start_idx:end_idx] if start_idx < len(all_entities) else []
        
        # Get entity IDs for this page
        entity_ids = [entity["id"] for entity in paginated_entities]
        
        # Filter relationships to only those connecting entities on this page
        relationships = [
            rel for rel in full_graph.get("relationships", [])
            if rel["source"] in entity_ids and rel["target"] in entity_ids
        ]
        
        # Create paginated graph
        paginated_graph = {
            "entities": paginated_entities,
            "relationships": relationships,
            "pagination": {
                "total_entities": len(all_entities),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(all_entities) + page_size - 1) // page_size
            }
        }
        
        return paginated_graph

    def _get_vector_db(self):
        """Get the vector database for entity indexing"""
        # If we don't have a vector db instance yet, initialize it
        if not hasattr(self, "_vector_db"):
            from models_app.rag_manager import RAGModelManager
            rag_manager = RAGModelManager()
            
            # Use the default model's vector store, or create one if needed
            default_model = rag_manager.get_model("default")
            if default_model:
                self._vector_db = default_model.vector_store
            else:
                # Create a new vector store if needed
                from langchain.vectorstores import FAISS
                from langchain.embeddings import HuggingFaceEmbeddings
                
                # Use the same embedding model as the RAG system
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
                self._vector_db = FAISS.from_texts(
                    texts=["Knowledge Graph Initialization"], 
                    embedding=embeddings
                )
        
        return self._vector_db

    def enhance_entities_with_external_kb(self, entities, enrich_threshold=0.7):
        """
        Enhance entities with information from external knowledge bases
        
        Args:
            entities: List of entities to enhance
            enrich_threshold: Confidence threshold for enrichment
            
        Returns:
            List of enhanced entities
        """
        from models_app.knowledge_graph.external_kb_connector import CascadingKBConnector
        
        # Initialize the cascading connector with priority for SwissAL and Wikidata
        kb_connector = CascadingKBConnector({
            "primary_connector": "swiss_al",
            "fallback_connectors": ["wikidata", "gnd", "dbpedia_german"],
            "confidence_threshold": 0.6
        })
        
        enhanced_entities = []
        
        for entity in entities:
            # Link entity to external KBs
            external_matches = kb_connector.link_entity(entity)
            
            if external_matches and len(external_matches) > 0:
                best_match = external_matches[0]
                
                # Only enrich if confidence is above threshold
                if best_match["confidence"] >= enrich_threshold:
                    # Format external ID with KB source
                    external_id = f"{best_match['source']}:{best_match['external_id']}"
                    
                    # Enrich entity with KB data
                    enriched_entity = kb_connector.enrich_entity(entity, external_id)
                    enhanced_entities.append(enriched_entity)
                else:
                    # Add external matches as references without full enrichment
                    if "external_references" not in entity:
                        entity["external_references"] = []
                    
                    for match in external_matches:
                        if match["confidence"] >= 0.5:  # Lower threshold for references
                            entity["external_references"].append({
                                "source": match["source"],
                                "id": match["external_id"],
                                "label": match["external_label"],
                                "confidence": match["confidence"]
                            })
                
                    enhanced_entities.append(entity)
            else:
                # No matches found, keep original entity
                enhanced_entities.append(entity)
        
        return enhanced_entities

    def process_documents_to_graph_parallel(self, document_paths, max_workers=4, **kwargs):
        """
        Process multiple documents to a knowledge graph in parallel.
        
        Args:
            document_paths: List of paths to documents to process
            max_workers: Maximum number of parallel workers
            **kwargs: Additional arguments for document processing
            
        Returns:
            The merged knowledge graph
        """
        # Input validation
        if not document_paths:
            logger.warning("No document paths provided for parallel processing")
            return None
        
        # Use ThreadPoolExecutor for parallel processing
        graphs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit document processing tasks
            future_to_path = {
                executor.submit(self.process_document_to_graph, path, **kwargs): path 
                for path in document_paths
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                document_path = future_to_path[future]
                try:
                    graph = future.result()
                    if graph:
                        graphs.append(graph)
                        logger.info(f"Successfully processed document: {document_path}")
                    else:
                        logger.warning(f"Document processing resulted in no graph: {document_path}")
                except Exception as e:
                    logger.error(f"Error processing document {document_path}: {str(e)}")
        
        # If no graphs were generated, return None
        if not graphs:
            logger.warning("No graphs were generated from parallel document processing")
            return None
        
        # Merge all graphs into one
        if len(graphs) == 1:
            return graphs[0]
        
        # Use graph builder to merge graphs
        merged_graph = self.graph_builder.merge_subgraphs(graphs)
        
        # Store the merged graph
        merged_graph_id = self.graph_storage.store_graph(merged_graph)
        merged_graph["id"] = merged_graph_id
        
        logger.info(f"Successfully merged {len(graphs)} graphs from parallel document processing")
        return merged_graph

    def semantic_search(self, query_text, graph_id=None):
        """
        Perform semantic search on a knowledge graph using embeddings.
        
        Args:
            query_text: The search query text
            graph_id: Optional ID of the graph to search (if None, uses the most recent)
        
        Returns:
            Dict with matched entities and their similarity scores
        """
        logger.info(f"Performing semantic search for: {query_text}")
        
        try:
            # Retrieve the graph data
            graph_data = self.retrieve_graph(graph_id) if graph_id else self._get_latest_graph()
            if not graph_data:
                logger.warning(f"No graph found for ID: {graph_id}")
                return {"entities": [], "relationships": []}
            
            # Use KG interface for semantic search
            kg_interface = KnowledgeGraphLLMInterface()
            
            # Perform semantic search
            relevant_entities = kg_interface.semantic_graph_query(query_text, graph_data)
            
            # If we have relevant entities, also get their relationships
            relevant_entity_ids = [entity.get("id") for entity in relevant_entities]
            relevant_relationships = []
            
            if relevant_entity_ids:
                # Filter relationships involving these entities
                for rel in graph_data.get("relationships", []):
                    if rel.get("source") in relevant_entity_ids or rel.get("target") in relevant_entity_ids:
                        relevant_relationships.append(rel)
            
            return {
                "entities": relevant_entities,
                "relationships": relevant_relationships
            }
        except Exception as e:
            # Use our specialized error handler
            from error_handlers.models_app_errors import handle_kg_search_error
            handle_kg_search_error(query_text, graph_id, e)
            return {"entities": [], "relationships": [], "error": str(e)}

class KGFeedbackManager:
    """Manages feedback from knowledge graph to document processing."""
    
    def __init__(self):
        self.kg_manager = None  # Will be initialized on first use
        self.next_layer = NextLayerInterface.get_instance()
        
    def initialize(self):
        """Initialize the knowledge graph manager."""
        if not self.kg_manager:
            from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
            self.kg_manager = KnowledgeGraphManager()
    
    def get_related_entities(self, entity_text, entity_type=None, limit=10):
        """Get entities related to the provided entity."""
        self.initialize()
        return self.kg_manager.get_related_entities(entity_text, entity_type, limit)
    
    def get_document_context(self, document_id):
        """Get contextual information about a document from the KG."""
        self.initialize()
        entities = self.kg_manager.get_document_entities(document_id)
        relationships = self.kg_manager.get_document_relationships(document_id)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "document_id": document_id
        }
    
    def enhance_document_metadata(self, metadata_context):
        """Enhance document metadata using knowledge graph insights."""
        self.initialize()
        
        # Get document ID
        document_id = metadata_context.document_metadata.get("processing_id")
        if not document_id:
            return False
            
        # Get existing entity mentions from metadata
        entities = []
        if metadata_context.has_analysis_result("entities"):
            entities = metadata_context.get_analysis_result("entities")
            
        # Nothing to enhance if no entities found
        if not entities:
            return False
            
        # Get related entities and relationships from KG
        kg_context = {}
        for entity in entities:
            related = self.get_related_entities(entity.get("text"), entity.get("type"))
            if related:
                if entity.get("text") not in kg_context:
                    kg_context[entity.get("text")] = []
                kg_context[entity.get("text")].extend(related)
        
        # Add KG context to metadata
        if kg_context:
            metadata_context.record_analysis_result(
                "kg_context", 
                kg_context,
                component="KGFeedbackManager"
            )
            return True
            
        return False
