import os
import logging
from django.conf import settings
from byaldi import RAGMultiModalModel
from .models import Evidence
import uuid
from langchain.memory import ConversationBufferMemory
import re
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGModelManager:
    """
    Manages RAG models for different chat sessions.
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.index_folder = getattr(settings, 'INDEX_FOLDER', os.path.join(os.getcwd(), '.byaldi'))
            os.makedirs(self.index_folder, exist_ok=True)
            self._initialized = True
            self.load_existing_indexes()
    
    def load_existing_indexes(self):
        """
        Loads all existing indexes from the .byaldi folder when the application starts.
        """
        if os.path.exists(self.index_folder):
            for session_id in os.listdir(self.index_folder):
                if os.path.isdir(os.path.join(self.index_folder, session_id)):
                    self.load_rag_model_for_session(session_id)
        else:
            logger.warning("No .byaldi folder found. No existing indexes to load.")
    
    def load_rag_model_for_session(self, session_id):
        """
        Loads the RAG model for the given session_id from the index on disk.
        """
        index_path = os.path.join(self.index_folder, session_id)
        
        if os.path.exists(index_path):
            try:
                rag = RAGMultiModalModel.from_index(index_path)
                self._models[session_id] = rag
                logger.info(f"RAG model for session {session_id} loaded from index.")
                return rag
            except Exception as e:
                logger.error(f"Error loading RAG model for session {session_id}: {e}")
                return None
        else:
            logger.warning(f"No index found for session {session_id}.")
            return None
    
    def get_model(self, session_id):
        """
        Gets the RAG model for the given session_id.
        If the model is not loaded, it will attempt to load it.
        """
        if session_id not in self._models:
            return self.load_rag_model_for_session(session_id)
        return self._models.get(session_id)
    
    def set_model(self, session_id, model):
        """
        Sets the RAG model for the given session_id.
        """
        self._models[session_id] = model
        return model
    
    def remove_model(self, session_id):
        """
        Removes the RAG model for the given session_id.
        """
        if session_id in self._models:
            del self._models[session_id]
            return True
        return False

class RAGManager:
    """
    Manages RAG models and processes for different chat sessions.
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.index_folder = getattr(settings, 'INDEX_FOLDER', os.path.join(os.getcwd(), '.byaldi'))
            os.makedirs(self.index_folder, exist_ok=True)
            self._initialized = True
            self.load_existing_indexes()
    
    def load_existing_indexes(self):
        """
        Loads all existing indexes from the .byaldi folder when the application starts.
        """
        if os.path.exists(self.index_folder):
            for session_id in os.listdir(self.index_folder):
                if os.path.isdir(os.path.join(self.index_folder, session_id)):
                    self.load_rag_model_for_session(session_id)
        else:
            logger.warning("No .byaldi folder found. No existing indexes to load.")
    
    def load_rag_model_for_session(self, session_id):
        """
        Loads the RAG model for the given session_id from the index on disk.
        """
        index_path = os.path.join(self.index_folder, session_id)
        
        if os.path.exists(index_path):
            try:
                rag = RAGMultiModalModel.from_index(index_path)
                self._models[session_id] = rag
                logger.info(f"RAG model for session {session_id} loaded from index.")
                return rag
            except Exception as e:
                logger.error(f"Error loading RAG model for session {session_id}: {e}")
                return None
        else:
            logger.warning(f"No index found for session {session_id}.")
            return None
    
    def get_model(self, session_id):
        """
        Gets the RAG model for the given session_id.
        If the model is not loaded, it will attempt to load it.
        """
        if session_id not in self._models:
            return self.load_rag_model_for_session(session_id)
        return self._models.get(session_id)
    
    def set_model(self, session_id, model):
        """
        Sets the RAG model for the given session_id.
        """
        self._models[session_id] = model
        return model
    
    def remove_model(self, session_id):
        """
        Removes the RAG model for the given session_id.
        """
        if session_id in self._models:
            del self._models[session_id]
            return True
        return False

    def _get_session_memory(self, session_id):
        """Get or create memory for a session"""
        if session_id not in self.memory_store:
            # Initialize memory with appropriate config
            self.memory_store[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.memory_store[session_id]

    async def generate_response(self, query, session_id, context=None):
        """Generate a response using the RAG process and track evidence"""
        # Get memory for this session
        memory = self._get_session_memory(session_id)
        
        # KEEP EXISTING CODE FOR QUERY ID GENERATION
        query_id = uuid.uuid4()
        
        # Get relevant documents
        relevant_docs = self.retrieve_documents(query)
        
        # KEEP EXISTING EVIDENCE TRACKING CODE
        for doc in relevant_docs:
            evidence = Evidence(
                query_id=query_id,
                source_type='document',
                content=doc.page_content,
                highlights=self._extract_highlights(doc, query)
            )
            evidence.save()
        
        # Format documents for context
        formatted_context = self._format_documents(relevant_docs)
        
        # Add history to context
        context_with_history = {
            "history": memory.chat_memory.messages,
            "formatted_context": formatted_context,
            **(context or {})
        }
        
        # Generate response with LLM using enhanced context
        response = await self.llm_service.generate(
            prompt=query,
            context=context_with_history,
            session_id=session_id,
            additional_context=context
        )
        
        # Track any generated evidence (like HyDE hypotheses)
        if hasattr(self.llm_service, 'generated_evidence'):
            for ev in self.llm_service.generated_evidence:
                evidence = Evidence(
                    query_id=query_id,
                    source_type='generated',
                    content=ev['content'],
                    highlights=ev.get('highlights', {})
                )
                evidence.save()

                # Add user query to memory
        memory.chat_memory.add_user_message(query)
        
        # Add response to memory
        memory.chat_memory.add_ai_message(response["answer"])
        
        
        # Keep existing response formatting with query_id
        response['query_id'] = str(query_id)
        
        return response
    
    def _extract_highlights(self, doc, query):
        """
        Extract relevant highlights from a document with confidence scores
        """
        highlights = []
        
        try:
            # Simple text matching for now (can be enhanced with NLP)
            content = doc.page_content.lower()
            query_terms = query.lower().split()
            
            for term in query_terms:
                if len(term) < 3:  # Skip very short terms
                    continue
                    
                start_idx = 0
                while True:
                    # Find the term in the content
                    pos = content.find(term, start_idx)
                    if pos == -1:
                        break
                        
                    # Get a window of context (50 chars before and after)
                    context_start = max(0, pos - 50)
                    context_end = min(len(content), pos + len(term) + 50)
                    
                    # Add to highlights with a basic confidence score
                    highlights.append({
                        'start': pos,
                        'end': pos + len(term),
                        'text': content[pos:pos + len(term)],
                        'context': content[context_start:context_end],
                        'confidence': 0.7  # Basic confidence score
                    })
                    
                    # Move to look for next occurrence
                    start_idx = pos + len(term)
        except Exception as e:
            print(f"Error extracting highlights: {e}")
        
        return highlights
    
    def _format_documents(self, documents):
        """
        Formats a list of documents into a single string for context
        """
        formatted_context = ""
        for doc in documents:
            formatted_context += f"{doc.page_content}\n\n"
        return formatted_context

    def retrieve_documents(self, query, session_id=None):
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The query text
            session_id: Optional session ID (defaults to "default")
            
        Returns:
            List of retrieved documents
        """
        # Get the RAG model for this session
        try:
            rag_model = self.get_model(session_id or "default")
            if not rag_model:
                logger.warning(f"No RAG model found for session {session_id}")
                return []
            
            # Use the model to retrieve documents
            documents = rag_model.get_relevant_documents(query)
            return documents
        except Exception as e:
            # Use our specialized error handler
            from error_handlers.models_app_errors import handle_rag_retrieval_error
            handle_rag_retrieval_error(query, session_id, e)
            return []

    def _extract_highlights(self, doc, query):
        """
        Extract relevant highlights from a document with confidence scores
        """
        highlights = []
        
        try:
            # Simple text matching for now (can be enhanced with NLP)
            content = doc.page_content.lower()
            query_terms = query.lower().split()
            
            for term in query_terms:
                if len(term) < 3:  # Skip very short terms
                    continue
                    
                start_idx = 0
                while True:
                    # Find the term in the content
                    pos = content.find(term, start_idx)
                    if pos == -1:
                        break
                        
                    # Get a window of context (50 chars before and after)
                    context_start = max(0, pos - 50)
                    context_end = min(len(content), pos + len(term) + 50)
                    
                    # Add to highlights with a basic confidence score
                    highlights.append({
                        'start': pos,
                        'end': pos + len(term),
                        'text': content[pos:pos + len(term)],
                        'context': content[context_start:context_end],
                        'confidence': 0.7  # Basic confidence score
                    })
                    
                    # Move to look for next occurrence
                    start_idx = pos + len(term)
        except Exception as e:
            print(f"Error extracting highlights: {e}")
        
        return highlights

    def _format_documents(self, documents):
        """
        Formats a list of documents into a single string for context
        """
        formatted_context = ""
        for doc in documents:
            formatted_context += f"{doc.page_content}\n\n"
        return formatted_context

    def rerank_results(self, query, documents, top_k=10, model="BAAI/bge-reranker-base"):
        """Re-rank results with a cross-encoder model"""
        logger.info(f"Re-ranking results for query: {query}")
        
        try:
            # Import cross-encoder
            from sentence_transformers import CrossEncoder
            
            # Initialize cross-encoder
            cross_encoder = CrossEncoder(model)
            
            # Prepare pairs for re-ranking
            pairs = []
            items = []
            
            # Extract text from documents
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    text = doc.page_content
                elif isinstance(doc, dict) and 'content' in doc:
                    text = doc['content']
                elif isinstance(doc, dict) and 'snippet' in doc:
                    text = doc['snippet']
                else:
                    text = str(doc)
                    
                pairs.append([query, text])
                items.append(doc)
            
            if not pairs:
                logger.warning("No valid items for re-ranking")
                return documents
                
            # Get scores from cross-encoder
            scores = cross_encoder.predict(pairs)
            
            # Create (document, score) tuples and sort
            scored_documents = [(doc, score) for doc, score in zip(items, scores)]
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k documents
            return [doc for doc, _ in scored_documents[:top_k]]
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {e}")
            # Fall back to original results on error
            return documents[:top_k]

    async def generate_kg_enhanced_response(self, query, session_id, graph_id=None):
        """Generate a response enhanced with knowledge graph data"""
        
        # If we have a graph ID, use knowledge graph for retrieval
        if graph_id:
            documents = self.retrieve_with_knowledge_graph(query, graph_id)
        else:
            # Fall back to regular vector retrieval
            documents = self.retrieve_documents(query)
        
        # Generate response with retrieved context
        response = await self._generate_response_with_context(query, documents)
        
        return response

    def retrieve_with_knowledge_graph(self, query, graph_id):
        """
        Enhance retrieval with knowledge graph connections
        
        Args:
            query: User query
            graph_id: ID of the knowledge graph to use
            
        Returns:
            List of retrieved documents
        """
        # Get knowledge graph components
        from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
        kg_manager = KnowledgeGraphManager()
        
        # Get graph data
        graph = kg_manager.retrieve_graph(graph_id)
        if not graph:
            # Fall back to standard retrieval if graph not found
            return self.retrieve_documents(query)
        
        # Using knowledge graph LLM interface to extract relevant graph data
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        
        # Extract relevant subgraph based on query
        relevant_graph_data = kg_interface._extract_relevant_graph_data(query, graph)
        relevant_entities = relevant_graph_data.get("entities", [])
        
        # Get bidirectional indexer
        from models_app.document_indexer import BidirectionalIndexer
        bidirectional_indexer = BidirectionalIndexer(
            kg_manager.graph_storage, 
            self.vector_store
        )
        
        # Find documents linked to relevant entities
        kg_documents = []
        for entity in relevant_entities:
            # Get documents for this entity
            entity_docs = bidirectional_indexer.find_documents_for_entity(entity["id"])
            if entity_docs:
                for doc in entity_docs:
                    kg_documents.append({
                        "document_path": doc["document_path"],
                        "relevance_score": 0.9,  # High score for KG-derived docs
                        "source": "knowledge_graph"
                    })
        
        # Combine with vector search results
        vector_docs = self.retrieve_documents(query)
        
        # Merge results, prioritizing KG-derived documents
        final_docs = self._merge_ranked_documents(kg_documents, vector_docs)
        
        return final_docs

    def _merge_ranked_documents(self, kg_docs, vector_docs, kg_weight=0.7, vector_weight=0.3):
        """
        Merge and rank documents from multiple sources
        
        Args:
            kg_docs: Documents from knowledge graph
            vector_docs: Documents from vector search
            kg_weight: Weight for KG-derived documents
            vector_weight: Weight for vector search documents
            
        Returns:
            Merged and ranked list of documents
        """
        # Create mapping to remove duplicates and combine scores
        doc_map = {}
        
        # Process KG documents
        for doc in kg_docs:
            doc_id = doc["document_path"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document": doc,
                    "kg_score": doc.get("relevance_score", 0.0) * kg_weight,
                    "vector_score": 0.0
                }
            else:
                # Update with higher score if found
                current_kg_score = doc_map[doc_id]["kg_score"]
                new_kg_score = doc.get("relevance_score", 0.0) * kg_weight
                doc_map[doc_id]["kg_score"] = max(current_kg_score, new_kg_score)
        
        # Process vector documents
        for i, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("source")
            # Calculate decreasing score based on position
            vector_score = (1.0 - (i/len(vector_docs))) * vector_weight
            
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document": doc,
                    "kg_score": 0.0,
                    "vector_score": vector_score
                }
            else:
                # Update vector score
                doc_map[doc_id]["vector_score"] = vector_score
            
        # Calculate combined scores and sort
        scored_docs = []
        for doc_id, info in doc_map.items():
            combined_score = info["kg_score"] + info["vector_score"]
            scored_docs.append((info["document"], combined_score))
        
        # Sort by descending score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return documents only
        return [doc for doc, _ in scored_docs]

    def hybrid_search(self, query, graph_id=None, vector_weight=0.7, graph_weight=0.3):
        """
        Perform hybrid search combining vector DB and knowledge graph
        
        Args:
            query: User query
            graph_id: ID of knowledge graph to use
            vector_weight: Weight for vector similarity (0-1)
            graph_weight: Weight for graph relevance (0-1)
        """
        # Get vector search results
        vector_results = self.vector_store.similarity_search(query, k=10)
        
        if not graph_id:
            return vector_results
        
        # Get knowledge graph components
        from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
        kg_manager = KnowledgeGraphManager()
        
        # Get graph
        graph = kg_manager.graph_storage.retrieve_graph(graph_id)
        
        # Extract entities from query
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        query_analysis = kg_interface.analyze_query(query)
        
        # Find relevant entities in graph
        relevant_entities = self._find_relevant_entities(graph, query_analysis["entities"])
        
        # Find documents related to these entities
        from models_app.document_indexer import BidirectionalIndexer
        bidirectional_indexer = BidirectionalIndexer(kg_manager.graph_storage, self.vector_store)
        
        graph_results = []
        for entity in relevant_entities:
            entity_docs = bidirectional_indexer.find_documents_for_entity(entity["id"])
            graph_results.extend(entity_docs)
        
        # Score and combine results
        scored_results = []
        all_docs = vector_results + graph_results
        
        # Remove duplicates
        doc_map = {}
        for doc in all_docs:
            if doc.metadata.get("doc_id") not in doc_map:
                doc_map[doc.metadata.get("doc_id")] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "graph_score": 0.0
                }
        
        # Assign vector scores
        for i, doc in enumerate(vector_results):
            if doc.metadata.get("doc_id") in doc_map:
                # Score based on position in results (10-i)/10
                doc_map[doc.metadata.get("doc_id")]["vector_score"] = (10 - i) / 10
        
        # Assign graph scores
        for i, doc in enumerate(graph_results):
            if doc.metadata.get("doc_id") in doc_map:
                # Score based on position in results (10-i)/10
                doc_map[doc.metadata.get("doc_id")]["graph_score"] = (10 - i) / 10
        
        # Calculate combined scores
        for doc_id, info in doc_map.items():
            combined_score = (info["vector_score"] * vector_weight) + (info["graph_score"] * graph_weight)
            scored_results.append((info["doc"], combined_score))
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return [doc for doc, _ in scored_results[:10]] 

    def enhanced_retrieval(self, query, session_id=None, top_k=20, rerank_k=5):
        """
        Enhanced retrieval with multiple techniques
        
        Args:
            query: User query
            session_id: Session ID for vectorstore access
            top_k: Number of results to retrieve initially
            rerank_k: Number of results after reranking
        
        Returns:
            List of enhanced retrieval results
        """
        logger.info(f"Performing enhanced retrieval for query: {query}")
        
        # Get RAG model for this session
        rag_model = self.get_model(session_id or "default")
        if not rag_model or not hasattr(rag_model, 'vectorstore'):
            logger.warning(f"No valid RAG model/vectorstore found for session {session_id}")
            return []
        
        # 1. Generate multiple query variations
        query_variations = self._generate_query_variations(query)
        
        # 2. Perform dense retrieval for each query variation
        all_results = []
        for q in query_variations:
            results = rag_model.vectorstore.similarity_search(q, k=top_k//len(query_variations))
            all_results.extend(results)
        
        # 3. Remove duplicates
        unique_results = self._deduplicate_results(all_results)
        
        # 4. Re-rank with cross-encoder
        reranked_results = self._rerank_results(query, unique_results)
        
        # 5. Apply KG enhancement if available
        try:
            kg_enhanced_results = self._enhance_with_knowledge_graph(query, reranked_results[:rerank_k])
            return kg_enhanced_results
        except Exception as e:
            logger.warning(f"KG enhancement failed: {str(e)}, returning reranked results")
            return reranked_results[:rerank_k]

    def _generate_query_variations(self, query, n_variations=3):
        """Generate variations of the query for multi-query retrieval"""
        try:
            # Use KG interface for query generation
            from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
            kg_interface = KnowledgeGraphLLMInterface()
            
            # Generate prompt for variations
            prompt = f"""Generate {n_variations} alternative search queries that would help answer this question.
            Each query should focus on a different aspect of the question.
            
            Original query: {query}
            
            Alternative queries (numbered):"""
            
            # Get LLM provider and generate text
            llm_provider = kg_interface.provider_factory.get_provider("default")
            variations_text, _ = llm_provider.generate_text(prompt, max_tokens=200)
            
            # Parse variations
            variations = []
            for line in variations_text.strip().split('\n'):
                # Remove numbering and clean up
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if clean_line and len(clean_line) > 5 and clean_line != query:
                    variations.append(clean_line)
            
            # Add original query and return
            return [query] + variations[:n_variations]
        except Exception as e:
            logger.warning(f"Error generating query variations: {e}")
            return [query]  # Fall back to just the original query

    def _deduplicate_results(self, results):
        """Remove duplicate documents based on content hash"""
        unique_docs = {}
        
        for doc in results:
            # Create content hash
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict) and 'content' in doc:
                content = doc['content']
            else:
                content = str(doc)
            
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Only keep if we haven't seen this content before
            if content_hash not in unique_docs:
                unique_docs[content_hash] = doc
        
        return list(unique_docs.values())

    def _rerank_results(self, query, documents, model="BAAI/bge-reranker-base"):
        """Re-rank results using a cross-encoder"""
        try:
            from sentence_transformers import CrossEncoder
            
            # Initialize cross-encoder
            cross_encoder = CrossEncoder(model)
            
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    text = doc.page_content
                elif isinstance(doc, dict) and 'content' in doc:
                    text = doc['content']
                else:
                    text = str(doc)
                
                pairs.append([query, text])
            
            # Get scores
            scores = cross_encoder.predict(pairs)
            
            # Create (document, score) tuples and sort
            scored_docs = [(doc, score) for doc, score in zip(documents, scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return just the documents
            return [doc for doc, _ in scored_docs]
        except Exception as e:
            logger.warning(f"Error in re-ranking: {e}")
            return documents

    def _enhance_with_knowledge_graph(self, query, documents):
        """Enhance retrieval with knowledge graph connections"""
        try:
            # Get KG manager
            from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
            kg_manager = KnowledgeGraphManager()
            
            # Get KG interface
            from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
            kg_interface = KnowledgeGraphLLMInterface()
            
            # Extract entities from documents
            from models_app.knowledge_graph.entity_extractor import EntityExtractor
            extractor = EntityExtractor()
            
            # Extract entities from each document
            all_entities = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, dict) and 'content' in doc:
                    content = doc['content']
                else:
                    content = str(doc)
                
                entities = extractor.extract_entities(content)
                all_entities.extend(entities)
            
            # Find related entities in knowledge graph
            related_entities = []
            for entity in all_entities:
                entity_label = entity.get("label", "")
                if entity_label:
                    similar_entities = kg_manager.graph_storage.find_similar_entities(entity_label, limit=3)
                    related_entities.extend(similar_entities)
            
            # Get documents related to these entities
            kg_documents = []
            for entity in related_entities:
                # Use bidirectional indexer to find related documents
                from models_app.document_indexer import BidirectionalIndexer
                bi_indexer = BidirectionalIndexer(kg_manager.graph_storage, self.vector_store)
                
                entity_docs = bi_indexer.find_documents_for_entity(entity.get("id"))
                kg_documents.extend(entity_docs)
            
            # Combine original documents with KG-derived documents
            all_documents = documents + kg_documents
            
            # Remove duplicates and return
            return self._deduplicate_results(all_documents)
        except Exception as e:
            logger.warning(f"Error in KG enhancement: {e}")
            return documents

    def bidirectional_search(self, query, session_id=None, top_k=10):
        """
        Search using bidirectional encoders for multilingual queries
        
        Args:
            query: User query
            session_id: Session ID
            top_k: Number of results to return
        """
        # Get RAG model
        rag_model = self.get_model(session_id or "default")
        if not rag_model:
            logger.warning(f"No RAG model for session {session_id}")
            return []
        
        try:
            # Initialize bidirectional encoder 
            from sentence_transformers import SentenceTransformer
            bi_encoder = SentenceTransformer("all-MiniLM-L12-v2")
            
            # Encode query
            query_embedding = bi_encoder.encode(query)
            
            # Use vector store similarity search
            results = rag_model.vectorstore.similarity_search_by_vector(
                embedding=query_embedding,
                k=top_k
            )
            
            return results
        except Exception as e:
            logger.warning(f"Bidirectional search error: {e}")
            return [] 

    def r3_reasoning(self, query, session_id=None):
        """
        Apply R³ reasoning to RAG results
        
        Args:
            query: User query
            session_id: Session ID
        """
        # 1. Retrieval - get documents
        documents = self.retrieve_documents(query, session_id)
        
        # 2 & 3. Reading & Reasoning
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        
        # Format documents for R³
        search_results = {
            'documents': documents
        }
        
        # Apply R³ reasoning
        reasoned_response = kg_interface.perform_r3_reasoning(query, search_results)
        
        return reasoned_response 

    def multimodal_rag_search(self, query, session_id=None):
        """
        Perform multimodal search using RAG and M-Retriever
        
        Args:
            query: Text query or image path
            session_id: Session ID
        """
        # Initialize M-Retriever
        from models_app.multimodal.m_retriever import MultimodalRetriever
        retriever = MultimodalRetriever()
        
        # Get RAG model
        rag_model = self.get_model(session_id or "default")
        if not rag_model:
            return []
        
        # Get text documents from RAG
        text_documents = self.retrieve_documents(query, session_id)
        text_contents = [doc.page_content for doc in text_documents]
        
        # Get image documents
        from models_app.models import UploadedFile
        image_files = UploadedFile.objects.filter(session_id=session_id)
        image_paths = [img.file.path for img in image_files if hasattr(img.file, 'path')]
        
        # Combine all items
        search_items = text_contents + image_paths
        
        # Perform multimodal search
        results = retriever.similarity_search(query, search_items)
        
        return results 

    def get_embeddings(self, session_id=None):
        """Get embeddings model for the session"""
        return SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    def update_retrieval_methods(self, model, query, session_id=None):
        """Update retrieval to use bidirectional encoders"""
        # Get embeddings model
        embeddings_model = self.get_embeddings(session_id)
        
        # Use consistent embedding approach
        query_embedding = embeddings_model.encode(query)
        
        # Search with embedding
        return model.search_by_embedding(query_embedding) 