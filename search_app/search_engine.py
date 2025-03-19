import os
import json
import logging
from django.conf import settings
from django.db.models import Q
from chat_app.models import Message, ChatSession
from models_app.models import UploadedFile

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Search engine for finding messages, files, and other content
    """
    def __init__(self):
        self.index = {}
        self.initialized = False
    
    def initialize(self):
        """Initialize the search engine"""
        if self.initialized:
            return
        
        # In a real implementation, this would build a search index
        # For now, we'll just set a flag
        self.initialized = True
        logger.info("Search engine initialized")
    
    def search(self, query, user=None, filters=None):
        """
        Search for content matching the query
        
        Args:
            query: Search query string
            user: User performing the search (for access control)
            filters: Dictionary of filters to apply
            
        Returns:
            Dictionary with search results
        """
        self.initialize()
        
        if not filters:
            filters = {}
        
        # Default filters
        content_types = filters.get('content_types', ['messages', 'files', 'sessions'])
        date_range = filters.get('date_range', None)
        
        results = {
            'messages': [],
            'files': [],
            'sessions': [],
            'total_count': 0
        }
        
        # Search messages
        if 'messages' in content_types:
            message_results = self._search_messages(query, user, date_range)
            results['messages'] = message_results
            results['total_count'] += len(message_results)
        
        # Search files
        if 'files' in content_types:
            file_results = self._search_files(query, user, date_range)
            results['files'] = file_results
            results['total_count'] += len(file_results)
        
        # Search sessions
        if 'sessions' in content_types:
            session_results = self._search_sessions(query, user, date_range)
            results['sessions'] = session_results
            results['total_count'] += len(session_results)
        
        return results
    
    def _search_messages(self, query, user, date_range=None):
        """Search for messages matching the query"""
        # Build base query
        message_query = Q(content__icontains=query)
        
        # Add user filter if provided
        if user:
            message_query &= Q(session__user=user)
        
        # Add date range filter if provided
        if date_range and 'start' in date_range:
            message_query &= Q(timestamp__gte=date_range['start'])
        if date_range and 'end' in date_range:
            message_query &= Q(timestamp__lte=date_range['end'])
        
        # Execute query
        messages = Message.objects.filter(message_query).order_by('-timestamp')[:20]
        
        # Format results
        results = []
        for msg in messages:
            results.append({
                'id': str(msg.id),
                'content': msg.content,
                'role': msg.role,
                'timestamp': msg.timestamp.isoformat(),
                'session_id': str(msg.session.id),
                'session_title': msg.session.title
            })
        
        return results
    
    def _search_files(self, query, user, date_range=None):
        """Search for files matching the query"""
        # Build base query
        file_query = Q(file__icontains=query) | Q(processing_results__icontains=query)
        
        # Add user filter if provided
        if user:
            file_query &= Q(user=user)
        
        # Add date range filter if provided
        if date_range and 'start' in date_range:
            file_query &= Q(upload_date__gte=date_range['start'])
        if date_range and 'end' in date_range:
            file_query &= Q(upload_date__lte=date_range['end'])
        
        # Execute query
        files = UploadedFile.objects.filter(file_query).order_by('-upload_date')[:20]
        
        # Format results
        results = []
        for file in files:
            results.append({
                'id': str(file.id),
                'file_name': os.path.basename(file.file.name),
                'file_type': file.file_type,
                'upload_date': file.upload_date.isoformat(),
                'processed': file.processed,
                'url': file.file.url if hasattr(file.file, 'url') else None
            })
        
        return results
    
    def _search_sessions(self, query, user, date_range=None):
        """Search for chat sessions matching the query"""
        # Build base query
        session_query = Q(title__icontains=query)
        
        # Add user filter if provided
        if user:
            session_query &= Q(user=user)
        
        # Add date range filter if provided
        if date_range and 'start' in date_range:
            session_query &= Q(created_at__gte=date_range['start'])
        if date_range and 'end' in date_range:
            session_query &= Q(created_at__lte=date_range['end'])
        
        # Execute query
        sessions = ChatSession.objects.filter(session_query).order_by('-created_at')[:20]
        
        # Format results
        results = []
        for session in sessions:
            results.append({
                'id': str(session.id),
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'message_count': Message.objects.filter(session=session).count()
            })
        
        return results
    
    def deep_seek_search(self, query, user=None, filters=None):
        """Implements the Deep-Seek 4-step search process"""
        
        # 1. PLAN - Analyze the query and create a search plan
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        search_plan = kg_interface.create_search_plan(query)
        
        # 2. SEARCH - Execute the search across multiple sources
        results = {}
        for search_step in search_plan["steps"]:
            step_results = self.search(
                search_step["query"],
                user=user,
                filters={**filters, **search_step.get("filters", {})}
            )
            results[search_step["id"]] = step_results
        
        # 3. EXTRACT - Extract entities and relationships from results
        from models_app.knowledge_graph.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        
        extracted_entities = []
        for step_id, step_results in results.items():
            for result_type, items in step_results.items():
                for item in items:
                    if "content" in item:
                        entities = extractor.extract_entities(item["content"])
                        for entity in entities:
                            entity["source"] = {
                                "type": result_type,
                                "id": item["id"]
                            }
                            extracted_entities.append(entity)
        
        # 4. ENRICH - Enrich entities with knowledge graph data
        from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
        kg_manager = KnowledgeGraphManager()
        
        enriched_results = {
            "original_results": results,
            "entities": [],
            "knowledge_graph": {}
        }
        
        # Create temporary graph for this search
        search_graph = kg_manager.create_temp_graph()
        
        for entity in extracted_entities:
            # Add to the graph
            entity_id = kg_manager.add_entity_to_graph(search_graph, entity)
            
            # Enrich with external knowledge
            kg_manager.enrich_entity(search_graph, entity_id)
            
            # Add to the results
            enriched_results["entities"].append(entity)
        
        # Export the graph data for visualization
        enriched_results["knowledge_graph"] = kg_manager.export_graph(search_graph)
        
        return enriched_results 
    
    def deep_research_search(self, query, user=None, max_iterations=3, depth=2, breadth=3):
        """
        Implements the Deep Research iterative exploration workflow.
        
        Deep Research explores a topic in depth by:
        1. Starting with an initial query
        2. Identifying key concepts and questions
        3. Iteratively exploring each concept in depth
        4. Building a comprehensive knowledge map
        5. Synthesizing findings from all explorations
        
        Args:
            query: Initial research query
            user: User performing the search
            max_iterations: Maximum number of research iterations
            depth: How deep to explore each concept (1-3)
            breadth: How many related concepts to explore (1-5)
            
        Returns:
            Comprehensive research results
        """
        logger.info(f"Starting Deep Research search for: {query}")
        
        # Apply HYDE to enhance the initial query
        from models_app.hyde_processor import HyDEProcessor
        hyde = HyDEProcessor(user=user)
        enhanced_query = hyde.process_query(query, strategy="weighted")
        
        # Use the enhanced query for initial search
        initial_results = self.search(enhanced_query, user=user)
        
        # 2. Extract key concepts for exploration
        from models_app.knowledge_graph.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        
        all_content = []
        for result_type, items in initial_results.items():
            for item in items:
                if "content" in item:
                    all_content.append(item["content"])
                elif "snippet" in item:
                    all_content.append(item["snippet"])
                    
        combined_content = " ".join(all_content)
        key_concepts = extractor.extract_entities(combined_content)
        
        # Sort concepts by confidence/relevance and take top N based on breadth
        key_concepts.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        exploration_concepts = key_concepts[:breadth]
        
        # 3. Iterative exploration of each concept
        exploration_paths = []
        for concept in exploration_concepts:
            # Build concept query
            concept_query = f"{query} {concept.get('label', '')}"
            
            # Initialize concept exploration path
            concept_path = {
                "concept": concept.get("label", ""),
                "iterations": [],
                "findings": []
            }
            
            # Iterative deepening
            current_query = concept_query
            for i in range(max_iterations):
                if i >= depth:
                    break
                    
                # Search with current query
                iteration_results = self.search(current_query, user=user)
                
                # Extract key findings
                findings = self._extract_key_findings(iteration_results)
                
                # Record this iteration
                concept_path["iterations"].append({
                    "query": current_query,
                    "results_count": sum(len(items) for items in iteration_results.values())
                })
                
                # Add findings
                concept_path["findings"].extend(findings)
                
                # Generate follow-up query based on findings
                if i < max_iterations - 1:  # Skip for last iteration
                    from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
                    kg_interface = KnowledgeGraphLLMInterface()
                    
                    findings_text = "\n".join([f["text"] for f in findings])
                    next_query = kg_interface.generate_follow_up_query(
                        original_query=current_query,
                        findings=findings_text
                    )
                    
                    # Update for next iteration
                    current_query = next_query
            
            # Add complete path for this concept
            exploration_paths.append(concept_path)
        
        # 4. Build knowledge graph from all explorations
        from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
        kg_manager = KnowledgeGraphManager()
        
        # Create temporary graph for this research
        research_graph = kg_manager.create_temp_graph()
        
        # Add all findings as entities and relationships
        for path in exploration_paths:
            for finding in path["findings"]:
                # Convert finding to entity
                entity = {
                    "label": finding["text"][:100],  # Use first 100 chars as label
                    "type": "research_finding",
                    "properties": {
                        "full_text": finding["text"],
                        "source": finding["source"],
                        "confidence": finding["confidence"],
                        "concept": path["concept"]
                    }
                }
                
                # Add to graph
                entity_id = kg_manager.add_entity_to_graph(research_graph, entity)
                
                # Add relationship to main concept
                kg_manager.graph_builder.add_relationship(
                    research_graph,
                    source_id=entity_id,
                    target_label=path["concept"],
                    relationship_type="supports",
                    properties={"strength": finding["confidence"]}
                )
        
        # 5. Synthesize research with comprehensive summary
        research_summary = self._synthesize_research(query, exploration_paths)
        
        # 6. Return comprehensive results
        return {
            "query": query,
            "depth": depth,
            "breadth": breadth,
            "iterations": sum(len(path["iterations"]) for path in exploration_paths),
            "exploration_paths": exploration_paths,
            "knowledge_graph": kg_manager.export_graph(research_graph),
            "summary": research_summary,
            "original_results": initial_results
        }
    
    def _extract_key_findings(self, search_results):
        """Extract key findings from search results"""
        findings = []
        
        # Process different result types
        for result_type, items in search_results.items():
            for item in items:
                # Extract text content based on result type
                if "content" in item:
                    text = item["content"]
                elif "snippet" in item:
                    text = item["snippet"]
                else:
                    continue
                
                # Calculate confidence based on result attributes
                confidence = item.get("score", 0.5)
                if "relevance" in item:
                    confidence = max(confidence, item["relevance"])
                
                # Add as finding
                findings.append({
                    "text": text,
                    "source": f"{result_type}:{item.get('id', '')}",
                    "confidence": confidence
                })
        
        # Sort by confidence and return
        findings.sort(key=lambda x: x["confidence"], reverse=True)
        return findings[:10]  # Return top 10 findings
    
    def _synthesize_research(self, query, exploration_paths):
        """Synthesize research findings into a comprehensive summary"""
        # Collate all findings
        all_findings = []
        for path in exploration_paths:
            for finding in path["findings"]:
                all_findings.append({
                    "text": finding["text"],
                    "concept": path["concept"],
                    "confidence": finding["confidence"]
                })
        
        # Sort by confidence
        all_findings.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Take top 20 findings
        top_findings = all_findings[:20]
        
        # Generate summary using KG LLM interface
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        
        findings_text = "\n\n".join([
            f"[CONCEPT: {f['concept']}]\n{f['text']}" for f in top_findings
        ])
        
        summary = kg_interface.generate_research_summary(
            query=query,
            findings=findings_text
        )
        
        return summary 

    def dense_retrieval_search(self, query, top_k=10):
        """Perform dense retrieval search using vector embeddings"""
        logger.info(f"Performing dense retrieval for query: {query}")
        
        # Get embeddings model - using SentenceTransformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create query embedding
        query_embedding = model.encode(query)
        
        # Get vector DB instance
        from models_app.rag_manager import RAGModelManager
        rag_manager = RAGModelManager()
        vector_db = None
        
        try:
            model = rag_manager.get_model("default")
            if model and hasattr(model, 'vector_store'):
                vector_db = model.vector_store
        except Exception as e:
            logger.error(f"Error accessing vector store: {e}")
        
        if not vector_db:
            logger.warning("No vector database available for dense retrieval")
            return []
        
        # Perform similarity search
        results = vector_db.similarity_search_by_vector(
            query_embedding, 
            k=top_k
        )
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0),
                "source": "dense_retrieval"
            })
        
        return formatted_results 

    def enhanced_multi_query_retrieval(self, original_query, n_queries=3):
        """Generate and enhance multiple queries with HYDE"""
        logger.info(f"Enhancing query with multi-query + HYDE: {original_query}")
        
        # 1. Generate alternative queries
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        
        # Generate alternative queries
        prompt = f"""Generate {n_queries} alternative search queries that would help answer the original question.
        Each query should focus on a different aspect or interpretation of the question.
        
        Original query: {original_query}
        
        Alternative queries (each on a new line):"""
        
        llm_provider = kg_interface.provider_factory.get_provider("default")
        alternatives_text, _ = llm_provider.generate_text(prompt, max_tokens=200)
        
        # Parse alternative queries
        alternative_queries = []
        for line in alternatives_text.strip().split('\n'):
            query = line.strip()
            if query and len(query) > 5 and query != original_query:
                alternative_queries.append(query)
        
        # Limit to requested number
        alternative_queries = alternative_queries[:n_queries]
        
        # 2. Apply HYDE to each query
        from models_app.hyde_processor import HyDEProcessor
        hyde = HyDEProcessor()
        
        enhanced_queries = []
        for query in alternative_queries:
            hyde_enhanced = hyde.process_query(query, strategy="weighted")
            enhanced_queries.append(hyde_enhanced)
        
        # 3. Add original HYDE-enhanced query 
        original_enhanced = hyde.process_query(original_query, strategy="weighted")
        all_queries = [original_enhanced] + enhanced_queries
        
        logger.info(f"Generated enhanced queries: {all_queries}")
        
        # 4. Search with each enhanced query
        all_results = []
        for query in all_queries:
            results = self.search(query)
            
            # Tag results with their source query
            for result_type, items in results.items():
                for item in items:
                    item['source_query'] = query
            
            all_results.append(results)
        
        # 5. Combine results using reciprocal rank fusion
        combined_results = self._combine_results_with_rrf(all_results, original_query)
        
        return combined_results

    def _combine_results_with_rrf(self, result_sets, original_query, k=60):
        """Combine results using Reciprocal Rank Fusion"""
        # Flatten all results into single list with source tracking
        all_items = []
        item_sources = {}  # To track duplicates
        
        for result_set in result_sets:
            for result_type, items in result_set.items():
                for item in items:
                    # Create unique ID for the item
                    item_id = None
                    if 'id' in item:
                        item_id = f"{result_type}:{item['id']}"
                    elif 'url' in item:
                        item_id = item['url']
                    elif 'content' in item:
                        # Hash content for ID
                        import hashlib
                        item_id = f"{result_type}:{hashlib.md5(item['content'].encode()).hexdigest()}"
                    else:
                        continue  # Skip items without ID
                    
                    # Add to list if not duplicate
                    if item_id not in item_sources:
                        item_sources[item_id] = item
                        all_items.append(item)
        
        # Score each item using RRF
        for item in all_items:
            item['rrf_score'] = 0
            source_queries = [item.get('source_query', original_query)]
            
            # Calculate rank-based score for each query
            for query in source_queries:
                # Simple relevance score based on text similarity
                if 'content' in item:
                    content = item['content']
                elif 'snippet' in item:
                    content = item['snippet']
                else:
                    content = str(item)
                    
                import difflib
                similarity = difflib.SequenceMatcher(None, query.lower(), content.lower()).ratio()
                
                # RRF formula: 1 / (k + rank)
                # Using inverted similarity as rank approximation
                rank = max(1, int((1 - similarity) * 10))
                item['rrf_score'] += 1 / (k + rank)
        
        # Sort by RRF score
        all_items.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
        
        # Reorganize into result types
        combined_results = {
            'messages': [],
            'files': [],
            'sessions': [],
            'web': [],
            'total_count': len(all_items)
        }
        
        for item in all_items:
            item_type = item.get('type', 'files')  # Default to files
            if item_type in combined_results:
                combined_results[item_type].append(item)
        
        return combined_results 

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
            # Use our specialized error handler
            from error_handlers.search_app_errors import handle_reranking_error
            handle_reranking_error(query, model, e)
            # Fall back to original results on error
            return documents[:top_k]

    def multimodal_search(self, query, include_images=True, include_text=True):
        """
        Perform multimodal search across text and images
        
        Args:
            query: Text query or image path
            include_images: Whether to include images
            include_text: Whether to include text
        """
        from models_app.multimodal.m_retriever import MultimodalRetriever
        
        # Initialize retriever
        retriever = MultimodalRetriever()
        
        # Collect search items
        search_items = []
        item_sources = {}
        
        # Add text content
        if include_text:
            text_results = self.search(query)
            for result_type, items in text_results.items():
                for item in items:
                    if 'content' in item:
                        search_items.append(item['content'])
                        item_sources[item['content']] = item
        
        # Add images
        if include_images:
            from models_app.models import UploadedFile
            image_files = UploadedFile.objects.filter(
                file_type__in=['image/jpeg', 'image/png', 'image/gif']
            )[:100]
            
            for img in image_files:
                if hasattr(img.file, 'path'):
                    path = img.file.path
                    search_items.append(path)
                    item_sources[path] = img
        
        # Perform multimodal search
        results = retriever.similarity_search(query, search_items)
        
        return {
            'multimodal_results': results,
            'sources': [item_sources.get(r['item'], r['item']) for r in results]
        }

    def bidirectional_encoder_search(self, query, top_k=10):
        """
        Perform search using bidirectional encoders for better multilingual support
        
        Args:
            query: Search query
            top_k: Number of top results to return
        """
        logger.info(f"Performing bidirectional encoder search for: {query}")
        
        try:
            # Initialize bidirectional encoder
            from sentence_transformers import SentenceTransformer
            bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
            
            # Encode query
            query_embedding = bi_encoder.encode(query)
            
            # Get documents to search (this depends on your implementation)
            # For example, get all indexed documents
            documents = self._get_all_indexed_documents()
            
            # Batch encode documents
            texts = [doc.get('content', '') for doc in documents]
            doc_embeddings = bi_encoder.encode(texts, batch_size=32)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # Create (document, similarity) pairs and sort
            results = [(doc, sim) for doc, sim in zip(documents, similarities)]
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error in bidirectional encoder search: {e}")
            return [] 

    def r3_search(self, query):
        """
        Implement R³ (Retrieval, Reading, Reasoning) for web search
        
        Args:
            query: User query
            
        Returns:
            Dict with reasoned answer and supporting evidence
        """
        logger.info(f"Performing R³ search for: {query}")
        
        # 1. Retrieval phase
        search_results = self.search(query)
        
        # 2. Reading phase - extract information from results
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        
        # 3. Reasoning phase - use R³ reasoning from KG interface
        reasoned_response = kg_interface.perform_r3_reasoning(query, search_results)
        
        return reasoned_response 