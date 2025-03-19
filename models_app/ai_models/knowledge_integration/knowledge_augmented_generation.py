import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np

from models_app.knowledge_graph.graph_manager import KnowledgeGraphManager
from models_app.knowledge_graph.graph_storage import GraphStorageFactory
from models_app.llm_providers.base import BaseLLMProvider
from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
from models_app.knowledge_graph.graph_visualization import GraphVisualizer
from analytics_app.utils import monitor_kg_llm_integration

logger = logging.getLogger(__name__)

class KAGBuilder:
    """
    Knowledge Augmented Generation Builder - Creates and enhances knowledge bases from documents
    
    The KAG-Builder component focuses on creating high-quality knowledge bases from documents,
    with bidirectional indexing between documents and graph elements.
    """
    
    def __init__(
        self, 
        kg_manager: KnowledgeGraphManager = None,
        llm_interface: KnowledgeGraphLLMInterface = None,
        storage_type: str = "json"
    ):
        """
        Initialize the KAG-Builder with components for graph management and LLM integration.
        
        Args:
            kg_manager: Knowledge graph manager for document processing
            llm_interface: Interface for LLM integration with knowledge graphs
            storage_type: Type of storage to use for the knowledge graph
        """
        self.kg_manager = kg_manager or KnowledgeGraphManager()
        self.llm_interface = llm_interface or KnowledgeGraphLLMInterface()
        self.storage = GraphStorageFactory.get_storage(storage_type)
        self.visualizer = GraphVisualizer()
        
    @monitor_kg_llm_integration
    def build_knowledge_base(
        self, 
        documents: List[Dict[str, Any]], 
        base_name: str = None,
        extraction_params: Dict[str, Any] = None,
        build_bidirectional_index: bool = True
    ) -> Dict[str, Any]:
        """
        Build a comprehensive knowledge base from a set of documents with bidirectional indexing.
        
        Args:
            documents: List of document dictionaries with content and metadata
            base_name: Name for the knowledge base
            extraction_params: Parameters for entity and relationship extraction
            build_bidirectional_index: Whether to build bidirectional index between documents and graph
            
        Returns:
            Dictionary with knowledge base information including graph_id and statistics
        """
        start_time = time.time()
        
        # Process documents to graph
        graph_data = self.kg_manager.process_documents_to_graph(
            documents, 
            extraction_params=extraction_params
        )
        
        if not graph_data or "id" not in graph_data:
            logger.error("Failed to build knowledge graph from documents")
            return {"success": False, "error": "Failed to build knowledge graph"}
            
        graph_id = graph_data["id"]
        
        # Create knowledge base metadata
        kb_metadata = {
            "name": base_name or f"Knowledge Base {graph_id}",
            "created_at": time.time(),
            "document_count": len(documents),
            "extraction_params": extraction_params or {},
            "bidirectional_index": build_bidirectional_index
        }
        
        # Update graph metadata
        graph_metadata = graph_data.get("metadata", {})
        graph_metadata.update({
            "knowledge_base": kb_metadata,
            "document_ids": [doc.get("id") for doc in documents if "id" in doc]
        })
        
        # Build bidirectional index if requested
        if build_bidirectional_index:
            self._build_bidirectional_index(graph_id, documents)
        
        # Update graph with enhanced metadata
        self.storage.update_graph_metadata(graph_id, graph_metadata)
        
        # Generate visualization
        try:
            visualization_url = self.visualizer.create_interactive_visualization(
                graph_data, 
                title=kb_metadata["name"]
            )
            kb_metadata["visualization_url"] = visualization_url
        except Exception as e:
            logger.warning(f"Failed to create visualization: {str(e)}")
        
        # Compute completion time
        completion_time = time.time() - start_time
        
        # Return knowledge base info
        return {
            "success": True,
            "graph_id": graph_id,
            "name": kb_metadata["name"],
            "document_count": len(documents),
            "entity_count": len(graph_data.get("entities", [])),
            "relationship_count": len(graph_data.get("relationships", [])),
            "completion_time": completion_time,
            "metadata": kb_metadata,
            "visualization_url": kb_metadata.get("visualization_url")
        }
    
    def _build_bidirectional_index(self, graph_id: str, documents: List[Dict[str, Any]]) -> None:
        """
        Build bidirectional index between documents and graph elements.
        
        Args:
            graph_id: ID of the graph to update
            documents: List of documents used to build the graph
        """
        if not graph_id or not documents:
            return
            
        try:
            # Retrieve graph
            graph_data = self.storage.retrieve_graph(graph_id)
            if not graph_data:
                logger.error(f"Could not retrieve graph {graph_id} for indexing")
                return
                
            # Create document index
            document_index = {}
            for doc in documents:
                doc_id = doc.get("id")
                if not doc_id:
                    continue
                    
                document_index[doc_id] = {
                    "title": doc.get("title", "Untitled Document"),
                    "source": doc.get("source", "Unknown Source"),
                    "entities": [],
                    "relationships": []
                }
            
            # Link entities to documents
            for entity in graph_data.get("entities", []):
                entity_id = entity.get("id")
                if not entity_id:
                    continue
                    
                source_docs = entity.get("source_documents", [])
                for doc_id in source_docs:
                    if doc_id in document_index:
                        document_index[doc_id]["entities"].append(entity_id)
                        
                        # Add document reference to entity if not present
                        if "document_references" not in entity:
                            entity["document_references"] = []
                        
                        if doc_id not in entity["document_references"]:
                            entity["document_references"].append(doc_id)
            
            # Link relationships to documents
            for relation in graph_data.get("relationships", []):
                relation_id = relation.get("id")
                if not relation_id:
                    continue
                    
                source_docs = relation.get("source_documents", [])
                for doc_id in source_docs:
                    if doc_id in document_index:
                        document_index[doc_id]["relationships"].append(relation_id)
                        
                        # Add document reference to relationship if not present
                        if "document_references" not in relation:
                            relation["document_references"] = []
                        
                        if doc_id not in relation["document_references"]:
                            relation["document_references"].append(doc_id)
            
            # Update graph metadata with document index
            metadata = graph_data.get("metadata", {})
            metadata["document_index"] = document_index
            
            # Update graph with bidirectional index
            self.storage.update_graph_metadata(graph_id, metadata)
            
            # Update entities and relationships
            for entity in graph_data.get("entities", []):
                if "document_references" in entity:
                    self.storage.update_entity(graph_id, entity["id"], entity)
                    
            for relation in graph_data.get("relationships", []):
                if "document_references" in relation:
                    self.storage.update_relationship(graph_id, relation["id"], relation)
                
            logger.info(f"Built bidirectional index for graph {graph_id} with {len(document_index)} documents")
            
        except Exception as e:
            logger.error(f"Error building bidirectional index: {str(e)}")

    def enhance_knowledge_base(
        self, 
        graph_id: str, 
        enhancement_type: str = "consistency",
        llm_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enhance an existing knowledge base using LLM capabilities.
        
        Args:
            graph_id: ID of the graph to enhance
            enhancement_type: Type of enhancement to perform:
                - "consistency": Improve consistency and fix contradictions
                - "enrichment": Add missing relationships and properties
                - "validation": Validate facts against documents
            llm_params: Parameters for LLM integration
            
        Returns:
            Dictionary with enhancement results
        """
        if not graph_id:
            return {"success": False, "error": "Graph ID is required"}
            
        try:
            # Retrieve graph
            graph_data = self.storage.retrieve_graph(graph_id)
            if not graph_data:
                return {"success": False, "error": f"Graph {graph_id} not found"}
            
            # Get graph metadata
            metadata = graph_data.get("metadata", {})
            
            # Create enhancement record
            enhancement = {
                "type": enhancement_type,
                "timestamp": time.time(),
                "changes": []
            }
            
            # Perform enhancement based on type
            if enhancement_type == "consistency":
                changes = self._enhance_consistency(graph_data, llm_params)
            elif enhancement_type == "enrichment":
                changes = self._enhance_enrichment(graph_data, llm_params)
            elif enhancement_type == "validation":
                changes = self._enhance_validation(graph_data, llm_params)
            else:
                return {"success": False, "error": f"Unknown enhancement type: {enhancement_type}"}
            
            enhancement["changes"] = changes
            
            # Update graph metadata with enhancement record
            if "enhancements" not in metadata:
                metadata["enhancements"] = []
                
            metadata["enhancements"].append(enhancement)
            self.storage.update_graph_metadata(graph_id, metadata)
            
            # Return enhancement results
            return {
                "success": True,
                "graph_id": graph_id,
                "enhancement_type": enhancement_type,
                "changes_count": len(changes),
                "changes": changes
            }
            
        except Exception as e:
            logger.error(f"Error enhancing knowledge base {graph_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _enhance_consistency(
        self, 
        graph_data: Dict[str, Any],
        llm_params: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhance consistency and fix contradictions in the knowledge graph.
        
        Args:
            graph_data: Graph data to enhance
            llm_params: Parameters for LLM integration
            
        Returns:
            List of changes made to the graph
        """
        changes = []
        
        # Get entities and relationships
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])
        
        # Find and resolve contradictions using LLM
        contradictions = self._find_contradictions(entities, relationships)
        
        # For each contradiction, use LLM to resolve
        for contradiction in contradictions:
            resolution = self.llm_interface.resolve_graph_contradiction(
                graph_data,
                contradiction,
                llm_params=llm_params
            )
            
            if resolution and "changes" in resolution:
                changes.extend(resolution["changes"])
        
        return changes
    
    def _enhance_enrichment(
        self, 
        graph_data: Dict[str, Any],
        llm_params: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhance the graph by adding missing relationships and properties.
        
        Args:
            graph_data: Graph data to enhance
            llm_params: Parameters for LLM integration
            
        Returns:
            List of changes made to the graph
        """
        changes = []
        
        # Use LLM to identify missing relationships
        enrichment = self.llm_interface.enrich_graph(
            graph_data,
            llm_params=llm_params
        )
        
        if enrichment and "additions" in enrichment:
            changes.extend(enrichment["additions"])
        
        return changes
    
    def _enhance_validation(
        self, 
        graph_data: Dict[str, Any],
        llm_params: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate facts in the graph against source documents.
        
        Args:
            graph_data: Graph data to validate
            llm_params: Parameters for LLM integration
            
        Returns:
            List of changes made to the graph
        """
        changes = []
        
        # Get metadata and document index
        metadata = graph_data.get("metadata", {})
        document_index = metadata.get("document_index", {})
        
        # For each entity and relationship, validate against documents
        for entity in graph_data.get("entities", []):
            entity_id = entity.get("id")
            doc_refs = entity.get("document_references", [])
            
            if not doc_refs:
                continue
                
            # Get document text
            doc_texts = []
            for doc_id in doc_refs:
                if doc_id in document_index:
                    doc_info = document_index[doc_id]
                    doc_texts.append({
                        "id": doc_id,
                        "title": doc_info.get("title", ""),
                        "text": doc_info.get("text", "")
                    })
            
            # Validate entity against documents
            validation = self.llm_interface.validate_graph_element(
                entity,
                doc_texts,
                llm_params=llm_params
            )
            
            if validation and not validation.get("is_valid", True):
                change = {
                    "type": "invalidate_entity",
                    "entity_id": entity_id,
                    "reason": validation.get("reason", "Not validated by source documents"),
                    "confidence": validation.get("confidence", 0.0)
                }
                
                changes.append(change)
        
        return changes
    
    def _find_contradictions(
        self, 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find contradictions in the knowledge graph.
        
        Args:
            entities: List of entities
            relationships: List of relationships
            
        Returns:
            List of contradictions found
        """
        # Simple implementation - look for conflicting relationships
        contradictions = []
        
        # Build relationship index
        rel_index = {}
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            
            if not source or not target or not rel_type:
                continue
                
            key = f"{source}_{rel_type}_{target}"
            if key not in rel_index:
                rel_index[key] = []
                
            rel_index[key].append(rel)
        
        # Find conflicting relationships (multiple relationships of same type between same entities)
        for key, rels in rel_index.items():
            if len(rels) > 1:
                contradiction = {
                    "type": "conflicting_relationships",
                    "relationships": [rel["id"] for rel in rels],
                    "description": f"Multiple {rels[0]['type']} relationships between same entities"
                }
                contradictions.append(contradiction)
        
        return contradictions


class KAGSolver:
    """
    Knowledge Augmented Generation Solver - Provides logic-based problem solving
    
    The KAG-Solver component focuses on using knowledge graphs for enhancing reasoning,
    fact-checking, and complex problem solving capabilities.
    """
    
    def __init__(
        self, 
        llm_provider: BaseLLMProvider = None,
        llm_interface: KnowledgeGraphLLMInterface = None,
        storage_type: str = "json"
    ):
        """
        Initialize the KAG-Solver with components for LLM integration and knowledge graphs.
        
        Args:
            llm_provider: LLM provider for direct interactions
            llm_interface: Interface for LLM integration with knowledge graphs
            storage_type: Type of storage to use for the knowledge graph
        """
        self.llm_interface = llm_interface or KnowledgeGraphLLMInterface()
        self.storage = GraphStorageFactory.get_storage(storage_type)
        self.visualizer = GraphVisualizer()
        
    @monitor_kg_llm_integration
    def solve_problem(
        self, 
        query: str,
        graph_id: str,
        solution_type: str = "reasoning",
        max_hops: int = 3,
        llm_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using knowledge graph and LLM integration.
        
        Args:
            query: Query or problem statement
            graph_id: ID of the knowledge graph to use
            solution_type: Type of solution to generate:
                - "reasoning": Step-by-step reasoning based on graph
                - "fact_checking": Verify facts against graph
                - "path_finding": Find path between concepts
            max_hops: Maximum hops to consider in the graph
            llm_params: Parameters for LLM integration
            
        Returns:
            Dictionary with solution and supporting information
        """
        if not query or not graph_id:
            return {"success": False, "error": "Query and graph ID are required"}
            
        try:
            # Retrieve graph (using progressive loading for large graphs)
            graph_data = self.storage.retrieve_graph(graph_id)
            if not graph_data:
                return {"success": False, "error": f"Graph {graph_id} not found"}
            
            # Apply solution strategy based on type
            if solution_type == "reasoning":
                result = self._apply_reasoning(query, graph_data, max_hops, llm_params)
            elif solution_type == "fact_checking":
                result = self._apply_fact_checking(query, graph_data, llm_params)
            elif solution_type == "path_finding":
                result = self._apply_path_finding(query, graph_data, max_hops, llm_params)
            else:
                return {"success": False, "error": f"Unknown solution type: {solution_type}"}
            
            # Generate subgraph visualization if available
            if "relevant_subgraph" in result:
                try:
                    subgraph = result["relevant_subgraph"]
                    viz_url = self.visualizer.create_interactive_visualization(
                        subgraph, 
                        title=f"Solution subgraph for: {query[:50]}..."
                    )
                    result["visualization_url"] = viz_url
                except Exception as e:
                    logger.warning(f"Failed to create visualization: {str(e)}")
            
            # Add operation metadata
            result.update({
                "success": True,
                "graph_id": graph_id,
                "query": query,
                "solution_type": solution_type,
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error solving problem with graph {graph_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _apply_reasoning(
        self, 
        query: str,
        graph_data: Dict[str, Any],
        max_hops: int = 3,
        llm_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Apply step-by-step reasoning based on the knowledge graph.
        
        Args:
            query: Query or problem statement
            graph_data: Knowledge graph data
            max_hops: Maximum hops to consider in the graph
            llm_params: Parameters for LLM integration
            
        Returns:
            Dictionary with reasoning steps and supporting information
        """
        # Extract relevant subgraph
        subgraph = self._extract_relevant_subgraph(query, graph_data, max_hops)
        
        # Generate augmented response with reasoning
        response = self.llm_interface.generate_graph_augmented_response(
            query,
            subgraph,
            response_format={
                "reasoning_steps": True,
                "citations": True,
                "confidence": True
            },
            llm_params=llm_params
        )
        
        # Structure the result
        result = {
            "answer": response.get("answer", ""),
            "reasoning_steps": response.get("reasoning_steps", []),
            "confidence": response.get("confidence", 0.0),
            "citations": response.get("citations", []),
            "relevant_subgraph": subgraph
        }
        
        return result
    
    def _apply_fact_checking(
        self, 
        statement: str,
        graph_data: Dict[str, Any],
        llm_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Verify facts in a statement against the knowledge graph.
        
        Args:
            statement: Statement to verify
            graph_data: Knowledge graph data
            llm_params: Parameters for LLM integration
            
        Returns:
            Dictionary with fact checking results
        """
        # Extract relevant subgraph
        subgraph = self._extract_relevant_subgraph(statement, graph_data, max_hops=2)
        
        # Generate fact checking response
        response = self.llm_interface.validate_statement(
            statement,
            subgraph,
            llm_params=llm_params
        )
        
        # Structure the result
        result = {
            "is_valid": response.get("is_valid", False),
            "confidence": response.get("confidence", 0.0),
            "supporting_facts": response.get("supporting_facts", []),
            "contradicting_facts": response.get("contradicting_facts", []),
            "explanation": response.get("explanation", ""),
            "relevant_subgraph": subgraph
        }
        
        return result
    
    def _apply_path_finding(
        self, 
        query: str,
        graph_data: Dict[str, Any],
        max_hops: int = 3,
        llm_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Find path between concepts mentioned in the query.
        
        Args:
            query: Query with concepts to find path between
            graph_data: Knowledge graph data
            max_hops: Maximum hops to consider in the graph
            llm_params: Parameters for LLM integration
            
        Returns:
            Dictionary with path finding results
        """
        # Extract entities from query
        entity_extraction = self.llm_interface.extract_query_entities(
            query,
            llm_params=llm_params
        )
        
        start_entity = entity_extraction.get("start_entity")
        end_entity = entity_extraction.get("end_entity")
        
        if not start_entity or not end_entity:
            return {
                "path_found": False,
                "error": "Could not identify start and end entities in the query"
            }
            
        # Find path in graph
        path = self._find_path_between_entities(
            start_entity,
            end_entity,
            graph_data,
            max_hops
        )
        
        if not path:
            return {
                "path_found": False,
                "start_entity": start_entity,
                "end_entity": end_entity,
                "explanation": f"No path found between {start_entity} and {end_entity} within {max_hops} hops"
            }
            
        # Create subgraph from path
        path_subgraph = self._create_subgraph_from_path(path, graph_data)
        
        # Generate path explanation
        explanation = self.llm_interface.explain_entity_path(
            start_entity,
            end_entity,
            path,
            path_subgraph,
            llm_params=llm_params
        )
        
        # Structure the result
        result = {
            "path_found": True,
            "start_entity": start_entity,
            "end_entity": end_entity,
            "path": path,
            "path_length": len(path) - 1,
            "explanation": explanation.get("explanation", ""),
            "relevant_subgraph": path_subgraph
        }
        
        return result
    
    def _extract_relevant_subgraph(
        self, 
        query: str,
        graph_data: Dict[str, Any],
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """
        Extract relevant subgraph related to the query.
        
        Args:
            query: Query to extract relevant subgraph for
            graph_data: Full knowledge graph
            max_hops: Maximum hops to include
            
        Returns:
            Subgraph with relevant entities and relationships
        """
        # This is a simplified implementation - in practice, more sophisticated techniques
        # would be used to identify the relevant subgraph
        
        try:
            # Extract entities from query using simple keyword matching
            entities = graph_data.get("entities", [])
            relationships = graph_data.get("relationships", [])
            
            # Find entities mentioned in query
            query_lower = query.lower()
            seed_entities = []
            
            for entity in entities:
                entity_name = entity.get("name", "").lower()
                if entity_name and entity_name in query_lower:
                    seed_entities.append(entity["id"])
            
            if not seed_entities:
                # If no direct mentions, try to find entities by properties
                for entity in entities:
                    props = entity.get("properties", {})
                    for prop_name, prop_value in props.items():
                        if isinstance(prop_value, str) and prop_value.lower() in query_lower:
                            seed_entities.append(entity["id"])
                            break
            
            # If still no entities found, return empty subgraph
            if not seed_entities:
                return {
                    "entities": [],
                    "relationships": [],
                    "metadata": {
                        "query": query,
                        "extraction_method": "keyword_matching",
                        "found_entities": False
                    }
                }
            
            # Expand graph from seed entities using breadth-first search
            included_entities = set(seed_entities)
            included_relationships = set()
            
            # BFS expansion
            frontier = set(seed_entities)
            for _ in range(max_hops):
                next_frontier = set()
                
                for entity_id in frontier:
                    # Find relationships involving this entity
                    for rel in relationships:
                        source = rel.get("source")
                        target = rel.get("target")
                        
                        if source == entity_id and target not in included_entities:
                            included_relationships.add(rel["id"])
                            included_entities.add(target)
                            next_frontier.add(target)
                        
                        if target == entity_id and source not in included_entities:
                            included_relationships.add(rel["id"])
                            included_entities.add(source)
                            next_frontier.add(source)
                
                frontier = next_frontier
                if not frontier:
                    break
            
            # Also include relationships between already included entities
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                
                if source in included_entities and target in included_entities:
                    included_relationships.add(rel["id"])
            
            # Build subgraph
            subgraph_entities = [e for e in entities if e["id"] in included_entities]
            subgraph_relationships = [r for r in relationships if r["id"] in included_relationships]
            
            subgraph = {
                "entities": subgraph_entities,
                "relationships": subgraph_relationships,
                "metadata": {
                    "query": query,
                    "extraction_method": "neighborhood_expansion",
                    "seed_entities": seed_entities,
                    "max_hops": max_hops
                }
            }
            
            return subgraph
            
        except Exception as e:
            logger.error(f"Error extracting relevant subgraph: {str(e)}")
            return {
                "entities": [],
                "relationships": [],
                "metadata": {
                    "query": query,
                    "extraction_method": "failed",
                    "error": str(e)
                }
            }
    
    def _find_path_between_entities(
        self, 
        start_entity_name: str,
        end_entity_name: str,
        graph_data: Dict[str, Any],
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find path between two entities in the graph.
        
        Args:
            start_entity_name: Name of the start entity
            end_entity_name: Name of the end entity
            graph_data: Knowledge graph data
            max_hops: Maximum hops in the path
            
        Returns:
            Path as a list of entities and relationships, or empty list if no path found
        """
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])
        
        # Find start and end entities by name
        start_entity_id = None
        end_entity_id = None
        
        for entity in entities:
            name = entity.get("name", "").lower()
            if name == start_entity_name.lower():
                start_entity_id = entity["id"]
            elif name == end_entity_name.lower():
                end_entity_id = entity["id"]
        
        if not start_entity_id or not end_entity_id:
            return []
        
        # Build relationship index
        rel_by_source = {}
        for rel in relationships:
            source = rel["source"]
            if source not in rel_by_source:
                rel_by_source[source] = []
            rel_by_source[source].append(rel)
        
        # BFS to find path
        queue = [(start_entity_id, [])]
        visited = set([start_entity_id])
        
        while queue:
            current, path = queue.pop(0)
            
            # If we reached the end entity, return the path
            if current == end_entity_id:
                # Collect the complete path with entity and relationship data
                result = []
                
                # Add start entity
                for entity in entities:
                    if entity["id"] == start_entity_id:
                        result.append({"type": "entity", "data": entity})
                        break
                
                # Add path elements (relationships and entities)
                for i, step in enumerate(path):
                    # Add relationship
                    for rel in relationships:
                        if rel["id"] == step["relationship"]:
                            result.append({"type": "relationship", "data": rel})
                            break
                    
                    # Add entity
                    for entity in entities:
                        if entity["id"] == step["entity"]:
                            result.append({"type": "entity", "data": entity})
                            break
                
                return result
            
            # If max hops reached, skip
            if len(path) >= max_hops:
                continue
            
            # Explore neighbors
            for rel in rel_by_source.get(current, []):
                target = rel["target"]
                if target not in visited:
                    visited.add(target)
                    new_path = path + [{"relationship": rel["id"], "entity": target}]
                    queue.append((target, new_path))
        
        # No path found
        return []
    
    def _create_subgraph_from_path(
        self, 
        path: List[Dict[str, Any]],
        graph_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a subgraph from a path in the graph.
        
        Args:
            path: Path as a list of entities and relationships
            graph_data: Full knowledge graph data
            
        Returns:
            Subgraph with entities and relationships from the path
        """
        included_entities = set()
        included_relationships = set()
        
        for item in path:
            item_type = item.get("type")
            item_data = item.get("data", {})
            
            if item_type == "entity":
                included_entities.add(item_data["id"])
            elif item_type == "relationship":
                included_relationships.add(item_data["id"])
                included_entities.add(item_data["source"])
                included_entities.add(item_data["target"])
        
        # Build subgraph
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])
        
        subgraph_entities = [e for e in entities if e["id"] in included_entities]
        subgraph_relationships = [r for r in relationships if r["id"] in included_relationships]
        
        return {
            "entities": subgraph_entities,
            "relationships": subgraph_relationships,
            "metadata": {
                "extraction_method": "path_extraction",
                "path_length": len(path)
            }
        } 