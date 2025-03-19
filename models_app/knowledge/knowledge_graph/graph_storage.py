from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging
import os
import uuid
import json
from datetime import datetime

from models_app.knowledge_graph.interfaces import GraphStorageInterface
from django.conf import settings

# Import Weaver client (adjust based on actual Weaver API)
# from weaver_client import WeaverClient

logger = logging.getLogger(__name__)

class WeaverGraphStorage(GraphStorageInterface):
    """
    Graph storage implementation using Weaver.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize Weaver client
        self.weaver_endpoint = self.config.get("endpoint", "http://localhost:8080")
        self.weaver_api_key = self.config.get("api_key", "")
        
        # Uncomment when actual Weaver client is available
        # self.client = WeaverClient(
        #     endpoint=self.weaver_endpoint,
        #     api_key=self.weaver_api_key
        # )
        
        # For now, we'll use a dummy implementation
        self.graphs = {}
        
    def store_graph(self, graph: Dict[str, Any], graph_id: Optional[str] = None) -> str:
        """
        Store a graph in Weaver
        
        Args:
            graph: The graph data
            graph_id: Optional ID for the graph
            
        Returns:
            The ID of the stored graph
        """
        if not graph_id:
            graph_id = str(uuid.uuid4())
            
        logger.info(f"Storing graph with ID: {graph_id}")
        
        # In a real implementation, convert graph to Weaver format and store
        # Example (pseudocode):
        # weaver_graph = self._convert_to_weaver_format(graph)
        # self.client.create_graph(graph_id, weaver_graph)
        
        # Dummy implementation
        self.graphs[graph_id] = graph
        
        return graph_id
        
    def retrieve_graph(self, graph_id: str) -> Dict[str, Any]:
        """
        Retrieve a graph from Weaver
        
        Args:
            graph_id: The ID of the graph to retrieve
            
        Returns:
            The graph data
        """
        logger.info(f"Retrieving graph with ID: {graph_id}")
        
        # In a real implementation:
        # weaver_graph = self.client.get_graph(graph_id)
        # return self._convert_from_weaver_format(weaver_graph)
        
        # Dummy implementation
        if graph_id not in self.graphs:
            logger.warning(f"Graph with ID {graph_id} not found")
            return {}
            
        return self.graphs[graph_id]
        
    def query_graph(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query a graph in Weaver
        
        Args:
            query: Query parameters
            
        Returns:
            List of query results
        """
        logger.info(f"Querying graph: {query}")
        
        graph_id = query.get("graph_id")
        if not graph_id:
            logger.error("No graph_id provided in query")
            return []
            
        # In a real implementation:
        # query_result = self.client.query_graph(
        #     graph_id=graph_id,
        #     query_type=query.get("query_type", "entity"),
        #     query_params=query.get("parameters", {})
        # )
        # return self._format_query_results(query_result)
        
        # Dummy implementation
        if graph_id not in self.graphs:
            logger.warning(f"Graph with ID {graph_id} not found")
            return []
            
        # Simple filtering based on entity type
        if query.get("entity_type"):
            return [
                entity for entity in self.graphs[graph_id].get("entities", [])
                if entity.get("type") == query.get("entity_type")
            ]
            
        # Return all entities by default
        return self.graphs[graph_id].get("entities", [])
        
    def update_graph(self, graph_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a graph in Weaver
        
        Args:
            graph_id: The ID of the graph to update
            updates: The updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating graph with ID: {graph_id}")
        
        if graph_id not in self.graphs:
            logger.warning(f"Graph with ID {graph_id} not found")
            return False
            
        # In a real implementation:
        # update_result = self.client.update_graph(
        #     graph_id=graph_id,
        #     updates=updates
        # )
        # return update_result.success
        
        # Dummy implementation
        current_graph = self.graphs[graph_id]
        
        # Add new entities
        if "add_entities" in updates:
            current_graph["entities"] = current_graph.get("entities", []) + updates["add_entities"]
            
        # Add new relationships
        if "add_relationships" in updates:
            current_graph["relationships"] = current_graph.get("relationships", []) + updates["add_relationships"]
            
        # Remove entities
        if "remove_entity_ids" in updates:
            current_graph["entities"] = [
                entity for entity in current_graph.get("entities", [])
                if entity.get("id") not in updates["remove_entity_ids"]
            ]
            
        # Remove relationships
        if "remove_relationship_ids" in updates:
            current_graph["relationships"] = [
                rel for rel in current_graph.get("relationships", [])
                if rel.get("id") not in updates["remove_relationship_ids"]
            ]
            
        return True

    def connect_to_weaver_vector_db(self, vector_db_config):
        """
        Connect this graph storage to a Weaver Vector Database
        
        Args:
            vector_db_config: Configuration for Weaver VD connection
        """
        self.vector_db_endpoint = vector_db_config.get("endpoint", "http://localhost:8000/v1")
        self.vector_db_api_key = vector_db_config.get("api_key", "")
        
        # Establish connection to Weaver VD
        # self.vector_db_client = WeaverVDClient(
        #     endpoint=self.vector_db_endpoint,
        #     api_key=self.vector_db_api_key
        # )
        
        # Create bidirectional links between graph and vector DB
        # For each graph entity, store its embedding in the vector DB
        # with a reference to the graph entity ID
        logger.info("Connected to Weaver Vector Database")

    def incremental_update(self, graph_id, new_entities=None, new_relationships=None,
                         removed_entity_ids=None, removed_relationship_ids=None):
        """
        Perform incremental update to an existing graph
        
        Args:
            graph_id: ID of graph to update
            new_entities: List of new entities to add
            new_relationships: List of new relationships to add
            removed_entity_ids: List of entity IDs to remove
            removed_relationship_ids: List of relationship IDs to remove
        
        Returns:
            Bool indicating success
        """
        # Retrieve existing graph
        graph = self.retrieve_graph(graph_id)
        if not graph:
            return False
        
        # Track changes
        changes = {
            "added_entities": [],
            "added_relationships": [],
            "removed_entities": [],
            "removed_relationships": []
        }
        
        # Add new entities
        if new_entities:
            existing_entity_ids = {e["id"] for e in graph.get("entities", [])}
            for entity in new_entities:
                if entity["id"] not in existing_entity_ids:
                    graph.setdefault("entities", []).append(entity)
                    changes["added_entities"].append(entity["id"])
        
        # Add new relationships
        if new_relationships:
            existing_rel_ids = {r["id"] for r in graph.get("relationships", [])}
            for rel in new_relationships:
                if rel["id"] not in existing_rel_ids:
                    graph.setdefault("relationships", []).append(rel)
                    changes["added_relationships"].append(rel["id"])
        
        # Remove entities
        if removed_entity_ids:
            graph["entities"] = [e for e in graph.get("entities", []) 
                               if e["id"] not in removed_entity_ids]
            changes["removed_entities"] = removed_entity_ids
            
            # Also remove relationships involving these entities
            if graph.get("relationships"):
                graph["relationships"] = [
                    r for r in graph.get("relationships", [])
                    if r["source"] not in removed_entity_ids and r["target"] not in removed_entity_ids
                ]
        
        # Remove relationships
        if removed_relationship_ids:
            graph["relationships"] = [r for r in graph.get("relationships", []) 
                                    if r["id"] not in removed_relationship_ids]
            changes["removed_relationships"] = removed_relationship_ids
        
        # Store updated graph
        self.store_graph(graph, graph_id)
        
        # Log changes
        logger.info(f"Incremental update to graph {graph_id}: {changes}")
        
        return True

    def retrieve_graph_progressive(self, graph_id, batch_size=100, include_entities=True, include_relationships=True, 
                               entity_types=None, relationship_types=None):
        """
        Progressively load a graph in smaller batches to reduce memory usage.
        
        Args:
            graph_id: ID of the graph to retrieve
            batch_size: Number of entities/relationships to load per batch
            include_entities: Whether to include entities
            include_relationships: Whether to include relationships
            entity_types: Filter entities by these types
            relationship_types: Filter relationships by these types
            
        Returns:
            Graph data iterator that yields batches of data
        """
        if not graph_id:
            logger.error("Graph ID is required for progressive retrieval")
            return None
        
        try:
            # Find graph in storage
            graph_key = self._get_graph_key(graph_id)
            if not self._exists(graph_key):
                logger.error(f"Graph {graph_id} not found in storage")
                return None
            
            # Load graph metadata
            metadata = self._load_metadata(graph_id)
            if not metadata:
                logger.error(f"Could not load metadata for graph {graph_id}")
                return None
            
            # Setup graph shell with metadata
            graph_shell = {
                "id": graph_id,
                "metadata": metadata,
                "entities": [],
                "relationships": []
            }
            
            # Count total entities and relationships
            total_entities = 0
            total_relationships = 0
            
            # Get entity keys
            entity_keys = self._get_entity_keys(graph_id)
            if entity_types:
                # Filter entities by type
                filtered_entity_keys = []
                for key in entity_keys:
                    entity = self._load_item(key)
                    if entity and entity.get("type") in entity_types:
                        filtered_entity_keys.append(key)
                entity_keys = filtered_entity_keys
            
            total_entities = len(entity_keys)
            
            # Get relationship keys
            relationship_keys = self._get_relationship_keys(graph_id)
            if relationship_types:
                # Filter relationships by type
                filtered_rel_keys = []
                for key in relationship_keys:
                    rel = self._load_item(key)
                    if rel and rel.get("type") in relationship_types:
                        filtered_rel_keys.append(key)
                relationship_keys = filtered_rel_keys
            
            total_relationships = len(relationship_keys)
            
            # Yield initial graph shell with metadata and counts
            yield {
                **graph_shell,
                "total_entities": total_entities,
                "total_relationships": total_relationships
            }
            
            # Load entities in batches
            if include_entities:
                for i in range(0, len(entity_keys), batch_size):
                    batch_keys = entity_keys[i:i+batch_size]
                    entities_batch = []
                    for key in batch_keys:
                        entity = self._load_item(key)
                        if entity:
                            entities_batch.append(entity)
                    
                    # Yield entity batch
                    yield {
                        "id": graph_id,
                        "batch_type": "entities",
                        "batch_start": i,
                        "batch_end": min(i+batch_size, total_entities),
                        "entities": entities_batch
                    }
            
            # Load relationships in batches
            if include_relationships:
                for i in range(0, len(relationship_keys), batch_size):
                    batch_keys = relationship_keys[i:i+batch_size]
                    relationships_batch = []
                    for key in batch_keys:
                        rel = self._load_item(key)
                        if rel:
                            relationships_batch.append(rel)
                    
                    # Yield relationship batch
                    yield {
                        "id": graph_id,
                        "batch_type": "relationships",
                        "batch_start": i,
                        "batch_end": min(i+batch_size, total_relationships),
                        "relationships": relationships_batch
                    }
            
        except Exception as e:
            logger.error(f"Error retrieving graph {graph_id} progressively: {str(e)}")
            return None

    def prune_graph(self, graph_id, criteria=None):
        """
        Prune a graph to remove irrelevant sections based on criteria.
        
        Args:
            graph_id: ID of the graph to prune
            criteria: Dict with pruning criteria:
                - min_confidence: Minimum confidence threshold for entities/relationships
                - entity_types: List of entity types to keep (all others removed)
                - relationship_types: List of relationship types to keep
                - disconnect_threshold: Remove entities with fewer connections than this
                - min_property_count: Remove entities with fewer properties than this
                - max_entities: Maximum number of entities to keep (keeps highest confidence)
                
        Returns:
            Pruned graph ID (new graph)
        """
        if not graph_id:
            logger.error("Graph ID is required for pruning")
            return None
        
        # Default criteria
        criteria = criteria or {}
        min_confidence = criteria.get("min_confidence", 0.0)
        entity_types = criteria.get("entity_types", None)
        relationship_types = criteria.get("relationship_types", None)
        disconnect_threshold = criteria.get("disconnect_threshold", 0)
        min_property_count = criteria.get("min_property_count", 0)
        max_entities = criteria.get("max_entities", None)
        
        try:
            # Retrieve original graph
            original_graph = self.retrieve_graph(graph_id)
            if not original_graph:
                logger.error(f"Graph {graph_id} not found for pruning")
                return None
            
            # Create a new graph with same metadata
            pruned_graph = {
                "metadata": original_graph.get("metadata", {}).copy(),
                "entities": [],
                "relationships": []
            }
            
            # Add pruning info to metadata
            pruned_graph["metadata"]["pruned_from"] = graph_id
            pruned_graph["metadata"]["pruning_criteria"] = criteria
            pruned_graph["metadata"]["pruning_timestamp"] = datetime.now().isoformat()
            
            # Filter entities by confidence and type
            entity_map = {}  # Map of entity IDs to indices in pruned_graph
            for entity in original_graph.get("entities", []):
                # Apply confidence filter
                if entity.get("confidence", 1.0) < min_confidence:
                    continue
                
                # Apply entity type filter
                if entity_types and entity.get("type") not in entity_types:
                    continue
                
                # Apply property count filter
                if min_property_count > 0:
                    property_count = len(entity.get("properties", {}))
                    if property_count < min_property_count:
                        continue
                
                # Add entity to pruned graph
                entity_map[entity.get("id")] = len(pruned_graph["entities"])
                pruned_graph["entities"].append(entity)
            
            # Truncate to max_entities if specified (sorting by confidence)
            if max_entities and len(pruned_graph["entities"]) > max_entities:
                # Sort by confidence (descending)
                pruned_graph["entities"].sort(
                    key=lambda e: e.get("confidence", 0.0),
                    reverse=True
                )
                # Keep only top max_entities
                pruned_graph["entities"] = pruned_graph["entities"][:max_entities]
                # Rebuild entity map
                entity_map = {
                    entity.get("id"): i 
                    for i, entity in enumerate(pruned_graph["entities"])
                }
            
            # Filter relationships
            for rel in original_graph.get("relationships", []):
                # Apply confidence filter
                if rel.get("confidence", 1.0) < min_confidence:
                    continue
                
                # Apply relationship type filter
                if relationship_types and rel.get("type") not in relationship_types:
                    continue
                
                # Check if both source and target entities exist in pruned graph
                source_id = rel.get("source")
                target_id = rel.get("target")
                
                if source_id in entity_map and target_id in entity_map:
                    pruned_graph["relationships"].append(rel)
            
            # Apply disconnect threshold filter if needed
            if disconnect_threshold > 0:
                # Count connections for each entity
                connection_count = {}
                for rel in pruned_graph["relationships"]:
                    source_id = rel.get("source")
                    target_id = rel.get("target")
                    
                    connection_count[source_id] = connection_count.get(source_id, 0) + 1
                    connection_count[target_id] = connection_count.get(target_id, 0) + 1
                
                # Filter entities by connection count
                connected_entities = []
                for entity in pruned_graph["entities"]:
                    entity_id = entity.get("id")
                    if connection_count.get(entity_id, 0) >= disconnect_threshold:
                        connected_entities.append(entity)
                
                # Update entities
                pruned_graph["entities"] = connected_entities
                
                # Rebuild entity map
                entity_map = {
                    entity.get("id"): i 
                    for i, entity in enumerate(pruned_graph["entities"])
                }
                
                # Re-filter relationships
                connected_relationships = []
                for rel in pruned_graph["relationships"]:
                    source_id = rel.get("source")
                    target_id = rel.get("target")
                    
                    if source_id in entity_map and target_id in entity_map:
                        connected_relationships.append(rel)
                
                pruned_graph["relationships"] = connected_relationships
            
            # Store pruned graph
            pruned_graph_id = self.store_graph(pruned_graph)
            
            # Add pruning details to original graph's metadata
            original_metadata = self._load_metadata(graph_id)
            if original_metadata:
                if "pruned_versions" not in original_metadata:
                    original_metadata["pruned_versions"] = []
                
                original_metadata["pruned_versions"].append({
                    "pruned_graph_id": pruned_graph_id,
                    "timestamp": datetime.now().isoformat(),
                    "criteria": criteria
                })
                
                # Update original graph metadata
                self._store_metadata(graph_id, original_metadata)
            
            logger.info(f"Pruned graph {graph_id} to {pruned_graph_id} " +
                       f"(entities: {len(original_graph.get('entities', []))} → {len(pruned_graph['entities'])}, " +
                       f"relationships: {len(original_graph.get('relationships', []))} → {len(pruned_graph['relationships'])})")
            
            return pruned_graph_id
            
        except Exception as e:
            logger.error(f"Error pruning graph {graph_id}: {str(e)}")
            return None

    def hybrid_search(self, query, graph_id=None, vector_weight=0.7, graph_weight=0.3):
        """
        Perform hybrid search combining structured KG search and vector similarity
        
        Args:
            query: User query
            graph_id: ID of the graph to search (if None, searches all graphs)
            vector_weight: Weight for vector similarity component
            graph_weight: Weight for graph-based relevance
        """
        # Get graph data
        if graph_id:
            graph = self.retrieve_graph(graph_id)
            if not graph:
                return []
            graphs = [graph]
        else:
            # Use all available graphs
            graphs = [self.retrieve_graph(g_id) for g_id in self.list_graph_ids()]
        
        # Use KG interface for semantic search
        from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
        kg_interface = KnowledgeGraphLLMInterface()
        
        # Find semantically relevant entities in all graphs
        all_entities = []
        for graph in graphs:
            if graph:
                entities = kg_interface.semantic_graph_query(query, graph)
                all_entities.extend(entities)
        
        # Return results
        return all_entities


# Placeholder classes for development - will be replaced with actual implementations
class DummyTripleStore:
    """Temporary in-memory triple store for development."""
    
    def __init__(self):
        self.data = {}
        
    def store(self, relationships, graph_id):
        self.data[graph_id] = relationships
        return graph_id
        
    def retrieve(self, graph_id):
        return self.data.get(graph_id, [])
        
    def query(self, pattern, limit=100):
        # Simple dummy implementation
        result = []
        for graph_id, rels in self.data.items():
            result.extend(rels[:limit])
        return result[:limit]
        
    def update(self, graph_id, relationships):
        if graph_id in self.data:
            self.data[graph_id] = relationships
        return True
        
    def get_relationships_for_entities(self, entity_ids):
        result = []
        for rels in self.data.values():
            for rel in rels:
                if rel['source'] in entity_ids or rel['target'] in entity_ids:
                    result.append(rel)
        return result


class DummyVectorStore:
    """Temporary in-memory vector store for development."""
    
    def __init__(self):
        self.data = {}
        self.vectors = {}
        
    def store(self, entities, embeddings, graph_id):
        self.data[graph_id] = entities
        self.vectors[graph_id] = embeddings
        return graph_id
        
    def retrieve(self, graph_id):
        return self.data.get(graph_id, [])
        
    def search(self, vector, limit=10, filters=None):
        # Simple dummy implementation
        result = []
        for graph_id, entities in self.data.items():
            result.extend(entities[:limit])
        return result[:limit]
        
    def update(self, graph_id, entities, embeddings):
        if graph_id in self.data:
            self.data[graph_id] = entities
            self.vectors[graph_id] = embeddings
        return True
        
    def get_by_ids(self, entity_ids):
        result = []
        for entities in self.data.values():
            for entity in entities:
                if entity['id'] in entity_ids:
                    result.append(entity)
        return result
