"""
Graph builder for knowledge graph construction.

This module provides the GraphBuilder class that constructs knowledge graphs
from entities and relationships extracted from various data sources.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

import networkx as nx
from django.conf import settings

from models_app.knowledge_graph.interfaces import GraphBuilderInterface
from models_app.knowledge_graph.entity_resolution import EntityResolver
from models_app.knowledge_graph.ontology_manager import OntologyManager
from analytics_app.utils import monitor_kg_performance
logger = logging.getLogger(__name__)

class GraphBuilder(GraphBuilderInterface):
    """
    Builds knowledge graphs from entities and relationships.
    
    The GraphBuilder is responsible for:
    - Creating a coherent knowledge graph structure
    - Managing entity and relationship IDs
    - Merging subgraphs into larger knowledge graphs
    - Handling graph metadata
    
    Configuration options:
    - DEFAULT_GRAPH_TYPE: Default type of graph to create (directed, undirected)
    - ENABLE_GRAPH_VALIDATION: Whether to validate graphs for consistency
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.entity_resolver = EntityResolver(config)
        
        # Load config from settings if not provided
        if not self.config:
            self.config = {
                'graph_type': getattr(settings, 'KNOWLEDGE_GRAPH_TYPE', 'directed'),
                'validate_graphs': getattr(settings, 'KNOWLEDGE_GRAPH_VALIDATE', True),
                'default_context': getattr(settings, 'KNOWLEDGE_GRAPH_DEFAULT_CONTEXT', 'general')
            }
            
        logger.info(f"GraphBuilder initialized with config: {self.config}")
    
    def add_entity(self, entity: Dict[str, Any]) -> str:
        """
        Add an entity to the knowledge graph with ontology validation.
        
        This method creates a standalone entity that can later be
        incorporated into a graph. It ensures the entity has a valid ID
        and all required attributes.
        
        Args:
            entity: The entity to add
            
        Returns:
            The ID of the added entity
        """
        # Validate against ontology
        ontology_manager = OntologyManager()
        is_valid, messages = ontology_manager.validate_entity(entity)
        
        if not is_valid:
            logger.warning(f"Adding invalid entity: {messages}")
            # You might want to add a validation_errors field to the entity
            if "metadata" not in entity:
                entity["metadata"] = {}
            entity["metadata"]["validation_errors"] = messages
        
        # Ensure the entity has an ID
        entity_id = entity.get('id')
        if not entity_id:
            entity_id = f"entity_{uuid.uuid4().hex[:8]}"
            entity['id'] = entity_id
            
        # Ensure the entity has a type
        if 'type' not in entity:
            entity['type'] = 'unknown'
            
        # Ensure the entity has a timestamp
        if 'timestamp' not in entity:
            entity['timestamp'] = datetime.now().isoformat()
            
        # Add metadata if missing
        if 'metadata' not in entity:
            entity['metadata'] = {}
            
        # Add attributes if missing
        if 'attributes' not in entity:
            entity['attributes'] = {}
            
        return entity_id
    
    def add_relationship(self, relationship: Dict[str, Any]) -> str:
        """
        Add a relationship to the knowledge graph.
        
        This method creates a standalone relationship that can later be
        incorporated into a graph. It ensures the relationship has a valid ID
        and all required attributes.
        
        Args:
            relationship: The relationship to add
            
        Returns:
            The ID of the added relationship
        """
        # Ensure the relationship has an ID
        relationship_id = relationship.get('id')
        if not relationship_id:
            relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
            relationship['id'] = relationship_id
            
        # Ensure the relationship has a type
        if 'type' not in relationship:
            relationship['type'] = 'unknown'
            
        # Ensure source and target are present
        if 'source' not in relationship or 'target' not in relationship:
            raise ValueError("Relationship must have 'source' and 'target' fields")
            
        # Ensure the relationship has a timestamp
        if 'timestamp' not in relationship:
            relationship['timestamp'] = datetime.now().isoformat()
            
        # Add confidence if missing
        if 'confidence' not in relationship:
            relationship['confidence'] = 0.5
            
        # Add attributes if missing
        if 'attributes' not in relationship:
            relationship['attributes'] = {}
            
        # Add modality if missing
        if 'modality' not in relationship:
            relationship['modality'] = 'default'
            
        return relationship_id
    
    @monitor_kg_performance
    def build_subgraph(self, entities: List[Dict[str, Any]], 
                      relationships: List[Dict[str, Any]],
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a subgraph from entities and relationships.
        
        This method creates a coherent subgraph from a set of entities
        and relationships, with optional contextual information.
        
        Args:
            entities: List of entities to include in the graph
            relationships: List of relationships to include in the graph
            context: Optional context dictionary
            
        Returns:
            A dictionary representing the knowledge graph
        """
        if not entities and not relationships:
            logger.warning("Attempted to build empty graph")
            return self._create_empty_graph(context)
            
        # First resolve duplicate entities
        resolved_entities = self.entity_resolver.resolve_entities(entities)
        
        # Process all entities to ensure they have valid IDs
        processed_entities = []
        entity_ids = set()
        entity_map = {}
        
        for entity in resolved_entities:
            # Ensure entity has a valid ID
            entity_id = self.add_entity(entity)
            entity_ids.add(entity_id)
            
            # Add to processed list and map
            processed_entities.append(entity)
            entity_map[entity_id] = entity
            
        # Process all relationships and validate references
        processed_relationships = []
        valid_relationships = []
        
        for relationship in relationships:
            # Ensure relationship has a valid ID
            relationship_id = self.add_relationship(relationship)
            
            # Check if source and target entities exist
            source_id = relationship.get('source')
            target_id = relationship.get('target')
            
            if source_id not in entity_ids:
                logger.warning(f"Relationship {relationship_id} references non-existent source entity {source_id}")
                if self.config.get('validate_graphs', True):
                    continue
                    
            if target_id not in entity_ids:
                logger.warning(f"Relationship {relationship_id} references non-existent target entity {target_id}")
                if self.config.get('validate_graphs', True):
                    continue
                    
            # Add to processed list and valid list
            processed_relationships.append(relationship)
            
            if source_id in entity_ids and target_id in entity_ids:
                valid_relationships.append(relationship)
                
        # Create the subgraph structure
        graph = {
            'id': f"graph_{uuid.uuid4().hex[:8]}",
            'type': self.config.get('graph_type', 'directed'),
            'entities': processed_entities,
            'relationships': valid_relationships,
            'metadata': {
                'entity_count': len(processed_entities),
                'relationship_count': len(valid_relationships),
                'context': context or self.config.get('default_context', 'general'),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add context information if provided
        if context:
            # Ensure graph metadata has required fields
            if 'metadata' not in graph:
                graph['metadata'] = {}
                
            # Update metadata with context
            graph['metadata'].update(context)
            
        # Optional graph consistency check
        if self.config.get('validate_graphs', True):
            self._validate_graph(graph)
            
        return graph
    
    @monitor_kg_performance
    def merge_subgraphs(self, subgraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple subgraphs into a single graph.
        
        This method combines multiple subgraphs, resolving entity and relationship
        conflicts, and creating a unified knowledge graph.
        
        Args:
            subgraphs: List of subgraphs to merge
            
        Returns:
            The merged knowledge graph
        """
        if not subgraphs:
            logger.warning("Attempted to merge empty graph list")
            return self._create_empty_graph()
            
        if len(subgraphs) == 1:
            return subgraphs[0]
            
        # Initialize with empty collections
        merged_entities = []
        merged_relationships = []
        entity_map = {}  # Maps entity IDs to their merged versions
        relationship_signatures = set()  # Tracks relationship signatures to avoid duplicates
        
        # Collect all graph metadata
        graph_metadata = {
            'sources': [],
            'timestamp': datetime.now().isoformat(),
            'entity_count': 0,
            'relationship_count': 0
        }
        
        # Process each subgraph
        for subgraph in subgraphs:
            # Skip invalid subgraphs
            if not isinstance(subgraph, dict):
                logger.warning(f"Invalid subgraph format: {type(subgraph)}")
                continue
                
            # Extract entities
            entities = subgraph.get('entities', [])
            
            # Extract relationships
            relationships = subgraph.get('relationships', [])
            
            # Extract metadata
            metadata = subgraph.get('metadata', {})
            
            # Add source to graph metadata
            graph_metadata['sources'].append(subgraph.get('id', 'unknown'))
            
            # Process entities
            for entity in entities:
                entity_id = entity.get('id')
                
                if not entity_id:
                    continue
                    
                # Check if entity already exists
                if entity_id in entity_map:
                    # Update existing entity with new information
                    merged_entity = entity_map[entity_id]
                    self._merge_entity_attributes(merged_entity, entity)
                else:
                    # Add new entity
                    merged_entities.append(entity)
                    entity_map[entity_id] = entity
                    
            # Process relationships
            for relationship in relationships:
                source_id = relationship.get('source')
                target_id = relationship.get('target')
                rel_type = relationship.get('type')
                
                if not source_id or not target_id or not rel_type:
                    continue
                    
                # Create a signature to detect duplicates
                signature = f"{source_id}|{target_id}|{rel_type}"
                
                if signature in relationship_signatures:
                    # Relationship already exists
                    continue
                    
                # Add new relationship
                merged_relationships.append(relationship)
                relationship_signatures.add(signature)
                
        # Update metadata counts
        graph_metadata['entity_count'] = len(merged_entities)
        graph_metadata['relationship_count'] = len(merged_relationships)
        
        # Create merged graph
        merged_graph = {
            'id': f"merged_{uuid.uuid4().hex[:8]}",
            'type': self.config.get('graph_type', 'directed'),
            'entities': merged_entities,
            'relationships': merged_relationships,
            'metadata': graph_metadata
        }
        
        # Optional graph consistency check
        if self.config.get('validate_graphs', True):
            self._validate_graph(merged_graph)
            
        return merged_graph
    
    def _create_empty_graph(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an empty graph structure."""
        graph = {
            'id': f"graph_{uuid.uuid4().hex[:8]}",
            'type': self.config.get('graph_type', 'directed'),
            'entities': [],
            'relationships': [],
            'metadata': {
                'entity_count': 0,
                'relationship_count': 0,
                'context': context or self.config.get('default_context', 'general'),
                'timestamp': datetime.now().isoformat(),
                'is_empty': True
            }
        }
        
        return graph
    
    def _validate_graph(self, graph: Dict[str, Any]) -> bool:
        """
        Validate graph for consistency.
        
        Args:
            graph: The graph to validate
            
        Returns:
            True if the graph is valid, False otherwise
        """
        try:
            # Check if required fields are present
            required_fields = ['id', 'entities', 'relationships', 'metadata']
            for field in required_fields:
                if field not in graph:
                    logger.warning(f"Graph missing required field: {field}")
                    return False
                    
            # Get entity IDs
            entity_ids = {entity['id'] for entity in graph['entities'] if 'id' in entity}
            
            # Check relationships reference valid entities
            for relationship in graph['relationships']:
                source_id = relationship.get('source')
                target_id = relationship.get('target')
                
                if source_id not in entity_ids:
                    logger.warning(f"Relationship references non-existent source entity: {source_id}")
                    return False
                    
                if target_id not in entity_ids:
                    logger.warning(f"Relationship references non-existent target entity: {target_id}")
                    return False
                    
            # Build a networkx graph to check for additional issues
            if self._create_networkx_graph(graph):
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error validating graph: {e}")
            return False
    
    def _create_networkx_graph(self, graph: Dict[str, Any]) -> Optional[nx.Graph]:
        """
        Convert the graph structure to a NetworkX graph.
        
        Args:
            graph: The graph dictionary
            
        Returns:
            NetworkX graph object or None if conversion fails
        """
        try:
            # Create directed or undirected graph
            if graph.get('type') == 'directed':
                G = nx.DiGraph()
            else:
                G = nx.Graph()
                
            # Add nodes (entities)
            for entity in graph['entities']:
                entity_id = entity.get('id')
                if entity_id:
                    G.add_node(entity_id, **entity)
                    
            # Add edges (relationships)
            for relationship in graph['relationships']:
                source_id = relationship.get('source')
                target_id = relationship.get('target')
                
                if source_id and target_id:
                    G.add_edge(source_id, target_id, **relationship)
                    
            return G
            
        except Exception as e:
            logger.error(f"Error creating NetworkX graph: {e}")
            return None
    
    def _merge_entity_attributes(self, target_entity: Dict[str, Any], source_entity: Dict[str, Any]) -> None:
        """
        Merge attributes from source entity into target entity.
        
        Args:
            target_entity: The entity to merge into
            source_entity: The entity to merge from
        """
        # For each attribute in source entity
        for key, value in source_entity.items():
            if key == 'id':
                # Skip ID field
                continue
                
            if key not in target_entity:
                # Copy missing attributes
                target_entity[key] = value
            elif key == 'attributes' and isinstance(value, dict) and isinstance(target_entity[key], dict):
                # Merge attributes dictionaries
                target_entity[key].update(value)
            elif key == 'metadata' and isinstance(value, dict) and isinstance(target_entity[key], dict):
                # Merge metadata dictionaries
                target_entity[key].update(value)
            elif key == 'confidence' and isinstance(value, (int, float)) and isinstance(target_entity[key], (int, float)):
                # Take the higher confidence value
                target_entity[key] = max(target_entity[key], value)
            elif key == 'timestamp' and target_entity[key] < value:
                # Take the more recent timestamp
                target_entity[key] = value
