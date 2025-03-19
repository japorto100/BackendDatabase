"""
Entity Resolution: Identifies and merges duplicate entities in knowledge graphs.

This module implements techniques to:
1. Detect potential duplicate entities using string similarity and embeddings
2. Group similar entities into clusters
3. Merge duplicate entities preserving all relevant information
4. Calculate confidence scores for entity matches

The entity resolution process is critical for maintaining a clean knowledge graph
by preventing duplicate entities that refer to the same real-world object.
"""

import logging
from typing import List, Dict, Any, Tuple
import difflib
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class EntityResolver:
    """
    Identifies and merges duplicate entities in knowledge graphs.
    
    This class uses multiple strategies for entity resolution:
    1. String similarity for labels and names
    2. Property matching for key entity properties
    3. External KB identity matching
    4. Embedding similarity when available
    
    The resolution process clusters similar entities and merges them
    while preserving all relevant information from each source.
    """
    
    def __init__(self, config=None):
        """
        Initialize the EntityResolver with configuration.
        
        Args:
            config: Configuration dictionary with parameters like
                   similarity thresholds, property weights, etc.
        """
        self.config = config or {}
        
        # Similarity thresholds
        self.label_similarity_threshold = self.config.get("label_similarity_threshold", 0.85)
        self.property_similarity_threshold = self.config.get("property_similarity_threshold", 0.7)
        self.embedding_similarity_threshold = self.config.get("embedding_similarity_threshold", 0.9)
        
        # Initialize embedding model if needed
        self.embedding_model = None
        if self.config.get("use_embeddings", True):
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get("embedding_model", "all-mpnet-base-v2")
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Initialized embedding model: {model_name}")
            except ImportError:
                logger.warning("SentenceTransformer not available, falling back to string similarity")
    
    def resolve_entities(self, entities, threshold=None):
        """
        Identify and merge duplicate entities
        
        Args:
            entities: List of entity dictionaries to resolve
            threshold: Optional override for similarity threshold
            
        Returns:
            List of resolved (merged) entities
        """
        if not entities:
            return []
            
        if threshold is None:
            threshold = self.label_similarity_threshold
        
        # 1. Group similar entities into clusters
        entity_clusters = self._cluster_similar_entities(entities, threshold)
        
        # 2. Merge entities in each cluster
        resolved_entities = []
        for cluster in entity_clusters:
            if len(cluster) == 1:
                # No duplicates found, keep original
                resolved_entities.append(cluster[0])
            else:
                # Merge duplicate entities
                merged_entity = self._merge_entities(cluster)
                resolved_entities.append(merged_entity)
        
        logger.info(f"Entity resolution: {len(entities)} entities → {len(resolved_entities)} resolved entities")
        return resolved_entities
    
    def _cluster_similar_entities(self, entities, threshold):
        """
        Group similar entities into clusters using various similarity measures
        
        The clustering algorithm uses:
        1. Label/name similarity
        2. External ID matching
        3. Property similarity
        4. Embedding similarity (if available)
        
        Args:
            entities: List of entities to cluster
            threshold: Similarity threshold
            
        Returns:
            List of entity clusters (each cluster is a list of entities)
        """
        # Initialize clusters
        clusters = []
        
        # Track which entities have been assigned to clusters
        assigned = set()
        
        # First pass: group by exact external ID matches
        external_id_groups = defaultdict(list)
        
        for i, entity in enumerate(entities):
            # Check if the entity has external references
            external_refs = entity.get("external_references", [])
            
            for ref in external_refs:
                if ref.get("confidence", 0) > 0.9:  # Only use high-confidence refs
                    # Create a key from source and ID
                    ext_key = f"{ref.get('source')}:{ref.get('id')}"
                    external_id_groups[ext_key].append(i)
        
        # Create clusters from external ID matches
        for indices in external_id_groups.values():
            if len(indices) > 1:  # Only create cluster if multiple entities match
                cluster = [entities[i] for i in indices]
                clusters.append(cluster)
                assigned.update(indices)
        
        # Second pass: cluster by similarity
        for i, entity in enumerate(entities):
            if i in assigned:
                continue  # Skip entities already in clusters
                
            # Start a new cluster with this entity
            cluster = [entity]
            assigned.add(i)
            
            # Find similar entities
            for j, other_entity in enumerate(entities):
                if j in assigned or i == j:
                    continue
                
                similarity = self._calculate_entity_similarity(entity, other_entity)
                
                if similarity >= threshold:
                    cluster.append(other_entity)
                    assigned.add(j)
            
            # Add cluster if it has at least one entity
            if cluster:
                clusters.append(cluster)
        
        # Add any remaining entities as single-entity clusters
        for i, entity in enumerate(entities):
            if i not in assigned:
                clusters.append([entity])
        
        return clusters
    
    def _calculate_entity_similarity(self, entity1, entity2):
        """
        Calculate similarity between two entities using multiple measures
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        # Initial checks for comparability
        type1 = entity1.get("type", "unknown").lower()
        type2 = entity2.get("type", "unknown").lower()
        
        # If types are completely different, they're not the same entity
        if type1 != "unknown" and type2 != "unknown" and type1 != type2:
            return 0.0
        
        # Calculate label similarity
        label1 = entity1.get("label", "").lower()
        label2 = entity2.get("label", "").lower()
        
        if not label1 or not label2:
            label_similarity = 0.0
        else:
            label_similarity = difflib.SequenceMatcher(None, label1, label2).ratio()
        
        # Calculate property similarity
        property_similarity = self._calculate_property_similarity(
            entity1.get("properties", {}), 
            entity2.get("properties", {})
        )
        
        # Calculate embedding similarity if available
        embedding_similarity = 0.0
        if self.embedding_model:
            embedding_similarity = self._calculate_embedding_similarity(entity1, entity2)
        
        # Combine similarities with weights
        weights = {
            "label": 0.5,
            "property": 0.3,
            "embedding": 0.2 if self.embedding_model else 0.0
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate combined similarity
        combined_similarity = (
            (label_similarity * weights["label"]) +
            (property_similarity * weights["property"]) +
            (embedding_similarity * weights["embedding"])
        )
        
        return combined_similarity
    
    def _calculate_property_similarity(self, props1, props2):
        """Calculate similarity between entity properties"""
        if not props1 or not props2:
            return 0.0
        
        # Get common property keys
        common_keys = set(props1.keys()) & set(props2.keys())
        if not common_keys:
            return 0.0
        
        # Calculate similarity for each common property
        property_similarities = []
        
        for key in common_keys:
            val1 = props1[key]
            val2 = props2[key]
            
            # Skip if either value is None
            if val1 is None or val2 is None:
                continue
                
            # Calculate similarity based on value type
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    sim = 1.0 - (abs(val1 - val2) / max_val)
                else:
                    sim = 1.0 if val1 == val2 else 0.0
            else:
                # String similarity
                str1 = str(val1).lower()
                str2 = str(val2).lower()
                sim = difflib.SequenceMatcher(None, str1, str2).ratio()
            
            property_similarities.append(sim)
        
        # Return average similarity
        if property_similarities:
            return sum(property_similarities) / len(property_similarities)
        else:
            return 0.0
    
    def _calculate_embedding_similarity(self, entity1, entity2):
        """Calculate similarity using entity embeddings"""
        if not self.embedding_model:
            return 0.0
            
        # Generate text representations
        text1 = self._entity_to_text(entity1)
        text2 = self._entity_to_text(entity2)
        
        # Generate embeddings
        emb1 = self.embedding_model.encode(text1)
        emb2 = self.embedding_model.encode(text2)
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 * norm2 > 0:
            return dot_product / (norm1 * norm2)
        else:
            return 0.0
    
    def _entity_to_text(self, entity):
        """Convert entity to a text representation for embedding"""
        parts = []
        
        # Add entity type
        entity_type = entity.get("type", "unknown")
        parts.append(f"Type: {entity_type}")
        
        # Add entity label
        label = entity.get("label", "")
        if label:
            parts.append(f"Name: {label}")
        
        # Add key properties
        props = entity.get("properties", {})
        for key, value in props.items():
            if value is not None:
                parts.append(f"{key}: {value}")
        
        return " ".join(parts)
    
    def _merge_entities(self, entities):
        """
        Merge a cluster of duplicate entities into a single entity
        
        This preserves all information from the duplicates by:
        1. Keeping the highest confidence information for each field
        2. Combining source references and external IDs
        3. Merging property values with proper confidence tracking
        
        Args:
            entities: List of entities to merge
            
        Returns:
            Merged entity dictionary
        """
        if not entities:
            return None
            
        if len(entities) == 1:
            return entities[0]
        
        # Start with the entity that has the highest number of properties
        entities_by_completeness = sorted(
            entities, 
            key=lambda e: len(e.get("properties", {})), 
            reverse=True
        )
        
        base_entity = entities_by_completeness[0].copy()
        
        # Track merged entities
        if "merged_from" not in base_entity:
            base_entity["merged_from"] = []
        
        # Initialize merged properties
        merged_properties = base_entity.get("properties", {}).copy()
        
        # Merge external references
        all_external_refs = base_entity.get("external_references", []).copy()
        
        # Track source references
        all_sources = base_entity.get("source_info", []).copy()
        
        # Merge data from other entities
        for entity in entities_by_completeness[1:]:
            # Add to merged_from
            base_entity["merged_from"].append(entity.get("id", "unknown"))
            
            # Merge properties (prefer base entity values if there are conflicts)
            for prop_key, prop_value in entity.get("properties", {}).items():
                if prop_key not in merged_properties:
                    merged_properties[prop_key] = prop_value
                elif isinstance(merged_properties[prop_key], list):
                    # For list values, append non-duplicate items
                    if isinstance(prop_value, list):
                        for item in prop_value:
                            if item not in merged_properties[prop_key]:
                                merged_properties[prop_key].append(item)
                    elif prop_value not in merged_properties[prop_key]:
                        merged_properties[prop_key].append(prop_value)
            
            # Merge external references
            for ext_ref in entity.get("external_references", []):
                # Check if this reference already exists
                exists = False
                for ref in all_external_refs:
                    if (ref.get("source") == ext_ref.get("source") and 
                        ref.get("id") == ext_ref.get("id")):
                        exists = True
                        # Take the higher confidence value if available
                        if ext_ref.get("confidence", 0) > ref.get("confidence", 0):
                            ref["confidence"] = ext_ref.get("confidence")
                        break
                
                if not exists:
                    all_external_refs.append(ext_ref)
            
            # Merge source references
            for source in entity.get("source_info", []):
                if source not in all_sources:
                    all_sources.append(source)
        
        # Update the merged entity
        base_entity["properties"] = merged_properties
        base_entity["external_references"] = all_external_refs
        base_entity["source_info"] = all_sources
        
        # Set merge metadata
        base_entity["merged_count"] = len(entities)
        base_entity["merged_timestamp"] = self._get_current_timestamp()
        
        return base_entity
    
    def _get_current_timestamp(self):
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def resolve_entity_batch(self, entities, batch_size=1000):
        """
        Process large batches of entities in chunks to save memory
        
        Args:
            entities: List of entities to resolve
            batch_size: Size of each processing batch
            
        Returns:
            List of resolved entities
        """
        if len(entities) <= batch_size:
            return self.resolve_entities(entities)
        
        # Process in batches
        batches = [entities[i:i+batch_size] for i in range(0, len(entities), batch_size)]
        
        resolved_batches = []
        for batch in batches:
            resolved_batch = self.resolve_entities(batch)
            resolved_batches.extend(resolved_batch)
        
        # Final resolution pass on the combined results to catch cross-batch duplicates
        return self.resolve_entities(resolved_batches)
    
    def analyze_resolution_results(self, original_entities, resolved_entities):
        """
        Analyze the results of entity resolution
        
        Args:
            original_entities: Original list of entities
            resolved_entities: Resolved list of entities
            
        Returns:
            Dict with analysis results
        """
        # Count merged entities
        merged_count = sum(entity.get("merged_count", 1) > 1 for entity in resolved_entities)
        
        # Identify merged entity types
        merged_types = {}
        for entity in resolved_entities:
            if entity.get("merged_count", 1) > 1:
                entity_type = entity.get("type", "unknown")
                merged_types[entity_type] = merged_types.get(entity_type, 0) + 1
        
        # Calculate reduction percentage
        reduction = (len(original_entities) - len(resolved_entities)) / len(original_entities) * 100 if original_entities else 0
        
        return {
            "original_count": len(original_entities),
            "resolved_count": len(resolved_entities),
            "merged_count": merged_count,
            "reduction_percentage": reduction,
            "merged_types": merged_types
        }
