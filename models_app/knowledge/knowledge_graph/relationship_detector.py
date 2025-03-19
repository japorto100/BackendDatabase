"""
Relationship detection for knowledge graph building.

This module provides functionality to detect relationships between entities
extracted from various data sources including text, images, and ColPali output.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import uuid
from datetime import datetime
import math

import numpy as np
from django.conf import settings

from models_app.knowledge_graph.interfaces import RelationshipDetectorInterface

logger = logging.getLogger(__name__)

class RelationshipDetector(RelationshipDetectorInterface):
    """
    Detects relationships between entities for knowledge graph construction.
    
    The detector can identify:
    - Text-based relationships: Using NLP techniques to extract relationships from text
    - Visual relationships: Using spatial and visual information to identify relationships
    - Cross-modal relationships: Connecting entities across modalities (text + visual)
    
    Configuration options:
    - MIN_RELATIONSHIP_CONFIDENCE: Minimum confidence threshold for relationships
    - SPATIAL_RELATIONSHIP_TYPES: Types of spatial relationships to detect
    - SEMANTIC_SIMILARITY_THRESHOLD: Threshold for considering entities semantically related
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the relationship detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load config from settings if not provided
        if not self.config:
            self.config = {
                'min_confidence': getattr(settings, 'RELATIONSHIP_DETECTOR_MIN_CONFIDENCE', 0.5),
                'spatial_relationship_types': getattr(settings, 'RELATIONSHIP_DETECTOR_SPATIAL_TYPES', 
                                                   ['contains', 'above', 'below', 'next_to', 'overlaps']),
                'semantic_similarity_threshold': getattr(settings, 'RELATIONSHIP_DETECTOR_SIM_THRESHOLD', 0.7),
                'max_textual_distance': getattr(settings, 'RELATIONSHIP_DETECTOR_MAX_TEXT_DISTANCE', 50),
            }
            
        logger.info(f"RelationshipDetector initialized with config: {self.config}")
    
    def detect_relationships(self, entities: List[Dict[str, Any]], 
                            context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect relationships between entities.
        
        This method focuses on text-based relationships, identifying connections
        between entities based on their proximity in text, semantic similarity,
        and syntactic patterns.
        
        Args:
            entities: List of entities to analyze
            context: Optional contextual information
            
        Returns:
            List of detected relationships
        """
        if not entities:
            return []
            
        relationships = []
        
        # Detect co-occurrence relationships
        cooccurrence_relationships = self._detect_cooccurrence_relationships(entities, context)
        relationships.extend(cooccurrence_relationships)
        
        # Detect semantic similarity relationships
        semantic_relationships = self._detect_semantic_relationships(entities, context)
        relationships.extend(semantic_relationships)
        
        # Detect syntactic relationships if text is available in context
        if context and 'text' in context:
            syntactic_relationships = self._detect_syntactic_relationships(entities, context['text'])
            relationships.extend(syntactic_relationships)
            
        # Add hierarchy relationships for organizational structures
        hierarchy_relationships = self._detect_hierarchy_relationships(entities, context)
        relationships.extend(hierarchy_relationships)
        
        # Deduplicate relationships
        return self._deduplicate_relationships(relationships)
    
    def detect_visual_relationships(self, visual_entities: List[Dict[str, Any]],
                                   layout_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect relationships between visual entities based on layout.
        
        This method analyzes spatial relationships between visual entities,
        identifying patterns like containment, adjacency, and alignment.
        
        Args:
            visual_entities: List of visual entities with region information
            layout_info: Information about the document layout
            
        Returns:
            List of detected visual relationships
        """
        if not visual_entities:
            return []
            
        relationships = []
        
        # Filter entities that have region information
        entities_with_regions = [
            entity for entity in visual_entities 
            if 'region' in entity or 'bbox' in entity
        ]
        
        if not entities_with_regions:
            return []
            
        # Normalize region format
        entities_with_normalized_regions = []
        for entity in entities_with_regions:
            normalized_entity = entity.copy()
            
            # If region is in bbox format, convert to normalized coordinates
            if 'bbox' in entity and 'region' not in entity:
                bbox = entity['bbox']
                
                # Get document dimensions
                doc_width = layout_info.get('width', 1.0)
                doc_height = layout_info.get('height', 1.0)
                
                # Normalize bbox to (x1, y1, x2, y2) in 0-1 range
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    normalized_entity['region'] = (
                        x1 / doc_width,
                        y1 / doc_height,
                        x2 / doc_width,
                        y2 / doc_height
                    )
                    
            entities_with_normalized_regions.append(normalized_entity)
            
        # Detect spatial relationships
        for i, entity1 in enumerate(entities_with_normalized_regions):
            for j, entity2 in enumerate(entities_with_normalized_regions):
                if i == j:
                    continue
                    
                # Get regions
                region1 = entity1.get('region')
                region2 = entity2.get('region')
                
                if not region1 or not region2:
                    continue
                    
                # Analyze spatial relationship
                spatial_relations = self._analyze_spatial_relationship(region1, region2)
                
                for relation_type, confidence in spatial_relations:
                    # Skip low confidence relationships
                    if confidence < self.config.get('min_confidence', 0.5):
                        continue
                        
                    # Create relationship record
                    relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
                    relationship = {
                        'id': relationship_id,
                        'type': relation_type,
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'confidence': confidence,
                        'modality': 'visual',
                        'attributes': {
                            'spatial': True,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    relationships.append(relationship)
                    
        # Detect alignment relationships (e.g., same row, same column)
        alignment_relationships = self._detect_alignment_relationships(
            entities_with_normalized_regions, layout_info
        )
        relationships.extend(alignment_relationships)
        
        # Detect visual similarity relationships
        similarity_relationships = self._detect_visual_similarity_relationships(visual_entities)
        relationships.extend(similarity_relationships)
        
        return relationships
    
    def merge_text_and_visual_relationships(self, 
                                          text_relationships: List[Dict[str, Any]],
                                          visual_relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge text-based and visual relationships.
        
        This method connects entities across modalities, identifying
        relationships between textual and visual entities.
        
        Args:
            text_relationships: Relationships from textual entities
            visual_relationships: Relationships from visual entities
            
        Returns:
            Merged list of relationships
        """
        if not text_relationships and not visual_relationships:
            return []
            
        # Start with all existing relationships
        all_relationships = text_relationships + visual_relationships
        
        # Extract all entities referenced in relationships
        text_entities = set()
        visual_entities = set()
        
        for rel in text_relationships:
            text_entities.add(rel['source'])
            text_entities.add(rel['target'])
            
        for rel in visual_relationships:
            visual_entities.add(rel['source'])
            visual_entities.add(rel['target'])
            
        # Create cross-modal relationships
        cross_modal_relationships = []
        
        # For a real implementation, this would involve more sophisticated
        # cross-modal matching, such as comparing embeddings or using 
        # contextual information to link text and visual entities
        
        # For this simplified implementation, we'll create relationships
        # between entities that might be related semantically
        # In a real system, this would use multi-modal embeddings for matching
        
        # Add all relationships together
        merged_relationships = all_relationships + cross_modal_relationships
        
        # Deduplicate relationships
        return self._deduplicate_relationships(merged_relationships)
    
    def _detect_cooccurrence_relationships(self, entities: List[Dict[str, Any]], 
                                         context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect co-occurrence relationships between entities based on proximity in text.
        
        Args:
            entities: List of entities
            context: Optional context information
            
        Returns:
            List of co-occurrence relationships
        """
        relationships = []
        
        # Only consider entities with position information
        entities_with_position = [
            entity for entity in entities
            if 'position' in entity and entity.get('source', '').startswith('text')
        ]
        
        # Sort entities by position
        sorted_entities = sorted(
            entities_with_position,
            key=lambda e: e['position'].get('start', 0)
        )
        
        # Maximum distance for considering entities co-occurring
        max_distance = self.config.get('max_textual_distance', 50)
        
        # Detect co-occurrences based on proximity
        for i, entity1 in enumerate(sorted_entities):
            for j in range(i+1, len(sorted_entities)):
                entity2 = sorted_entities[j]
                
                # Calculate distance between entities
                end_pos1 = entity1['position'].get('end', 0)
                start_pos2 = entity2['position'].get('start', 0)
                distance = start_pos2 - end_pos1
                
                # Skip if distance is too large
                if distance > max_distance:
                    break
                    
                # Calculate confidence based on distance
                confidence = max(0.5, 1.0 - (distance / max_distance))
                
                # Create relationship record
                relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
                relationship = {
                    'id': relationship_id,
                    'type': 'co_occurs_with',
                    'source': entity1['id'],
                    'target': entity2['id'],
                    'confidence': confidence,
                    'modality': 'text',
                    'attributes': {
                        'distance': distance,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                relationships.append(relationship)
                
        return relationships
    
    def _detect_semantic_relationships(self, entities: List[Dict[str, Any]], 
                                     context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect semantic relationships based on entity embeddings.
        
        Args:
            entities: List of entities
            context: Optional context information
            
        Returns:
            List of semantic relationships
        """
        relationships = []
        
        # Only consider entities with embeddings
        entities_with_embedding = [
            entity for entity in entities
            if 'embedding' in entity
        ]
        
        # Compute pairwise similarities
        similarity_threshold = self.config.get('semantic_similarity_threshold', 0.7)
        
        for i, entity1 in enumerate(entities_with_embedding):
            for j in range(i+1, len(entities_with_embedding)):
                entity2 = entities_with_embedding[j]
                
                # Skip entities of the same type (to avoid trivial relationships)
                if entity1.get('type') == entity2.get('type'):
                    continue
                    
                # Calculate semantic similarity
                similarity = self._compute_cosine_similarity(
                    entity1['embedding'], entity2['embedding']
                )
                
                # Skip if similarity is below threshold
                if similarity < similarity_threshold:
                    continue
                    
                # Create relationship record
                relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
                relationship = {
                    'id': relationship_id,
                    'type': 'semantically_related_to',
                    'source': entity1['id'],
                    'target': entity2['id'],
                    'confidence': similarity,
                    'modality': 'embedding',
                    'attributes': {
                        'similarity': similarity,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                relationships.append(relationship)
                
        return relationships
    
    def _detect_syntactic_relationships(self, entities: List[Dict[str, Any]], 
                                      text: str) -> List[Dict[str, Any]]:
        """
        Detect syntactic relationships based on text patterns.
        
        Args:
            entities: List of entities
            text: The source text
            
        Returns:
            List of syntactic relationships
        """
        # This is a placeholder for sophisticated syntactic relationship detection
        # In a real implementation, this would use dependency parsing or other
        # NLP techniques to extract relationships from sentence structure
        
        # For now, we'll use some simple patterns to identify relationships
        relationships = []
        text_entities = [
            entity for entity in entities
            if 'text' in entity and entity.get('source', '').startswith('text')
        ]
        
        # Create a mapping of entity text to entity IDs
        text_to_ids = {}
        for entity in text_entities:
            entity_text = entity['text']
            if entity_text not in text_to_ids:
                text_to_ids[entity_text] = []
            text_to_ids[entity_text].append(entity['id'])
            
        # Define some simple patterns for relationships
        patterns = [
            # "X is part of Y"
            (r'([A-Za-z\s]+)\s+is\s+part\s+of\s+([A-Za-z\s]+)', 'part_of'),
            # "X belongs to Y"
            (r'([A-Za-z\s]+)\s+belongs\s+to\s+([A-Za-z\s]+)', 'belongs_to'),
            # "X contains Y"
            (r'([A-Za-z\s]+)\s+contains\s+([A-Za-z\s]+)', 'contains'),
            # "X works for Y"
            (r'([A-Za-z\s]+)\s+works\s+for\s+([A-Za-z\s]+)', 'works_for'),
            # "X is located in Y"
            (r'([A-Za-z\s]+)\s+is\s+located\s+in\s+([A-Za-z\s]+)', 'located_in'),
        ]
        
        # Search for patterns in text
        for pattern, relationship_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                # Find entities matching the extracted text
                source_ids = []
                target_ids = []
                
                for entity_text, entity_ids in text_to_ids.items():
                    if entity_text.lower() in source_text.lower() or source_text.lower() in entity_text.lower():
                        source_ids.extend(entity_ids)
                    if entity_text.lower() in target_text.lower() or target_text.lower() in entity_text.lower():
                        target_ids.extend(entity_ids)
                        
                # Create relationships between matching entities
                for source_id in source_ids:
                    for target_id in target_ids:
                        if source_id != target_id:
                            relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
                            relationship = {
                                'id': relationship_id,
                                'type': relationship_type,
                                'source': source_id,
                                'target': target_id,
                                'confidence': 0.8,  # Fixed confidence for pattern matches
                                'modality': 'text',
                                'attributes': {
                                    'pattern': pattern,
                                    'match_text': match.group(0),
                                    'timestamp': datetime.now().isoformat()
                                }
                            }
                            relationships.append(relationship)
                            
        return relationships
    
    def _detect_hierarchy_relationships(self, entities: List[Dict[str, Any]], 
                                      context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect hierarchical relationships between entities.
        
        Args:
            entities: List of entities
            context: Optional context information
            
        Returns:
            List of hierarchical relationships
        """
        # This method would implement domain-specific hierarchy detection
        # For example, in an organizational context:
        # - Person works for Organization
        # - Department is part of Organization
        # - Manager supervises Employee
        
        # For now, return an empty list as a placeholder
        return []
    
    def _analyze_spatial_relationship(self, region1: Tuple[float, float, float, float], 
                                    region2: Tuple[float, float, float, float]) -> List[Tuple[str, float]]:
        """
        Analyze spatial relationship between two regions.
        
        Args:
            region1: First region as (x1, y1, x2, y2)
            region2: Second region as (x1, y1, x2, y2)
            
        Returns:
            List of (relationship_type, confidence) tuples
        """
        # Unpack regions
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        # Calculate region areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection
        intersect_x1 = max(x1_1, x1_2)
        intersect_y1 = max(y1_1, y1_2)
        intersect_x2 = min(x2_1, x2_2)
        intersect_y2 = min(y2_1, y2_2)
        
        # Check if regions overlap
        if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
            # Calculate intersection area
            intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            
            # Calculate containment ratios
            ratio1 = intersect_area / area1
            ratio2 = intersect_area / area2
            
            relationships = []
            
            # Check containment
            if ratio2 > 0.9:  # region1 contains region2
                relationships.append(('contains', ratio2))
            elif ratio1 > 0.9:  # region2 contains region1
                relationships.append(('contained_by', ratio1))
            else:  # Partial overlap
                relationships.append(('overlaps', min(ratio1, ratio2)))
                
            return relationships
        else:
            # No overlap - check relative positions
            relationships = []
            
            # Check if region1 is above region2
            if y2_1 < y1_2:
                # Calculate horizontal overlap
                h_overlap = min(x2_1, x2_2) - max(x1_1, x1_2)
                h_overlap = max(0, h_overlap)
                
                # Calculate confidence based on horizontal alignment
                confidence = h_overlap / min((x2_1 - x1_1), (x2_2 - x1_2))
                relationships.append(('above', confidence))
                
            # Check if region1 is below region2
            elif y1_1 > y2_2:
                # Calculate horizontal overlap
                h_overlap = min(x2_1, x2_2) - max(x1_1, x1_2)
                h_overlap = max(0, h_overlap)
                
                # Calculate confidence based on horizontal alignment
                confidence = h_overlap / min((x2_1 - x1_1), (x2_2 - x1_2))
                relationships.append(('below', confidence))
                
            # Check if region1 is left of region2
            if x2_1 < x1_2:
                # Calculate vertical overlap
                v_overlap = min(y2_1, y2_2) - max(y1_1, y1_2)
                v_overlap = max(0, v_overlap)
                
                # Calculate confidence based on vertical alignment
                confidence = v_overlap / min((y2_1 - y1_1), (y2_2 - y1_2))
                relationships.append(('left_of', confidence))
                
            # Check if region1 is right of region2
            elif x1_1 > x2_2:
                # Calculate vertical overlap
                v_overlap = min(y2_1, y2_2) - max(y1_1, y1_2)
                v_overlap = max(0, v_overlap)
                
                # Calculate confidence based on vertical alignment
                confidence = v_overlap / min((y2_1 - y1_1), (y2_2 - y1_2))
                relationships.append(('right_of', confidence))
                
            return relationships
    
    def _detect_alignment_relationships(self, entities: List[Dict[str, Any]], 
                                      layout_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect alignment relationships (same row, same column).
        
        Args:
            entities: List of entities with region information
            layout_info: Document layout information
            
        Returns:
            List of alignment relationships
        """
        relationships = []
        
        # Group entities by rows and columns
        row_groups = self._group_entities_by_row(entities, layout_info)
        column_groups = self._group_entities_by_column(entities, layout_info)
        
        # Create 'same_row' relationships
        for row_idx, row_entities in enumerate(row_groups):
            if len(row_entities) >= 2:
                for i, entity1 in enumerate(row_entities):
                    for j in range(i+1, len(row_entities)):
                        entity2 = row_entities[j]
                        
                        relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
                        relationship = {
                            'id': relationship_id,
                            'type': 'same_row',
                            'source': entity1['id'],
                            'target': entity2['id'],
                            'confidence': 0.9,  # High confidence for layout alignment
                            'modality': 'visual',
                            'attributes': {
                                'row_index': row_idx,
                                'spatial': True,
                                'timestamp': datetime.now().isoformat()
                            }
                        }
                        relationships.append(relationship)
                        
        # Create 'same_column' relationships
        for col_idx, col_entities in enumerate(column_groups):
            if len(col_entities) >= 2:
                for i, entity1 in enumerate(col_entities):
                    for j in range(i+1, len(col_entities)):
                        entity2 = col_entities[j]
                        
                        relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
                        relationship = {
                            'id': relationship_id,
                            'type': 'same_column',
                            'source': entity1['id'],
                            'target': entity2['id'],
                            'confidence': 0.9,  # High confidence for layout alignment
                            'modality': 'visual',
                            'attributes': {
                                'column_index': col_idx,
                                'spatial': True,
                                'timestamp': datetime.now().isoformat()
                            }
                        }
                        relationships.append(relationship)
                        
        return relationships
    
    def _group_entities_by_row(self, entities: List[Dict[str, Any]], 
                             layout_info: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Group entities that are aligned horizontally (same row)."""
        # This is a simplified implementation - in reality, this would use
        # more sophisticated layout analysis
        
        # Extract entities with region information
        entities_with_region = [
            entity for entity in entities
            if 'region' in entity
        ]
        
        # Sort entities by y-coordinate (vertical position)
        sorted_entities = sorted(
            entities_with_region,
            key=lambda e: (e['region'][1] + e['region'][3]) / 2  # Center y-coordinate
        )
        
        # Group entities by row
        row_tolerance = layout_info.get('row_tolerance', 0.05)  # 5% of height
        
        row_groups = []
        current_row = []
        prev_y = None
        
        for entity in sorted_entities:
            center_y = (entity['region'][1] + entity['region'][3]) / 2
            
            if prev_y is None:
                # First entity in a row
                current_row = [entity]
                prev_y = center_y
            elif abs(center_y - prev_y) <= row_tolerance:
                # Entity in the same row
                current_row.append(entity)
            else:
                # New row
                row_groups.append(current_row)
                current_row = [entity]
                prev_y = center_y
                
        # Add the last row
        if current_row:
            row_groups.append(current_row)
            
        return row_groups
    
    def _group_entities_by_column(self, entities: List[Dict[str, Any]], 
                                layout_info: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Group entities that are aligned vertically (same column)."""
        # This is a simplified implementation - in reality, this would use
        # more sophisticated layout analysis
        
        # Extract entities with region information
        entities_with_region = [
            entity for entity in entities
            if 'region' in entity
        ]
        
        # Sort entities by x-coordinate (horizontal position)
        sorted_entities = sorted(
            entities_with_region,
            key=lambda e: (e['region'][0] + e['region'][2]) / 2  # Center x-coordinate
        )
        
        # Group entities by column
        column_tolerance = layout_info.get('column_tolerance', 0.05)  # 5% of width
        
        column_groups = []
        current_column = []
        prev_x = None
        
        for entity in sorted_entities:
            center_x = (entity['region'][0] + entity['region'][2]) / 2
            
            if prev_x is None:
                # First entity in a column
                current_column = [entity]
                prev_x = center_x
            elif abs(center_x - prev_x) <= column_tolerance:
                # Entity in the same column
                current_column.append(entity)
            else:
                # New column
                column_groups.append(current_column)
                current_column = [entity]
                prev_x = center_x
                
        # Add the last column
        if current_column:
            column_groups.append(current_column)
            
        return column_groups
    
    def _detect_visual_similarity_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect visual similarity relationships between entities.
        
        Args:
            entities: List of visual entities
            
        Returns:
            List of visual similarity relationships
        """
        relationships = []
        
        # Only consider entities with embeddings
        entities_with_embedding = [
            entity for entity in entities
            if 'embedding' in entity and entity.get('source', '') == 'colpali'
        ]
        
        # Calculate pairwise visual similarity
        similarity_threshold = self.config.get('semantic_similarity_threshold', 0.7)
        
        for i, entity1 in enumerate(entities_with_embedding):
            for j in range(i+1, len(entities_with_embedding)):
                entity2 = entities_with_embedding[j]
                
                # Calculate visual similarity
                similarity = self._compute_cosine_similarity(
                    entity1['embedding'], entity2['embedding']
                )
                
                # Skip if similarity is below threshold
                if similarity < similarity_threshold:
                    continue
                    
                # Create relationship record
                relationship_id = f"rel_{uuid.uuid4().hex[:8]}"
                relationship = {
                    'id': relationship_id,
                    'type': 'visually_similar_to',
                    'source': entity1['id'],
                    'target': entity2['id'],
                    'confidence': similarity,
                    'modality': 'visual',
                    'attributes': {
                        'similarity': similarity,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                relationships.append(relationship)
                
        return relationships
    
    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not vec1 or not vec2:
            return 0.0
            
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Check for zero vectors
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
            
        # Calculate cosine similarity
        return np.dot(a, b) / (a_norm * b_norm)
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Deduplicated list of relationships
        """
        # Create a set of unique relationship signatures
        unique_signatures = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a signature for the relationship
            # Order source and target to ensure consistency
            source, target = sorted([rel['source'], rel['target']])
            signature = f"{source}|{target}|{rel['type']}"
            
            # Only add if signature is new
            if signature not in unique_signatures:
                unique_signatures.add(signature)
                unique_relationships.append(rel)
                
        return unique_relationships
