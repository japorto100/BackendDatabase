"""
Entity extraction for knowledge graph building.

This module provides functionality to extract named entities and concepts
from various data types including text, images, and ColPali visual embeddings.
"""

import logging
import re
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid
from datetime import datetime

import numpy as np
from PIL import Image
import nltk
from django.conf import settings

from models_app.knowledge_graph.interfaces import EntityExtractorInterface

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

logger = logging.getLogger(__name__)

class EntityExtractor(EntityExtractorInterface):
    """
    Extracts entities from various data sources for knowledge graph construction.
    
    The extractor can handle:
    - Text data: Using NLP techniques to extract named entities
    - Image data: Using OCR and vision models to extract visual entities
    - ColPali output: Leveraging ColPali's visual embeddings for advanced entity extraction
    
    Configuration options:
    - USE_SPACY: Whether to use spaCy for NLP (if available)
    - MIN_ENTITY_CONFIDENCE: Minimum confidence threshold for entities
    - CUSTOM_ENTITY_TYPES: Additional domain-specific entity types to extract
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the entity extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load config from settings if not provided
        if not self.config:
            self.config = {
                'use_spacy': getattr(settings, 'ENTITY_EXTRACTOR_USE_SPACY', False),
                'min_confidence': getattr(settings, 'ENTITY_EXTRACTOR_MIN_CONFIDENCE', 0.5),
                'custom_entity_types': getattr(settings, 'ENTITY_EXTRACTOR_CUSTOM_TYPES', {}),
                'embedding_dim': getattr(settings, 'KNOWLEDGE_GRAPH_EMBEDDING_DIM', 768),
            }
        
        # Initialize NLP components
        self._init_nlp()
        
        logger.info(f"EntityExtractor initialized with config: {self.config}")
    
    def _init_nlp(self):
        """Initialize NLP components based on configuration."""
        # Try to use spaCy if configured and available
        self.use_spacy = self.config.get('use_spacy', False)
        self.spacy_model = None
        
        if self.use_spacy:
            try:
                import spacy
                # Load a medium-sized model with Named Entity Recognition
                self.spacy_model = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy model for entity extraction")
            except (ImportError, OSError) as e:
                logger.warning(f"Failed to load spaCy: {e}. Falling back to NLTK.")
                self.use_spacy = False
    
    def extract_from_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from text content.
        
        Args:
            text: The text to extract entities from
            metadata: Optional metadata about the text source
            
        Returns:
            List of extracted entities with their attributes
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {type(text)}")
            return []
            
        # Extract entities using the appropriate NLP library
        if self.use_spacy and self.spacy_model:
            return self._extract_with_spacy(text, metadata)
        else:
            return self._extract_with_nltk(text, metadata)
    
    def extract_from_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from image content.
        
        This method relies on OCR to extract text from images and then
        processes that text for entities. For advanced visual entity extraction,
        use extract_from_colpali instead.
        
        Args:
            image_path: Path to the image file
            metadata: Optional metadata about the image
            
        Returns:
            List of extracted entities with their attributes
        """
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Invalid image path: {image_path}")
            return []
            
        # We'll use a simple approach here - in a real implementation,
        # this would connect to the OCR pipeline
        try:
            # Import OCR components only when needed
            from models_app.vision.ocr.ocr_model_selector import OCRModelSelector
            
            # Select an appropriate OCR model
            ocr_selector = OCRModelSelector()
            ocr_adapter = ocr_selector.select_ocr_model(image_path)
            
            # Process the image with OCR
            ocr_result = ocr_adapter.process_image(image_path)
            
            # Extract text and blocks
            extracted_text = ocr_result.get('text', '')
            text_blocks = ocr_result.get('blocks', [])
            
            # First, extract entities from the whole text
            entities = self.extract_from_text(extracted_text, metadata)
            
            # Add position information from blocks where possible
            self._enrich_entities_with_positions(entities, text_blocks)
            
            # Add source information
            for entity in entities:
                entity['source'] = 'ocr_image'
                if metadata:
                    entity['metadata'].update(metadata)
                
            return entities
            
        except ImportError:
            logger.warning("OCR components not available. Using basic image processing.")
            # Basic fallback if OCR is not available
            return []
        except Exception as e:
            logger.error(f"Error extracting entities from image: {e}")
            return []
    
    def extract_from_colpali(self, colpali_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from ColPali model output.
        
        This specialized method leverages the rich visual understanding of the
        ColPali model to extract meaningful entities from images beyond what
        basic OCR can provide.
        
        Args:
            colpali_output: The output dictionary from ColPali processing
            
        Returns:
            List of extracted visual entities with their attributes
        """
        if not colpali_output or not isinstance(colpali_output, dict):
            logger.warning(f"Invalid ColPali output: {type(colpali_output)}")
            return []
            
        entities = []
        
        # Extract embeddings from ColPali output
        embeddings = colpali_output.get('embeddings')
        attention_map = colpali_output.get('attention_map')
        
        if embeddings is None:
            logger.warning("No embeddings found in ColPali output")
            return []
            
        # Identify regions of interest from attention map
        regions = self._identify_regions_of_interest(attention_map) if attention_map is not None else []
        
        # If no regions identified but we have global embeddings, create a single entity
        if not regions and embeddings is not None:
            # Create entity for the whole image
            entity_id = f"visual_{uuid.uuid4().hex[:8]}"
            
            entity = {
                'id': entity_id,
                'type': 'visual_element',
                'confidence': 0.9,  # High confidence for the whole image
                'region': (0, 0, 1.0, 1.0),  # Normalized coordinates
                'embedding': self._get_embedding_vector(embeddings),
                'attributes': {
                    'global': True,
                    'timestamp': datetime.now().isoformat()
                },
                'source': 'colpali',
                'metadata': {
                    'model': colpali_output.get('model_name', 'unknown'),
                }
            }
            entities.append(entity)
            
        # Process each region of interest
        for region in regions:
            # Extract region embedding
            region_embedding = self._get_region_embedding(embeddings, region)
            
            # Classify the region content
            entity_type, confidence = self._classify_region(region_embedding)
            
            # Create entity record
            entity_id = f"visual_{uuid.uuid4().hex[:8]}"
            entity = {
                'id': entity_id,
                'type': entity_type,
                'confidence': confidence,
                'region': region,
                'embedding': region_embedding,
                'attributes': self._extract_entity_attributes(region_embedding),
                'source': 'colpali',
                'metadata': {
                    'model': colpali_output.get('model_name', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            entities.append(entity)
            
        return entities
        
    def _extract_with_spacy(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NLP library."""
        entities = []
        
        # Process the text with spaCy
        doc = self.spacy_model(text)
        
        # Extract named entities
        for ent in doc.ents:
            # Map spaCy entity types to our schema
            entity_type = self._map_spacy_entity_type(ent.label_)
            
            # Skip entities with low confidence or unwanted types
            if entity_type is None:
                continue
                
            # Create entity record
            entity_id = f"text_{uuid.uuid4().hex[:8]}"
            entity = {
                'id': entity_id,
                'type': entity_type,
                'text': ent.text,
                'confidence': 0.8,  # spaCy doesn't provide confidence scores
                'position': {
                    'start': ent.start_char,
                    'end': ent.end_char
                },
                'embedding': self._get_text_embedding(ent.text),
                'attributes': {},
                'source': 'spacy',
                'metadata': metadata or {}
            }
            entities.append(entity)
            
        return entities
        
    def _extract_with_nltk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract entities using NLTK library."""
        entities = []
        
        # Tokenize text
        tokens = nltk.word_tokenize(text)
        
        # Part-of-speech tagging
        tagged_tokens = nltk.pos_tag(tokens)
        
        # Named entity chunking
        ne_tree = nltk.ne_chunk(tagged_tokens)
        
        # Extract entities from the parse tree
        for subtree in ne_tree:
            if hasattr(subtree, 'label'):
                # This is a named entity
                entity_text = ' '.join([token for token, pos in subtree.leaves()])
                entity_type = self._map_nltk_entity_type(subtree.label())
                
                # Skip unwanted entity types
                if entity_type is None:
                    continue
                    
                # Find the position in the original text
                position = self._find_text_position(text, entity_text)
                
                # Create entity record
                entity_id = f"text_{uuid.uuid4().hex[:8]}"
                entity = {
                    'id': entity_id,
                    'type': entity_type,
                    'text': entity_text,
                    'confidence': 0.7,  # NLTK doesn't provide confidence scores
                    'position': position,
                    'embedding': self._get_text_embedding(entity_text),
                    'attributes': {},
                    'source': 'nltk',
                    'metadata': metadata or {}
                }
                entities.append(entity)
                
        return entities
    
    def _map_spacy_entity_type(self, spacy_type: str) -> Optional[str]:
        """Map spaCy entity types to our knowledge graph schema."""
        # Standard mapping for common entity types
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'PRODUCT': 'product',
            'EVENT': 'event',
            'WORK_OF_ART': 'creative_work',
            'LAW': 'legal',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'money',
            'QUANTITY': 'quantity',
            'PERCENT': 'percentage',
            'FACILITY': 'facility',
            'NORP': 'group',  # Nationalities, religious or political groups
        }
        
        # Add custom entity type mappings if configured
        custom_mapping = self.config.get('custom_entity_types', {})
        mapping.update(custom_mapping)
        
        return mapping.get(spacy_type)
    
    def _map_nltk_entity_type(self, nltk_type: str) -> Optional[str]:
        """Map NLTK entity types to our knowledge graph schema."""
        # Standard mapping for NLTK entity types
        mapping = {
            'PERSON': 'person',
            'ORGANIZATION': 'organization',
            'GPE': 'location',  # Geo-political entity
            'GSP': 'location',  # Geo-social political group
            'FACILITY': 'facility',
            'LOCATION': 'location',
        }
        
        # Add custom entity type mappings if configured
        custom_mapping = self.config.get('custom_entity_types', {})
        mapping.update(custom_mapping)
        
        return mapping.get(nltk_type)
    
    def _find_text_position(self, text: str, entity_text: str) -> Dict[str, int]:
        """Find the start and end positions of entity_text within text."""
        start = text.find(entity_text)
        if start == -1:
            # Try case-insensitive search as fallback
            start = text.lower().find(entity_text.lower())
            
        if start == -1:
            # If still not found, return estimated position
            return {'start': 0, 'end': 0}
            
        end = start + len(entity_text)
        return {'start': start, 'end': end}
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text entities."""
        # This is a simplified placeholder - in a real implementation,
        # this would use a text embedding model
        
        # For now, return a random vector to simulate an embedding
        embedding_dim = self.config.get('embedding_dim', 768)
        np.random.seed(hash(text) % 2**32)
        return np.random.uniform(-1, 1, embedding_dim).tolist()
    
    def _enrich_entities_with_positions(self, entities: List[Dict[str, Any]], blocks: List[Dict[str, Any]]):
        """Add bounding box information to entities when possible."""
        # Create a mapping from text block content to its bbox
        block_positions = {}
        for block in blocks:
            if 'text' in block and 'bbox' in block:
                block_positions[block['text']] = block['bbox']
                
        # Try to find matching blocks for each entity
        for entity in entities:
            entity_text = entity.get('text', '')
            
            # Check for exact matches first
            if entity_text in block_positions:
                entity['bbox'] = block_positions[entity_text]
                continue
                
            # Otherwise, look for blocks containing the entity text
            for block_text, bbox in block_positions.items():
                if entity_text in block_text:
                    entity['bbox'] = bbox
                    break
    
    def _identify_regions_of_interest(self, attention_map: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Identify regions of interest from an attention map.
        
        Args:
            attention_map: 2D numpy array representing attention scores
            
        Returns:
            List of regions as (x1, y1, x2, y2) normalized coordinates
        """
        if attention_map is None:
            return []
            
        # Simple thresholding approach
        # In a real implementation, this would use more sophisticated
        # computer vision techniques for region proposal
        
        # Normalize attention map
        if attention_map.max() > 0:
            norm_map = attention_map / attention_map.max()
        else:
            return []
            
        # Threshold to find high-attention regions
        threshold = 0.5
        binary_map = (norm_map > threshold).astype(np.uint8)
        
        # Simple connected components to find regions
        # In a real implementation, this would use scikit-image or OpenCV
        regions = []
        h, w = binary_map.shape
        
        # Very simplified region detection - just identify connected high-attention areas
        # This is just a placeholder for the real implementation
        visited = np.zeros_like(binary_map)
        
        for y in range(h):
            for x in range(w):
                if binary_map[y, x] == 1 and visited[y, x] == 0:
                    # Found a new region, perform simple flood fill
                    min_x, min_y, max_x, max_y = x, y, x, y
                    stack = [(x, y)]
                    visited[y, x] = 1
                    
                    while stack:
                        cx, cy = stack.pop()
                        
                        # Update region bounds
                        min_x = min(min_x, cx)
                        min_y = min(min_y, cy)
                        max_x = max(max_x, cx)
                        max_y = max(max_y, cy)
                        
                        # Check neighbors
                        for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                            if (0 <= nx < w and 0 <= ny < h and 
                                binary_map[ny, nx] == 1 and visited[ny, nx] == 0):
                                stack.append((nx, ny))
                                visited[ny, nx] = 1
                    
                    # Convert to normalized coordinates
                    region = (min_x/w, min_y/h, max_x/w, max_y/h)
                    regions.append(region)
        
        return regions
    
    def _get_region_embedding(self, embeddings: np.ndarray, region: Tuple[float, float, float, float]) -> List[float]:
        """
        Extract embedding vector for a specific region.
        
        Args:
            embeddings: Embedding tensor from ColPali
            region: (x1, y1, x2, y2) normalized coordinates
            
        Returns:
            Embedding vector for the region
        """
        # This is a simplified implementation
        # In a real system, this would use spatial pooling over the region
        
        if embeddings is None:
            embedding_dim = self.config.get('embedding_dim', 768)
            return [0.0] * embedding_dim
            
        # For 2D embedding maps, extract the region
        if len(embeddings.shape) >= 3:
            h, w = embeddings.shape[0:2]
            x1, y1, x2, y2 = region
            
            # Convert to pixel coordinates
            x1_px, y1_px = int(x1 * w), int(y1 * h)
            x2_px, y2_px = int(x2 * w), int(y2 * h)
            
            # Ensure valid coordinates
            x1_px = max(0, min(x1_px, w-1))
            y1_px = max(0, min(y1_px, h-1))
            x2_px = max(x1_px+1, min(x2_px, w))
            y2_px = max(y1_px+1, min(y2_px, h))
            
            # Extract region embeddings and average them
            region_emb = embeddings[y1_px:y2_px, x1_px:x2_px]
            if region_emb.size > 0:
                return np.mean(region_emb, axis=(0, 1)).tolist()
        
        # Fallback to global embedding
        return self._get_embedding_vector(embeddings)
    
    def _get_embedding_vector(self, embeddings: np.ndarray) -> List[float]:
        """Extract a single embedding vector from the embeddings tensor."""
        if embeddings is None:
            embedding_dim = self.config.get('embedding_dim', 768)
            return [0.0] * embedding_dim
            
        # For single vector
        if len(embeddings.shape) == 1:
            return embeddings.tolist()
            
        # For 2D embedding maps, average all embeddings
        if len(embeddings.shape) >= 2:
            return np.mean(embeddings, axis=tuple(range(len(embeddings.shape)-1))).tolist()
            
        # Fallback
        embedding_dim = self.config.get('embedding_dim', 768)
        return [0.0] * embedding_dim
    
    def _classify_region(self, embedding: List[float]) -> Tuple[str, float]:
        """
        Classify a region based on its embedding vector.
        
        Args:
            embedding: Region embedding vector
            
        Returns:
            Tuple of (entity_type, confidence)
        """
        # This is a placeholder implementation
        # In a real system, this would use a classifier trained on embeddings
        
        # For now, return a generic visual element type
        return "visual_element", 0.7
    
    def _extract_entity_attributes(self, embedding: List[float]) -> Dict[str, Any]:
        """
        Extract additional attributes about an entity from its embedding.
        
        Args:
            embedding: Entity embedding vector
            
        Returns:
            Dictionary of entity attributes
        """
        # This is a placeholder implementation
        # In a real system, this would extract meaningful attributes
        
        return {
            "visual_prominence": 0.8,
            "context_relevance": 0.7,
            "timestamp": datetime.now().isoformat()
        }

    def extract_entities(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from structured document data.
        
        Args:
            structured_data: Data from document processor's prepare_for_extraction method
            
        Returns:
            List of extracted entities
        """
        # Implementation details...

    def extract_and_enrich_entities(self, text, metadata=None):
        """
        Extract entities from text and enrich them with external knowledge bases
        
        This method performs:
        1. Initial entity extraction from text
        2. Entity linking with external KBs (primarily SwissAL and Wikidata)
        3. Entity enrichment with KB data (additional properties, types, etc.)
        4. Entity resolution to merge duplicates
        
        Args:
            text: Text to extract entities from
            metadata: Optional metadata about the text
            
        Returns:
            List of enriched entities
        """
        # Extract initial entities
        entities = self.extract_from_text(text, metadata)
        
        # Initialize the cascading connector with SwissAL and Wikidata prioritized
        from models_app.knowledge_graph.external_kb_connector import CascadingKBConnector
        kb_connector = CascadingKBConnector({
            "primary_connector": "swiss_al",
            "fallback_connectors": ["wikidata", "gnd", "dbpedia_german"],
            "confidence_threshold": 0.6
        })
        
        # Enrich entities
        enriched_entities = []
        for entity in entities:
            # Get entity type-specific config
            entity_type = entity.get("type", "unknown").lower()
            
            # Determine KB priority based on entity type
            kb_priority = self._get_kb_priority_for_type(entity_type)
            
            # Link entity to external KBs
            external_matches = kb_connector.link_entity(entity)
            
            if external_matches and len(external_matches) > 0:
                # Find the best match according to priority and confidence
                best_match = self._select_best_kb_match(external_matches, kb_priority)
                
                # Enrich entity if we have a good match
                if best_match and best_match["confidence"] >= 0.7:
                    external_id = f"{best_match['source']}:{best_match['external_id']}"
                    enriched_entity = kb_connector.enrich_entity(entity, external_id)
                    enriched_entities.append(enriched_entity)
                else:
                    # Add any reasonable matches as references
                    if "external_references" not in entity:
                        entity["external_references"] = []
                    
                    for match in external_matches:
                        if match["confidence"] >= 0.5:
                            entity["external_references"].append({
                                "source": match["source"],
                                "id": match["external_id"],
                                "url": match.get("external_url", ""),
                                "confidence": match["confidence"]
                            })
                    
                    enriched_entities.append(entity)
            else:
                # No matches found
                enriched_entities.append(entity)
        
        # Resolve duplicates
        from models_app.knowledge_graph.entity_resolution import EntityResolver
        resolver = EntityResolver()
        resolved_entities = resolver.resolve_entities(enriched_entities)
        
        return resolved_entities

    def _get_kb_priority_for_type(self, entity_type):
        """Get KB priority based on entity type"""
        # Government, administrative, locations prefer Swiss-AL
        if entity_type in ["government", "administration", "location", "place", "organization"]:
            return ["swiss_al", "wikidata", "gnd", "dbpedia_german"]
        
        # People, events, concepts prefer Wikidata
        elif entity_type in ["person", "event", "concept"]:
            return ["wikidata", "gnd", "swiss_al", "dbpedia_german"]
        
        # Academic content prefers GND
        elif entity_type in ["academic", "publication", "book"]:
            return ["gnd", "wikidata", "dbpedia_german", "swiss_al"]
        
        # Default priority
        return ["swiss_al", "wikidata", "gnd", "dbpedia_german"]

    def _select_best_kb_match(self, matches, kb_priority):
        """
        Select best KB match based on priority and confidence
        
        This implements a priority-weighted confidence scoring system
        that prefers matches from higher-priority KBs while still
        considering match quality
        """
        if not matches:
            return None
        
        # Calculate priority-weighted scores
        weighted_matches = []
        for i, match in enumerate(matches):
            # Get source priority (position in priority list)
            source = match["source"]
            try:
                priority_index = kb_priority.index(source)
            except ValueError:
                priority_index = len(kb_priority)  # Put at end if not in priority list
            
            # Calculate priority weight (higher priority = higher weight)
            priority_weight = 1.0 - (priority_index / len(kb_priority))
            
            # Calculate weighted score
            confidence = match["confidence"]
            weighted_score = (confidence * 0.7) + (priority_weight * 0.3)
            
            weighted_matches.append((match, weighted_score))
        
        # Sort by weighted score
        weighted_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best match
        return weighted_matches[0][0] if weighted_matches else None
