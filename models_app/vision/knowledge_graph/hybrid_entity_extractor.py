"""
Hybrid entity extractor for knowledge graph construction.

This module provides a comprehensive entity extraction system that combines
document-based and visual entity extraction to create a unified knowledge graph
from multimodal data sources.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set

# Import base classes and specialized extractors
from models_app.knowledge_graph.entity_extractor import EntityExtractor
from models_app.vision.knowledge_graph.document_entity_extractor import DocumentEntityExtractor
from models_app.vision.knowledge_graph.visual_entity_extractor import VisualEntityExtractor
from models_app.knowledge_graph.interfaces import KnowledgeGraphEntity

logger = logging.getLogger(__name__)

class HybridEntityExtractor(EntityExtractor):
    """
    Hybrid entity extractor for comprehensive multimodal entity extraction.
    
    This class combines document and visual entity extraction capabilities to
    provide a unified approach to knowledge graph construction from multiple
    data modalities. It handles cross-modal entity correlation and merging.
    
    Features:
    - Document entity extraction (text, metadata, structure)
    - Visual entity extraction (images, charts, layouts)
    - Cross-modal entity correlation and deduplication
    - Spatial-semantic relationship detection
    - Confidence scoring and entity merging
    - Quality-aware extraction strategies based on image quality metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the hybrid entity extractor.
        
        Args:
            config: Configuration dictionary with hybrid-specific settings
        """
        super().__init__(config)
        
        # Initialize specialized extractors
        self.document_extractor = DocumentEntityExtractor(config)
        self.visual_extractor = VisualEntityExtractor(config)
        
        # Configure hybrid settings
        self.cross_modal_threshold = config.get('cross_modal_threshold', 0.7) if config else 0.7
        self.enable_entity_merging = config.get('enable_entity_merging', True) if config else True
        self.merge_confidence_boost = config.get('merge_confidence_boost', 0.1) if config else 0.1
        
        # Quality-aware extraction settings
        self.quality_threshold_low = config.get('quality_threshold_low', 0.3) if config else 0.3
        self.quality_threshold_medium = config.get('quality_threshold_medium', 0.6) if config else 0.6
        self.confidence_penalty_low_quality = config.get('confidence_penalty_low_quality', 0.3) if config else 0.3
        self.confidence_penalty_medium_quality = config.get('confidence_penalty_medium_quality', 0.1) if config else 0.1
    
    def extract_from_document(self, document: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from a structured hybrid document.
        This method is designed to work with the standardized output from
        DocumentProcessorFactory.prepare_document_for_extraction method.
        
        Args:
            document: Structured document data from prepare_for_extraction
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract quality metrics from metadata if available
        metadata = document.get("metadata", {})
        quality_metrics = metadata.get("quality_metrics", {})
        kg_extraction_hints = metadata.get("kg_extraction_hints", {})
        
        # Log quality metrics for debugging
        if quality_metrics:
            logger.info(f"Quality metrics detected: {quality_metrics}")
            if "overall_quality" in quality_metrics:
                logger.info(f"Overall document quality: {quality_metrics['overall_quality']}")
                
        # Apply quality-aware extraction strategy
        extraction_strategy = self._determine_extraction_strategy(
            quality_metrics, kg_extraction_hints
        )
        
        # Check if this is a hybrid document
        if document.get("document_type") != "hybrid":
            logger.warning(f"Document type is not 'hybrid': {document.get('document_type')}")
            return entities
        
        # Extract from document text - adjust based on text reliability
        if "document_text" in document and document["document_text"]:
            text_confidence_factor = extraction_strategy.get("text_confidence_factor", 1.0)
            
            # Log our extraction strategy
            logger.info(f"Applying text extraction with confidence factor: {text_confidence_factor}")
            
            text_entities = self.extract_from_text(
                document["document_text"],
                metadata={**metadata, "confidence_factor": text_confidence_factor}
            )
            
            # Apply confidence adjustment based on quality
            for entity in text_entities:
                entity["confidence"] = entity.get("confidence", 0.5) * text_confidence_factor
                if "metadata" not in entity:
                    entity["metadata"] = {}
                entity["metadata"]["quality_adjusted"] = True
                
            entities.extend(text_entities)
        
        # Extract from document sections - apply quality-aware processing
        if "document_sections" in document and isinstance(document["document_sections"], list):
            for section in document["document_sections"]:
                section_type = section.get("type", "unknown")
                
                if section_type == "text":
                    # Already processed in full document text
                    continue
                elif section_type == "image":
                    # Process image section with quality consideration
                    if "content" in section and section["content"]:
                        image_confidence_factor = extraction_strategy.get("visual_confidence_factor", 1.0)
                        
                        # For low quality images, if extraction_strategy suggests visual prioritization
                        if extraction_strategy.get("prioritize_visual", False):
                            logger.info(f"Prioritizing visual extraction for low quality image")
                            section_metadata = {
                                **metadata, 
                                "confidence_factor": image_confidence_factor,
                                "prioritize_visual": True
                            }
                        else:
                            section_metadata = {
                                **metadata, 
                                "confidence_factor": image_confidence_factor
                            }
                            
                        image_entities = self._extract_from_images(
                            [section], section_metadata
                        )
                        
                        # Apply confidence adjustment
                        for entity in image_entities:
                            entity["confidence"] = entity.get("confidence", 0.5) * image_confidence_factor
                            if "metadata" not in entity:
                                entity["metadata"] = {}
                            entity["metadata"]["quality_adjusted"] = True
                            
                        entities.extend(image_entities)
                        
                elif section_type == "table":
                    # Process table section with quality consideration
                    if "content" in section and section["content"]:
                        table_confidence_factor = extraction_strategy.get("table_confidence_factor", 1.0)
                        
                        table_entities = self._extract_from_colpali_results(
                            [{"table": section["content"]}], 
                            {**metadata, "confidence_factor": table_confidence_factor}
                        )
                        
                        # Apply confidence adjustment
                        for entity in table_entities:
                            entity["confidence"] = entity.get("confidence", 0.5) * table_confidence_factor
                            if "metadata" not in entity:
                                entity["metadata"] = {}
                            entity["metadata"]["quality_adjusted"] = True
                            
                        entities.extend(table_entities)
        
        # Extract from document structure
        if "document_structure" in document and document["document_structure"]:
            # Process document structure sections
            if "sections" in document["document_structure"]:
                for section in document["document_structure"]["sections"]:
                    # Process based on section type
                    if section.get("type") in ["image", "chart", "diagram"]:
                        # Adjust confidence based on quality
                        visual_confidence_factor = extraction_strategy.get("visual_confidence_factor", 1.0)
                        base_confidence = 0.8 * visual_confidence_factor
                        
                        visual_entity = {
                            "id": f"visual-section-{len(entities)}",
                            "text": section.get("content", ""),
                            "entity_type": section.get("type", "VISUAL_SECTION"),
                            "start_pos": 0,
                            "end_pos": 0,
                            "confidence": base_confidence,
                            "metadata": {
                                "position": section.get("position", {}),
                                "metadata": section.get("metadata", {}),
                                "quality_adjusted": True,
                                "quality_factor": visual_confidence_factor
                            }
                        }
                        entities.append(visual_entity)
        
        # Cross-modal correlation - merge entities from different sources
        entities = self._correlate_and_merge_entities(entities)
        
        # Detect semantic relationships
        semantic_entities = self.detect_semantic_relationships(entities)
        
        # Final quality check - mark uncertain entities when quality is very low
        if extraction_strategy.get("mark_uncertain", False):
            for entity in semantic_entities:
                if entity.get("confidence", 0) < 0.4:
                    if "metadata" not in entity:
                        entity["metadata"] = {}
                    entity["metadata"]["uncertain_due_to_quality"] = True
        
        return semantic_entities
    
    def extract_from_hybrid_source(self, text: Optional[str] = None, 
                                 image_data: Any = None,
                                 ocr_result: Optional[Dict[str, Any]] = None,
                                 colpali_result: Optional[Dict[str, Any]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from multiple data sources simultaneously.
        
        This method allows extraction from various sources in a single call,
        enabling correlation between different modalities.
        
        Args:
            text: Text content
            image_data: Image data
            ocr_result: OCR analysis result
            colpali_result: ColPali analysis result
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        metadata = metadata or {}
        all_entities = []
        
        # Extract from text if available
        if text:
            text_entities = self.document_extractor.extract_from_text(text, metadata)
            all_entities.extend(text_entities)
            
        # Extract from image if available
        if image_data is not None:
            image_entities = self.visual_extractor.extract_from_image(image_data, metadata)
            all_entities.extend(image_entities)
            
        # Extract from OCR result if available
        if ocr_result:
            ocr_entities = self.visual_extractor.extract_from_ocr_result(ocr_result, metadata)
            all_entities.extend(ocr_entities)
            
        # Extract from ColPali result if available
        if colpali_result:
            colpali_entities = self.visual_extractor.extract_from_colpali(colpali_result, metadata)
            all_entities.extend(colpali_entities)
            
        # Correlate and merge entities across modalities
        if self.enable_entity_merging:
            merged_entities = self._correlate_and_merge_entities(all_entities)
            return merged_entities
            
        return all_entities
    
    def _extract_from_images(self, images: List[Dict[str, Any]], 
                           metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from multiple images.
        
        Args:
            images: List of image data objects
            metadata: Document metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, image in enumerate(images):
            # Create image-specific metadata
            image_metadata = metadata.copy()
            image_metadata.update({
                'image_index': i,
                'total_images': len(images)
            })
            
            # Add image filename if available
            if 'filename' in image:
                image_metadata['filename'] = image['filename']
                
            # Add image dimensions if available
            if 'width' in image and 'height' in image:
                image_metadata['width'] = image['width']
                image_metadata['height'] = image['height']
                
            # Extract entities from image
            image_data = image.get('data') or image.get('path')
            if image_data:
                image_entities = self.visual_extractor.extract_from_image(image_data, image_metadata)
                
                # Add image context to entities
                for entity in image_entities:
                    if 'metadata' not in entity:
                        entity['metadata'] = {}
                    entity['metadata']['image_index'] = i
                    
                entities.extend(image_entities)
                
        return entities
    
    def _extract_from_ocr_results(self, ocr_results: List[Dict[str, Any]], 
                                metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from multiple OCR results.
        
        Args:
            ocr_results: List of OCR result objects
            metadata: Document metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, ocr_result in enumerate(ocr_results):
            # Create OCR-specific metadata
            ocr_metadata = metadata.copy()
            ocr_metadata.update({
                'ocr_result_index': i,
                'total_ocr_results': len(ocr_results)
            })
            
            # Add page number if available
            if 'page_number' in ocr_result:
                ocr_metadata['page_number'] = ocr_result['page_number']
                
            # Extract entities from OCR result
            ocr_entities = self.visual_extractor.extract_from_ocr_result(ocr_result, ocr_metadata)
            
            # Add OCR context to entities
            for entity in ocr_entities:
                if 'metadata' not in entity:
                    entity['metadata'] = {}
                entity['metadata']['ocr_result_index'] = i
                
            entities.extend(ocr_entities)
            
        return entities
    
    def _extract_from_colpali_results(self, colpali_results: List[Dict[str, Any]], 
                                    metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from multiple ColPali results.
        
        Args:
            colpali_results: List of ColPali result objects
            metadata: Document metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, colpali_result in enumerate(colpali_results):
            # Create ColPali-specific metadata
            colpali_metadata = metadata.copy()
            colpali_metadata.update({
                'colpali_result_index': i,
                'total_colpali_results': len(colpali_results)
            })
            
            # Add page number if available
            if 'page_number' in colpali_result:
                colpali_metadata['page_number'] = colpali_result['page_number']
                
            # Extract entities from ColPali result
            colpali_entities = self.visual_extractor.extract_from_colpali(colpali_result, colpali_metadata)
            
            # Add ColPali context to entities
            for entity in colpali_entities:
                if 'metadata' not in entity:
                    entity['metadata'] = {}
                entity['metadata']['colpali_result_index'] = i
                
            entities.extend(colpali_entities)
            
            # Detect spatial relationships between entities
            spatial_relationships = self.visual_extractor.detect_spatial_relationships(colpali_entities)
            
            # Create relationship entities
            for relationship in spatial_relationships:
                relationship_entity = {
                    'id': relationship['id'],
                    'type': f"relationship_{relationship['type']}",
                    'text': relationship['type'],
                    'confidence': relationship['confidence'],
                    'metadata': relationship['metadata'],
                    'attributes': {
                        'source_entity_id': relationship['source'],
                        'target_entity_id': relationship['target']
                    }
                }
                
                entities.append(relationship_entity)
                
        return entities
    
    def _correlate_and_merge_entities(self, entities: List[KnowledgeGraphEntity]) -> List[KnowledgeGraphEntity]:
        """
        Correlate and merge entities across modalities.
        
        This method identifies entities that refer to the same real-world object
        across different modalities (text and visual) and merges them into a
        single, more comprehensive entity.
        
        Args:
            entities: List of entities from all sources
            
        Returns:
            List of merged entities
        """
        # Group similar entities by type and text
        entity_groups = {}
        
        for entity in entities:
            entity_type = entity['type']
            entity_text = entity['text']
            
            # Create a key for grouping similar entities
            key = (entity_type, entity_text)
            
            if key not in entity_groups:
                entity_groups[key] = []
                
            entity_groups[key].append(entity)
            
        # Merge entities within each group
        merged_entities = []
        
        for (entity_type, entity_text), group in entity_groups.items():
            if len(group) == 1:
                # No need to merge if there's only one entity
                merged_entities.append(group[0])
            else:
                # Merge multiple entities
                merged_entity = self._merge_entity_group(group)
                merged_entities.append(merged_entity)
                
        # Perform cross-modal entity correlation
        return self._perform_cross_modal_correlation(merged_entities)
    
    def _merge_entity_group(self, entities: List[KnowledgeGraphEntity]) -> KnowledgeGraphEntity:
        """
        Merge a group of similar entities into a single entity.
        
        Args:
            entities: List of similar entities to merge
            
        Returns:
            Merged entity
        """
        if not entities:
            return None
            
        # Start with the highest confidence entity as the base
        entities_sorted = sorted(entities, key=lambda e: e.get('confidence', 0.0), reverse=True)
        base_entity = entities_sorted[0]
        
        # Create a new merged entity
        merged_entity = {
            'id': base_entity['id'],
            'type': base_entity['type'],
            'text': base_entity['text'],
            'confidence': min(base_entity.get('confidence', 0.0) + self.merge_confidence_boost, 1.0),
            'metadata': base_entity.get('metadata', {}).copy(),
            'attributes': base_entity.get('attributes', {}).copy()
        }
        
        # Add source IDs
        merged_entity['metadata']['source_entity_ids'] = [e['id'] for e in entities]
        merged_entity['metadata']['merge_count'] = len(entities)
        
        # Update metadata with a list of sources
        sources = set()
        for entity in entities:
            if 'metadata' in entity and 'source' in entity['metadata']:
                sources.add(entity['metadata']['source'])
                
        if sources:
            merged_entity['metadata']['sources'] = list(sources)
            
        # Merge attributes from all entities
        for entity in entities[1:]:  # Skip the base entity
            if 'attributes' in entity:
                for key, value in entity['attributes'].items():
                    if key not in merged_entity['attributes']:
                        merged_entity['attributes'][key] = value
                        
        return merged_entity
    
    def _perform_cross_modal_correlation(self, entities: List[KnowledgeGraphEntity]) -> List[KnowledgeGraphEntity]:
        """
        Perform cross-modal correlation between text and visual entities.
        
        This method establishes relationships between entities detected in
        different modalities (text and visual) based on spatial and semantic relationships.
        
        Args:
            entities: List of entities
            
        Returns:
            Enhanced list of entities with cross-modal relationships
        """
        # Separate text and visual entities
        text_entities = []
        visual_entities = []
        other_entities = []
        
        for entity in entities:
            entity_type = entity['type']
            
            if entity_type.startswith('document_') or entity_type in ['person', 'organization', 'location', 'date']:
                text_entities.append(entity)
            elif entity_type.startswith('visual_') or entity_type in ['table_cell', 'chart_axis', 'image']:
                visual_entities.append(entity)
            else:
                other_entities.append(entity)
                
        # Create relationships between text and visual entities
        relationships = []
        
        # Find visual entities that correspond to text entities
        for text_entity in text_entities:
            text_entity_text = text_entity['text'].lower()
            
            for visual_entity in visual_entities:
                # Check if the visual entity has text that matches the text entity
                visual_entity_text = visual_entity.get('text', '').lower()
                
                # Skip if no text in visual entity or text doesn't match
                if not visual_entity_text or text_entity_text not in visual_entity_text:
                    continue
                    
                # Calculate match confidence
                text_length_ratio = len(text_entity_text) / max(len(visual_entity_text), 1)
                match_confidence = min(text_length_ratio * 0.9, 0.9)  # Cap at 0.9
                
                # Create relationship if confidence is high enough
                if match_confidence >= self.cross_modal_threshold:
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'type': 'cross_modal_match',
                        'text': 'Cross-Modal Match',
                        'confidence': match_confidence,
                        'metadata': {
                            'source': 'cross_modal_correlation',
                            'text_entity_id': text_entity['id'],
                            'visual_entity_id': visual_entity['id']
                        },
                        'attributes': {
                            'text_entity_type': text_entity['type'],
                            'visual_entity_type': visual_entity['type'],
                            'match_type': 'text_content'
                        }
                    }
                    
                    relationships.append(relationship)
                    
        # Add cross-modal relationships to the entity list
        enhanced_entities = entities.copy()
        enhanced_entities.extend(relationships)
        
        return enhanced_entities
    
    def detect_semantic_relationships(self, entities: List[KnowledgeGraphEntity]) -> List[KnowledgeGraphEntity]:
        """
        Detect semantic relationships between entities.
        
        This method analyzes the semantic relationships between entities
        and creates explicit relationship entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Enhanced list of entities with semantic relationships
        """
        # This would typically use NLP to detect relationships
        # For now, return the original entities
        return entities
    
    def _determine_extraction_strategy(self, quality_metrics: Dict[str, Any], 
                                       kg_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the optimal extraction strategy based on quality metrics and hints.
        
        Args:
            quality_metrics: Document quality metrics
            kg_hints: Knowledge graph extraction hints
            
        Returns:
            Dict with strategy parameters
        """
        strategy = {
            "text_confidence_factor": 1.0,
            "visual_confidence_factor": 1.0,
            "table_confidence_factor": 1.0,
            "prioritize_visual": False,
            "mark_uncertain": False
        }
        
        # If no quality metrics available, return default strategy
        if not quality_metrics:
            return strategy
            
        overall_quality = quality_metrics.get("overall_quality", 0.7)
        blur_score = quality_metrics.get("blur_score", 0.7)
        contrast_score = quality_metrics.get("contrast_score", 0.7)
        
        # Apply hint-based adjustments if available
        if kg_hints:
            if kg_hints.get("quality_concerns", False):
                logger.info("KG extraction hints indicate quality concerns")
                strategy["mark_uncertain"] = True
                
            if kg_hints.get("prioritize_visual_extraction", False):
                logger.info("KG extraction hints suggest prioritizing visual extraction")
                strategy["prioritize_visual"] = True
                strategy["text_confidence_factor"] = 0.7
                
            if "text_reliability" in kg_hints:
                text_reliability = kg_hints["text_reliability"]
                logger.info(f"Using text reliability from hints: {text_reliability}")
                strategy["text_confidence_factor"] = text_reliability
                return strategy  # Return early as we have explicit hints
        
        # Quality-based adjustments
        if overall_quality < self.quality_threshold_low:
            # Very low quality
            strategy["text_confidence_factor"] = 1.0 - self.confidence_penalty_low_quality
            strategy["mark_uncertain"] = True
            
            # For very blurry images, visual analysis might be more reliable than OCR
            if blur_score < 0.3:
                strategy["prioritize_visual"] = True
                strategy["text_confidence_factor"] = max(0.3, strategy["text_confidence_factor"] - 0.2)
                strategy["visual_confidence_factor"] = min(1.0, strategy["visual_confidence_factor"] + 0.1)
                
            # For low contrast images, reduce confidence in all extractions
            if contrast_score < 0.3:
                factor = max(0.3, contrast_score)
                strategy["text_confidence_factor"] *= factor
                strategy["visual_confidence_factor"] *= factor
                strategy["table_confidence_factor"] *= factor
                
        elif overall_quality < self.quality_threshold_medium:
            # Medium quality
            strategy["text_confidence_factor"] = 1.0 - self.confidence_penalty_medium_quality
            
            # Slight adjustments based on specific issues
            if blur_score < 0.4:
                strategy["text_confidence_factor"] = max(0.5, strategy["text_confidence_factor"] - 0.1)
            
            if contrast_score < 0.4:
                strategy["text_confidence_factor"] = max(0.5, strategy["text_confidence_factor"] - 0.1)
                
        # For tables, additional quality considerations
        if "noise_level" in quality_metrics:
            noise_level = quality_metrics["noise_level"]
            if noise_level > 0.6:  # High noise
                strategy["table_confidence_factor"] = max(0.4, strategy["table_confidence_factor"] - 0.3)
                
        logger.debug(f"Determined extraction strategy: {strategy}")
        return strategy 