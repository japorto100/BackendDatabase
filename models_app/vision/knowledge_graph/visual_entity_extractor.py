"""
Visual entity extractor for knowledge graph construction.

This module provides visual-specific entity extraction capabilities,
extending the base EntityExtractor to handle images and visual outputs
from the ColPali module.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np

# Import base classes and utilities
from models_app.knowledge_graph.entity_extractor import EntityExtractor
from models_app.knowledge_graph.interfaces import KnowledgeGraphEntity

logger = logging.getLogger(__name__)

class VisualEntityExtractor(EntityExtractor):
    """
    Entity extractor specialized for visual content.
    
    This class extends the base EntityExtractor to provide specialized
    capabilities for extracting entities from images and visual analysis
    results, including:
    - Visual elements (charts, tables, images, etc.)
    - Spatial relationships between elements
    - Visual attributes (color, size, position)
    - OCR results in context of visual layout
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the visual entity extractor.
        
        Args:
            config: Configuration dictionary with visual-specific settings
        """
        super().__init__(config)
        
        # Visual-specific entity types
        self.visual_entity_types = {
            'image': 'visual_image',
            'chart': 'visual_chart',
            'graph': 'visual_graph',
            'table': 'visual_table',
            'diagram': 'visual_diagram',
            'logo': 'visual_logo',
            'signature': 'visual_signature',
            'text_block': 'visual_text_block',
            'title': 'visual_title',
            'header': 'visual_header',
            'footer': 'visual_footer',
            'page_number': 'visual_page_number',
            'bullet_point': 'visual_bullet_point',
            'checkbox': 'visual_checkbox',
            'form_field': 'visual_form_field'
        }
        
        # Load any additional entity types from config
        if config and 'entity_types' in config:
            self.visual_entity_types.update(config['entity_types'])
            
        # Visual relationship types
        self.visual_relationship_types = {
            'contains': 'contains',
            'above': 'above',
            'below': 'below',
            'left_of': 'left_of',
            'right_of': 'right_of',
            'aligned_with': 'aligned_with',
            'part_of': 'part_of',
            'visually_similar_to': 'visually_similar_to'
        }
        
        # Confidence thresholds
        self.confidence_threshold = config.get('confidence_threshold', 0.5) if config else 0.5
    
    def extract_from_image(self, image_data: Any, metadata: Dict[str, Any] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from an image using visual analysis techniques.
        
        Args:
            image_data: Image data as numpy array, PIL Image, or path
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        if metadata is None:
            metadata = {}
            
        # Get quality metrics if available
        quality_metrics = metadata.get("quality_metrics", {})
        kg_extraction_hints = metadata.get("kg_extraction_hints", {})
        confidence_factor = metadata.get("confidence_factor", 1.0)
        prioritize_visual = metadata.get("prioritize_visual", False)
        
        # Log quality information if available
        if quality_metrics:
            logger.info(f"Visual entity extraction with quality metrics: {quality_metrics}")
        
        # Load and preprocess image
        try:
            # Get actual image data based on input type
            import numpy as np
            from PIL import Image as PILImage
            
            if isinstance(image_data, str):
                # Path to image
                pil_image = PILImage.open(image_data)
                img_array = np.array(pil_image)
            elif isinstance(image_data, np.ndarray):
                # Numpy array
                img_array = image_data
                pil_image = PILImage.fromarray(img_array)
            elif hasattr(image_data, 'resize') and callable(getattr(image_data, 'resize')):
                # PIL Image
                pil_image = image_data
                img_array = np.array(pil_image)
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return []
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return []
        
        entities = []
        
        # Adjust visual analysis approach based on quality metrics
        if quality_metrics and "overall_quality" in quality_metrics:
            overall_quality = quality_metrics["overall_quality"]
            
            # For low quality images, adjust our approaches
            if overall_quality < 0.4 or prioritize_visual:
                logger.info("Low quality image or prioritized visual extraction - using specialized approach")
                # Use more robust visual detection settings
                entities.extend(self._extract_visual_entities_robust(img_array, metadata))
            else:
                # Use standard detection
                entities.extend(self._extract_visual_entities(img_array, metadata))
        else:
            # Default to standard detection when no quality info
            entities.extend(self._extract_visual_entities(img_array, metadata))
            
        # Apply confidence adjustment based on quality
        for entity in entities:
            if confidence_factor != 1.0:
                entity["confidence"] = entity.get("confidence", 0.5) * confidence_factor
                
            # Add quality information to entity metadata
            if "metadata" not in entity:
                entity["metadata"] = {}
                
            if quality_metrics:
                entity["metadata"]["quality_metrics"] = {
                    "overall_quality": quality_metrics.get("overall_quality", 1.0),
                    "confidence_adjusted": confidence_factor != 1.0
                }
            
            # Mark potentially unreliable extractions
            if quality_metrics and quality_metrics.get("overall_quality", 1.0) < 0.3:
                entity["metadata"]["low_quality_source"] = True
                
        return entities
    
    def _extract_visual_entities_robust(self, image_array: Any, metadata: Dict[str, Any] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract visual entities using more robust techniques for low quality images.
        
        Uses more aggressive preprocessing and lower confidence thresholds
        for better recall at the expense of precision.
        
        Args:
            image_array: Image as numpy array
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        if metadata is None:
            metadata = {}
            
        entities = []
        
        try:
            import cv2
            import numpy as np
            
            # Apply contrast enhancement and noise reduction for low quality images
            image = image_array.copy()
            
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Apply adaptive histogram equalization to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 11, 31, 9)
            
            # Detect edges with lower thresholds
            edges = cv2.Canny(denoised, 30, 100)
            
            # Find contours with relaxed parameters
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Identify potential visual elements based on contour analysis
            min_area = 100  # Lower threshold than usual
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Analyze shape to determine entity type
                    aspect_ratio = float(w) / h if h > 0 else 0
                    shape_type = self._analyze_contour_shape(contour, aspect_ratio)
                    
                    # Create entity with reduced confidence
                    entity = {
                        "id": f"visual-element-{i}",
                        "text": f"{shape_type} at position ({x}, {y})",
                        "entity_type": f"visual_{shape_type.lower()}",
                        "start_pos": 0,
                        "end_pos": 0,
                        "confidence": 0.5,  # Lower base confidence
                        "metadata": {
                            "position": {
                                "x": int(x),
                                "y": int(y),
                                "width": int(w),
                                "height": int(h)
                            },
                            "visual_properties": {
                                "area": float(area),
                                "aspect_ratio": float(aspect_ratio),
                                "shape_type": shape_type
                            },
                            "robust_extraction": True
                        }
                    }
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in robust visual entity extraction: {str(e)}")
            return entities
            
    def _analyze_contour_shape(self, contour, aspect_ratio: float) -> str:
        """
        Analyze contour shape to classify the visual element.
        
        Args:
            contour: OpenCV contour
            aspect_ratio: Width/height ratio
            
        Returns:
            Shape classification
        """
        import cv2
        
        # Approximate the contour to simplify
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Count corners
        corners = len(approx)
        
        # Circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * 3.14159 * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Classify based on corners and shape
        if corners == 4 and 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        elif corners == 4:
            return "Rectangle"
        elif corners == 3:
            return "Triangle"
        elif corners > 4 and circularity > 0.8:
            return "Circle"
        elif corners > 4:
            return "Polygon"
        else:
            return "Shape"
    
    def extract_from_colpali(self, colpali_output: Dict[str, Any], 
                           metadata: Dict[str, Any] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from ColPali output.
        
        This method processes ColPali's structured output to extract
        meaningful visual entities and their relationships.
        
        Args:
            colpali_output: Output from ColPali analysis
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        metadata = metadata or {}
        entities = []
        
        # Process different parts of ColPali output
        if 'elements' in colpali_output:
            element_entities = self._extract_from_colpali_elements(
                colpali_output['elements'], metadata)
            entities.extend(element_entities)
            
        if 'layout' in colpali_output:
            layout_entities = self._extract_from_colpali_layout(
                colpali_output['layout'], metadata)
            entities.extend(layout_entities)
            
        if 'blocks' in colpali_output:
            block_entities = self._extract_from_colpali_blocks(
                colpali_output['blocks'], metadata)
            entities.extend(block_entities)
            
        if 'tables' in colpali_output:
            table_entities = self._extract_from_colpali_tables(
                colpali_output['tables'], metadata)
            entities.extend(table_entities)
            
        if 'charts' in colpali_output:
            chart_entities = self._extract_from_colpali_charts(
                colpali_output['charts'], metadata)
            entities.extend(chart_entities)
            
        return entities
    
    def extract_from_ocr_result(self, ocr_result: Dict[str, Any],
                              metadata: Dict[str, Any] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from OCR results with spatial context.
        
        This method processes OCR results, preserving the spatial
        arrangement of text elements for knowledge graph construction.
        
        Args:
            ocr_result: OCR result with text and bounding boxes
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        metadata = metadata or {}
        entities = []
        
        # Process OCR text blocks
        if 'blocks' in ocr_result and isinstance(ocr_result['blocks'], list):
            for i, block in enumerate(ocr_result['blocks']):
                # Skip blocks with low confidence
                if 'confidence' in block and block['confidence'] < self.confidence_threshold:
                    continue
                    
                block_text = block.get('text', '')
                if not block_text.strip():
                    continue
                    
                # Create block entity
                block_id = str(uuid.uuid4())
                block_entity = {
                    'id': block_id,
                    'type': self.visual_entity_types['text_block'],
                    'text': block_text,
                    'confidence': block.get('confidence', 0.7),
                    'metadata': {
                        'source': 'ocr_analysis',
                        'block_index': i
                    },
                    'attributes': {}
                }
                
                # Add bounding box if available
                if 'bbox' in block:
                    bbox = block['bbox']
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        block_entity['metadata']['bbox'] = {
                            'x': bbox[0],
                            'y': bbox[1],
                            'width': bbox[2] - bbox[0],
                            'height': bbox[3] - bbox[1]
                        }
                        
                        # Add spatial attributes
                        block_entity['attributes']['position_x'] = bbox[0]
                        block_entity['attributes']['position_y'] = bbox[1]
                        block_entity['attributes']['width'] = bbox[2] - bbox[0]
                        block_entity['attributes']['height'] = bbox[3] - bbox[1]
                        
                # Add block type if available
                if 'type' in block:
                    block_entity['metadata']['block_type'] = block['type']
                    
                # Analyze text to determine if it's a title, header, etc.
                block_entity = self._analyze_text_block(block_entity, block_text)
                
                entities.append(block_entity)
                
                # Process lines and words if available
                if 'lines' in block and isinstance(block['lines'], list):
                    for line in block['lines']:
                        if 'words' in line and isinstance(line['words'], list):
                            for word in line['words']:
                                # Extract named entities from word text
                                word_text = word.get('text', '')
                                if word_text.strip():
                                    # We could extract named entities here, but for now
                                    # we'll rely on the text-based entity extraction
                                    pass
                                    
        # If OCR result has a different structure, handle it here
        # This is just one example format
        
        return entities
    
    def _extract_from_colpali_elements(self, elements: List[Dict[str, Any]], 
                                     metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from ColPali visual elements.
        
        Args:
            elements: List of visual elements from ColPali
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, element in enumerate(elements):
            # Skip elements with low confidence
            if 'confidence' in element and element['confidence'] < self.confidence_threshold:
                continue
                
            element_type = element.get('type', 'unknown')
            element_id = str(uuid.uuid4())
            
            # Map ColPali element type to entity type
            entity_type = self.visual_entity_types.get(
                element_type.lower(), self.visual_entity_types['image'])
                
            # Create entity
            element_entity = {
                'id': element_id,
                'type': entity_type,
                'text': element.get('caption', f"{element_type.capitalize()} {i+1}"),
                'confidence': element.get('confidence', 0.7),
                'metadata': {
                    'source': 'colpali_analysis',
                    'element_index': i,
                    'element_type': element_type
                },
                'attributes': {}
            }
            
            # Add bounding box if available
            if 'bbox' in element:
                bbox = element['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    element_entity['metadata']['bbox'] = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1]
                    }
                    
                    # Add spatial attributes
                    element_entity['attributes']['position_x'] = bbox[0]
                    element_entity['attributes']['position_y'] = bbox[1]
                    element_entity['attributes']['width'] = bbox[2] - bbox[0]
                    element_entity['attributes']['height'] = bbox[3] - bbox[1]
                    
            # Add other attributes
            for key, value in element.items():
                if key not in ['type', 'bbox', 'confidence', 'caption'] and isinstance(value, (str, int, float, bool)):
                    element_entity['attributes'][key] = value
                    
            entities.append(element_entity)
            
        return entities
    
    def _extract_from_colpali_layout(self, layout: Dict[str, Any], 
                                   metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from ColPali layout analysis.
        
        Args:
            layout: Layout analysis from ColPali
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Create layout entity
        layout_id = str(uuid.uuid4())
        layout_entity = {
            'id': layout_id,
            'type': 'document_layout',
            'text': 'Document Layout',
            'confidence': layout.get('confidence', 0.8),
            'metadata': {
                'source': 'colpali_layout_analysis'
            },
            'attributes': {}
        }
        
        # Add layout attributes
        if 'columns' in layout:
            layout_entity['attributes']['columns'] = layout['columns']
            
        if 'orientation' in layout:
            layout_entity['attributes']['orientation'] = layout['orientation']
            
        if 'page_size' in layout:
            layout_entity['attributes']['page_size'] = layout['page_size']
            
        entities.append(layout_entity)
        
        # Process layout regions if available
        if 'regions' in layout and isinstance(layout['regions'], list):
            for i, region in enumerate(layout['regions']):
                region_id = str(uuid.uuid4())
                region_type = region.get('type', 'unknown')
                
                # Map region type to entity type
                entity_type = 'layout_region'
                if region_type.lower() in ['header', 'footer', 'body', 'margin']:
                    entity_type = f"layout_{region_type.lower()}"
                    
                # Create region entity
                region_entity = {
                    'id': region_id,
                    'type': entity_type,
                    'text': region.get('description', f"{region_type.capitalize()} Region"),
                    'confidence': region.get('confidence', 0.7),
                    'metadata': {
                        'source': 'colpali_layout_analysis',
                        'region_index': i,
                        'region_type': region_type
                    },
                    'attributes': {}
                }
                
                # Add bounding box if available
                if 'bbox' in region:
                    bbox = region['bbox']
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        region_entity['metadata']['bbox'] = {
                            'x': bbox[0],
                            'y': bbox[1],
                            'width': bbox[2] - bbox[0],
                            'height': bbox[3] - bbox[1]
                        }
                        
                entities.append(region_entity)
                
        return entities
    
    def _extract_from_colpali_blocks(self, blocks: List[Dict[str, Any]], 
                                   metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from ColPali text blocks.
        
        Args:
            blocks: Text blocks from ColPali
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, block in enumerate(blocks):
            # Skip blocks with low confidence
            if 'confidence' in block and block['confidence'] < self.confidence_threshold:
                continue
                
            block_text = block.get('text', '')
            if not block_text.strip():
                continue
                
            # Create block entity
            block_id = str(uuid.uuid4())
            block_entity = {
                'id': block_id,
                'type': self.visual_entity_types['text_block'],
                'text': block_text[:100] + '...' if len(block_text) > 100 else block_text,
                'confidence': block.get('confidence', 0.7),
                'metadata': {
                    'source': 'colpali_block_analysis',
                    'block_index': i
                },
                'attributes': {
                    'full_text': block_text
                }
            }
            
            # Add bounding box if available
            if 'bbox' in block:
                bbox = block['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    block_entity['metadata']['bbox'] = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1]
                    }
                    
                    # Add spatial attributes
                    block_entity['attributes']['position_x'] = bbox[0]
                    block_entity['attributes']['position_y'] = bbox[1]
                    block_entity['attributes']['width'] = bbox[2] - bbox[0]
                    block_entity['attributes']['height'] = bbox[3] - bbox[1]
                    
            # Analyze text to determine if it's a title, header, etc.
            block_entity = self._analyze_text_block(block_entity, block_text)
            
            entities.append(block_entity)
            
            # Extract named entities from block text
            # This will be handled by the text-based entity extraction
            
        return entities
    
    def _extract_from_colpali_tables(self, tables: List[Dict[str, Any]], 
                                   metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from ColPali table detection.
        
        Args:
            tables: Tables detected by ColPali
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, table in enumerate(tables):
            # Skip tables with low confidence
            if 'confidence' in table and table['confidence'] < self.confidence_threshold:
                continue
                
            # Create table entity
            table_id = str(uuid.uuid4())
            table_entity = {
                'id': table_id,
                'type': self.visual_entity_types['table'],
                'text': table.get('caption', f"Table {i+1}"),
                'confidence': table.get('confidence', 0.7),
                'metadata': {
                    'source': 'colpali_table_analysis',
                    'table_index': i
                },
                'attributes': {}
            }
            
            # Add bounding box if available
            if 'bbox' in table:
                bbox = table['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    table_entity['metadata']['bbox'] = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1]
                    }
                    
                    # Add spatial attributes
                    table_entity['attributes']['position_x'] = bbox[0]
                    table_entity['attributes']['position_y'] = bbox[1]
                    table_entity['attributes']['width'] = bbox[2] - bbox[0]
                    table_entity['attributes']['height'] = bbox[3] - bbox[1]
                    
            # Add table properties
            if 'rows' in table:
                table_entity['attributes']['rows'] = table['rows']
                
            if 'columns' in table:
                table_entity['attributes']['columns'] = table['columns']
                
            if 'cells' in table:
                table_entity['attributes']['cell_count'] = len(table['cells'])
                
            entities.append(table_entity)
            
            # Process table cells if available
            if 'cells' in table and isinstance(table['cells'], list):
                for j, cell in enumerate(table['cells']):
                    cell_text = cell.get('text', '')
                    if not cell_text.strip():
                        continue
                        
                    # Create cell entity
                    cell_id = str(uuid.uuid4())
                    cell_entity = {
                        'id': cell_id,
                        'type': 'table_cell',
                        'text': cell_text,
                        'confidence': cell.get('confidence', 0.7),
                        'metadata': {
                            'source': 'colpali_table_analysis',
                            'table_index': i,
                            'cell_index': j,
                            'row': cell.get('row', -1),
                            'column': cell.get('column', -1)
                        },
                        'attributes': {}
                    }
                    
                    # Add bounding box if available
                    if 'bbox' in cell:
                        bbox = cell['bbox']
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            cell_entity['metadata']['bbox'] = {
                                'x': bbox[0],
                                'y': bbox[1],
                                'width': bbox[2] - bbox[0],
                                'height': bbox[3] - bbox[1]
                            }
                            
                    entities.append(cell_entity)
                    
                    # Extract named entities from cell text
                    # This will be handled by the text-based entity extraction
                    
        return entities
    
    def _extract_from_colpali_charts(self, charts: List[Dict[str, Any]], 
                                   metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from ColPali chart detection.
        
        Args:
            charts: Charts detected by ColPali
            metadata: Additional metadata
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, chart in enumerate(charts):
            # Skip charts with low confidence
            if 'confidence' in chart and chart['confidence'] < self.confidence_threshold:
                continue
                
            chart_type = chart.get('type', 'unknown')
            
            # Create chart entity
            chart_id = str(uuid.uuid4())
            chart_entity = {
                'id': chart_id,
                'type': self.visual_entity_types['chart'],
                'text': chart.get('caption', f"{chart_type.capitalize()} Chart"),
                'confidence': chart.get('confidence', 0.7),
                'metadata': {
                    'source': 'colpali_chart_analysis',
                    'chart_index': i,
                    'chart_type': chart_type
                },
                'attributes': {}
            }
            
            # Add bounding box if available
            if 'bbox' in chart:
                bbox = chart['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    chart_entity['metadata']['bbox'] = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1]
                    }
                    
                    # Add spatial attributes
                    chart_entity['attributes']['position_x'] = bbox[0]
                    chart_entity['attributes']['position_y'] = bbox[1]
                    chart_entity['attributes']['width'] = bbox[2] - bbox[0]
                    chart_entity['attributes']['height'] = bbox[3] - bbox[1]
                    
            # Add chart properties
            if 'data_points' in chart:
                chart_entity['attributes']['data_points'] = len(chart['data_points'])
                
            if 'axis_labels' in chart:
                chart_entity['attributes']['axis_labels'] = chart['axis_labels']
                
            entities.append(chart_entity)
            
            # Process chart elements if available
            if 'elements' in chart and isinstance(chart['elements'], list):
                for j, element in enumerate(chart['elements']):
                    element_type = element.get('type', 'unknown')
                    element_text = element.get('text', '')
                    
                    # Create element entity
                    element_id = str(uuid.uuid4())
                    element_entity = {
                        'id': element_id,
                        'type': f"chart_{element_type}",
                        'text': element_text or f"{element_type.capitalize()} {j+1}",
                        'confidence': element.get('confidence', 0.7),
                        'metadata': {
                            'source': 'colpali_chart_analysis',
                            'chart_index': i,
                            'element_index': j,
                            'element_type': element_type
                        },
                        'attributes': {}
                    }
                    
                    # Add bounding box if available
                    if 'bbox' in element:
                        bbox = element['bbox']
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            element_entity['metadata']['bbox'] = {
                                'x': bbox[0],
                                'y': bbox[1],
                                'width': bbox[2] - bbox[0],
                                'height': bbox[3] - bbox[1]
                            }
                            
                    entities.append(element_entity)
                    
        return entities
    
    def _analyze_text_block(self, block_entity: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Analyze a text block to determine its characteristics.
        
        Args:
            block_entity: The block entity to analyze
            text: The text content of the block
            
        Returns:
            Updated block entity
        """
        # Simple heuristics to classify text blocks
        lines = text.splitlines()
        first_line = lines[0] if lines else ""
        
        # Check if it might be a title (short, first block, etc.)
        if (len(text) < 100 and len(lines) <= 2 and 
            (block_entity['metadata'].get('block_index', 0) == 0 or 
             block_entity['attributes'].get('position_y', 0) < 100)):
            block_entity['type'] = self.visual_entity_types['title']
            block_entity['confidence'] = 0.8
            
        # Check if it might be a header
        elif (len(text) < 150 and len(lines) <= 3 and 
              (text.isupper() or text.endswith(':') or 
               block_entity['attributes'].get('position_y', 0) < 200)):
            block_entity['type'] = self.visual_entity_types['header']
            block_entity['confidence'] = 0.7
            
        # Check if it might be a footer
        elif ('page' in text.lower() or 
              block_entity['attributes'].get('position_y', 1000) > 700):
            block_entity['type'] = self.visual_entity_types['footer']
            block_entity['confidence'] = 0.7
            
        # Check if it might be a bullet point list
        elif any(line.strip().startswith(('•', '-', '*', '○', '✓', '✗')) for line in lines):
            block_entity['type'] = self.visual_entity_types['bullet_point']
            block_entity['confidence'] = 0.8
            
        return block_entity
    
    def detect_spatial_relationships(self, entities: List[KnowledgeGraphEntity]) -> List[Dict[str, Any]]:
        """
        Detect spatial relationships between visual entities.
        
        Args:
            entities: List of entities with spatial information
            
        Returns:
            List of relationships between entities
        """
        relationships = []
        
        # Filter entities with bounding box information
        spatial_entities = []
        for entity in entities:
            if ('metadata' in entity and 'bbox' in entity['metadata'] and
                all(k in entity['metadata']['bbox'] for k in ['x', 'y', 'width', 'height'])):
                spatial_entities.append(entity)
                
        # Analyze spatial relationships
        for i, entity1 in enumerate(spatial_entities):
            bbox1 = entity1['metadata']['bbox']
            x1, y1 = bbox1['x'], bbox1['y']
            w1, h1 = bbox1['width'], bbox1['height']
            right1, bottom1 = x1 + w1, y1 + h1
            
            for j, entity2 in enumerate(spatial_entities):
                if i == j:
                    continue
                    
                bbox2 = entity2['metadata']['bbox']
                x2, y2 = bbox2['x'], bbox2['y']
                w2, h2 = bbox2['width'], bbox2['height']
                right2, bottom2 = x2 + w2, y2 + h2
                
                # Check for containment
                if (x1 <= x2 and y1 <= y2 and right1 >= right2 and bottom1 >= bottom2):
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'type': self.visual_relationship_types['contains'],
                        'confidence': 0.9,
                        'metadata': {
                            'source': 'spatial_analysis'
                        }
                    }
                    relationships.append(relationship)
                    
                # Check for above/below
                if (x1 < right2 and right1 > x2):  # Horizontal overlap
                    if bottom1 <= y2:  # entity1 is above entity2
                        relationship = {
                            'id': str(uuid.uuid4()),
                            'source': entity1['id'],
                            'target': entity2['id'],
                            'type': self.visual_relationship_types['above'],
                            'confidence': 0.9,
                            'metadata': {
                                'source': 'spatial_analysis',
                                'distance': y2 - bottom1
                            }
                        }
                        relationships.append(relationship)
                    elif bottom2 <= y1:  # entity2 is above entity1
                        relationship = {
                            'id': str(uuid.uuid4()),
                            'source': entity1['id'],
                            'target': entity2['id'],
                            'type': self.visual_relationship_types['below'],
                            'confidence': 0.9,
                            'metadata': {
                                'source': 'spatial_analysis',
                                'distance': y1 - bottom2
                            }
                        }
                        relationships.append(relationship)
                        
                # Check for left/right
                if (y1 < bottom2 and bottom1 > y2):  # Vertical overlap
                    if right1 <= x2:  # entity1 is left of entity2
                        relationship = {
                            'id': str(uuid.uuid4()),
                            'source': entity1['id'],
                            'target': entity2['id'],
                            'type': self.visual_relationship_types['left_of'],
                            'confidence': 0.9,
                            'metadata': {
                                'source': 'spatial_analysis',
                                'distance': x2 - right1
                            }
                        }
                        relationships.append(relationship)
                    elif right2 <= x1:  # entity2 is left of entity1
                        relationship = {
                            'id': str(uuid.uuid4()),
                            'source': entity1['id'],
                            'target': entity2['id'],
                            'type': self.visual_relationship_types['right_of'],
                            'confidence': 0.9,
                            'metadata': {
                                'source': 'spatial_analysis',
                                'distance': x1 - right2
                            }
                        }
                        relationships.append(relationship)
                        
                # Check for alignment
                if abs(y1 - y2) < 10:  # Horizontally aligned (top edges)
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'type': self.visual_relationship_types['aligned_with'],
                        'confidence': 0.8,
                        'metadata': {
                            'source': 'spatial_analysis',
                            'alignment_type': 'top'
                        }
                    }
                    relationships.append(relationship)
                elif abs(bottom1 - bottom2) < 10:  # Horizontally aligned (bottom edges)
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'type': self.visual_relationship_types['aligned_with'],
                        'confidence': 0.8,
                        'metadata': {
                            'source': 'spatial_analysis',
                            'alignment_type': 'bottom'
                        }
                    }
                    relationships.append(relationship)
                elif abs(x1 - x2) < 10:  # Vertically aligned (left edges)
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'type': self.visual_relationship_types['aligned_with'],
                        'confidence': 0.8,
                        'metadata': {
                            'source': 'spatial_analysis',
                            'alignment_type': 'left'
                        }
                    }
                    relationships.append(relationship)
                elif abs(right1 - right2) < 10:  # Vertically aligned (right edges)
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'type': self.visual_relationship_types['aligned_with'],
                        'confidence': 0.8,
                        'metadata': {
                            'source': 'spatial_analysis',
                            'alignment_type': 'right'
                        }
                    }
                    relationships.append(relationship)
                    
        return relationships
    
    def extract_from_document(self, document: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from a structured image document.
        This method is designed to work with the standardized output from
        DocumentProcessorFactory.prepare_document_for_extraction method.
        
        Args:
            document: Structured document data from prepare_for_extraction
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Check if this is an image document
        if document.get("document_type") != "image":
            logger.warning(f"Document type is not 'image': {document.get('document_type')}")
            return entities
            
        # Extract from document text (OCR results)
        if "document_text" in document and document["document_text"]:
            # Extract entities from OCR text
            ocr_entities = self.extract_from_text(
                document["document_text"],
                metadata=document.get("metadata", {})
            )
            entities.extend(ocr_entities)
        
        # Extract from visual elements if available
        if "document_structure" in document and "visual_elements" in document["document_structure"]:
            visual_elements = document["document_structure"]["visual_elements"]
            for element in visual_elements:
                # Process based on element type
                if element.get("type") == "text":
                    # Text regions already handled by OCR extraction
                    continue
                elif element.get("type") in ["image", "figure", "diagram", "chart"]:
                    # Extract visual entities
                    if "bounding_box" in element:
                        visual_entity = {
                            "id": f"visual-{len(entities)}",
                            "text": element.get("text", ""),
                            "entity_type": element.get("type", "VISUAL_ELEMENT"),
                            "start_pos": 0,
                            "end_pos": 0,
                            "confidence": element.get("confidence", 0.0),
                            "metadata": {
                                "bounding_box": element.get("bounding_box"),
                                "attributes": element.get("attributes", {})
                            }
                        }
                        entities.append(visual_entity)
        
        # Extract from tables if present
        if "document_structure" in document and "tables" in document["document_structure"]:
            table_entities = self._extract_from_colpali_tables(
                document["document_structure"]["tables"],
                document.get("metadata", {})
            )
            entities.extend(table_entities)
        
        # Detect spatial relationships between entities
        spatial_relationships = self.detect_spatial_relationships(entities)
        for entity in entities:
            if "metadata" not in entity:
                entity["metadata"] = {}
            entity["metadata"]["spatial_relationships"] = spatial_relationships
        
        return entities 