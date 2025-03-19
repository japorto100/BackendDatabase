"""
Document-specific entity extractor for knowledge graph construction.

This module provides document-specific entity extraction capabilities,
extending the base EntityExtractor to handle document text and metadata.
"""

import logging
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set

# Import base classes and utilities
from models_app.knowledge_graph.entity_extractor import EntityExtractor
from models_app.vision.utils.text_processing import preprocess_text, chunk_text
from models_app.knowledge_graph.interfaces import KnowledgeGraphEntity

logger = logging.getLogger(__name__)

class DocumentEntityExtractor(EntityExtractor):
    """
    Entity extractor specialized for document content.
    
    This class extends the base EntityExtractor to provide document-specific
    entity extraction capabilities, including document metadata, structure,
    and content extraction.
    
    Specialized capabilities:
    - Document metadata extraction (title, author, date, etc.)
    - Document structure extraction (headers, sections, paragraphs)
    - Document content extraction (text blocks, tables, lists)
    - Document type-specific entity extraction (invoices, contracts, reports)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document entity extractor.
        
        Args:
            config: Configuration dictionary with document-specific settings
        """
        super().__init__(config)
        
        # Document-specific entity types
        self.document_entity_types = {
            'title': 'document_title',
            'header': 'document_header',
            'paragraph': 'document_paragraph',
            'table': 'document_table',
            'list': 'document_list',
            'page': 'document_page',
            'footer': 'document_footer',
            'metadata': 'document_metadata',
            'signature': 'document_signature',
            'date': 'date',
            'person': 'person',
            'organization': 'organization',
            'location': 'location',
            'money': 'money',
            'percentage': 'percentage',
            'time': 'time',
            'product': 'product',
            'section': 'document_section',
            'form': 'document_form',
            'form_field': 'form_field',
            'checkbox': 'form_checkbox',
            'radio_button': 'form_radio_button',
            'text_field': 'form_text_field',
            'form_label': 'form_label',
            'form_value': 'form_value'
        }
        
        # Load any additional entity types from config
        if config and 'entity_types' in config:
            self.document_entity_types.update(config['entity_types'])
        
        # Regular expressions for common entity types
        self.entity_patterns = {
            'date': r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
            'money': r'\$\s?\d+(?:\.\d{2})?\b|\b\d+\s?(?:USD|EUR|GBP|JPY)\b',
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b'
        }
    
    def extract_from_text(self, text: str, metadata: Dict[str, Any] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from text.
        
        This method extracts entities from plain text using various techniques.
        
        Args:
            text: The text to extract entities from
            metadata: Additional metadata and context
            
        Returns:
            List of extracted entities
        """
        if not text:
            return []
            
        if metadata is None:
            metadata = {}
            
        # Get quality metrics if available
        quality_metrics = metadata.get("quality_metrics", {})
        kg_extraction_hints = metadata.get("kg_extraction_hints", {})
        confidence_factor = metadata.get("confidence_factor", 1.0)
        
        # Log quality information if available
        if quality_metrics:
            logger.info(f"Text entity extraction with quality metrics: {quality_metrics}")
            if "overall_quality" in quality_metrics:
                logger.info(f"Overall text quality: {quality_metrics['overall_quality']}")
                
        # Initialize entity list
        entities = []
        
        # Text preprocessing based on quality
        if quality_metrics and quality_metrics.get("overall_quality", 1.0) < 0.5:
            # For low quality text, apply more aggressive preprocessing
            logger.info(f"Applying enhanced text preprocessing for low quality text")
            text = self._preprocess_low_quality_text(text)
        else:
            # Standard preprocessing
            text = preprocess_text(text)
            
        # Extract entities from text structure
        structure_entities = self._extract_document_structure(text, metadata)
        entities.extend(structure_entities)
        
        # Extract entities using regex patterns
        regex_entities = self._extract_regex_entities(text)
        entities.extend(regex_entities)
        
        # Apply confidence adjustment based on quality
        if confidence_factor != 1.0:
            for entity in entities:
                entity["confidence"] = entity.get("confidence", 0.5) * confidence_factor
                if "metadata" not in entity:
                    entity["metadata"] = {}
                entity["metadata"]["quality_adjusted"] = True
                
                # Add quality information if available
                if quality_metrics:
                    entity["metadata"]["text_quality"] = quality_metrics.get("overall_quality", 1.0)
                
                # Mark entities as potentially unreliable if quality is very low
                if quality_metrics and quality_metrics.get("overall_quality", 1.0) < 0.3:
                    entity["metadata"]["low_quality_source"] = True
                    
                # Add text reliability if available in hints
                if kg_extraction_hints and "text_reliability" in kg_extraction_hints:
                    entity["metadata"]["text_reliability"] = kg_extraction_hints["text_reliability"]
        
        return entities
    
    def _preprocess_low_quality_text(self, text: str) -> str:
        """
        Enhanced preprocessing for low quality text.
        
        Args:
            text: Original text
            
        Returns:
            Preprocessed text
        """
        # Start with standard preprocessing
        preprocessed = preprocess_text(text)
        
        # Additional cleaning for OCR errors
        
        # 1. Fix common OCR errors with substitutions
        ocr_fixes = {
            # Character confusions
            'l1': 'h', 'Il': 'H', 'rn': 'm', 'cl': 'd',
            # Broken words (common patterns)
            ' - ': '-', ' , ': ', ', ' . ': '. ',
            # Remove excessive punctuation
            '..': '.', ',,': ',', '::': ':',
            # Fix spaces
            '  ': ' '
        }
        
        for error, fix in ocr_fixes.items():
            preprocessed = preprocessed.replace(error, fix)
        
        # 2. Remove short fragments that are likely noise
        lines = preprocessed.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if len(line.strip()) > 2:  # Keep lines with more than 2 characters
                cleaned_lines.append(line)
                
        preprocessed = '\n'.join(cleaned_lines)
        
        return preprocessed
    
    def extract_from_document(self, document: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from a structured document.
        This method is designed to work with the standardized output from
        DocumentProcessorFactory.prepare_document_for_extraction method.
        
        Args:
            document: Structured document data from prepare_for_extraction
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract from document text
        if "document_text" in document and document["document_text"]:
            text_entities = self.extract_from_text(
                document["document_text"], 
                metadata=document.get("metadata", {})
            )
            entities.extend(text_entities)
        
        # Extract from document structure if available
        if "document_structure" in document and document["document_structure"]:
            structure_entities = self._extract_document_structure(
                document.get("document_text", ""),
                document["document_structure"]
            )
            entities.extend(structure_entities)
        
        # Extract from metadata
        if "metadata" in document and document["metadata"]:
            metadata_entities = self._extract_from_metadata(document["metadata"])
            entities.extend(metadata_entities)
            
        # Extract from tables if present
        if "document_structure" in document and "tables" in document["document_structure"]:
            table_entities = self._extract_from_tables(
                document["document_structure"]["tables"],
                document.get("metadata", {})
            )
            entities.extend(table_entities)
            
        # Extract from form elements if present
        if "form_elements" in document:
            form_entities = self._extract_from_form_elements(
                document["form_elements"],
                document.get("metadata", {})
            )
            entities.extend(form_entities)
            
        # Check in document analysis results for form elements
        if "metadata" in document and "document_analysis" in document["metadata"]:
            analysis = document["metadata"]["document_analysis"]
            if "form_elements" in analysis:
                form_entities = self._extract_from_form_elements(
                    analysis["form_elements"],
                    document.get("metadata", {})
                )
                entities.extend(form_entities)
        
        # Check KG hints for form data instruction
        if "metadata" in document and "kg_extraction_hints" in document["metadata"]:
            kg_hints = document["metadata"]["kg_extraction_hints"]
            if kg_hints.get("has_form_data", False):
                logger.info(f"Found form data hint in KG extraction hints")
                
                # Apply specialized form entity extraction based on form type
                if kg_hints.get("form_type") == "checklist":
                    logger.info("Applying checklist extraction strategy")
                    # Already handled in _extract_from_form_elements
                    
                elif kg_hints.get("form_type") == "data_entry":
                    logger.info("Applying data entry extraction strategy")
                    # Already handled in _extract_from_form_elements
        
        # Deduplicate entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_document_structure(self, text: str, metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract document structure entities from text.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of structure entities
        """
        entities = []
        
        # Extract title (assume first line might be title if it's shorter than 100 chars)
        lines = text.splitlines()
        if lines and len(lines[0]) < 100:
            title_text = lines[0].strip()
            if title_text:
                title_entity = {
                    'id': str(uuid.uuid4()),
                    'type': self.document_entity_types['title'],
                    'text': title_text,
                    'confidence': 0.8,
                    'metadata': {
                        'line_number': 1,
                        'source': 'structure_analysis'
                    }
                }
                entities.append(title_entity)
                
        # Extract headers (lines ending with : or all caps lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line.endswith(':') or line.isupper()) and len(line) < 100:
                header_entity = {
                    'id': str(uuid.uuid4()),
                    'type': self.document_entity_types['header'],
                    'text': line,
                    'confidence': 0.7,
                    'metadata': {
                        'line_number': i + 1,
                        'source': 'structure_analysis'
                    }
                }
                entities.append(header_entity)
                
        # Extract paragraphs (blocks of text separated by empty lines)
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            if line.strip():
                current_paragraph.append(line)
            elif current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = []
                
        if current_paragraph:
            paragraphs.append('\n'.join(current_paragraph))
            
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 100:  # Only consider substantial paragraphs
                paragraph_entity = {
                    'id': str(uuid.uuid4()),
                    'type': self.document_entity_types['paragraph'],
                    'text': paragraph[:200] + '...' if len(paragraph) > 200 else paragraph,
                    'confidence': 0.9,
                    'metadata': {
                        'paragraph_index': i,
                        'paragraph_length': len(paragraph),
                        'source': 'structure_analysis'
                    }
                }
                entities.append(paragraph_entity)
                
        # Extract sections (defined by headers and their following content)
        sections = []
        current_section = None
        
        for entity in entities:
            if entity['type'] == self.document_entity_types['header']:
                if current_section:
                    sections.append(current_section)
                current_section = {
                    'header': entity,
                    'content': []
                }
            elif current_section and entity['type'] == self.document_entity_types['paragraph']:
                current_section['content'].append(entity)
                
        if current_section:
            sections.append(current_section)
            
        for i, section in enumerate(sections):
            if section['content']:
                # Create a section entity that contains the header and references to paragraphs
                section_entity = {
                    'id': str(uuid.uuid4()),
                    'type': self.document_entity_types['section'],
                    'text': section['header']['text'],
                    'confidence': 0.8,
                    'metadata': {
                        'section_index': i,
                        'paragraph_count': len(section['content']),
                        'source': 'structure_analysis'
                    },
                    'attributes': {
                        'header_id': section['header']['id'],
                        'paragraph_ids': [p['id'] for p in section['content']]
                    }
                }
                entities.append(section_entity)
                
        return entities
    
    def _extract_regex_entities(self, text: str) -> List[KnowledgeGraphEntity]:
        """
        Extract entities using regular expressions.
        
        Args:
            text: Document text
            
        Returns:
            List of regex-matched entities
        """
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group(0)
                entity = {
                    'id': str(uuid.uuid4()),
                    'type': entity_type,
                    'text': entity_text,
                    'confidence': 0.7,  # Regex matches are fairly reliable but not perfect
                    'metadata': {
                        'match_position': match.start(),
                        'match_length': len(entity_text),
                        'source': 'regex_extraction'
                    }
                }
                entities.append(entity)
                
        return entities
    
    def _extract_from_metadata(self, metadata: Dict[str, Any]) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from document metadata.
        
        Args:
            metadata: Document metadata dictionary
            
        Returns:
            List of metadata entities
        """
        entities = []
        
        # Common metadata fields to extract as entities
        metadata_fields = {
            'title': 'document_title',
            'author': 'person',
            'creator': 'person',
            'subject': 'document_metadata',
            'keywords': 'document_metadata',
            'created_date': 'date',
            'modified_date': 'date',
            'document_type': 'document_metadata',
            'organization': 'organization',
            'publisher': 'organization'
        }
        
        for field, entity_type in metadata_fields.items():
            if field in metadata and metadata[field]:
                # Handle list values (like keywords)
                if isinstance(metadata[field], list):
                    for item in metadata[field]:
                        if isinstance(item, (str, int, float)) and str(item).strip():
                            entity = {
                                'id': str(uuid.uuid4()),
                                'type': entity_type,
                                'text': str(item),
                                'confidence': 0.9,  # Metadata is usually reliable
                                'metadata': {
                                    'metadata_field': field,
                                    'source': 'document_metadata'
                                }
                            }
                            entities.append(entity)
                # Handle string, number, or date values
                elif isinstance(metadata[field], (str, int, float)) and str(metadata[field]).strip():
                    entity = {
                        'id': str(uuid.uuid4()),
                        'type': entity_type,
                        'text': str(metadata[field]),
                        'confidence': 0.9,  # Metadata is usually reliable
                        'metadata': {
                            'metadata_field': field,
                            'source': 'document_metadata'
                        }
                    }
                    entities.append(entity)
                    
        # Create a metadata container entity
        metadata_entity = {
            'id': str(uuid.uuid4()),
            'type': 'document_metadata',
            'text': 'Document Metadata',
            'confidence': 1.0,
            'metadata': {
                'source': 'document_metadata'
            },
            'attributes': {}
        }
        
        # Add important metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata_entity['attributes'][key] = value
            elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                metadata_entity['attributes'][key] = ', '.join(str(item) for item in value)
                
        entities.append(metadata_entity)
        
        return entities
    
    def _extract_from_tables(self, tables: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from document tables.
        
        Args:
            tables: List of table dictionaries
            metadata: Document metadata
            
        Returns:
            List of table entities
        """
        entities = []
        
        for i, table in enumerate(tables):
            # Create a table entity
            table_entity = {
                'id': str(uuid.uuid4()),
                'type': self.document_entity_types['table'],
                'text': table.get('caption', f'Table {i+1}'),
                'confidence': 0.9,
                'metadata': {
                    'table_index': i,
                    'rows': len(table.get('data', [])),
                    'columns': len(table.get('headers', [])),
                    'source': 'table_extraction'
                },
                'attributes': {}
            }
            
            # Add table headers
            if 'headers' in table and table['headers']:
                table_entity['attributes']['headers'] = table['headers']
                
            # Add table location if available
            if 'location' in table:
                table_entity['metadata']['location'] = table['location']
                
            # Add page number if available
            if 'page_number' in table:
                table_entity['metadata']['page_number'] = table['page_number']
                
            entities.append(table_entity)
            
            # Extract entities from table data
            if 'data' in table and isinstance(table['data'], list):
                for row_idx, row in enumerate(table['data']):
                    if isinstance(row, list):
                        for col_idx, cell in enumerate(row):
                            if isinstance(cell, (str, int, float)) and str(cell).strip():
                                cell_text = str(cell)
                                
                                # Extract regex entities from cell text
                                cell_entities = self._extract_regex_entities(cell_text)
                                
                                for entity in cell_entities:
                                    entity['metadata']['table_index'] = i
                                    entity['metadata']['row_index'] = row_idx
                                    entity['metadata']['column_index'] = col_idx
                                    
                                    # Add header if available
                                    if ('headers' in table and 
                                        isinstance(table['headers'], list) and 
                                        col_idx < len(table['headers'])):
                                        entity['metadata']['column_header'] = table['headers'][col_idx]
                                        
                                    entities.append(entity)
                                    
        return entities
    
    def _extract_from_form_elements(self, form_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> List[KnowledgeGraphEntity]:
        """
        Extract entities from form elements and their relationships.
        
        Args:
            form_data: Form elements data
            metadata: Document metadata
            
        Returns:
            List of form-related entities
        """
        entities = []
        
        # Skip if no form elements or not a form
        if not form_data or not form_data.get("has_form_elements", False):
            return entities
            
        logger.info(f"Extracting entities from form elements, type: {form_data.get('form_type', 'unknown')}")
        
        # Create a form container entity
        form_id = str(uuid.uuid4())
        form_entity = {
            'id': form_id,
            'type': self.document_entity_types['form'],
            'text': f"Form: {form_data.get('form_type', 'unknown')}",
            'confidence': form_data.get('confidence', 0.7),
            'metadata': {
                'form_type': form_data.get('form_type', 'unknown'),
                'has_form_elements': True,
                'element_count': len(form_data.get('elements', [])),
                'source': 'form_analysis'
            },
            'attributes': {
                'form_type': form_data.get('form_type', 'unknown'),
                'confidence': form_data.get('confidence', 0.7)
            },
            'relationships': []
        }
        
        # Process each form element
        for element in form_data.get('elements', []):
            element_type = element.get('type', 'unknown')
            element_id = str(uuid.uuid4())
            
            # Map form element type to entity type
            entity_type = self.document_entity_types.get(
                element_type, 
                self.document_entity_types['form_field']
            )
            
            # Create the element entity
            element_entity = {
                'id': element_id,
                'type': entity_type,
                'text': element.get('name', f"{element_type}_{len(entities)}"),
                'confidence': element.get('confidence', 0.7),
                'metadata': {
                    'bbox': element.get('bbox', []),
                    'page': element.get('page', 0),
                    'source': 'form_analysis'
                },
                'attributes': {},
                'relationships': [
                    {
                        'type': 'belongs_to',
                        'target': form_id,
                        'confidence': 1.0
                    }
                ]
            }
            
            # Add element-specific attributes
            if element_type == 'checkbox':
                element_entity['attributes']['checked'] = element.get('checked', False)
                element_entity['text'] = f"{element.get('name', 'Checkbox')}: {'Checked' if element.get('checked', False) else 'Unchecked'}"
                
            elif element_type == 'radio':
                element_entity['attributes']['selected'] = element.get('selected', False)
                element_entity['text'] = f"{element.get('name', 'Radio button')}: {'Selected' if element.get('selected', False) else 'Unselected'}"
                
            elif element_type == 'text_field':
                element_entity['attributes']['value'] = element.get('value', '')
                
                # For text fields with values, create a separate value entity
                if element.get('value'):
                    value_id = str(uuid.uuid4())
                    value_entity = {
                        'id': value_id,
                        'type': self.document_entity_types['form_value'],
                        'text': element.get('value', ''),
                        'confidence': element.get('confidence', 0.7),
                        'metadata': {
                            'field_name': element.get('name', ''),
                            'source': 'form_analysis'
                        },
                        'relationships': [
                            {
                                'type': 'value_of',
                                'target': element_id,
                                'confidence': 1.0
                            }
                        ]
                    }
                    entities.append(value_entity)
                    
                    # Add relationship from element to value
                    element_entity['relationships'].append({
                        'type': 'has_value',
                        'target': value_id,
                        'confidence': 1.0
                    })
            
            # Add label-related information if available
            if 'name' in element and element.get('name') != f"{element_type}_{len(entities)}":
                # Create a label entity
                label_id = str(uuid.uuid4())
                label_entity = {
                    'id': label_id,
                    'type': self.document_entity_types['form_label'],
                    'text': element.get('name', ''),
                    'confidence': element.get('confidence', 0.7) * 0.9,  # Slightly lower confidence for derived label
                    'metadata': {
                        'element_type': element_type,
                        'source': 'form_analysis'
                    },
                    'relationships': [
                        {
                            'type': 'labels',
                            'target': element_id,
                            'confidence': 0.9
                        }
                    ]
                }
                entities.append(label_entity)
                
                # Add relationship from element to label
                element_entity['relationships'].append({
                    'type': 'labeled_by',
                    'target': label_id,
                    'confidence': 0.9
                })
            
            # Add relationship from form to element
            form_entity['relationships'].append({
                'type': 'contains',
                'target': element_id,
                'confidence': 1.0
            })
            
            entities.append(element_entity)
        
        # Process structured data if available
        if 'structured_data' in form_data and form_data['structured_data']:
            for field_name, field_value in form_data['structured_data'].items():
                # Skip empty values
                if not field_value:
                    continue
                    
                # Create key-value pair entity
                kv_id = str(uuid.uuid4())
                kv_entity = {
                    'id': kv_id,
                    'type': 'key_value_pair',
                    'text': f"{field_name}: {field_value}",
                    'confidence': form_data.get('confidence', 0.7),
                    'metadata': {
                        'source': 'form_structured_data'
                    },
                    'attributes': {
                        'key': field_name,
                        'value': field_value
                    },
                    'relationships': [
                        {
                            'type': 'extracted_from',
                            'target': form_id,
                            'confidence': 1.0
                        }
                    ]
                }
                entities.append(kv_entity)
                
                # Add relationship from form to key-value pair
                form_entity['relationships'].append({
                    'type': 'has_data',
                    'target': kv_id,
                    'confidence': 1.0
                })
        
        # Add the form entity itself
        entities.append(form_entity)
        
        return entities
        
    def extract_form_relationships(self, entities: List[KnowledgeGraphEntity]) -> List[Dict[str, Any]]:
        """
        Extract relationships between form elements and other entities.
        
        This method analyzes entities to find semantic relationships between
        form elements and other entity types, such as people, organizations,
        dates, etc.
        
        Args:
            entities: List of entities extracted from a document
            
        Returns:
            List of relationships between entities
        """
        relationships = []
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get('type')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Find form fields
        form_fields = []
        for entity_type in ['form_checkbox', 'form_radio_button', 'form_text_field', 'form_field']:
            if entity_type in entities_by_type:
                form_fields.extend(entities_by_type[entity_type])
        
        # Find form values
        form_values = entities_by_type.get('form_value', [])
        
        # Find form labels
        form_labels = entities_by_type.get('form_label', [])
        
        # Process form fields
        for field in form_fields:
            field_name = field.get('text', '').lower()
            field_id = field.get('id')
            
            # Skip if no ID
            if not field_id:
                continue
                
            # Check if this field might refer to a person
            person_related = any(keyword in field_name for keyword in 
                ['name', 'person', 'customer', 'client', 'employee', 'author'])
            
            # Check if this field might refer to an organization
            org_related = any(keyword in field_name for keyword in 
                ['company', 'organization', 'business', 'employer', 'vendor'])
            
            # Check if this field might refer to a date
            date_related = any(keyword in field_name for keyword in 
                ['date', 'day', 'month', 'year', 'birthday', 'issued', 'created'])
            
            # Connect to relevant entities based on field name
            if person_related and 'person' in entities_by_type:
                for person in entities_by_type['person']:
                    rel = {
                        'source': field_id,
                        'target': person.get('id'),
                        'type': 'refers_to',
                        'confidence': 0.7
                    }
                    relationships.append(rel)
            
            if org_related and 'organization' in entities_by_type:
                for org in entities_by_type['organization']:
                    rel = {
                        'source': field_id,
                        'target': org.get('id'),
                        'type': 'refers_to',
                        'confidence': 0.7
                    }
                    relationships.append(rel)
            
            if date_related and 'date' in entities_by_type:
                for date in entities_by_type['date']:
                    rel = {
                        'source': field_id,
                        'target': date.get('id'),
                        'type': 'refers_to_date',
                        'confidence': 0.7
                    }
                    relationships.append(rel)
        
        return relationships
    
    def _deduplicate_entities(self, entities: List[KnowledgeGraphEntity]) -> List[KnowledgeGraphEntity]:
        """
        Remove duplicate entities based on type and text.
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated list of entities
        """
        deduplicated = []
        seen = set()  # (type, text) tuples
        
        for entity in entities:
            key = (entity['type'], entity['text'])
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
                
        return deduplicated 