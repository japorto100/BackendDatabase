"""
Document Vision Adapter

Adapter that processes documents containing text and images using vision models.
This module bridges document processing with vision capabilities.

Key features:
1. Extracts text and images from various document formats (PDF, DOCX, etc.)
2. Processes images using vision models for comprehensive document understanding
3. Seamlessly integrates with different vision providers through the VisionProviderFactory
4. Supports ColPali model for efficient document understanding by default

Use cases:
- Multimodal document analysis
- Document-to-knowledge extraction
- Compliance verification
- Layout-based document classification
- Automatic summarization of complex documents
"""

import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from PIL import Image
import fitz  # PyMuPDF
import io
import numpy as np
from datetime import datetime

from models_app.ai_models.vision.vision_factory import VisionProviderFactory
from models_app.ai_models.knowledge_integration.knowledge_augmented_generation import KAGBuilder
from models_app.knowledge.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
from models_app.knowledge.indexing.document_indexer import BidirectionalIndexer

logger = logging.getLogger(__name__)

@dataclass
class ProcessedPage:
    """Class representing a processed document page with text and image content."""
    page_num: int
    text: str
    images: List[Image.Image]
    image_positions: List[Dict[str, float]]  # Position info for each image


class DocumentVisionAdapter:
    """
    Adapter for processing documents with text and images using vision models.
    
    This adapter:
    1. Extracts text and images from documents (PDF, DOCX, etc.)
    2. Processes the images using vision models
    3. Combines text and image analysis for comprehensive document understanding
    
    By default, it uses ColPali model which is specifically designed for document understanding
    and can process both textual and visual content efficiently without complex OCR pipelines.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document vision adapter.
        
        Args:
            config: Configuration for the adapter and underlying vision models
        """
        self.config = config or {}
        self.vision_factory = VisionProviderFactory()
        
        # Default configuration
        self.extract_images = self.config.get('extract_images', True)
        self.max_images_per_page = self.config.get('max_images_per_page', 5)
        self.min_image_size = self.config.get('min_image_size', 100)  # Min dimension in pixels
        
        # Default to document_processing task with ColPali model
        self.vision_task = self.config.get('vision_task', 'document_processing')
        
        # Configure vision provider based on task
        vision_config = self.config.get('vision_config', None)
        if not vision_config:
            # Use ColPali by default for document processing
            vision_config = {
                'provider_type': 'vision',
                'vision_type': 'lightweight',
                'model_type': 'colpali',  # Use ColPali model by default
                'quantization_level': '4bit'  # Optimal quantization for most devices
            }
            
            # If custom provider is requested, use factory recommendations
            if self.config.get('use_recommended_provider', False):
                vision_config = self.vision_factory.get_recommended_provider(
                    task=self.vision_task
                )
        
        # We don't initialize the vision provider yet to save resources
        self.vision_config = vision_config
        self.vision_provider = None
    
    def _get_vision_provider(self):
        """Get or initialize the vision provider."""
        if self.vision_provider is None:
            logger.info("Initializing vision provider for document processing")
            self.vision_provider = self.vision_factory.create_provider(self.vision_config)
            self.vision_provider.initialize()
        return self.vision_provider
    
    def process_document(self, document_path: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with text and images.
        
        This method extracts text and images from various document formats and processes
        them together to understand the document in its entirety. The ColPali model 
        (default) is particularly effective for this task as it processes documents by 
        considering both textual and visual content together.
        
        Args:
            document_path: Path to the document
            query: Optional query to guide the processing
            
        Returns:
            Dict[str, Any]: Processing results including text content and image analysis
            
        Use cases:
            - Document understanding/summarization
            - Visual question answering on documents
            - Information extraction from complex layouts
            - Document classification based on content and structure
        """
        # Check file existence
        if not os.path.exists(document_path):
            return {
                "error": f"Document not found: {document_path}",
                "success": False
            }
        
        try:
            # Extract text and images based on document type
            file_ext = os.path.splitext(document_path)[1].lower()
            
            if file_ext == '.pdf':
                pages = self._process_pdf(document_path)
            elif file_ext in ['.docx', '.doc']:
                pages = self._process_docx(document_path)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff']:
                # Single image document
                pages = self._process_image_as_document(document_path)
            else:
                # Try to process as text document
                pages = self._process_text_document(document_path)
            
            # No pages were extracted
            if not pages:
                return {
                    "error": f"Could not extract content from document: {document_path}",
                    "success": False
                }
            
            # Combine all page texts
            full_text = "\n\n".join([page.text for page in pages])
            
            # Collect all images (limit to reasonable number)
            MAX_TOTAL_IMAGES = 10
            all_images = []
            for page in pages:
                all_images.extend(page.images[:min(len(page.images), self.max_images_per_page)])
                if len(all_images) >= MAX_TOTAL_IMAGES:
                    all_images = all_images[:MAX_TOTAL_IMAGES]
                    break
            
            # If no images or extraction disabled, return just the text
            if not all_images or not self.extract_images:
                return {
                    "text": full_text,
                    "page_count": len(pages),
                    "has_images": False,
                    "success": True
                }
            
            # Process with vision provider
            vision_provider = self._get_vision_provider()
            
            # Prepare the document for vision processing
            document_data = {
                "text": full_text,
                "images": all_images
            }
            
            # Process with the vision provider
            if query:
                result = vision_provider.process_document_with_images(document_data, query)
            else:
                # Default to document summarization if no query
                summary_prompt = "Provide a concise summary of this document based on both text and images."
                result = vision_provider.process_document_with_images(document_data, summary_prompt)
            
            # Add document metadata to the result
            result.update({
                "page_count": len(pages),
                "has_images": True,
                "image_count": len(all_images),
                "file_path": document_path,
                "file_type": file_ext,
                "success": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document with vision: {str(e)}")
            return {
                "error": f"Error processing document: {str(e)}",
                "file_path": document_path,
                "success": False
            }
    
    def _process_pdf(self, pdf_path: str) -> List[ProcessedPage]:
        """
        Extract text and images from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List[ProcessedPage]: Processed pages with text and images
        """
        pages = []
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                
                # Extract text
                text = page.get_text()
                
                # Extract images if enabled
                images = []
                image_positions = []
                
                if self.extract_images:
                    # Get images
                    image_list = page.get_images(full=True)
                    
                    for img_idx, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Convert to PIL Image
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Check if image is big enough
                            if min(image.size) >= self.min_image_size:
                                images.append(image)
                                
                                # Get image position on page
                                # This is simplified - getting exact positions is complex
                                image_rect = page.get_image_bbox(img_info[0])
                                if image_rect:
                                    # Normalize to page coordinates (0-1)
                                    page_rect = page.rect
                                    position = {
                                        "x": (image_rect.x0 - page_rect.x0) / page_rect.width,
                                        "y": (image_rect.y0 - page_rect.y0) / page_rect.height,
                                        "width": image_rect.width / page_rect.width,
                                        "height": image_rect.height / page_rect.height
                                    }
                                    image_positions.append(position)
                                else:
                                    # If position can't be determined, use a placeholder
                                    image_positions.append({
                                        "x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5
                                    })
                                
                        except Exception as e:
                            logger.warning(f"Error extracting image {img_idx} from page {page_idx}: {e}")
                
                # Create ProcessedPage
                processed_page = ProcessedPage(
                    page_num=page_idx + 1,
                    text=text,
                    images=images,
                    image_positions=image_positions
                )
                
                pages.append(processed_page)
            
            return pages
            
        except Exception as e:
            logger.error(f"Error processing PDF document: {e}")
            # Return a single empty page on error
            return [ProcessedPage(page_num=1, text=f"Error processing PDF: {str(e)}", images=[], image_positions=[])]
    
    def _process_docx(self, docx_path: str) -> List[ProcessedPage]:
        """
        Extract text and images from a DOCX document.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            List[ProcessedPage]: Processed pages with text and images
        """
        try:
            import docx
            from docx.document import Document as DocxDocument
            from docx.parts.image import ImagePart
            
            # Open the document
            doc = docx.Document(docx_path)
            
            # Extract all text
            full_text = "\n".join([para.text for para in doc.paragraphs])
            
            # Extract images if enabled
            images = []
            image_positions = []
            
            if self.extract_images:
                # Extract images (positions are approximate)
                rel_ids = []
                for rel_id, rel in doc.part.rels.items():
                    if "image" in rel.target_ref:
                        rel_ids.append(rel_id)
                
                for rel_id in rel_ids:
                    try:
                        image_part = doc.part.related_parts[rel_id]
                        if isinstance(image_part, ImagePart):
                            image_bytes = image_part.blob
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Check if image is big enough
                            if min(image.size) >= self.min_image_size:
                                images.append(image)
                                # We don't have accurate position info in this basic implementation
                                image_positions.append({
                                    "x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5
                                })
                    except Exception as e:
                        logger.warning(f"Error extracting image with rel_id {rel_id}: {e}")
            
            # Create a single processed page (DOCX doesn't have page info in this basic implementation)
            return [ProcessedPage(
                page_num=1,
                text=full_text,
                images=images,
                image_positions=image_positions
            )]
            
        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
            # Return a single page with just an error message
            return [ProcessedPage(
                page_num=1, 
                text="Unable to process DOCX: python-docx not installed", 
                images=[],
                image_positions=[]
            )]
        except Exception as e:
            logger.error(f"Error processing DOCX document: {e}")
            # Return a single empty page on error
            return [ProcessedPage(
                page_num=1,
                text=f"Error processing DOCX: {str(e)}",
                images=[],
                image_positions=[]
            )]
    
    def _process_image_as_document(self, image_path: str) -> List[ProcessedPage]:
        """
        Process a single image as a document.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List[ProcessedPage]: Single processed page with the image
        """
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Get OCR text if possible
            text = self._extract_text_from_image(image)
            
            # Create a single processed page
            return [ProcessedPage(
                page_num=1,
                text=text,
                images=[image],
                image_positions=[{"x": 0, "y": 0, "width": 1, "height": 1}]
            )]
            
        except Exception as e:
            logger.error(f"Error processing image as document: {e}")
            # Return a single empty page on error
            return [ProcessedPage(
                page_num=1,
                text=f"Error processing image: {str(e)}",
                images=[],
                image_positions=[]
            )]
    
    def _process_text_document(self, text_path: str) -> List[ProcessedPage]:
        """
        Process a plain text document.
        
        Args:
            text_path: Path to the text file
            
        Returns:
            List[ProcessedPage]: Single processed page with the text
        """
        try:
            # Read the text file
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create a single processed page
            return [ProcessedPage(
                page_num=1,
                text=text,
                images=[],
                image_positions=[]
            )]
            
        except Exception as e:
            logger.error(f"Error processing text document: {e}")
            # Return a single empty page on error
            return [ProcessedPage(
                page_num=1,
                text=f"Error processing text document: {str(e)}",
                images=[],
                image_positions=[]
            )]
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from an image using OCR.
        
        Note: For document images, ColPali can often capture both the visual and textual
        elements without requiring a separate OCR step, which makes it particularly effective
        for document understanding.
        
        Args:
            image: PIL Image to extract text from
            
        Returns:
            str: Extracted text or empty string if OCR fails
        """
        try:
            # Try to use pytesseract if available
            import pytesseract
            return pytesseract.image_to_string(image)
        except ImportError:
            # Try to use easyocr if available
            try:
                import easyocr
                reader = easyocr.Reader(['en'])
                result = reader.readtext(np.array(image))
                return '\n'.join([item[1] for item in result])
            except ImportError:
                logger.warning("No OCR library available. Install pytesseract or easyocr.")
                return ""
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
    
    def analyze_document(self, document_path: str) -> Dict[str, Any]:
        """
        Analyze a document without processing it with the vision model.
        
        This is useful for getting document structure and metadata.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dict[str, Any]: Document analysis results
            
        Use cases:
            - Quick document inspection
            - Document categorization by structure
            - Preprocessing for document ingestion pipelines
            - Document metadata extraction
        """
        # Check file existence
        if not os.path.exists(document_path):
            return {
                "error": f"Document not found: {document_path}",
                "success": False
            }
        
        try:
            # Extract pages based on document type
            file_ext = os.path.splitext(document_path)[1].lower()
            
            if file_ext == '.pdf':
                pages = self._process_pdf(document_path)
            elif file_ext in ['.docx', '.doc']:
                pages = self._process_docx(document_path)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff']:
                pages = self._process_image_as_document(document_path)
            else:
                pages = self._process_text_document(document_path)
            
            # Count words and characters
            full_text = "\n\n".join([page.text for page in pages])
            word_count = len(full_text.split())
            char_count = len(full_text)
            
            # Count total images
            image_count = sum(len(page.images) for page in pages)
            
            # Get file size
            file_size = os.path.getsize(document_path)
            
            # Generate page summaries
            page_summaries = []
            for page in pages:
                page_words = len(page.text.split())
                page_summaries.append({
                    "page_num": page.page_num,
                    "word_count": page_words,
                    "image_count": len(page.images),
                    "text_preview": page.text[:100] + "..." if len(page.text) > 100 else page.text
                })
            
            return {
                "file_path": document_path,
                "file_type": file_ext,
                "file_size_bytes": file_size,
                "page_count": len(pages),
                "word_count": word_count,
                "character_count": char_count,
                "has_images": image_count > 0,
                "image_count": image_count,
                "page_summaries": page_summaries,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {
                "error": f"Error analyzing document: {str(e)}",
                "file_path": document_path,
                "success": False
            }
            
    def process_for_knowledge_graph(self, document_path: str, extraction_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document specifically for knowledge graph extraction.
        
        This method bridges the gap between document processing and knowledge graph integration,
        by extracting structured information from documents that can be used to populate
        or enhance knowledge graphs.
        
        Args:
            document_path: Path to the document
            extraction_params: Parameters to guide the extraction process
            
        Returns:
            Dict[str, Any]: Structured data extracted from the document suitable for KG ingestion
            
        Use cases:
            - Extracting entities and relationships from documents for KG population
            - Building document-based knowledge bases
            - Creating semantic connections between documents and knowledge
            - Enabling document-based knowledge reasoning
        """
        extraction_params = extraction_params or {}
        
        try:
            # First process the document normally
            processed_doc = self.process_document(document_path)
            
            if not processed_doc.get("success", False):
                return processed_doc
                
            # Prepare document data for KG integration
            document_data = {
                "id": os.path.basename(document_path),
                "title": os.path.basename(document_path),
                "content": processed_doc.get("text", ""),
                "metadata": processed_doc.get("metadata", {}),
                "source": document_path
            }
            
            # Use KAGBuilder for KG integration if available
            try:
                # Integriere direkt mit KAGBuilder
                kag_builder = KAGBuilder()
                
                # Verwende optimierte Granularität für KG-Extraktion
                extraction_granularity = extraction_params.get('granularity', 'low')
                
                # Spezifische Extraktionsparameter für KG-Builder
                kg_extraction_params = {
                    'granularity': extraction_granularity,
                    'build_bidirectional_index': True,
                    **extraction_params
                }
                
                # Baue Knowledge Base auf
                kg_result = kag_builder.build_knowledge_base(
                    documents=[document_data],
                    base_name=f"KB_{os.path.basename(document_path)}",
                    extraction_params=kg_extraction_params,
                    build_bidirectional_index=True
                )
                
                # Kombiniere Ergebnisse
                result = {
                    "success": True,
                    "document": processed_doc,
                    "knowledge_graph": kg_result,
                    "graph_id": kg_result.get("graph_id")
                }
                
                # Logging für erfolgreiche KG-Integration
                logger.info(f"Document successfully processed for KG: {document_path}")
                
                return result
                
            except ImportError:
                # Fallback, wenn KAGBuilder nicht verfügbar ist
                logger.warning("KAGBuilder not available, falling back to vision module integration")
                
                # Fall back to using the @vision module's KG capabilities
                from models_app.vision.knowledge_graph import DocumentKnowledgeExtractor
                
                kg_extractor = DocumentKnowledgeExtractor()
                kg_elements = kg_extractor.extract_from_document(
                    processed_doc,
                    extraction_params
                )
                
                return {
                    "success": True,
                    "document": processed_doc,
                    "knowledge_elements": kg_elements,
                    "note": "Used fallback knowledge graph extraction via models_app.vision.knowledge_graph"
                }
            
        except Exception as e:
            logger.error(f"Error extracting knowledge elements from document: {str(e)}")
            return {
                "error": f"Error extracting knowledge elements: {str(e)}",
                "file_path": document_path,
                "success": False
            }

    def process_hybrid(self, document_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document using both KG and RAG in a single pass.
        
        This method combines knowledge graph extraction with RAG indexing
        for comprehensive document processing and knowledge integration.
        
        Args:
            document_path: Path to the document
            session_id: Optional session ID for RAG indexing
            
        Returns:
            Dict[str, Any]: Combined KG and RAG processing results
        """
        try:
            # Process document for knowledge graph
            kg_result = self.process_for_knowledge_graph(document_path)
            
            if not kg_result.get("success", False):
                return kg_result
                
            # Get graph ID from KG processing
            graph_id = kg_result.get("knowledge_graph", {}).get("graph_id")
            
            # Process document for RAG
            # First process the document normally if not already done
            document_data = kg_result.get("document") or self.process_document(document_path)
            
            # Initialize RAG integration components
            from models_app.knowledge.rag.rag_manager import RAGManager
            rag_manager = RAGManager()
            session_id = session_id or "default"
            
            # Get RAG model for session
            rag_model = rag_manager.get_model(session_id)
            if not rag_model:
                logger.warning(f"No RAG model found for session {session_id}, attempting to initialize")
                # Could implement RAG model initialization here
            
            # Add document to RAG index if model exists
            if rag_model:
                # Prepare document for RAG indexing
                rag_document = {
                    "text": document_data.get("text", ""),
                    "metadata": {
                        "source": document_path,
                        "title": os.path.basename(document_path)
                    }
                }
                
                # Add images if available
                if "images" in document_data and document_data["images"]:
                    rag_document["images"] = document_data["images"]
                
                # Add to RAG index
                try:
                    index_result = rag_model.add_document(rag_document)
                    
                    # If graph_id exists, create bidirectional linking
                    if graph_id and hasattr(rag_model, 'vectorstore'):
                        # Initialize components for bidirectional indexing
                        kg_manager = KnowledgeGraphManager()
                        
                        # Create bidirectional indexer
                        bi_indexer = BidirectionalIndexer(
                            kg_manager.graph_storage,
                            rag_model.vectorstore
                        )
                        
                        # Link document to graph
                        linking_result = bi_indexer.link_document_to_graph(
                            document_path,
                            graph_id
                        )
                        
                        # Complete result with all components
                        return {
                            "success": True,
                            "document": document_data,
                            "knowledge_graph": kg_result.get("knowledge_graph"),
                            "rag_index": {
                                "session_id": session_id,
                                "indexed": True,
                                "index_info": index_result
                            },
                            "linking": linking_result
                        }
                except Exception as e:
                    logger.error(f"Error in RAG indexing: {str(e)}")
                    # Continue with partial result
            
            # Return combined result (even if RAG indexing failed)
            return {
                "success": True,
                "document": document_data,
                "knowledge_graph": kg_result.get("knowledge_graph"),
                "rag_index": {
                    "session_id": session_id,
                    "indexed": False,
                    "error": "RAG indexing failed or not available"
                }
            }
        except Exception as e:
            logger.error(f"Error in hybrid processing: {str(e)}")
            return {
                "error": f"Error in hybrid processing: {str(e)}",
                "file_path": document_path,
                "success": False
            } 