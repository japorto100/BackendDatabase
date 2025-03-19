"""
Zentrale ChunkingService-Klasse für alle Dokumentenverarbeitungs-Komponenten.

Diese Klasse zentralisiert alle Chunking-Methoden, die bisher zwischen DocumentProcessingManager
und verschiedenen Adaptern dupliziert waren.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
from PIL import Image
import fitz  # PyMuPDF
import math

from models_app.vision.document.utils.core.processing_metadata_context import ProcessingMetadataContext

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    start_offset: int
    end_offset: int
    chunk_type: str
    content_summary: Dict[str, Any]
    parent_document: str

@dataclass
class ChunkingStrategy:
    """Strategy for document chunking."""
    method: str  # "size", "content", "hybrid"
    chunk_size: int
    overlap: int
    preserve_boundaries: bool
    content_types: List[str]

class ChunkingService:
    """
    Zentraler Dienst für Dokument-Chunking-Operationen.
    
    Diese Klasse stellt einheitliche Methoden zum Chunking von Dokumenten bereit,
    basierend auf verschiedenen Strategien und Anforderungen. Sie ersetzt mehrere
    redundante Implementierungen in verschiedenen Komponenten.
    """
    
    VERSION = "2.0.0"
    
    # Add size-based chunking constants
    LARGE_SIZE_THRESHOLD = 10 * 1024 * 1024  # 10MB
    LARGE_PAGE_THRESHOLD = 10  # 10 pages
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert den ChunkingService.
        
        Args:
            config: Konfigurationseinstellungen für das Chunking
        """
        self.config = config or {}
        
        # Standard-Chunking-Konfiguration
        self.default_config = {
            "max_page_count": 20,         # Maximale Seitenanzahl für eine Datei ohne Chunking
            "max_file_size_mb": 25,       # Maximale Dateigröße in MB ohne Chunking
            "enable_chunking": True,      # Ob Chunking aktiviert ist
            "default_chunk_size": 2000,   # Standardgröße für Chunks (in Zeichen)
            "default_chunk_overlap": 200, # Standardüberlappung zwischen Chunks
            "min_chunk_size": 500,        # Minimale Chunk-Größe
            "max_chunk_size": 5000,       # Maximale Chunk-Größe
            "respect_semantic_boundaries": True,  # Respektiert semantische Grenzen
            "default_strategy": "adaptive"  # Standard-Chunking-Strategie
        }
        
        # Konfiguration mit benutzerdefinierten Werten aktualisieren
        self.chunking_config = {**self.default_config, **self.config}
        
        self.default_chunk_size = self.config.get("default_chunk_size", 1024 * 1024)  # 1MB
        self.min_chunk_size = self.config.get("min_chunk_size", 100 * 1024)  # 100KB
        self.max_chunk_size = self.config.get("max_chunk_size", 5 * 1024 * 1024)  # 5MB
        self.default_overlap = self.config.get("default_overlap", 0.1)  # 10% overlap
    
    def check_if_chunking_needed(self, document_path: str, metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Überprüft, ob ein Dokument in Chunks verarbeitet werden sollte, basierend auf Dateigröße und Seitenanzahl.
        
        Args:
            document_path: Pfad zum Dokument
            metadata_context: Optional, MetadataContext zur Aufzeichnung von Entscheidungen
            
        Returns:
            Dict: Ergebnis der Chunking-Bewertung
        """
        # Dateigröße in MB
        file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
        
        # Überprüfe Dateierweiterung, um festzustellen, ob es sich um ein mehrseitiges Dokument handelt
        _, ext = os.path.splitext(document_path)
        is_potential_multipage = ext.lower() in ['.pdf', '.docx', '.doc', '.pptx', '.ppt']
        
        # Für PDF-Dateien: tatsächliche Seitenanzahl prüfen
        page_count = 1
        if ext.lower() == '.pdf':
            try:
                doc = fitz.open(document_path)
                page_count = len(doc)
                doc.close()
            except:
                # Wenn wir die Seitenanzahl nicht lesen können, schätzen wir sie anhand der Dateigröße
                page_count = max(1, int(file_size_mb / 2))  # Grobe Schätzung
        
        # Bestimmen, ob Chunking erforderlich ist
        chunking_needed = (
            is_potential_multipage and
            (file_size_mb > self.chunking_config["max_file_size_mb"] or
             page_count > self.chunking_config["max_page_count"])
        )
        
        # Aufzeichnen der Entscheidung in Metadaten, falls vorhanden
        if metadata_context and chunking_needed:
            metadata_context.record_decision(
                component="ChunkingService",
                decision="Document requires chunking",
                reason=f"Large document: {file_size_mb:.1f}MB, {page_count} pages",
                confidence=0.95
            )
        
        return {
            "chunking_needed": chunking_needed,
            "file_size_mb": file_size_mb,
            "page_count": page_count,
            "is_potential_multipage": is_potential_multipage
        }
    
    def should_apply_chunking(self, document_path: str, analysis_result: Dict[str, Any]) -> bool:
        """
        Bestimmt, ob ein Dokument in Chunks verarbeitet werden sollte, basierend auf seiner Größe und Komplexität.
        
        Args:
            document_path: Pfad zur Dokumentdatei
            analysis_result: Ergebnisse der Dokumentanalyse
            
        Returns:
            bool: True, wenn Chunking angewendet werden sollte, sonst False
        """
        # Dateigröße in MB
        file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
        
        # Seitenanzahl, falls verfügbar
        page_count = analysis_result.get("page_count", 1)
        
        # Dokument ist zu groß
        if file_size_mb > 20:
            return True
            
        # Dokument hat viele Seiten
        if page_count > 50:
            return True
            
        # Dokument ist komplex und ziemlich groß
        if analysis_result.get("complexity", 0) > 0.7 and (file_size_mb > 10 or page_count > 30):
            return True
            
        return False
    
    def determine_chunking_strategy(self, document_info: Dict[str, Any]) -> ChunkingStrategy:
        """
        Determine the optimal chunking strategy based on document characteristics.
        
        Args:
            document_info: Document analysis information
            
        Returns:
            ChunkingStrategy for the document
        """
        file_size = document_info.get("file_size", 0)
        content_types = document_info.get("content_types", [])
        
        # Determine base chunk size
        if file_size < 1024 * 1024:  # < 1MB
            chunk_size = self.min_chunk_size
        elif file_size > 50 * 1024 * 1024:  # > 50MB
            chunk_size = self.max_chunk_size
        else:
            chunk_size = self.default_chunk_size
        
        # Adjust strategy based on content
        if "image" in content_types:
            method = "content"
            chunk_size = max(chunk_size, 2 * 1024 * 1024)  # Minimum 2MB for images
        elif "text" in content_types and file_size > 10 * 1024 * 1024:
            method = "hybrid"
            chunk_size = min(chunk_size, 3 * 1024 * 1024)  # Maximum 3MB for text
        else:
            method = "size"
        
        return ChunkingStrategy(
            method=method,
            chunk_size=chunk_size,
            overlap=int(chunk_size * self.default_overlap),
            preserve_boundaries=True,
            content_types=content_types
        )
    
    def create_chunks(
        self,
        document_path: str,
        strategy: ChunkingStrategy,
        metadata_context: Optional[Any] = None
    ) -> List[ChunkMetadata]:
        """
        Create document chunks based on the determined strategy.
        
        Args:
            document_path: Path to the document
            strategy: Chunking strategy to use
            metadata_context: Optional metadata context
            
        Returns:
            List of chunk metadata
        """
        chunks = []
        
        try:
            if strategy.method == "content":
                chunks = self._create_content_based_chunks(document_path, strategy)
            elif strategy.method == "hybrid":
                chunks = self._create_hybrid_chunks(document_path, strategy)
            else:  # size-based
                chunks = self._create_size_based_chunks(document_path, strategy)
            
            # Record chunking in metadata
            if metadata_context:
                metadata_context.record_preprocessing_step(
                    step_name="chunking",
                    details={
                        "strategy": strategy.__dict__,
                        "chunk_count": len(chunks),
                        "average_chunk_size": sum(c.end_offset - c.start_offset for c in chunks) / len(chunks)
                    }
                )
            
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Chunking failed: {str(e)}")
    
    def _create_content_based_chunks(
        self,
        document_path: str,
        strategy: ChunkingStrategy
    ) -> List[ChunkMetadata]:
        """Create chunks based on content boundaries."""
        chunks = []
        
        try:
            # Handle PDF documents
            if document_path.lower().endswith(".pdf"):
                doc = fitz.open(document_path)
                current_chunk = []
                current_size = 0
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_dict = page.get_text("dict")
                    
                    # Process each block in the page
                    for block in page_dict["blocks"]:
                        block_size = len(str(block).encode('utf-8'))
                        
                        if current_size + block_size > strategy.chunk_size:
                            # Create new chunk
                            chunk_id = f"chunk_{len(chunks)}"
                            chunks.append(ChunkMetadata(
                                chunk_id=chunk_id,
                                start_offset=sum(len(str(b).encode('utf-8')) for b in current_chunk),
                                end_offset=current_size,
                                chunk_type="content",
                                content_summary=self._summarize_chunk_content(current_chunk),
                                parent_document=document_path
                            ))
                            
                            current_chunk = [block]
                            current_size = block_size
                        else:
                            current_chunk.append(block)
                            current_size += block_size
                
                # Handle last chunk
                if current_chunk:
                    chunk_id = f"chunk_{len(chunks)}"
                    chunks.append(ChunkMetadata(
                        chunk_id=chunk_id,
                        start_offset=sum(len(str(b).encode('utf-8')) for b in current_chunk[:-1]),
                        end_offset=current_size,
                        chunk_type="content",
                        content_summary=self._summarize_chunk_content(current_chunk),
                        parent_document=document_path
                    ))
                
            # Handle image documents
            elif any(document_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff']):
                chunks = self._chunk_image(document_path, strategy)
            
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Content-based chunking failed: {str(e)}")
    
    def _create_hybrid_chunks(
        self,
        document_path: str,
        strategy: ChunkingStrategy
    ) -> List[ChunkMetadata]:
        """Create chunks using a hybrid approach (content + size)."""
        try:
            # Start with content-based chunking
            content_chunks = self._create_content_based_chunks(document_path, strategy)
            
            # Adjust chunk sizes if needed
            adjusted_chunks = []
            current_chunk = []
            current_size = 0
            
            for chunk in content_chunks:
                chunk_size = chunk.end_offset - chunk.start_offset
                
                if current_size + chunk_size > strategy.chunk_size:
                    # Create new adjusted chunk
                    if current_chunk:
                        adjusted_chunks.append(self._merge_chunks(current_chunk, document_path))
                    current_chunk = [chunk]
                    current_size = chunk_size
                else:
                    current_chunk.append(chunk)
                    current_size += chunk_size
            
            # Handle last chunk
            if current_chunk:
                adjusted_chunks.append(self._merge_chunks(current_chunk, document_path))
            
            return adjusted_chunks
            
        except Exception as e:
            raise ChunkingError(f"Hybrid chunking failed: {str(e)}")
    
    def _create_size_based_chunks(
        self,
        document_path: str,
        strategy: ChunkingStrategy
    ) -> List[ChunkMetadata]:
        """Create chunks based on size."""
        chunks = []
        file_size = os.path.getsize(document_path)
        
        # Calculate number of chunks
        num_chunks = math.ceil(file_size / strategy.chunk_size)
        
        for i in range(num_chunks):
            start_offset = i * strategy.chunk_size
            end_offset = min((i + 1) * strategy.chunk_size, file_size)
            
            chunks.append(ChunkMetadata(
                chunk_id=f"chunk_{i}",
                start_offset=start_offset,
                end_offset=end_offset,
                chunk_type="size",
                content_summary={"size": end_offset - start_offset},
                parent_document=document_path
            ))
        
        return chunks
    
    def _chunk_image(
        self,
        image_path: str,
        strategy: ChunkingStrategy
    ) -> List[ChunkMetadata]:
        """Chunk an image document."""
        chunks = []
        
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Calculate grid size based on image dimensions and chunk size
            aspect_ratio = width / height
            grid_size = math.ceil(math.sqrt(image.size[0] * image.size[1] / strategy.chunk_size))
            
            grid_width = math.ceil(grid_size * aspect_ratio)
            grid_height = math.ceil(grid_size / aspect_ratio)
            
            chunk_width = width // grid_width
            chunk_height = height // grid_height
            
            for i in range(grid_height):
                for j in range(grid_width):
                    # Calculate chunk boundaries
                    left = j * chunk_width
                    top = i * chunk_height
                    right = min((j + 1) * chunk_width, width)
                    bottom = min((i + 1) * chunk_height, height)
                    
                    chunk_id = f"chunk_{i}_{j}"
                    chunks.append(ChunkMetadata(
                        chunk_id=chunk_id,
                        start_offset=left + top * width,
                        end_offset=right + bottom * width,
                        chunk_type="image",
                        content_summary={
                            "dimensions": (right - left, bottom - top),
                            "position": (left, top)
                        },
                        parent_document=image_path
                    ))
            
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Image chunking failed: {str(e)}")
    
    def _merge_chunks(
        self,
        chunks: List[ChunkMetadata],
        document_path: str
    ) -> ChunkMetadata:
        """Merge multiple chunks into one."""
        if not chunks:
            raise ValueError("No chunks to merge")
        
        merged_id = f"merged_{chunks[0].chunk_id}"
        start_offset = min(chunk.start_offset for chunk in chunks)
        end_offset = max(chunk.end_offset for chunk in chunks)
        
        # Combine content summaries
        content_summary = {}
        for chunk in chunks:
            content_summary.update(chunk.content_summary)
        
        return ChunkMetadata(
            chunk_id=merged_id,
            start_offset=start_offset,
            end_offset=end_offset,
            chunk_type="merged",
            content_summary=content_summary,
            parent_document=document_path
        )
    
    def _summarize_chunk_content(self, content: List[Any]) -> Dict[str, Any]:
        """Generate a summary of chunk content."""
        summary = {
            "size": sum(len(str(item).encode('utf-8')) for item in content),
            "item_count": len(content)
        }
        
        # Add more detailed analysis if needed
        # This could include text statistics, image analysis, etc.
        
        return summary
    
    def process_chunked_document(
        self, 
        document_path: str, 
        processor_func, 
        metadata_context: Optional[ProcessingMetadataContext] = None,
        chunking_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verarbeitet ein Dokument in Chunks und aggregiert die Ergebnisse.
        
        Args:
            document_path: Pfad zum Dokument
            processor_func: Funktion zur Verarbeitung einzelner Chunks
            metadata_context: Optional, MetadataContext für Entscheidungen
            chunking_result: Optional, vorhandenes Chunking-Ergebnis
            
        Returns:
            Dict mit aggregierten Verarbeitungsergebnissen
        """
        # Prüfe, ob Chunking erforderlich ist
        if chunking_result is None:
            chunking_result = self.check_if_chunking_needed(document_path, metadata_context)
        
        if not chunking_result["chunking_needed"]:
            # Verarbeite das gesamte Dokument als einen Chunk
            return processor_func(document_path)
        
        # Dokument in Chunks zerlegen
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        chunks = self.chunk_document(content)
        
        # Verarbeite jeden Chunk
        results = []
        for chunk in chunks:
            try:
                chunk_result = processor_func(chunk["text"])
                if chunk_result:
                    results.append(chunk_result)
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                if metadata_context:
                    metadata_context.record_error(
                        component="ChunkingService",
                        message=f"Chunk processing error: {str(e)}",
                        error_type=type(e).__name__
                    )
        
        # Aggregiere Ergebnisse
        # TODO: Implementiere spezifische Aggregationslogik basierend auf Ergebnistyp
        
        return {
            "chunked_processing": True,
            "chunk_count": len(chunks),
            "successful_chunks": len(results),
            "results": results
        }
    
    def classify_document(self, document_path, document_info=None):
        """Classify document into size categories for optimal processing."""
        # Get document info if not provided
        if not document_info:
            document_info = self.get_document_info(document_path)
            
        file_size = document_info.get('size_bytes', 0)
        page_count = document_info.get('page_count', 0)
        
        # Classify document
        if file_size > self.LARGE_SIZE_THRESHOLD or page_count > self.LARGE_PAGE_THRESHOLD:
            return "large"
        elif page_count == 1:
            return "single_page"
        else:
            return "small"
    
    def process_document(self, document_path, chunking_config=None, metadata_context=None):
        """Process document with size-appropriate strategy."""
        # Default chunking config
        config = {
            "chunk_size": 5,  # default chunk size in pages
            "overlap": 1,      # default overlap between chunks
            "respect_sections": True,  # respect document sections
            "max_chunk_size_bytes": 5 * 1024 * 1024  # 5MB max per chunk
        }
        if chunking_config:
            config.update(chunking_config)
            
        # Get document classification
        doc_classification = self.classify_document(document_path)
        
        # Select processing strategy based on classification
        if doc_classification == "large":
            return self._process_large_document(document_path, config, metadata_context)
        elif doc_classification == "single_page":
            return self._process_single_page_document(document_path, metadata_context)
        else:
            return self._process_small_document(document_path, config, metadata_context) 