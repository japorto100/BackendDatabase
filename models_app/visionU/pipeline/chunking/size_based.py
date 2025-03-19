import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import fitz  # PyMuPDF
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_index: int
    start_page: int
    end_page: int
    size_bytes: int
    format: str
    path: str

class SizeBasedChunker:
    """Chunks documents based on size and content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize chunker with configuration."""
        self.config = config or {}
        self.chunk_sizes = {
            "small": 5 * 1024 * 1024,  # 5MB
            "medium": 10 * 1024 * 1024,  # 10MB
            "large": 20 * 1024 * 1024   # 20MB
        }
        self.max_pages_per_chunk = self.config.get("max_pages_per_chunk", 10)
        self.temp_dir = self.config.get("temp_dir", tempfile.gettempdir())
    
    def should_chunk(self, document_path: str) -> Tuple[bool, str]:
        """
        Determine if document should be chunked.
        
        Args:
            document_path: Path to document
            
        Returns:
            Tuple of (should_chunk, reason)
        """
        try:
            file_size = os.path.getsize(document_path)
            file_ext = os.path.splitext(document_path)[1].lower()
            
            # Check file size
            if file_size > self.chunk_sizes["small"]:
                return True, f"File size {file_size} bytes exceeds threshold"
            
            # Check page count for PDFs
            if file_ext == ".pdf":
                doc = fitz.open(document_path)
                page_count = doc.page_count
                doc.close()
                
                if page_count > self.max_pages_per_chunk:
                    return True, f"Page count {page_count} exceeds threshold"
            
            # Check image dimensions
            if file_ext in [".png", ".jpg", ".jpeg", ".tiff"]:
                with Image.open(document_path) as img:
                    width, height = img.size
                    if width * height > 25000000:  # 25MP
                        return True, "Image dimensions too large"
            
            return False, "Document within acceptable limits"
            
        except Exception as e:
            logger.error(f"Error checking chunk necessity: {str(e)}")
            return False, f"Error during check: {str(e)}"
    
    def create_chunks(self, document_path: str) -> List[ChunkMetadata]:
        """
        Create chunks from document.
        
        Args:
            document_path: Path to document
            
        Returns:
            List of chunk metadata
        """
        try:
            file_ext = os.path.splitext(document_path)[1].lower()
            
            if file_ext == ".pdf":
                return self._chunk_pdf(document_path)
            elif file_ext in [".png", ".jpg", ".jpeg", ".tiff"]:
                return self._chunk_image(document_path)
            else:
                raise ValueError(f"Unsupported format for chunking: {file_ext}")
            
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            raise
    
    def _chunk_pdf(self, pdf_path: str) -> List[ChunkMetadata]:
        """Chunk PDF document into smaller parts."""
        chunks = []
        doc = fitz.open(pdf_path)
        
        try:
            total_pages = doc.page_count
            current_chunk = 0
            start_page = 0
            
            while start_page < total_pages:
                # Determine end page for this chunk
                end_page = min(start_page + self.max_pages_per_chunk, total_pages)
                
                # Create new PDF for chunk
                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
                
                # Save chunk
                chunk_path = os.path.join(
                    self.temp_dir,
                    f"chunk_{current_chunk}_{os.path.basename(pdf_path)}"
                )
                chunk_doc.save(chunk_path)
                chunk_size = os.path.getsize(chunk_path)
                
                # Create metadata
                chunk_metadata = ChunkMetadata(
                    chunk_index=current_chunk,
                    start_page=start_page,
                    end_page=end_page,
                    size_bytes=chunk_size,
                    format="pdf",
                    path=chunk_path
                )
                chunks.append(chunk_metadata)
                
                # Clean up
                chunk_doc.close()
                current_chunk += 1
                start_page = end_page
            
            return chunks
            
        finally:
            doc.close()
    
    def _chunk_image(self, image_path: str) -> List[ChunkMetadata]:
        """Chunk large image into smaller parts."""
        chunks = []
        
        with Image.open(image_path) as img:
            width, height = img.size
            format_name = img.format.lower()
            
            # Calculate grid size based on image dimensions
            grid_size = self._calculate_grid_size(width, height)
            chunk_width = width // grid_size[0]
            chunk_height = height // grid_size[1]
            
            current_chunk = 0
            
            # Split image into grid
            for y in range(grid_size[1]):
                for x in range(grid_size[0]):
                    # Calculate chunk boundaries
                    left = x * chunk_width
                    top = y * chunk_height
                    right = min((x + 1) * chunk_width, width)
                    bottom = min((y + 1) * chunk_height, height)
                    
                    # Extract chunk
                    chunk = img.crop((left, top, right, bottom))
                    
                    # Save chunk
                    chunk_path = os.path.join(
                        self.temp_dir,
                        f"chunk_{current_chunk}_{os.path.basename(image_path)}"
                    )
                    chunk.save(chunk_path)
                    chunk_size = os.path.getsize(chunk_path)
                    
                    # Create metadata
                    chunk_metadata = ChunkMetadata(
                        chunk_index=current_chunk,
                        start_page=0,
                        end_page=0,
                        size_bytes=chunk_size,
                        format=format_name,
                        path=chunk_path
                    )
                    chunks.append(chunk_metadata)
                    
                    current_chunk += 1
        
        return chunks
    
    def _calculate_grid_size(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate optimal grid size for image chunking."""
        area = width * height
        if area <= 1000000:  # 1MP
            return (1, 1)
        elif area <= 4000000:  # 4MP
            return (2, 2)
        elif area <= 16000000:  # 16MP
            return (3, 3)
        else:
            return (4, 4)
    
    def cleanup_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """Clean up temporary chunk files."""
        for chunk in chunks:
            try:
                if os.path.exists(chunk.path):
                    os.remove(chunk.path)
            except Exception as e:
                logger.error(f"Failed to remove chunk {chunk.path}: {str(e)}")
    
    def merge_chunks(self, chunks: List[ChunkMetadata], output_path: str) -> bool:
        """
        Merge processed chunks back into single document.
        
        Args:
            chunks: List of chunk metadata
            output_path: Path for merged output
            
        Returns:
            bool: Success status
        """
        try:
            # Sort chunks by index
            chunks = sorted(chunks, key=lambda x: x.chunk_index)
            
            if chunks[0].format == "pdf":
                return self._merge_pdf_chunks(chunks, output_path)
            else:
                return self._merge_image_chunks(chunks, output_path)
            
        except Exception as e:
            logger.error(f"Chunk merging failed: {str(e)}")
            return False
    
    def _merge_pdf_chunks(self, chunks: List[ChunkMetadata], output_path: str) -> bool:
        """Merge PDF chunks."""
        merged_doc = fitz.open()
        
        try:
            for chunk in chunks:
                chunk_doc = fitz.open(chunk.path)
                merged_doc.insert_pdf(chunk_doc)
                chunk_doc.close()
            
            merged_doc.save(output_path)
            return True
            
        except Exception as e:
            logger.error(f"PDF chunk merging failed: {str(e)}")
            return False
            
        finally:
            merged_doc.close()
    
    def _merge_image_chunks(self, chunks: List[ChunkMetadata], output_path: str) -> bool:
        """Merge image chunks."""
        try:
            # Get dimensions of first chunk to determine format
            with Image.open(chunks[0].path) as first_chunk:
                format_name = first_chunk.format
            
            # Calculate final image dimensions
            total_width = 0
            total_height = 0
            chunk_positions = []
            
            grid_size = int(len(chunks) ** 0.5)
            chunk_width = 0
            chunk_height = 0
            
            for i, chunk in enumerate(chunks):
                with Image.open(chunk.path) as img:
                    if i == 0:
                        chunk_width = img.width
                        chunk_height = img.height
                        total_width = chunk_width * grid_size
                        total_height = chunk_height * grid_size
                    
                    # Calculate position
                    x = (i % grid_size) * chunk_width
                    y = (i // grid_size) * chunk_height
                    chunk_positions.append((x, y))
            
            # Create merged image
            merged_img = Image.new('RGB', (total_width, total_height))
            
            # Paste chunks
            for chunk, pos in zip(chunks, chunk_positions):
                with Image.open(chunk.path) as img:
                    merged_img.paste(img, pos)
            
            # Save merged image
            merged_img.save(output_path, format=format_name)
            return True
            
        except Exception as e:
            logger.error(f"Image chunk merging failed: {str(e)}")
            return False 