"""
WebDocumentAdapter: Processes web content (HTML, MHT files).
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Any
import traceback
from datetime import datetime
import hashlib
import re
from urllib.parse import urlparse, urljoin

from models_app.vision.document.adapters.document_format_adapter import DocumentFormatAdapter
from models_app.vision.document.utils.core.processing_metadata_context import ProcessingMetadataContext
from models_app.vision.document.utils.error_handling.errors import (
    DocumentProcessingError,
    DocumentValidationError
)
from models_app.vision.document.utils.error_handling.handlers import (
    handle_document_errors,
    measure_processing_time
)

logger = logging.getLogger(__name__)

class WebDocumentAdapter(DocumentFormatAdapter):
    """
    Adapter for processing web content like HTML files, MHT archives, etc.
    Extracts text, hyperlinks, images, and tables from web documents.
    """
    
    # Class-level constants
    VERSION = "1.0.0"
    CAPABILITIES = {
        "text_extraction": 0.9,
        "structure_preservation": 0.7,
        "metadata_extraction": 0.6,
        "image_extraction": 0.5,
        "link_extraction": 0.9,
        "table_extraction": 0.7
    }
    SUPPORTED_FORMATS = ['.html', '.htm', '.mht', '.mhtml', '.xhtml']
    PRIORITY = 60
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Web document adapter.
        
        Args:
            config: Configuration for the adapter
        """
        super().__init__(config)
        self.bs4_module = None
        self.html_document = None
    
    def _initialize_adapter_components(self) -> None:
        """Initialize adapter-specific components."""
        try:
            # Use dynamic imports to avoid hard dependencies
            import bs4
            
            self.bs4_module = bs4
            logger.info("Successfully initialized web content processing components")
            
            # Optional imports for enhanced capabilities
            try:
                import html2text
                self.html2text = html2text
                self.has_html2text = True
            except ImportError:
                self.has_html2text = False
                logger.info("html2text not available - using fallback text extraction")
                
        except ImportError as e:
            logger.warning(f"Could not import web processing libraries: {str(e)}. Using fallback mode.")
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self) -> None:
        """Initialize fallback components when main libraries are unavailable."""
        self.can_process_html = True  # Basic HTML processing is still possible
        logger.info("Initialized fallback mode - HTML processing will be limited")
    
    @handle_document_errors
    @measure_processing_time
    def _process_file(self, file_path: str, options: Dict[str, Any], 
                    metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process a web document file and extract information.
        
        Args:
            file_path: Path to the web document file
            options: Processing options
            metadata_context: Context for tracking metadata during processing
            
        Returns:
            Dictionary with extracted information
        """
        if metadata_context:
            metadata_context.start_timing("web_processing")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # Process based on file extension
            if file_extension in ['.html', '.htm', '.xhtml']:
                result = self._process_html_file(file_path, options, metadata_context)
            elif file_extension in ['.mht', '.mhtml']:
                result = self._process_mht_file(file_path, options, metadata_context)
            else:
                raise DocumentValidationError(
                    f"Unsupported file format: {file_extension}",
                    document_path=file_path
                )
            
            # Add metadata
            if "metadata" not in result:
                result["metadata"] = self.extract_metadata(file_path, metadata_context)
            
            if metadata_context:
                metadata_context.end_timing("web_processing")
                
            return result
            
        except Exception as e:
            if metadata_context:
                metadata_context.record_error(
                    component=self._processor_name,
                    message=f"Error processing web document: {str(e)}",
                    error_type=type(e).__name__
                )
                metadata_context.end_timing("web_processing")
            
            logger.error(f"Error processing web document: {str(e)}")
            logger.debug(traceback.format_exc())
            
            raise DocumentProcessingError(f"Error processing web document: {str(e)}")
    
    def _process_html_file(self, file_path: str, options: Dict[str, Any],
                          metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process an HTML file using BeautifulSoup if available.
        
        Args:
            file_path: Path to the HTML file
            options: Processing options
            metadata_context: Context for tracking metadata
            
        Returns:
            Dictionary with extracted content and structure
        """
        try:
            # Read the HTML content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            # Parse with BeautifulSoup if available
            if self.bs4_module:
                return self._process_with_beautifulsoup(html_content, file_path, options)
            else:
                return self._process_html_fallback(html_content, file_path, options)
                
        except Exception as e:
            logger.error(f"Error processing HTML file {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process HTML file: {str(e)}")
    
    def _process_with_beautifulsoup(self, html_content: str, file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process HTML content using BeautifulSoup.
        
        Args:
            html_content: HTML content to process
            file_path: Original file path
            options: Processing options
            
        Returns:
            Dictionary with extracted content
        """
        # Parse HTML with BeautifulSoup
        soup = self.bs4_module.BeautifulSoup(html_content, 'html.parser')
        self.html_document = soup
        
        # Extract metadata from HTML
        metadata = self._extract_html_metadata(soup)
        
        # Extract text
        if self.has_html2text:
            # Use html2text for better formatting
            h = self.html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.escape_all = False
            main_text = h.handle(html_content)
        else:
            # Basic text extraction
            main_text = soup.get_text(separator='\n', strip=True)
        
        # Extract links
        links = self._extract_links(soup)
        
        # Extract images
        images = self._extract_images(soup)
        
        # Extract tables
        tables = self._extract_tables(soup)
        
        # Extract structured content (headings, paragraphs)
        structure = self._extract_structure(soup)
        
        # Prepare result
        result = {
            "document_type": "html",
            "file_path": file_path,
            "text": main_text,
            "metadata": metadata,
            "links": links,
            "images": images,
            "tables": tables,
            "structure": structure,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _process_html_fallback(self, html_content: str, file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process HTML content using basic string operations (fallback).
        
        Args:
            html_content: HTML content to process
            file_path: Original file path
            options: Processing options
            
        Returns:
            Dictionary with basic extracted content
        """
        # Simple metadata extraction
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Very basic text extraction
        text = re.sub(r'<[^>]+>', ' ', html_content)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)              # Normalize whitespace
        text = text.strip()
        
        # Extract links with regex
        links = []
        link_pattern = re.compile(r'<a\s+(?:[^>]*?\s+)?href=["\']([^"\']+)["\']', re.IGNORECASE)
        for match in link_pattern.finditer(html_content):
            links.append({"url": match.group(1)})
        
        # Extract image sources with regex
        images = []
        img_pattern = re.compile(r'<img\s+(?:[^>]*?\s+)?src=["\']([^"\']+)["\']', re.IGNORECASE)
        for match in img_pattern.finditer(html_content):
            images.append({"src": match.group(1)})
        
        return {
            "document_type": "html",
            "file_path": file_path,
            "text": text,
            "metadata": {
                "title": title,
                "filename": os.path.basename(file_path)
            },
            "links": links,
            "images": images,
            "tables": [],  # Cannot extract tables without BeautifulSoup
            "structure": {
                "headings": [],
                "paragraphs": []
            },
            "processing_timestamp": datetime.now().isoformat()
        }
    
    def _process_mht_file(self, file_path: str, options: Dict[str, Any],
                        metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process an MHT file.
        
        Args:
            file_path: Path to the MHT file
            options: Processing options
            metadata_context: Context for tracking metadata
            
        Returns:
            Dictionary with extracted content
        """
        try:
            # MHT files are MIME archives. We need to extract the HTML part
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Find the HTML content part
            html_part_match = re.search(r'Content-Type: text/html(.*?)(?:\r?\n\r?\n)(.*?)(?:\r?\n--)', 
                                        content, re.DOTALL | re.IGNORECASE)
            
            if html_part_match:
                # Extract HTML content
                html_content = html_part_match.group(2)
                # Clean up any encoding
                if 'Content-Transfer-Encoding: quoted-printable' in html_part_match.group(1):
                    html_content = self._decode_quoted_printable(html_content)
                
                # Process the HTML part
                if self.bs4_module:
                    result = self._process_with_beautifulsoup(html_content, file_path, options)
                else:
                    result = self._process_html_fallback(html_content, file_path, options)
                
                # Update document type
                result["document_type"] = "mht"
                return result
            else:
                # Fallback to treating it as text
                return {
                    "document_type": "mht",
                    "file_path": file_path,
                    "text": content,
                    "metadata": {
                        "filename": os.path.basename(file_path)
                    },
                    "links": [],
                    "images": [],
                    "tables": [],
                    "structure": {
                        "headings": [],
                        "paragraphs": []
                    },
                    "processing_timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error processing MHT file {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process MHT file: {str(e)}")
    
    def _decode_quoted_printable(self, text: str) -> str:
        """
        Decode quoted-printable text (used in MIME files).
        
        Args:
            text: Quoted-printable encoded text
            
        Returns:
            Decoded text
        """
        try:
            import quopri
            from io import BytesIO
            
            # Decode quoted-printable
            decoded = quopri.decode(BytesIO(text.encode('utf-8')))
            return decoded.decode('utf-8', errors='replace')
        except ImportError:
            # Basic fallback
            result = text
            result = re.sub(r'=\r\n', '', result)  # Remove soft line breaks
            result = re.sub(r'=([0-9A-F]{2})', lambda m: chr(int(m.group(1), 16)), result)
            return result
    
    def _extract_html_metadata(self, soup) -> Dict[str, Any]:
        """
        Extract metadata from HTML document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            
            if name and content:
                metadata[name] = content
        
        # Extract charset/encoding
        charset = None
        meta_charset = soup.find('meta', charset=True)
        if meta_charset:
            charset = meta_charset.get('charset')
        
        if charset:
            metadata['charset'] = charset
        
        return metadata
    
    def _extract_links(self, soup) -> List[Dict[str, Any]]:
        """
        Extract hyperlinks from HTML document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of dictionaries with link information
        """
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            link_info = {
                "url": a_tag['href'],
                "text": a_tag.get_text().strip(),
                "title": a_tag.get('title', ''),
                "rel": a_tag.get('rel', '')
            }
            links.append(link_info)
        
        return links
    
    def _extract_images(self, soup) -> List[Dict[str, Any]]:
        """
        Extract images from HTML document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of dictionaries with image information
        """
        images = []
        
        for img_tag in soup.find_all('img'):
            img_info = {
                "src": img_tag.get('src', ''),
                "alt": img_tag.get('alt', ''),
                "title": img_tag.get('title', ''),
                "width": img_tag.get('width', ''),
                "height": img_tag.get('height', '')
            }
            images.append(img_info)
        
        return images
    
    def _extract_tables(self, soup) -> List[Dict[str, Any]]:
        """
        Extract tables from HTML document.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of dictionaries with table information
        """
        tables = []
        
        for idx, table_tag in enumerate(soup.find_all('table')):
            rows = []
            
            # Process table rows
            for tr_tag in table_tag.find_all('tr'):
                row = []
                
                # Process cells (td or th)
                for cell_tag in tr_tag.find_all(['td', 'th']):
                    row.append(cell_tag.get_text().strip())
                
                if row:
                    rows.append(row)
            
            # Determine if there's a header row
            has_header = False
            header_row = []
            if rows and table_tag.find('th'):
                has_header = True
                
                # Extract header cells
                header_tr = table_tag.find('tr')
                if header_tr:
                    for th in header_tr.find_all('th'):
                        header_row.append(th.get_text().strip())
            
            # Create table info
            table_info = {
                "index": idx,
                "rows": len(rows),
                "columns": len(rows[0]) if rows else 0,
                "has_header": has_header,
                "header": header_row if has_header else [],
                "data": rows
            }
            
            tables.append(table_info)
        
        return tables
    
    def _extract_structure(self, soup) -> Dict[str, Any]:
        """
        Extract document structure (headings, paragraphs) from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with document structure
        """
        structure = {
            "headings": [],
            "paragraphs": []
        }
        
        # Extract headings
        for heading_level in range(1, 7):
            for heading in soup.find_all(f'h{heading_level}'):
                structure["headings"].append({
                    "level": heading_level,
                    "text": heading.get_text().strip()
                })
        
        # Extract paragraphs
        for p in soup.find_all('p'):
            p_text = p.get_text().strip()
            if p_text:
                structure["paragraphs"].append({
                    "text": p_text,
                    "length": len(p_text)
                })
        
        return structure

# Register this adapter
DocumentFormatAdapter.register_adapter("web", WebDocumentAdapter) 