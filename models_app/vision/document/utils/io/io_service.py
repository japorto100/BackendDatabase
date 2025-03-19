"""
Centralized I/O service for file operations and format conversions.

This module provides unified methods for file I/O operations and format conversions
used across different components of the document processing system.
"""

import os
import logging
import tempfile
import mimetypes
import shutil
import subprocess
from typing import Dict, List, Any, Optional, Union, BinaryIO, Tuple
from datetime import datetime
import json
import pickle
import base64

logger = logging.getLogger(__name__)

class IOService:
    """
    Centralized service for file operations and format conversions.
    
    This class provides unified methods for:
    - Basic file operations (read, write, copy, move)
    - File metadata extraction
    - Temporary file management
    - Format conversions (PDF, images, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the IO service."""
        self.config = config or {}
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        
        # Check external tool availability
        self.libreoffice_available = self._check_tool_availability(['libreoffice', '--version'])
        self.pandoc_available = self._check_tool_availability(['pandoc', '--version'])
        self.imagemagick_available = self._check_tool_availability(['convert', '--version'])
        
        logger.info(f"IOService initialized. LibreOffice: {self.libreoffice_available}, "
                   f"Pandoc: {self.pandoc_available}, ImageMagick: {self.imagemagick_available}")
    
    def _check_tool_availability(self, cmd: List[str]) -> bool:
        """Check if an external tool is available."""
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    # File Operations
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract file metadata."""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return {}
        
        try:
            stats = os.stat(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            _, file_extension = os.path.splitext(file_path)
            
            return {
                "file_name": os.path.basename(file_path),
                "file_extension": file_extension.lower(),
                "file_path": file_path,
                "file_size": stats.st_size,
                "file_size_mb": stats.st_size / (1024 * 1024),
                "mime_type": mime_type or "application/octet-stream",
                "creation_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modification_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "access_time": datetime.fromtimestamp(stats.st_atime).isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting metadata for {file_path}: {str(e)}")
            return {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "error": str(e)
            }
    
    def read_text(self, file_path: str, encoding: str = 'utf-8') -> str:
        """Read text file with fallback encoding."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                raise IOError(f"Could not read file with latin-1 encoding: {str(e)}")
        except Exception as e:
            raise IOError(f"Could not read file: {str(e)}")
    
    def read_binary(self, file_path: str) -> bytes:
        """Read binary file."""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Could not read binary file: {str(e)}")
    
    def write_text(self, file_path: str, content: str, encoding: str = 'utf-8') -> str:
        """Write text to file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return file_path
        except Exception as e:
            raise IOError(f"Could not write file: {str(e)}")
    
    def write_binary(self, file_path: str, content: bytes) -> str:
        """Write binary data to file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(content)
            return file_path
        except Exception as e:
            raise IOError(f"Could not write binary file: {str(e)}")
    
    # Temporary File Management
    
    def create_temp_file(
        self,
        prefix: str = 'doc_',
        suffix: str = '',
        content: Optional[Union[str, bytes]] = None
    ) -> str:
        """Create temporary file with optional content."""
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(fd)
            
            if content is not None:
                if isinstance(content, str):
                    self.write_text(temp_path, content)
                else:
                    self.write_binary(temp_path, content)
            
            return temp_path
        except Exception as e:
            raise IOError(f"Could not create temporary file: {str(e)}")
    
    def create_temp_directory(self, prefix: str = 'doc_') -> str:
        """Create temporary directory."""
        try:
            return tempfile.mkdtemp(prefix=prefix)
        except Exception as e:
            raise IOError(f"Could not create temporary directory: {str(e)}")
    
    # File System Operations
    
    def ensure_directory(self, dir_path: str) -> str:
        """Create directory if it doesn't exist."""
        try:
            os.makedirs(dir_path, exist_ok=True)
            return dir_path
        except Exception as e:
            raise IOError(f"Could not create directory: {str(e)}")
    
    def list_files(
        self,
        directory: str,
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> List[str]:
        """List files in directory with optional pattern matching."""
        try:
            if recursive:
                matches = []
                for root, _, files in os.walk(directory):
                    for filename in files:
                        if not pattern or self._matches_pattern(filename, pattern):
                            matches.append(os.path.join(root, filename))
                return matches
            else:
                files = os.listdir(directory)
                return [
                    os.path.join(directory, f) for f in files
                    if os.path.isfile(os.path.join(directory, f))
                    and (not pattern or self._matches_pattern(f, pattern))
                ]
        except Exception as e:
            raise IOError(f"Could not list files: {str(e)}")
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(filename.lower(), pattern.lower())
    
    # Format Conversion
    
    def convert_to_pdf(
        self,
        input_file: str,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """Convert file to PDF."""
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return None
        
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + '.pdf'
        
        _, ext = os.path.splitext(input_file)
        ext = ext.lower()
        
        try:
            if ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp']:
                if not self.libreoffice_available:
                    raise IOError("LibreOffice not available for conversion")
                
                cmd = [
                    'libreoffice', '--headless', '--convert-to', 'pdf',
                    '--outdir', os.path.dirname(output_file),
                    input_file
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                
                expected_output = os.path.join(
                    os.path.dirname(output_file),
                    os.path.splitext(os.path.basename(input_file))[0] + '.pdf'
                )
                
                if os.path.exists(expected_output) and expected_output != output_file:
                    os.rename(expected_output, output_file)
                
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']:
                if not self.imagemagick_available:
                    raise IOError("ImageMagick not available for conversion")
                
                cmd = ['convert', input_file, output_file]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                
            elif ext in ['.md', '.html', '.txt', '.rtf', '.epub']:
                if not self.pandoc_available:
                    raise IOError("Pandoc not available for conversion")
                
                cmd = ['pandoc', input_file, '-o', output_file]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                
            elif ext == '.pdf':
                shutil.copy2(input_file, output_file)
            else:
                logger.error(f"Unsupported file format: {ext}")
                return None
            
            return output_file if os.path.exists(output_file) else None
            
        except Exception as e:
            logger.error(f"Error converting to PDF: {str(e)}")
            return None
    
    def convert_to_images(
        self,
        input_file: str,
        output_dir: Optional[str] = None,
        dpi: int = 300,
        format: str = 'png'
    ) -> List[str]:
        """Convert document to images (one per page)."""
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return []
        
        if output_dir is None:
            output_dir = os.path.join(
                self.temp_dir,
                os.path.splitext(os.path.basename(input_file))[0] + '_images'
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to PDF first if needed
        pdf_file = input_file
        temp_pdf = None
        
        if not input_file.lower().endswith('.pdf'):
            pdf_file = self.convert_to_pdf(input_file)
            if pdf_file is None:
                logger.error(f"PDF conversion failed: {input_file}")
                return []
            temp_pdf = pdf_file
        
        try:
            if not self.imagemagick_available:
                raise IOError("ImageMagick not available for conversion")
            
            output_pattern = os.path.join(output_dir, f"page_%04d.{format}")
            cmd = [
                'convert', '-density', str(dpi),
                pdf_file, '-quality', '90',
                output_pattern
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            
            # List and sort created images
            image_files = []
            for file in os.listdir(output_dir):
                if file.startswith('page_') and file.endswith(f'.{format}'):
                    image_files.append(os.path.join(output_dir, file))
            
            return sorted(image_files)
            
        except Exception as e:
            logger.error(f"Error converting to images: {str(e)}")
            return []
            
        finally:
            if temp_pdf and os.path.exists(temp_pdf) and temp_pdf != input_file:
                try:
                    os.remove(temp_pdf)
                except:
                    pass
    
    # Serialization
    
    def save_json(
        self,
        file_path: str,
        data: Any,
        encoding: str = 'utf-8',
        indent: int = 4
    ) -> str:
        """Save data as JSON file."""
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=indent)
            return file_path
        except Exception as e:
            raise IOError(f"Could not save JSON: {str(e)}")
    
    def load_json(self, file_path: str, encoding: str = 'utf-8') -> Any:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Could not load JSON: {str(e)}")
    
    def save_pickle(self, file_path: str, data: Any) -> str:
        """Save data as pickle file."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return file_path
        except Exception as e:
            raise IOError(f"Could not save pickle: {str(e)}")
    
    def load_pickle(self, file_path: str) -> Any:
        """Load data from pickle file."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError(f"Could not load pickle: {str(e)}") 