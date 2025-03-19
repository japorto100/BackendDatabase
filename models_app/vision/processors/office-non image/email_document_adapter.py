"""
EmailDocumentAdapter: Processes email files (.eml, .msg).
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Any
import traceback
from datetime import datetime
import hashlib
import re
import email
from email.header import decode_header

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

class EmailDocumentAdapter(DocumentFormatAdapter):
    """
    Adapter for processing email files (.eml, .msg).
    Extracts message content, headers, attachments, and metadata.
    """
    
    # Class-level constants
    VERSION = "1.0.0"
    CAPABILITIES = {
        "text_extraction": 0.9,
        "structure_preservation": 0.8,
        "metadata_extraction": 0.9,
        "attachment_handling": 0.7,
        "header_extraction": 0.9
    }
    SUPPORTED_FORMATS = ['.eml', '.msg']
    PRIORITY = 65
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Email document adapter.
        
        Args:
            config: Configuration for the adapter
        """
        super().__init__(config)
        self.extract_attachments = True
        self.python_msg = None
        self.message = None
    
    def _initialize_adapter_components(self) -> None:
        """Initialize adapter-specific components."""
        try:
            # For MSG files
            try:
                import extract_msg
                self.python_msg = extract_msg
                logger.info("Successfully initialized MSG file processing components")
                self.can_process_msg = True
            except ImportError:
                self.can_process_msg = False
                logger.warning("extract_msg not available - .msg file handling will be limited")
            
            # Configure attachment handling
            self.extract_attachments = self.config.get("extract_attachments", True)
            self.attachment_temp_dir = self.config.get("attachment_temp_dir", tempfile.gettempdir())
            
        except ImportError as e:
            logger.warning(f"Could not import email processing libraries: {str(e)}. Using fallback mode.")
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self) -> None:
        """Initialize fallback components when main libraries are unavailable."""
        # Standard library email module is always available
        self.can_process_eml = True
        logger.info("Initialized fallback mode - email processing will be limited")
    
    @handle_document_errors
    @measure_processing_time
    def _process_file(self, file_path: str, options: Dict[str, Any], 
                    metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process an email file and extract information.
        
        Args:
            file_path: Path to the email file
            options: Processing options
            metadata_context: Context for tracking metadata during processing
            
        Returns:
            Dictionary with extracted information
        """
        if metadata_context:
            metadata_context.start_timing("email_processing")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # Process based on file extension
            if file_extension == '.eml':
                result = self._process_eml_file(file_path, options, metadata_context)
            elif file_extension == '.msg':
                result = self._process_msg_file(file_path, options, metadata_context)
            else:
                raise DocumentValidationError(
                    f"Unsupported file format: {file_extension}",
                    document_path=file_path
                )
            
            # Add metadata if not already present
            if "metadata" not in result:
                result["metadata"] = self.extract_metadata(file_path, metadata_context)
            
            if metadata_context:
                metadata_context.end_timing("email_processing")
                
            return result
            
        except Exception as e:
            if metadata_context:
                metadata_context.record_error(
                    component=self._processor_name,
                    message=f"Error processing email document: {str(e)}",
                    error_type=type(e).__name__
                )
                metadata_context.end_timing("email_processing")
            
            logger.error(f"Error processing email document: {str(e)}")
            logger.debug(traceback.format_exc())
            
            raise DocumentProcessingError(f"Error processing email document: {str(e)}")
    
    def _process_eml_file(self, file_path: str, options: Dict[str, Any],
                         metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process an .eml file using Python's email module.
        
        Args:
            file_path: Path to the .eml file
            options: Processing options
            metadata_context: Context for tracking metadata
            
        Returns:
            Dictionary with extracted email data
        """
        try:
            # Read the email file
            with open(file_path, 'rb') as f:
                raw_email = f.read()
            
            # Parse the email
            email_message = email.message_from_bytes(raw_email)
            self.message = email_message
            
            # Extract headers
            headers = self._extract_email_headers(email_message)
            
            # Extract body parts
            body_parts = self._extract_email_body_parts(email_message)
            
            # Extract attachments
            attachments = []
            if self.extract_attachments:
                attachments = self._extract_email_attachments(email_message, file_path)
            
            # Combine text parts for full text
            all_text = []
            for part in body_parts:
                if part.get("content_type", "").startswith("text/"):
                    all_text.append(part["content"])
            
            # Create final result
            result = {
                "document_type": "email",
                "file_path": file_path,
                "headers": headers,
                "text": "\n\n".join(all_text),
                "body_parts": body_parts,
                "attachments": attachments,
                "metadata": {
                    "subject": headers.get("subject", ""),
                    "from": headers.get("from", ""),
                    "to": headers.get("to", ""),
                    "date": headers.get("date", ""),
                    "filename": os.path.basename(file_path)
                },
                "processing_timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing .eml file {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process .eml file: {str(e)}")
    
    def _process_msg_file(self, file_path: str, options: Dict[str, Any],
                         metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process a .msg file using extract_msg if available.
        
        Args:
            file_path: Path to the .msg file
            options: Processing options
            metadata_context: Context for tracking metadata
            
        Returns:
            Dictionary with extracted email data
        """
        if self.python_msg and self.can_process_msg:
            try:
                # Load the MSG file
                msg = self.python_msg.Message(file_path)
                
                # Extract headers
                headers = {
                    "subject": msg.subject,
                    "from": msg.sender,
                    "to": msg.to,
                    "cc": msg.cc,
                    "date": msg.date,
                    "message_id": msg.message_id
                }
                
                # Get body content
                body_text = msg.body
                html_body = msg.htmlBody
                
                body_parts = []
                
                # Add plain text part
                if body_text:
                    body_parts.append({
                        "content_type": "text/plain",
                        "content": body_text,
                        "size": len(body_text)
                    })
                
                # Add HTML part
                if html_body:
                    body_parts.append({
                        "content_type": "text/html",
                        "content": html_body,
                        "size": len(html_body)
                    })
                
                # Extract attachments
                attachments = []
                if self.extract_attachments:
                    for attachment in msg.attachments:
                        attachment_info = {
                            "filename": attachment.longFilename or attachment.shortFilename,
                            "content_type": attachment.mimetype,
                            "size": len(attachment.data) if hasattr(attachment, "data") else 0
                        }
                        attachments.append(attachment_info)
                
                # Create result
                result = {
                    "document_type": "email",
                    "file_path": file_path,
                    "headers": headers,
                    "text": body_text or "",
                    "body_parts": body_parts,
                    "attachments": attachments,
                    "metadata": {
                        "subject": headers.get("subject", ""),
                        "from": headers.get("from", ""),
                        "to": headers.get("to", ""),
                        "date": headers.get("date", ""),
                        "filename": os.path.basename(file_path)
                    },
                    "processing_timestamp": datetime.now().isoformat()
                }
                
                # Close the message to release resources
                msg.close()
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing .msg file {file_path}: {str(e)}")
                logger.info("Falling back to basic MSG file processing")
        
        # Fallback for MSG files without extract_msg
        try:
            # Basic extraction of readable strings
            with open(file_path, 'rb') as f:
                content = f.read()
                
            # Extract readable strings
            text = ""
            current_string = ""
            for byte in content:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                elif byte in [9, 10, 13]:  # Tab, LF, CR
                    current_string += chr(byte)
                elif current_string:
                    if len(current_string) > 4:  # Consider strings longer than 4 chars
                        text += current_string + "\n"
                    current_string = ""
                    
            if current_string and len(current_string) > 4:
                text += current_string
            
            # Try to extract subject, from, to from strings
            subject = ""
            sender = ""
            recipient = ""
            
            subject_match = re.search(r"Subject:(.*?)(?:\r?\n\S|$)", text, re.IGNORECASE)
            if subject_match:
                subject = subject_match.group(1).strip()
                
            from_match = re.search(r"From:(.*?)(?:\r?\n\S|$)", text, re.IGNORECASE)
            if from_match:
                sender = from_match.group(1).strip()
                
            to_match = re.search(r"To:(.*?)(?:\r?\n\S|$)", text, re.IGNORECASE)
            if to_match:
                recipient = to_match.group(1).strip()
            
            # Create simplified result
            return {
                "document_type": "email",
                "file_path": file_path,
                "headers": {
                    "subject": subject,
                    "from": sender,
                    "to": recipient
                },
                "text": text,
                "body_parts": [
                    {
                        "content_type": "text/plain",
                        "content": text,
                        "size": len(text)
                    }
                ],
                "attachments": [],  # Cannot extract attachments in fallback mode
                "metadata": {
                    "subject": subject,
                    "from": sender,
                    "to": recipient,
                    "filename": os.path.basename(file_path)
                },
                "processing_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in fallback .msg processing for {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process .msg file: {str(e)}")
    
    def _extract_email_headers(self, email_message) -> Dict[str, str]:
        """
        Extract and decode email headers.
        
        Args:
            email_message: Email message object
            
        Returns:
            Dictionary with header names and values
        """
        headers = {}
        
        # Extract common headers
        for header in ['subject', 'from', 'to', 'cc', 'bcc', 'date', 'message-id', 'in-reply-to', 'references']:
            value = email_message.get(header, '')
            
            # Decode encoded headers
            if value:
                decoded_parts = []
                for part, encoding in decode_header(value):
                    if isinstance(part, bytes):
                        if encoding:
                            try:
                                decoded_parts.append(part.decode(encoding))
                            except UnicodeDecodeError:
                                decoded_parts.append(part.decode('utf-8', errors='replace'))
                        else:
                            decoded_parts.append(part.decode('utf-8', errors='replace'))
                    else:
                        decoded_parts.append(part)
                
                headers[header.lower()] = ''.join(decoded_parts)
        
        return headers
    
    def _extract_email_body_parts(self, email_message) -> List[Dict[str, Any]]:
        """
        Extract email body parts.
        
        Args:
            email_message: Email message object
            
        Returns:
            List of dictionaries with body part information
        """
        body_parts = []
        
        # Process all parts
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = part.get("Content-Disposition", "")
                
                # Only include body parts, not attachments
                if "attachment" not in content_disposition:
                    content = self._get_part_content(part)
                    if content:
                        body_parts.append({
                            "content_type": content_type,
                            "content": content,
                            "size": len(content)
                        })
        else:
            # Single part email
            content_type = email_message.get_content_type()
            content = self._get_part_content(email_message)
            if content:
                body_parts.append({
                    "content_type": content_type,
                    "content": content,
                    "size": len(content)
                })
        
        return body_parts
    
    def _get_part_content(self, part) -> str:
        """
        Get content from an email part, handling encoding.
        
        Args:
            part: Email message part
            
        Returns:
            Decoded content as string
        """
        content = part.get_payload(decode=True)
        if not content:
            return ""
            
        # Get charset
        charset = part.get_content_charset()
        if charset:
            try:
                return content.decode(charset)
            except UnicodeDecodeError:
                pass
        
        # Try common encodings
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # Last resort
        return content.decode('utf-8', errors='replace')
    
    def _extract_email_attachments(self, email_message, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract attachments from email.
        
        Args:
            email_message: Email message object
            file_path: Path to the email file
            
        Returns:
            List of dictionaries with attachment information
        """
        attachments = []
        
        # Process all parts to find attachments
        for part in email_message.walk():
            content_disposition = part.get("Content-Disposition", "")
            
            # Check if it's an attachment
            if "attachment" in content_disposition or "inline" in content_disposition:
                filename = part.get_filename()
                if not filename:
                    continue
                
                # Decode filename if necessary
                if isinstance(filename, bytes):
                    filename = filename.decode('utf-8', errors='replace')
                
                # Get content type
                content_type = part.get_content_type() or "application/octet-stream"
                
                # Get content
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                
                # Create attachment info
                attachment_info = {
                    "filename": filename,
                    "content_type": content_type,
                    "size": len(payload)
                }
                
                # Save attachment to temp file if configured
                if self.extract_attachments:
                    try:
                        # Create a safe filename
                        safe_filename = re.sub(r'[^\w\-\.]', '_', filename)
                        temp_path = os.path.join(self.attachment_temp_dir, 
                                               f"{os.path.basename(file_path)}_{safe_filename}")
                        
                        with open(temp_path, 'wb') as f:
                            f.write(payload)
                        
                        attachment_info["temp_path"] = temp_path
                    except Exception as e:
                        logger.warning(f"Failed to save attachment {filename}: {str(e)}")
                
                attachments.append(attachment_info)
        
        return attachments

# Register this adapter
DocumentBaseAdapter.register_adapter("email", EmailDocumentAdapter) 