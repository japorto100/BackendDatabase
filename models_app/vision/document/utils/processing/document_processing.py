"""
Document processing utilities for handling document results and section fusion.

This module combines document result creation/formatting with section fusion capabilities
for hybrid documents containing both text and image content.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime
import json
import os
import glob

from models_app.vision.fusion.hybrid_fusion import HybridFusion
from models_app.interfaces.next_layer_interface import NextLayerInterface
from models_app.interfaces.processing_event_type import ProcessingEventType

logger = logging.getLogger(__name__)

class DocumentResult:
    """Handles document result creation and formatting."""
    
    @staticmethod
    def create(
        text: str = "",
        structure: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        knowledge_graph: Optional[Dict] = None,
        processing_info: Optional[Dict] = None,
        document_path: str = "",
        sections: Optional[List] = None,
        tables: Optional[List] = None,
        images: Optional[List] = None,
        visual_understanding: Optional[Dict] = None,
        source_adapter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a standardized document result."""
        structure = structure or {}
        metadata = metadata or {}
        processing_info = processing_info or {}
        
        result = {
            "document_path": document_path,
            "text": text,
            "structure": structure,
            "metadata": metadata,
            "processing_info": {
                **processing_info,
                "timestamp": datetime.now().isoformat(),
                "source_adapter": source_adapter
            }
        }
        
        # Add optional fields if present
        if knowledge_graph:
            result["knowledge_graph"] = knowledge_graph
        if sections:
            result["sections"] = sections
        if tables:
            result["tables"] = tables
        if images:
            result["images"] = images
        if visual_understanding:
            result["visual_understanding"] = visual_understanding
            
        return result

    @staticmethod
    def merge(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merges multiple document results into one."""
        if not results:
            return DocumentResult.create()
        if len(results) == 1:
            return results[0]
        
        # Merge text with paragraph separation
        all_text = "\n\n".join(r.get("text", "") for r in results if r.get("text"))
        
        # Merge structures
        combined_structure = DocumentResult._merge_structures([r.get("structure", {}) for r in results])
        
        # Merge metadata with adapter attribution
        combined_metadata = {}
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            adapter_name = result.get("processing_info", {}).get("source_adapter", f"adapter_{i}")
            combined_metadata[adapter_name] = metadata
        
        # Combine sections
        all_sections = []
        for result in results:
            if sections := result.get("sections"):
                all_sections.extend(sections)
        
        return DocumentResult.create(
            text=all_text,
            structure=combined_structure,
            metadata=combined_metadata,
            sections=all_sections
        )

    @staticmethod
    def _merge_structures(structures: List[Dict]) -> Dict[str, Any]:
        """Helper method to merge document structures."""
        if not structures:
            return {}
        if len(structures) == 1:
            return structures[0]
        
        combined = {}
        for structure in structures:
            for key, value in structure.items():
                if key not in combined:
                    combined[key] = value
                    continue
                
                # Extend lists
                if isinstance(value, list) and isinstance(combined[key], list):
                    combined[key].extend(value)
                    # Remove duplicates for simple types
                    if all(isinstance(item, (str, int, float)) for item in combined[key]):
                        combined[key] = list(set(combined[key]))
                
                # Merge dictionaries recursively
                elif isinstance(value, dict) and isinstance(combined[key], dict):
                    combined[key].update(value)
                
                # For other types, use newer value
                else:
                    combined[key] = value
        
        return combined

    @staticmethod
    def format_as_json(result: Dict[str, Any]) -> str:
        """Formats document result as JSON."""
        return json.dumps(result, indent=2)

    @staticmethod
    def format_as_html(result: Dict[str, Any]) -> str:
        """Formats document result as HTML with styling."""
        if not result:
            return "<html><body><p>No result available.</p></body></html>"
        
        # HTML template with styling
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='UTF-8'>",
            f"<title>Document: {result.get('document_path', 'Untitled')}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #333; }",
            "pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".metadata { background-color: #f8f8f8; padding: 10px; margin: 10px 0; }",
            ".section { margin-bottom: 20px; border-bottom: 1px solid #eee; }",
            ".image-container { margin: 10px 0; }",
            ".image-container img { max-width: 100%; border: 1px solid #ddd; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        # Document title
        doc_path = result.get("document_path", "")
        html.append(f"<h1>Document: {os.path.basename(doc_path) if doc_path else 'Untitled'}</h1>")
        
        # Metadata section
        html.append("<h2>Metadata</h2>")
        html.append("<div class='metadata'>")
        if metadata := result.get("metadata"):
            html.append("<table>")
            html.append("<tr><th>Property</th><th>Value</th></tr>")
            for key, value in metadata.items():
                html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            html.append("</table>")
        else:
            html.append("<p>No metadata available.</p>")
        html.append("</div>")
        
        # Main content
        html.append("<h2>Content</h2>")
        if text := result.get("text"):
            for paragraph in text.split("\n\n"):
                if paragraph.strip():
                    html.append(f"<p>{paragraph}</p>")
        else:
            html.append("<p>No text available.</p>")
        
        # Sections
        if sections := result.get("sections"):
            html.append("<h2>Sections</h2>")
            for i, section in enumerate(sections):
                html.append(f"<div class='section' id='section-{i}'>")
                html.append(f"<h3>Section {i+1}: {section.get('type', 'Unknown')}</h3>")
                if section_text := section.get("text"):
                    html.append(f"<pre>{section_text}</pre>")
                html.append("</div>")
        
        html.extend(["</body>", "</html>"])
        return "\n".join(html)

class SectionFusion:
    """Handles fusion of different document sections."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the section fusion handler."""
        self.config = config or {}
        self.hybrid_fusion = HybridFusion()
    
    def fuse_sections(
        self,
        text_sections: List[Dict[str, Any]],
        image_sections: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fuses text and image sections into a coherent document."""
        if not text_sections and not image_sections:
            return DocumentResult.create()
        
        # Handle single section type cases
        if not text_sections:
            return self._create_image_only_result(image_sections, metadata)
        if not image_sections:
            return self._create_text_only_result(text_sections, metadata)
        
        # Combine text with proper spacing
        combined_text = self._combine_text_sections(text_sections, image_sections)
        
        # Merge structural information
        combined_structure = self._combine_structures(text_sections, image_sections)
        
        # Fuse embeddings if available
        embedding_info = self._fuse_embeddings(text_sections, image_sections)
        
        # Collect tables and images
        tables = self._collect_tables(text_sections, image_sections)
        images = self._collect_images(text_sections, image_sections)
        
        # Merge sections maintaining order
        all_sections = self._merge_sections_with_order(text_sections, image_sections)
        
        # Create final result
        result = DocumentResult.create(
            text=combined_text,
            structure=combined_structure,
            metadata=metadata or {},
            sections=all_sections,
            tables=tables,
            images=images
        )
        
        # Add embedding information if available
        if embedding_info:
            result["embedding_info"] = embedding_info
        
        return result
    
    def _create_text_only_result(
        self,
        text_sections: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creates a result from text sections only."""
        results = []
        for section in text_sections:
            section_result = DocumentResult.create(
                text=section.get("text", ""),
                structure=section.get("structure", {}),
                metadata=section.get("metadata", {}),
                tables=section.get("tables", []),
                images=section.get("images", [])
            )
            results.append(section_result)
        
        merged_result = DocumentResult.merge(results)
        if metadata:
            merged_result["metadata"].update(metadata)
        
        return merged_result
    
    def _create_image_only_result(
        self,
        image_sections: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creates a result from image sections only."""
        results = []
        for section in image_sections:
            section_result = DocumentResult.create(
                text=section.get("text", ""),
                structure=section.get("structure", {}),
                metadata=section.get("metadata", {}),
                tables=section.get("tables", []),
                images=section.get("images", []),
                visual_understanding=section.get("visual_understanding", {})
            )
            results.append(section_result)
        
        merged_result = DocumentResult.merge(results)
        if metadata:
            merged_result["metadata"].update(metadata)
        
        return merged_result
    
    def _combine_text_sections(
        self,
        text_sections: List[Dict[str, Any]],
        image_sections: List[Dict[str, Any]]
    ) -> str:
        """Combines text from all sections with proper spacing."""
        all_text_parts = []
        
        # Extract text from text sections
        for section in text_sections:
            if section_text := section.get("text"):
                all_text_parts.append(section_text)
        
        # Extract text from image sections
        for section in image_sections:
            if section_text := section.get("text"):
                all_text_parts.append(section_text)
        
        return "\n\n".join(all_text_parts)
    
    def _combine_structures(
        self,
        text_sections: List[Dict[str, Any]],
        image_sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combines structure information from all sections."""
        combined_structure = {}
        
        # Process text sections
        for section in text_sections:
            structure = section.get("structure", {})
            for key, value in structure.items():
                if key not in combined_structure:
                    combined_structure[key] = value
                elif isinstance(value, list) and isinstance(combined_structure[key], list):
                    combined_structure[key].extend(value)
                elif isinstance(value, dict) and isinstance(combined_structure[key], dict):
                    combined_structure[key].update(value)
                else:
                    combined_structure[key] = value
        
        # Process image sections
        for section in image_sections:
            structure = section.get("structure", {})
            for key, value in structure.items():
                if key not in combined_structure:
                    combined_structure[key] = value
                elif isinstance(value, list) and isinstance(combined_structure[key], list):
                    combined_structure[key].extend(value)
        
        return combined_structure
    
    def _fuse_embeddings(
        self,
        text_sections: List[Dict[str, Any]],
        image_sections: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Fuses embeddings from text and image sections."""
        text_embeddings = self._extract_embeddings_from_sections(text_sections)
        image_embeddings = self._extract_embeddings_from_sections(image_sections)
        
        if not text_embeddings or not image_embeddings:
            return None
        
        try:
            fused_embeddings, strategy, confidence = self.hybrid_fusion.fuse_with_best_strategy(
                image_embeddings,
                text_embeddings,
                document_metadata={"document_type": "hybrid"}
            )
            
            return {
                "strategy": strategy,
                "confidence": confidence,
                "embeddings": fused_embeddings.tolist() if isinstance(fused_embeddings, np.ndarray) else fused_embeddings
            }
        except Exception as e:
            logger.warning(f"Error in hybrid fusion: {str(e)}")
            return None
    
    def _extract_embeddings_from_sections(
        self,
        sections: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Extracts embeddings from sections."""
        all_embeddings = []
        
        for section in sections:
            if "embeddings" in section:
                all_embeddings.append(section["embeddings"])
            elif "embedding_info" in section and "embeddings" in section["embedding_info"]:
                all_embeddings.append(section["embedding_info"]["embeddings"])
        
        if not all_embeddings:
            return None
        
        # Return single embedding or mean of multiple embeddings
        return all_embeddings[0] if len(all_embeddings) == 1 else np.mean(all_embeddings, axis=0)
    
    def _collect_tables(
        self,
        text_sections: List[Dict[str, Any]],
        image_sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collects tables from all sections."""
        tables = []
        
        for section in text_sections + image_sections:
            if section_tables := section.get("tables", []):
                tables.extend(section_tables)
        
        return tables
    
    def _collect_images(
        self,
        text_sections: List[Dict[str, Any]],
        image_sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collects images from all sections."""
        images = []
        
        for section in text_sections + image_sections:
            if section_images := section.get("images", []):
                images.extend(section_images)
        
        return images
    
    def _merge_sections_with_order(
        self,
        text_sections: List[Dict[str, Any]],
        image_sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merges sections while maintaining logical order."""
        # Simple concatenation for now - could be enhanced with position info
        return text_sections + image_sections 

class DLQManager:
    """Manages Dead Letter Queue for failed document processing."""
    
    def __init__(self, storage_path="dlq"):
        self.storage_path = storage_path
        self.next_layer = NextLayerInterface.get_instance()
        os.makedirs(storage_path, exist_ok=True)
        
    def add_to_dlq(self, document_path, error_info, metadata_context=None):
        """Add failed document to DLQ with error information."""
        document_id = os.path.basename(document_path)
        dlq_entry = {
            "document_path": document_path,
            "error": str(error_info),
            "error_type": type(error_info).__name__,
            "timestamp": datetime.now().isoformat(),
            "retry_count": 0
        }
        
        if metadata_context:
            dlq_entry["metadata_context"] = metadata_context.get_current_status()
        
        # Save DLQ entry
        dlq_path = os.path.join(self.storage_path, f"{document_id}.json")
        with open(dlq_path, 'w') as f:
            json.dump(dlq_entry, f, indent=2)
            
        # Emit DLQ event
        self.next_layer.emit_simple_event(
            ProcessingEventType.ERROR_OCCURRED,
            document_path,
            {
                "error": str(error_info),
                "added_to_dlq": True,
                "dlq_path": dlq_path
            }
        )
        
        return dlq_path
    
    def process_dlq(self, max_retries=3):
        """Process documents in the DLQ up to max_retries."""
        dlq_entries = glob.glob(os.path.join(self.storage_path, "*.json"))
        results = {"processed": 0, "failed": 0, "remaining": 0}
        
        for entry_path in dlq_entries:
            with open(entry_path, 'r') as f:
                entry = json.load(f)
                
            # Skip if max retries reached
            if entry["retry_count"] >= max_retries:
                results["remaining"] += 1
                continue
                
            # Try processing again
            try:
                # Update retry count
                entry["retry_count"] += 1
                with open(entry_path, 'w') as f:
                    json.dump(entry, f, indent=2)
                
                # Process document with recovery options
                result = self._process_dlq_document(entry)
                
                # If successful, remove from DLQ
                if result.get("success", False):
                    os.remove(entry_path)
                    results["processed"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
                
        return results 