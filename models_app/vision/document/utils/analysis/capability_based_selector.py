"""
Capability-based adapter selection system for document processing.

This module provides a selection system that chooses document adapters
based on their capabilities and the requirements of specific documents.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from models_app.vision.document.document_base_adapter import DocumentBaseAdapter
from models_app.vision.document.utils.core.processing_metadata_context import ProcessingMetadataContext

logger = logging.getLogger(__name__)

class CapabilityBasedSelector:
    """
    Selects document adapters based on their capabilities and document requirements.
    
    This selector evaluates available adapters against the capabilities required
    for processing a specific document, ensuring the most appropriate adapter
    is selected for each processing task.
    """
    
    def __init__(self, registry):
        """
        Initialize the capability selector.
        
        Args:
            registry: The DocumentAdapterRegistry instance
        """
        self.registry = registry
        
        # Default capability weights (importance of each capability)
        self.capability_weights = {
            "images": 1.0,
            "tables": 1.0,
            "forms": 1.0,
            "text_extraction": 1.0,
            "scanned_documents": 1.0,
            "pdfs": 0.8,
            "office_documents": 0.8,
            "mixed_content": 0.7,
            "complex_layouts": 0.7,
            "photos": 0.6,
            "diagrams": 0.6,
            "charts": 0.6
        }
    
    def derive_required_capabilities(self, document_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Derive required capabilities from document analysis results.
        
        Args:
            document_analysis: Results of document analysis
            
        Returns:
            Dict mapping capability names to minimum confidence values
        """
        requirements = {}
        
        # Document type info
        document_type = document_analysis.get("document_type", {})
        is_scanned = document_type.get("is_scanned", False)
        doc_type = document_type.get("type", "unknown")
        
        # Document content features
        has_images = document_analysis.get("has_images", False)
        has_tables = document_analysis.get("has_tables", False)
        has_forms = document_analysis.get("is_form", False)
        has_complex_layout = document_analysis.get("has_complex_layout", False)
        has_charts = document_analysis.get("has_charts", False)
        has_diagrams = document_analysis.get("has_diagrams", False)
        
        # Set required capabilities based on document features
        if is_scanned:
            requirements["scanned_documents"] = 0.7
        
        if has_images:
            image_ratio = document_analysis.get("image_ratio", 0.0)
            if image_ratio > 0.5:
                requirements["images"] = 0.8
            else:
                requirements["images"] = 0.5
        
        if has_tables:
            requirements["tables"] = 0.7
            
        if has_forms:
            requirements["forms"] = 0.8
            
        if has_complex_layout:
            requirements["complex_layouts"] = 0.7
            
        if has_charts:
            requirements["charts"] = 0.6
            
        if has_diagrams:
            requirements["diagrams"] = 0.6
            
        # Set capabilities based on document type
        if doc_type == "image":
            requirements["images"] = 0.9
            requirements["text_extraction"] = 0.6
            
        elif doc_type == "pdf":
            requirements["pdfs"] = 0.8
            requirements["text_extraction"] = 0.7
            
        elif doc_type in ["document", "spreadsheet", "presentation"]:
            requirements["office_documents"] = 0.8
            
        # Special case for mixed content
        if has_images and has_tables and has_complex_layout:
            requirements["mixed_content"] = 0.8
            
        # Ensure text extraction is included
        if "text_extraction" not in requirements:
            requirements["text_extraction"] = 0.5
            
        return requirements
    
    def select_adapter(
        self,
        document_path: str, 
        document_analysis: Dict[str, Any],
        metadata_context: ProcessingMetadataContext,
        required_capabilities: Optional[Dict[str, float]] = None
    ) -> DocumentBaseAdapter:
        """
        Select the best adapter based on document requirements and adapter capabilities.
        
        Args:
            document_path: Path to the document
            document_analysis: Document analysis results
            metadata_context: Metadata context for recording decisions
            required_capabilities: Optional explicit capability requirements
            
        Returns:
            The best matching adapter
        """
        start_time = time.time()
        
        # Determine required capabilities if not provided
        if required_capabilities is None:
            required_capabilities = self.derive_required_capabilities(document_analysis)
            
        # Record requirements in metadata context
        metadata_context.record_capability_requirements(
            component="CapabilityBasedSelector",
            capabilities=required_capabilities
        )
        
        # Get all available adapters
        adapters = self.registry.get_all_adapters()
        
        # Score each adapter based on capabilities
        adapter_scores = []
        for adapter in adapters:
            try:
                # Get adapter capabilities
                capabilities = adapter.get_capabilities()
                
                # Calculate matching score
                score = self._calculate_capability_score(capabilities, required_capabilities)
                
                # Record capability match
                metadata_context.record_capability_match(
                    adapter_name=adapter.__class__.__name__,
                    capabilities=capabilities,
                    match_score=score,
                    required_capabilities=required_capabilities
                )
                
                adapter_scores.append((adapter, score))
                
            except Exception as e:
                logger.warning(f"Error evaluating adapter {adapter.__class__.__name__}: {str(e)}")
                continue
                
        # Sort by score (descending)
        adapter_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not adapter_scores:
            # Fallback to universal adapter if no matches
            logger.warning("No matching adapters found, using universal adapter")
            universal_adapter = self.registry.get_adapter("universal")
            metadata_context.record_decision(
                component="CapabilityBasedSelector",
                decision=f"Selected fallback adapter: {universal_adapter.__class__.__name__}",
                reason="No adapters matched the required capabilities",
                confidence=0.3
            )
            return universal_adapter
            
        # Select the best matching adapter
        selected_adapter, score = adapter_scores[0]
        
        # Prepare alternative options for decision recording
        alternatives = []
        for adapter, alt_score in adapter_scores[1:3]:  # Top 2 alternatives
            if alt_score > 0:
                alternatives.append({
                    "adapter": adapter.__class__.__name__,
                    "score": alt_score,
                    "score_diff": score - alt_score
                })
        
        # Record selection decision
        metadata_context.record_decision(
            component="CapabilityBasedSelector",
            decision=f"Selected adapter: {selected_adapter.__class__.__name__}",
            reason=f"Best capability match with score {score:.2f}",
            confidence=min(1.0, score),
            alternatives=alternatives
        )
        
        # Record timing
        selection_time = time.time() - start_time
        logger.info(f"Adapter selection completed in {selection_time:.4f}s, "
                   f"selected {selected_adapter.__class__.__name__} with score {score:.2f}")
        
        return selected_adapter
    
    def _calculate_capability_score(
        self, 
        adapter_capabilities: Dict[str, float],
        required_capabilities: Dict[str, float]
    ) -> float:
        """
        Calculate the capability match score for an adapter.
        
        Args:
            adapter_capabilities: Adapter's capabilities with confidence values
            required_capabilities: Required capabilities with minimum confidence
            
        Returns:
            float: Match score (0.0-1.0)
        """
        if not required_capabilities:
            return 0.5  # Neutral score if no requirements
            
        total_score = 0.0
        total_weight = 0.0
        
        # Check each required capability
        for capability, min_confidence in required_capabilities.items():
            # Get capability weight (importance)
            weight = self.capability_weights.get(capability, 1.0)
            
            # Get adapter's confidence for this capability
            adapter_confidence = adapter_capabilities.get(capability, 0.0)
            
            # Calculate score for this capability
            if adapter_confidence >= min_confidence:
                # Full score if confidence meets or exceeds requirement
                capability_score = 1.0
            else:
                # Partial score based on how close we are to requirement
                capability_score = max(0.0, adapter_confidence / min_confidence)
            
            # Add weighted score
            total_score += capability_score * weight
            total_weight += weight
        
        # Calculate final normalized score
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0  # No matching capabilities 