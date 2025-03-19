"""
Special Vision AI Models

This module provides specialized vision AI model integrations such as 
document-processing adapters with knowledge graph integration capabilities.
"""

from models_app.ai_models.vision.special.document_vision_adapter import DocumentVisionAdapter, ProcessedPage

# Define public API
__all__ = [
    'DocumentVisionAdapter',
    'ProcessedPage'
] 