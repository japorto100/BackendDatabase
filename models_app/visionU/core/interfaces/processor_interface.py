from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class ProcessorInterface(ABC):
    """Base interface for all document processors."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the processor with required resources."""
        pass
    
    @abstractmethod
    def validate_input(self, document_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Validate if the processor can handle the input."""
        pass
    
    @abstractmethod
    def process(self, document_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process the document and return results."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, float]:
        """Return processor capabilities with confidence scores."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the processor."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return the processor version."""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported formats."""
        pass
    
    @property
    @abstractmethod
    def supported_content_types(self) -> List[str]:
        """Return list of supported content types."""
        pass 