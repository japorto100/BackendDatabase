from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class EntityExtractorInterface(ABC):
    """Interface for entity extraction components."""
    
    @abstractmethod
    def extract_from_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract entities from text content."""
        pass
        
    @abstractmethod
    def extract_from_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract entities from image content."""
        pass
    
    @abstractmethod
    def extract_from_colpali(self, colpali_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from ColPali model output.
        
        This is a specialized method for visual feature extraction from the
        ColPali vision model, which can detect visual entities, layouts,
        and other rich visual features.
        """
        pass

class RelationshipDetectorInterface(ABC):
    """Interface for relationship detection components."""
    
    @abstractmethod
    def detect_relationships(self, entities: List[Dict[str, Any]], 
                            context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect relationships between entities."""
        pass
    
    @abstractmethod
    def detect_visual_relationships(self, visual_entities: List[Dict[str, Any]],
                                   layout_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect relationships between visual entities based on layout."""
        pass
    
    @abstractmethod
    def merge_text_and_visual_relationships(self, 
                                          text_relationships: List[Dict[str, Any]],
                                          visual_relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text-based and visual relationships."""
        pass

class GraphBuilderInterface(ABC):
    """Interface for graph building components."""
    
    @abstractmethod
    def add_entity(self, entity: Dict[str, Any]) -> str:
        """Add an entity to the knowledge graph."""
        pass
    
    @abstractmethod
    def add_relationship(self, relationship: Dict[str, Any]) -> str:
        """Add a relationship to the knowledge graph."""
        pass
    
    @abstractmethod
    def build_subgraph(self, entities: List[Dict[str, Any]], 
                      relationships: List[Dict[str, Any]],
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a subgraph from entities and relationships."""
        pass
    
    @abstractmethod
    def merge_subgraphs(self, subgraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple subgraphs into a single graph."""
        pass

class GraphStorageInterface(ABC):
    """Interface for graph storage components."""
    
    @abstractmethod
    def store_graph(self, graph: Dict[str, Any], graph_id: Optional[str] = None) -> str:
        """Store a knowledge graph."""
        pass
    
    @abstractmethod
    def retrieve_graph(self, graph_id: str) -> Dict[str, Any]:
        """Retrieve a knowledge graph by ID."""
        pass
    
    @abstractmethod
    def query_graph(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the knowledge graph."""
        pass
    
    @abstractmethod
    def update_graph(self, graph_id: str, updates: Dict[str, Any]) -> bool:
        """Update a stored knowledge graph."""
        pass

class KnowledgeGraphManagerInterface(ABC):
    """Master interface for the knowledge graph system."""
    
    @abstractmethod
    def process_document(self, document_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a document and add its content to the knowledge graph."""
        pass
    
    @abstractmethod
    def process_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process an image and add its content to the knowledge graph."""
        pass
    
    @abstractmethod
    def process_colpali_output(self, colpali_output: Dict[str, Any], 
                              document_id: str) -> str:
        """Process ColPali output and enrich the knowledge graph with visual data."""
        pass
    
    @abstractmethod
    def merge_document_graphs(self, document_ids: List[str]) -> str:
        """Merge multiple document graphs into a single knowledge graph."""
        pass
