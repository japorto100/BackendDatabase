"""
Hypergraph Knowledge Representation

This module provides an implementation of a hypergraph-based knowledge representation
that extends beyond the traditional binary relationships in standard knowledge graphs.

Hypergraphs allow for modeling complex relationships that involve multiple entities
simultaneously, which is particularly useful for representing construction project 
documents, where multiple entities, conditions, and specifications are often 
interconnected in ways that go beyond simple binary relationships.
"""

import logging
import uuid
from typing import Dict, List, Any, Set, Tuple, Optional, Union
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HyperedgeData:
    """
    Data associated with a hyperedge in the hypergraph.
    """
    type: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeData:
    """
    Data associated with a node in the hypergraph.
    """
    type: str
    name: str
    source: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


class Hypergraph:
    """
    A hypergraph representation of knowledge.
    
    Hypergraphs extend traditional graphs by allowing edges to connect any number
    of nodes, not just two. This is useful for representing complex relationships
    in construction documents and technical specifications.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize a new hypergraph.
        
        Args:
            name: Name of the hypergraph
        """
        self.name = name
        self.nodes: Dict[str, NodeData] = {}
        self.hyperedges: Dict[str, Tuple[Set[str], HyperedgeData]] = {}
        logger.info(f"Initialized new hypergraph: {name}")
    
    def add_node(self, node_id: str, node_type: str, name: str, 
                source: Optional[str] = None, attributes: Dict[str, Any] = None) -> str:
        """
        Add a node to the hypergraph.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node (e.g., 'Person', 'Document', 'Requirement')
            name: Name or label of the node
            source: Source of the node information (e.g., document path)
            attributes: Additional attributes for the node
            
        Returns:
            The node_id
        """
        if attributes is None:
            attributes = {}
            
        self.nodes[node_id] = NodeData(
            type=node_type,
            name=name,
            source=source,
            attributes=attributes
        )
        return node_id
    
    def add_hyperedge(self, nodes: List[str], edge_type: str, 
                     weight: float = 1.0, attributes: Dict[str, Any] = None) -> str:
        """
        Add a hyperedge connecting multiple nodes.
        
        Args:
            nodes: List of node IDs to connect
            edge_type: Type of the relationship
            weight: Weight of the relationship
            attributes: Additional attributes for the relationship
            
        Returns:
            The generated hyperedge ID
        """
        if attributes is None:
            attributes = {}
            
        # Validate that all nodes exist
        for node_id in nodes:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} does not exist in the hypergraph")
        
        # Generate a unique ID for the hyperedge
        edge_id = str(uuid.uuid4())
        
        # Add the hyperedge
        self.hyperedges[edge_id] = (
            set(nodes),
            HyperedgeData(
                type=edge_type,
                weight=weight,
                attributes=attributes
            )
        )
        
        logger.debug(f"Added hyperedge {edge_id} of type {edge_type} connecting {len(nodes)} nodes")
        return edge_id
    
    def get_nodes_in_hyperedge(self, edge_id: str) -> List[str]:
        """Get all node IDs in a hyperedge."""
        if edge_id not in self.hyperedges:
            raise ValueError(f"Hyperedge {edge_id} does not exist")
        
        return list(self.hyperedges[edge_id][0])
    
    def get_hyperedges_containing_node(self, node_id: str) -> List[str]:
        """Get all hyperedge IDs that contain a specific node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        return [
            edge_id for edge_id, (nodes, _) in self.hyperedges.items()
            if node_id in nodes
        ]
    
    def query_by_node_type(self, node_type: str) -> List[str]:
        """Get all nodes of a specific type."""
        return [
            node_id for node_id, data in self.nodes.items()
            if data.type == node_type
        ]
    
    def query_by_edge_type(self, edge_type: str) -> List[str]:
        """Get all hyperedges of a specific type."""
        return [
            edge_id for edge_id, (_, data) in self.hyperedges.items()
            if data.type == edge_type
        ]
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the hypergraph to a dictionary format."""
        return {
            "name": self.name,
            "nodes": {
                node_id: {
                    "type": data.type,
                    "name": data.name,
                    "source": data.source,
                    "attributes": data.attributes
                } for node_id, data in self.nodes.items()
            },
            "hyperedges": {
                edge_id: {
                    "nodes": list(nodes),
                    "type": data.type,
                    "weight": data.weight,
                    "attributes": data.attributes
                } for edge_id, (nodes, data) in self.hyperedges.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypergraph':
        """Create a hypergraph from a dictionary representation."""
        hypergraph = cls(name=data.get("name", "imported"))
        
        # Add nodes
        for node_id, node_data in data.get("nodes", {}).items():
            hypergraph.add_node(
                node_id=node_id,
                node_type=node_data["type"],
                name=node_data["name"],
                source=node_data.get("source"),
                attributes=node_data.get("attributes", {})
            )
        
        # Add hyperedges
        for edge_id, edge_data in data.get("hyperedges", {}).items():
            hypergraph.hyperedges[edge_id] = (
                set(edge_data["nodes"]),
                HyperedgeData(
                    type=edge_data["type"],
                    weight=edge_data.get("weight", 1.0),
                    attributes=edge_data.get("attributes", {})
                )
            )
        
        return hypergraph


class ConstructionDocumentHypergraph(Hypergraph):
    """
    Specialized hypergraph for representing construction documents.
    
    This class extends the base Hypergraph with construction-specific
    functionality and relationship types.
    """
    
    def __init__(self, project_name: str):
        """
        Initialize a new construction document hypergraph.
        
        Args:
            project_name: Name of the construction project
        """
        super().__init__(name=f"construction_{project_name}")
        
        # Standard node types for construction
        self.NODE_TYPES = {
            "REQUIREMENT": "Requirement",
            "MATERIAL": "Material",
            "PERSON": "Person",
            "LOCATION": "Location",
            "DOCUMENT": "Document",
            "REGULATION": "Regulation",
            "EQUIPMENT": "Equipment",
            "SAFETY_MEASURE": "SafetyMeasure",
            "TIMELINE": "Timeline",
            "COST": "Cost"
        }
        
        # Standard hyperedge types for construction
        self.EDGE_TYPES = {
            "SPECIFIES": "Specifies",
            "REQUIRES": "Requires",
            "REFERENCES": "References",
            "LOCATED_AT": "LocatedAt",
            "RESPONSIBLE_FOR": "ResponsibleFor",
            "COMPLIES_WITH": "CompliesWith",
            "SCHEDULED_FOR": "ScheduledFor",
            "COSTS": "Costs",
            "SAFETY_REQUIREMENT": "SafetyRequirement"
        }
    
    def add_document(self, doc_id: str, title: str, path: str, 
                    doc_type: str = "Specification", attributes: Dict[str, Any] = None) -> str:
        """
        Add a document node to the hypergraph.
        
        Args:
            doc_id: Document identifier
            title: Document title
            path: Path to the document
            doc_type: Type of document
            attributes: Additional document attributes
            
        Returns:
            The node ID
        """
        attributes = attributes or {}
        attributes["doc_type"] = doc_type
        
        return self.add_node(
            node_id=doc_id,
            node_type=self.NODE_TYPES["DOCUMENT"],
            name=title,
            source=path,
            attributes=attributes
        )
    
    def add_requirement(self, req_id: str, description: str, 
                       source_doc: str = None, attributes: Dict[str, Any] = None) -> str:
        """
        Add a requirement node to the hypergraph.
        
        Args:
            req_id: Requirement identifier
            description: Description of the requirement
            source_doc: Source document node ID
            attributes: Additional requirement attributes
            
        Returns:
            The node ID
        """
        attributes = attributes or {}
        
        node_id = self.add_node(
            node_id=req_id,
            node_type=self.NODE_TYPES["REQUIREMENT"],
            name=description,
            source=source_doc,
            attributes=attributes
        )
        
        # If source document is provided, create a references relationship
        if source_doc and source_doc in self.nodes:
            self.add_hyperedge(
                nodes=[source_doc, node_id],
                edge_type=self.EDGE_TYPES["REFERENCES"]
            )
        
        return node_id
    
    def add_complex_requirement(self, req_id: str, description: str, 
                               materials: List[str], locations: List[str],
                               regulations: List[str] = None, 
                               responsible_persons: List[str] = None,
                               source_doc: str = None) -> Tuple[str, str]:
        """
        Add a complex requirement involving multiple entities.
        
        Args:
            req_id: Requirement identifier
            description: Description of the requirement
            materials: List of material node IDs
            locations: List of location node IDs
            regulations: List of regulation node IDs
            responsible_persons: List of responsible person node IDs
            source_doc: Source document node ID
            
        Returns:
            Tuple of (requirement node ID, hyperedge ID)
        """
        # Add the requirement node
        req_node_id = self.add_requirement(req_id, description, source_doc)
        
        # Collect all nodes for the complex relationship
        related_nodes = [req_node_id] + materials + locations
        
        if regulations:
            related_nodes.extend(regulations)
        
        if responsible_persons:
            related_nodes.extend(responsible_persons)
        
        # Create a hyperedge connecting all these entities
        hyperedge_id = self.add_hyperedge(
            nodes=related_nodes,
            edge_type="ComplexRequirement",
            attributes={
                "description": description,
                "source_document": source_doc
            }
        )
        
        return req_node_id, hyperedge_id


# Example usage demonstration
def create_sample_construction_hypergraph() -> ConstructionDocumentHypergraph:
    """
    Create a sample construction hypergraph for demonstration.
    
    Returns:
        ConstructionDocumentHypergraph instance
    """
    # Initialize hypergraph for a construction project
    hypergraph = ConstructionDocumentHypergraph(project_name="Office Building")
    
    # Add document nodes
    doc1 = hypergraph.add_document(
        doc_id="DOC001",
        title="Structural Specifications",
        path="/documents/structural_specs.pdf",
        doc_type="Technical Specification"
    )
    
    doc2 = hypergraph.add_document(
        doc_id="DOC002",
        title="Safety Regulations",
        path="/documents/safety_regs.pdf",
        doc_type="Regulatory Document"
    )
    
    # Add material nodes
    concrete = hypergraph.add_node(
        node_id="MAT001",
        node_type=hypergraph.NODE_TYPES["MATERIAL"],
        name="High-Strength Concrete",
        attributes={"grade": "C30/37", "supplier": "ConcreteSupplier Inc."}
    )
    
    steel = hypergraph.add_node(
        node_id="MAT002",
        node_type=hypergraph.NODE_TYPES["MATERIAL"],
        name="Reinforcement Steel",
        attributes={"type": "Rebar", "grade": "B500B"}
    )
    
    # Add location nodes
    foundation = hypergraph.add_node(
        node_id="LOC001",
        node_type=hypergraph.NODE_TYPES["LOCATION"],
        name="Building Foundation",
        attributes={"level": "-1", "area": "500m²"}
    )
    
    # Add person nodes
    engineer = hypergraph.add_node(
        node_id="PER001",
        node_type=hypergraph.NODE_TYPES["PERSON"],
        name="John Smith",
        attributes={"role": "Structural Engineer", "license": "SE12345"}
    )
    
    # Add regulation nodes
    reg = hypergraph.add_node(
        node_id="REG001",
        node_type=hypergraph.NODE_TYPES["REGULATION"],
        name="Building Code Section 3.4.2",
        attributes={"code": "IBC 2018", "section": "3.4.2"}
    )
    
    # Add a complex requirement involving multiple entities
    req_id, edge_id = hypergraph.add_complex_requirement(
        req_id="REQ001",
        description="Foundation must use high-strength concrete with steel reinforcement, inspected by the structural engineer and compliant with building code 3.4.2",
        materials=[concrete, steel],
        locations=[foundation],
        regulations=[reg],
        responsible_persons=[engineer],
        source_doc=doc1
    )
    
    # Add a timeline node
    timeline = hypergraph.add_node(
        node_id="TIME001",
        node_type=hypergraph.NODE_TYPES["TIMELINE"],
        name="Foundation Construction Phase",
        attributes={"start_date": "2023-06-01", "end_date": "2023-07-15"}
    )
    
    # Add a scheduling relationship
    hypergraph.add_hyperedge(
        nodes=[req_id, timeline],
        edge_type=hypergraph.EDGE_TYPES["SCHEDULED_FOR"]
    )
    
    # Add cost information
    cost = hypergraph.add_node(
        node_id="COST001",
        node_type=hypergraph.NODE_TYPES["COST"],
        name="Foundation Materials Cost",
        attributes={"amount": 45000, "currency": "EUR"}
    )
    
    # Connect cost to materials
    hypergraph.add_hyperedge(
        nodes=[cost, concrete, steel],
        edge_type=hypergraph.EDGE_TYPES["COSTS"]
    )
    
    return hypergraph


def print_hypergraph_info(hypergraph: Hypergraph):
    """Print information about a hypergraph."""
    print(f"\nHypergraph: {hypergraph.name}")
    print(f"Nodes: {len(hypergraph.nodes)}")
    print(f"Hyperedges: {len(hypergraph.hyperedges)}")
    
    # Print node types and counts
    node_type_counts = {}
    for node_id, data in hypergraph.nodes.items():
        node_type_counts[data.type] = node_type_counts.get(data.type, 0) + 1
    
    print("\nNode types:")
    for node_type, count in node_type_counts.items():
        print(f"  {node_type}: {count}")
    
    # Print hyperedge types and counts
    edge_type_counts = {}
    for edge_id, (nodes, data) in hypergraph.hyperedges.items():
        edge_type_counts[data.type] = edge_type_counts.get(data.type, 0) + 1
    
    print("\nHyperedge types:")
    for edge_type, count in edge_type_counts.items():
        print(f"  {edge_type}: {count}")
    
    # Print a sample complex relationship
    if hypergraph.hyperedges:
        # Find a hyperedge with more than 2 nodes
        complex_edges = [(edge_id, nodes) for edge_id, (nodes, _) in hypergraph.hyperedges.items() if len(nodes) > 2]
        
        if complex_edges:
            edge_id, nodes = complex_edges[0]
            edge_data = hypergraph.hyperedges[edge_id][1]
            
            print("\nSample complex relationship:")
            print(f"  Type: {edge_data.type}")
            print(f"  Nodes ({len(nodes)}):")
            
            for node_id in nodes:
                node_data = hypergraph.nodes[node_id]
                print(f"    - {node_data.name} ({node_data.type})")


if __name__ == "__main__":
    # Create a sample hypergraph for a construction project
    construction_hg = create_sample_construction_hypergraph()
    
    # Print information about the hypergraph
    print_hypergraph_info(construction_hg)
    
    # Export the hypergraph to a dictionary
    export_data = construction_hg.export_to_dict()
    print(f"\nExported hypergraph to dictionary with {len(export_data['nodes'])} nodes and {len(export_data['hyperedges'])} hyperedges")
    
    # Create a new hypergraph from the exported data
    imported_hg = Hypergraph.from_dict(export_data)
    print(f"\nImported hypergraph '{imported_hg.name}' with {len(imported_hg.nodes)} nodes and {len(imported_hg.hyperedges)} hyperedges") 