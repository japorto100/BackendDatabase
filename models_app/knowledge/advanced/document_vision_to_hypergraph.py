"""
DocumentVisionAdapter to Hypergraph Integration

This module provides integration between the DocumentVisionAdapter and the Hypergraph
knowledge representation, enabling document-extracted information to be directly
mapped to hypergraph structures.

This bridges the gap between document processing and advanced knowledge representation,
allowing complex document relationships to be modeled in a more expressive way than
traditional knowledge graphs.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from models_app.ai_models.vision.document_vision_adapter import DocumentVisionAdapter
from models_app.knowledge.advanced.hypergraph_kg import Hypergraph, ConstructionDocumentHypergraph


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentToHypergraphConverter:
    """
    Converts document information processed by DocumentVisionAdapter to a hypergraph structure.
    
    This class bridges document processing with advanced knowledge representation
    by mapping extracted document information into hypergraph structures.
    """
    
    def __init__(self, project_name: str = None):
        """
        Initialize the converter.
        
        Args:
            project_name: Optional name for the construction project
        """
        self.document_adapter = DocumentVisionAdapter()
        self.project_name = project_name or f"project_{datetime.now().strftime('%Y%m%d')}"
        self.hypergraph = ConstructionDocumentHypergraph(project_name=self.project_name)
        
        # Track document nodes for reference
        self.document_nodes = {}
        
        logger.info(f"Initialized DocumentToHypergraphConverter for project: {self.project_name}")
    
    def process_document(self, document_path: str) -> Tuple[Dict[str, Any], str]:
        """
        Process a document and add its contents to the hypergraph.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            Tuple of (processing_result, document_node_id)
        """
        # Process the document using the adapter
        logger.info(f"Processing document: {document_path}")
        result = self.document_adapter.process_document(document_path)
        
        if not result.get("success", False):
            logger.error(f"Failed to process document: {document_path}")
            return result, None
        
        # Add the document to the hypergraph
        doc_filename = os.path.basename(document_path)
        doc_id = f"DOC_{doc_filename.replace('.', '_')}"
        
        doc_metadata = result.get("metadata", {})
        doc_attributes = {
            "processed_date": datetime.now().isoformat(),
            "page_count": result.get("page_count", 0),
            "image_count": len(result.get("images", [])),
            **doc_metadata
        }
        
        # Add the document node
        doc_node_id = self.hypergraph.add_document(
            doc_id=doc_id,
            title=doc_metadata.get("title", doc_filename),
            path=document_path,
            doc_type=doc_metadata.get("doc_type", "Document"),
            attributes=doc_attributes
        )
        
        # Store the document node for reference
        self.document_nodes[document_path] = doc_node_id
        
        # Extract and add text content as concept nodes
        if "text" in result and result["text"]:
            self._add_text_content(result["text"], doc_node_id)
        
        # Process images if available
        if "images" in result and result["images"]:
            self._add_image_content(result["images"], doc_node_id)
        
        return result, doc_node_id
    
    def process_for_knowledge_graph(self, document_path: str) -> Tuple[Dict[str, Any], str]:
        """
        Process a document specifically for knowledge graph extraction and convert to hypergraph.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            Tuple of (kg_processing_result, document_node_id)
        """
        # Process for knowledge graph
        logger.info(f"Processing document for KG: {document_path}")
        result = self.document_adapter.process_for_knowledge_graph(document_path)
        
        if not result.get("success", False):
            logger.error(f"Failed to process document for KG: {document_path}")
            return result, None
        
        # Check if document node already exists
        doc_node_id = self.document_nodes.get(document_path)
        
        # If not, create a document node
        if not doc_node_id:
            doc_info = result.get("document", {})
            doc_filename = os.path.basename(document_path)
            doc_id = f"DOC_{doc_filename.replace('.', '_')}"
            
            doc_metadata = doc_info.get("metadata", {})
            doc_attributes = {
                "processed_date": datetime.now().isoformat(),
                "kg_processed": True,
                **doc_metadata
            }
            
            # Add the document node
            doc_node_id = self.hypergraph.add_document(
                doc_id=doc_id,
                title=doc_metadata.get("title", doc_filename),
                path=document_path,
                doc_type=doc_metadata.get("doc_type", "Document"),
                attributes=doc_attributes
            )
            
            # Store the document node for reference
            self.document_nodes[document_path] = doc_node_id
        
        # Extract knowledge elements if available
        if "knowledge_elements" in result and result["knowledge_elements"]:
            self._add_knowledge_elements(result["knowledge_elements"], doc_node_id)
        
        # Process knowledge graph data if available
        if "knowledge_graph" in result and result["knowledge_graph"]:
            self._add_knowledge_graph_data(result["knowledge_graph"], doc_node_id)
        
        return result, doc_node_id
    
    def _add_text_content(self, text: str, doc_node_id: str):
        """
        Add text content from the document as concept nodes.
        
        Args:
            text: Text content from the document
            doc_node_id: Document node ID in the hypergraph
        """
        # Simple approach: add important sections as concept nodes
        # In a real implementation, this would use NLP to extract meaningful concepts
        
        # Split into paragraphs and select non-empty ones
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Take up to 5 paragraphs to avoid overwhelming the graph
        for i, paragraph in enumerate(paragraphs[:5]):
            # Create a concept node for each paragraph
            concept_id = f"CONCEPT_{doc_node_id}_{i}"
            
            # Generate a brief name for the concept (first 50 chars)
            concept_name = paragraph[:50] + "..." if len(paragraph) > 50 else paragraph
            
            # Add the concept node
            concept_node_id = self.hypergraph.add_node(
                node_id=concept_id,
                node_type="Concept",
                name=concept_name,
                source=doc_node_id,
                attributes={"full_text": paragraph, "paragraph_index": i}
            )
            
            # Connect the concept to the document
            self.hypergraph.add_hyperedge(
                nodes=[doc_node_id, concept_node_id],
                edge_type="ContainsConcept",
                attributes={"position": i}
            )
    
    def _add_image_content(self, images: List[Dict[str, Any]], doc_node_id: str):
        """
        Add image content from the document as image nodes.
        
        Args:
            images: List of image data from the document
            doc_node_id: Document node ID in the hypergraph
        """
        for i, image_data in enumerate(images):
            # Create an image node for each image
            image_id = f"IMAGE_{doc_node_id}_{i}"
            
            # Extract image position if available
            position = image_data.get("position", None)
            page_num = image_data.get("page_num", 0)
            
            # Add the image node
            image_node_id = self.hypergraph.add_node(
                node_id=image_id,
                node_type="Image",
                name=f"Image {i+1} in {self.hypergraph.nodes[doc_node_id].name}",
                source=doc_node_id,
                attributes={
                    "page_num": page_num,
                    "position": position,
                    "has_image_data": "image_data" in image_data
                }
            )
            
            # Connect the image to the document
            self.hypergraph.add_hyperedge(
                nodes=[doc_node_id, image_node_id],
                edge_type="ContainsImage",
                attributes={"page_num": page_num, "position": position}
            )
    
    def _add_knowledge_elements(self, elements: List[Dict[str, Any]], doc_node_id: str):
        """
        Add knowledge elements to the hypergraph.
        
        Args:
            elements: List of knowledge elements extracted from the document
            doc_node_id: Document node ID in the hypergraph
        """
        # Track entities for creating relationships
        entity_nodes = {}
        
        # First, create nodes for all entities
        for i, element in enumerate(elements):
            if "entity" in element:
                entity_type = element.get("type", "Entity")
                entity_name = element["entity"]
                
                # Generate an ID for the entity
                entity_id = f"ENTITY_{doc_node_id}_{i}"
                
                # Add the entity node
                entity_node_id = self.hypergraph.add_node(
                    node_id=entity_id,
                    node_type=entity_type,
                    name=entity_name,
                    source=doc_node_id,
                    attributes=element.get("attributes", {})
                )
                
                # Store for relationship creation
                entity_nodes[entity_name] = entity_node_id
                
                # Connect the entity to the document
                self.hypergraph.add_hyperedge(
                    nodes=[doc_node_id, entity_node_id],
                    edge_type="MentionsEntity",
                    attributes={"confidence": element.get("confidence", 1.0)}
                )
        
        # Now create relationships between entities
        for element in elements:
            if "relationship" in element and "source" in element and "target" in element:
                source_name = element["source"]
                target_name = element["target"]
                relationship_type = element["relationship"]
                
                # Check if both entities exist
                if source_name in entity_nodes and target_name in entity_nodes:
                    source_node_id = entity_nodes[source_name]
                    target_node_id = entity_nodes[target_name]
                    
                    # Create the relationship
                    self.hypergraph.add_hyperedge(
                        nodes=[source_node_id, target_node_id],
                        edge_type=relationship_type,
                        attributes={
                            "source_document": doc_node_id,
                            "confidence": element.get("confidence", 1.0)
                        }
                    )
    
    def _add_knowledge_graph_data(self, kg_data: Dict[str, Any], doc_node_id: str):
        """
        Add data from a knowledge graph to the hypergraph.
        
        Args:
            kg_data: Knowledge graph data
            doc_node_id: Document node ID in the hypergraph
        """
        # Track entity mappings from KG to hypergraph
        entity_mapping = {}
        
        # Add all entities
        entities = kg_data.get("entities", [])
        for entity in entities:
            entity_id = entity.get("id", f"KG_ENTITY_{len(entity_mapping)}")
            entity_type = entity.get("type", "Entity")
            entity_name = entity.get("name", f"Entity {entity_id}")
            
            # Add the entity to the hypergraph
            hg_entity_id = self.hypergraph.add_node(
                node_id=f"KG_{entity_id}",
                node_type=entity_type,
                name=entity_name,
                source=doc_node_id,
                attributes=entity.get("properties", {})
            )
            
            # Map KG entity ID to hypergraph entity ID
            entity_mapping[entity_id] = hg_entity_id
            
            # Connect entity to document
            self.hypergraph.add_hyperedge(
                nodes=[doc_node_id, hg_entity_id],
                edge_type="ExtractedEntity",
                attributes={"confidence": entity.get("confidence", 1.0)}
            )
        
        # Add relationships
        relationships = kg_data.get("relationships", [])
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            rel_type = rel.get("type", "Relates")
            
            # Check if both entities exist in our mapping
            if source_id in entity_mapping and target_id in entity_mapping:
                hg_source_id = entity_mapping[source_id]
                hg_target_id = entity_mapping[target_id]
                
                # Add the relationship
                self.hypergraph.add_hyperedge(
                    nodes=[hg_source_id, hg_target_id],
                    edge_type=rel_type,
                    attributes={
                        "source_document": doc_node_id,
                        "properties": rel.get("properties", {}),
                        "confidence": rel.get("confidence", 1.0)
                    }
                )
    
    def add_construction_entities(self, doc_node_id: str, materials: List[Dict[str, Any]] = None,
                                locations: List[Dict[str, Any]] = None, 
                                requirements: List[Dict[str, Any]] = None):
        """
        Add construction-specific entities to the hypergraph.
        
        Args:
            doc_node_id: Document node ID in the hypergraph
            materials: List of material information
            locations: List of location information
            requirements: List of requirement information
        """
        # Track created nodes by type for complex relationship creation
        created_nodes = {
            "materials": [],
            "locations": [],
            "requirements": []
        }
        
        # Add materials
        if materials:
            for i, material in enumerate(materials):
                material_id = f"MAT_{doc_node_id}_{i}"
                material_node_id = self.hypergraph.add_node(
                    node_id=material_id,
                    node_type=self.hypergraph.NODE_TYPES["MATERIAL"],
                    name=material["name"],
                    source=doc_node_id,
                    attributes=material.get("attributes", {})
                )
                created_nodes["materials"].append(material_node_id)
        
        # Add locations
        if locations:
            for i, location in enumerate(locations):
                location_id = f"LOC_{doc_node_id}_{i}"
                location_node_id = self.hypergraph.add_node(
                    node_id=location_id,
                    node_type=self.hypergraph.NODE_TYPES["LOCATION"],
                    name=location["name"],
                    source=doc_node_id,
                    attributes=location.get("attributes", {})
                )
                created_nodes["locations"].append(location_node_id)
        
        # Add requirements
        if requirements:
            for i, requirement in enumerate(requirements):
                req_id = f"REQ_{doc_node_id}_{i}"
                
                # Check if the requirement specifies related entities
                req_materials = requirement.get("materials", [])
                req_locations = requirement.get("locations", [])
                
                # Map material and location names to node IDs if they exist
                material_nodes = []
                for mat_name in req_materials:
                    for mat_id in created_nodes["materials"]:
                        if self.hypergraph.nodes[mat_id].name == mat_name:
                            material_nodes.append(mat_id)
                            break
                
                location_nodes = []
                for loc_name in req_locations:
                    for loc_id in created_nodes["locations"]:
                        if self.hypergraph.nodes[loc_id].name == loc_name:
                            location_nodes.append(loc_id)
                            break
                
                # Add the requirement as a complex relationship if it has related entities
                if material_nodes or location_nodes:
                    req_node_id, _ = self.hypergraph.add_complex_requirement(
                        req_id=req_id,
                        description=requirement["description"],
                        materials=material_nodes,
                        locations=location_nodes,
                        source_doc=doc_node_id
                    )
                else:
                    # Simple requirement without related entities
                    req_node_id = self.hypergraph.add_requirement(
                        req_id=req_id,
                        description=requirement["description"],
                        source_doc=doc_node_id,
                        attributes=requirement.get("attributes", {})
                    )
                
                created_nodes["requirements"].append(req_node_id)
    
    def save_hypergraph(self, output_path: str):
        """
        Save the hypergraph to a JSON file.
        
        Args:
            output_path: Path to save the hypergraph
        """
        try:
            # Export the hypergraph to a dictionary
            export_data = self.hypergraph.export_to_dict()
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Saved hypergraph to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving hypergraph: {str(e)}")
            return False


# Example usage demonstration
def process_construction_documents(document_paths: List[str], output_path: str = None):
    """
    Process multiple construction documents and create a hypergraph.
    
    Args:
        document_paths: List of paths to construction documents
        output_path: Path to save the resulting hypergraph
    """
    # Initialize the converter
    converter = DocumentToHypergraphConverter(project_name="Construction_Project")
    
    # Process each document
    for doc_path in document_paths:
        try:
            # Process for knowledge graph and convert to hypergraph
            result, doc_node_id = converter.process_for_knowledge_graph(doc_path)
            
            if not result.get("success", False):
                logger.warning(f"Failed to process document: {doc_path}")
                continue
            
            logger.info(f"Successfully processed document: {doc_path}")
            
            # Example: Add construction-specific entities
            # In a real implementation, these would be extracted from the document
            if "foundation" in doc_path.lower() or "structural" in doc_path.lower():
                converter.add_construction_entities(
                    doc_node_id=doc_node_id,
                    materials=[
                        {"name": "Concrete", "attributes": {"grade": "C30/37"}},
                        {"name": "Reinforcement Steel", "attributes": {"type": "Rebar"}}
                    ],
                    locations=[
                        {"name": "Foundation", "attributes": {"area": "500m²"}}
                    ],
                    requirements=[
                        {
                            "description": "Foundation must use high-strength concrete",
                            "materials": ["Concrete"],
                            "locations": ["Foundation"]
                        }
                    ]
                )
        except Exception as e:
            logger.exception(f"Error processing document: {doc_path}")
    
    # Save the hypergraph if output path is provided
    if output_path:
        converter.save_hypergraph(output_path)
    
    return converter.hypergraph


if __name__ == "__main__":
    import sys
    
    # Check if document paths were provided
    if len(sys.argv) < 2:
        print("Usage: python document_vision_to_hypergraph.py <doc_path1> <doc_path2> ... [--output OUTPUT_PATH]")
        sys.exit(1)
    
    # Parse command line arguments
    doc_paths = []
    output_path = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        else:
            doc_paths.append(sys.argv[i])
            i += 1
    
    # Process the documents
    hypergraph = process_construction_documents(doc_paths, output_path)
    
    # Print information about the resulting hypergraph
    print(f"\nProcessed {len(doc_paths)} documents into a hypergraph with:")
    print(f"  - {len(hypergraph.nodes)} nodes")
    print(f"  - {len(hypergraph.hyperedges)} relationships")
    
    if output_path:
        print(f"\nHypergraph saved to: {output_path}")
    else:
        print("\nHypergraph was not saved (no output path provided)") 