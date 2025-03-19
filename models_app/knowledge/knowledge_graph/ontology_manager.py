"""
Ontology Manager for Knowledge Graph Schema Management.

Handles ontology loading, validation, and schema enforcement for knowledge graphs.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class OntologyManager:
    """
    Manages ontologies for the knowledge graph system.
    
    Responsibilities:
    - Load and parse ontology definitions
    - Validate entities and relationships against ontologies
    - Provide schema information to graph builders
    - Suggest entity types and relationships based on context
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ontology manager"""
        self.config = config or {}
        self.ontologies = {}
        self.active_ontology = None
        self.default_ontology_path = self.config.get(
            "default_ontology_path", 
            os.path.join(os.path.dirname(__file__), "ontologies", "default_ontology.json")
        )
        
        # Initialize default ontology
        self._load_default_ontology()
    
    def _load_default_ontology(self):
        """Load the default ontology"""
        try:
            if os.path.exists(self.default_ontology_path):
                with open(self.default_ontology_path, 'r', encoding='utf-8') as f:
                    ontology = json.load(f)
                    self.ontologies["default"] = ontology
                    self.active_ontology = "default"
            else:
                # Create a minimal default ontology
                default_ontology = self._create_minimal_ontology()
                self.ontologies["default"] = default_ontology
                self.active_ontology = "default"
                
                # Save it for future use
                os.makedirs(os.path.dirname(self.default_ontology_path), exist_ok=True)
                with open(self.default_ontology_path, 'w', encoding='utf-8') as f:
                    json.dump(default_ontology, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to load default ontology: {str(e)}")
            # Create in-memory minimal ontology
            self.ontologies["default"] = self._create_minimal_ontology()
            self.active_ontology = "default"
    
    def _create_minimal_ontology(self) -> Dict[str, Any]:
        """Create a minimal default ontology"""
        return {
            "name": "Default Knowledge Graph Ontology",
            "version": "1.0.0",
            "description": "Default ontology with basic entity and relationship types",
            "entity_types": {
                "Person": {
                    "description": "An individual person",
                    "required_properties": ["name"],
                    "optional_properties": ["age", "birth_date", "occupation"]
                },
                "Organization": {
                    "description": "A company, institution, or other organization",
                    "required_properties": ["name"],
                    "optional_properties": ["industry", "founded_date", "location"]
                },
                "Location": {
                    "description": "A physical place",
                    "required_properties": ["name"],
                    "optional_properties": ["coordinates", "country", "population"]
                },
                "Event": {
                    "description": "An occurrence at a particular time",
                    "required_properties": ["name"],
                    "optional_properties": ["start_date", "end_date", "location"]
                },
                "Document": {
                    "description": "A document or publication",
                    "required_properties": ["title"],
                    "optional_properties": ["author", "publication_date", "content"]
                },
                "Concept": {
                    "description": "An abstract idea or term",
                    "required_properties": ["name"],
                    "optional_properties": ["definition", "field"]
                }
            },
            "relationship_types": {
                "WORKS_FOR": {
                    "description": "Employment relationship",
                    "source_types": ["Person"],
                    "target_types": ["Organization"]
                },
                "LOCATED_IN": {
                    "description": "Physical location relationship",
                    "source_types": ["Organization", "Event", "Person"],
                    "target_types": ["Location"]
                },
                "KNOWS": {
                    "description": "Acquaintance relationship",
                    "source_types": ["Person"],
                    "target_types": ["Person"]
                },
                "PART_OF": {
                    "description": "Composition relationship",
                    "source_types": ["Location", "Organization", "Event"],
                    "target_types": ["Location", "Organization", "Event"]
                },
                "AUTHORED": {
                    "description": "Creation relationship",
                    "source_types": ["Person", "Organization"],
                    "target_types": ["Document", "Concept"]
                },
                "MENTIONS": {
                    "description": "Reference relationship",
                    "source_types": ["Document"],
                    "target_types": ["Person", "Organization", "Location", "Event", "Concept"]
                },
                "RELATED_TO": {
                    "description": "Generic relationship",
                    "source_types": ["Person", "Organization", "Location", "Event", "Document", "Concept"],
                    "target_types": ["Person", "Organization", "Location", "Event", "Document", "Concept"]
                }
            }
        }
    
    def load_ontology(self, ontology_path: str, ontology_id: Optional[str] = None) -> bool:
        """
        Load an ontology from a file.
        
        Args:
            ontology_path: Path to the ontology file
            ontology_id: Optional ID for the ontology (defaults to filename)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(ontology_path, 'r', encoding='utf-8') as f:
                ontology = json.load(f)
                
                # Use filename as ID if not provided
                if not ontology_id:
                    ontology_id = Path(ontology_path).stem
                
                # Validate the ontology structure
                if not self._validate_ontology_structure(ontology):
                    logger.error(f"Invalid ontology structure in {ontology_path}")
                    return False
                
                # Add to ontologies
                self.ontologies[ontology_id] = ontology
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to load ontology {ontology_path}: {str(e)}")
            return False
    
    def _validate_ontology_structure(self, ontology: Dict[str, Any]) -> bool:
        """Validate the structure of an ontology"""
        # Check for required fields
        if not all(key in ontology for key in ["name", "entity_types", "relationship_types"]):
            return False
        
        # Check entity types
        if not isinstance(ontology["entity_types"], dict):
            return False
        
        # Check relationship types
        if not isinstance(ontology["relationship_types"], dict):
            return False
        
        return True
    
    def set_active_ontology(self, ontology_id: str) -> bool:
        """
        Set the active ontology.
        
        Args:
            ontology_id: ID of the ontology to activate
            
        Returns:
            bool: True if successful, False if ontology not found
        """
        if ontology_id in self.ontologies:
            self.active_ontology = ontology_id
            return True
        return False
    
    def get_ontology(self, ontology_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get an ontology by ID.
        
        Args:
            ontology_id: ID of the ontology to get (defaults to active ontology)
            
        Returns:
            Dict: The ontology or None if not found
        """
        if not ontology_id:
            ontology_id = self.active_ontology
            
        return self.ontologies.get(ontology_id)
    
    def validate_entity(self, entity: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate an entity against the active ontology.
        
        Args:
            entity: The entity to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_validation_messages)
        """
        ontology = self.get_ontology()
        if not ontology:
            return True, ["No active ontology defined"]
        
        messages = []
        entity_type = entity.get("type")
        
        # Check if entity type exists in ontology
        if not entity_type:
            messages.append("Entity has no type")
            return False, messages
        
        entity_type_def = ontology["entity_types"].get(entity_type)
        if not entity_type_def:
            messages.append(f"Entity type '{entity_type}' not defined in ontology")
            return False, messages
        
        # Check required properties
        required_props = entity_type_def.get("required_properties", [])
        entity_props = entity.get("properties", {})
        
        missing_props = []
        for prop in required_props:
            if prop not in entity_props:
                missing_props.append(prop)
        
        if missing_props:
            messages.append(f"Missing required properties: {', '.join(missing_props)}")
            return False, messages
        
        return True, ["Entity is valid"]
    
    def validate_relationship(self, relationship: Dict[str, Any], 
                             source_entity: Optional[Dict[str, Any]] = None, 
                             target_entity: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Validate a relationship against the active ontology.
        
        Args:
            relationship: The relationship to validate
            source_entity: Optional source entity (if available)
            target_entity: Optional target entity (if available)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_validation_messages)
        """
        ontology = self.get_ontology()
        if not ontology:
            return True, ["No active ontology defined"]
        
        messages = []
        rel_type = relationship.get("type")
        
        # Check if relationship type exists in ontology
        if not rel_type:
            messages.append("Relationship has no type")
            return False, messages
        
        rel_type_def = ontology["relationship_types"].get(rel_type)
        if not rel_type_def:
            messages.append(f"Relationship type '{rel_type}' not defined in ontology")
            return False, messages
        
        # Check source and target types if entities are provided
        if source_entity and target_entity:
            source_type = source_entity.get("type")
            target_type = target_entity.get("type")
            
            if source_type and target_type:
                valid_source_types = rel_type_def.get("source_types", [])
                valid_target_types = rel_type_def.get("target_types", [])
                
                if valid_source_types and source_type not in valid_source_types:
                    messages.append(f"Invalid source type '{source_type}' for relationship '{rel_type}'")
                    return False, messages
                
                if valid_target_types and target_type not in valid_target_types:
                    messages.append(f"Invalid target type '{target_type}' for relationship '{rel_type}'")
                    return False, messages
        
        return True, ["Relationship is valid"]
    
    def suggest_entity_type(self, entity_properties: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Suggest entity types based on properties.
        
        Args:
            entity_properties: Properties of the entity
            
        Returns:
            List[Tuple[str, float]]: List of (entity_type, confidence) tuples
        """
        ontology = self.get_ontology()
        if not ontology:
            return []
        
        suggestions = []
        
        for entity_type, type_def in ontology["entity_types"].items():
            # Calculate match score based on properties
            required_props = set(type_def.get("required_properties", []))
            optional_props = set(type_def.get("optional_properties", []))
            all_type_props = required_props.union(optional_props)
            
            entity_prop_keys = set(entity_properties.keys())
            
            # Count matches
            matching_props = entity_prop_keys.intersection(all_type_props)
            
            # Calculate confidence score
            if not all_type_props:
                confidence = 0.0
            else:
                # More weight to required properties
                req_matches = len(entity_prop_keys.intersection(required_props))
                opt_matches = len(entity_prop_keys.intersection(optional_props))
                
                if required_props:
                    req_score = req_matches / len(required_props)
                else:
                    req_score = 0.0
                    
                if optional_props:
                    opt_score = opt_matches / len(optional_props)
                else:
                    opt_score = 0.0
                
                # Combined score (70% required, 30% optional)
                confidence = 0.7 * req_score + 0.3 * opt_score
            
            suggestions.append((entity_type, confidence))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions
    
    def suggest_relationship_type(self, source_type: str, target_type: str) -> List[Tuple[str, float]]:
        """
        Suggest relationship types based on source and target entity types.
        
        Args:
            source_type: Type of the source entity
            target_type: Type of the target entity
            
        Returns:
            List[Tuple[str, float]]: List of (relationship_type, confidence) tuples
        """
        ontology = self.get_ontology()
        if not ontology:
            return []
        
        suggestions = []
        
        for rel_type, rel_def in ontology["relationship_types"].items():
            source_types = rel_def.get("source_types", [])
            target_types = rel_def.get("target_types", [])
            
            # Skip if source or target types are restricted and don't match
            if source_types and source_type not in source_types:
                continue
                
            if target_types and target_type not in target_types:
                continue
            
            # Calculate confidence based on specificity
            # More specific relationships (fewer valid source/target types) get higher confidence
            if not source_types or not target_types:
                # Generic relationship with no type restrictions
                confidence = 0.3
            else:
                source_specificity = 1.0 / len(source_types)
                target_specificity = 1.0 / len(target_types)
                confidence = 0.5 * (source_specificity + target_specificity)
            
            suggestions.append((rel_type, confidence))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions
    
    def create_schema_visualization(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a visualization of the ontology schema.
        
        Args:
            output_path: Optional path to save the visualization
            
        Returns:
            Dict: Visualization data in a format suitable for rendering
        """
        ontology = self.get_ontology()
        if not ontology:
            return {"error": "No active ontology"}
        
        # Create nodes for entity types
        nodes = []
        for entity_type, type_def in ontology["entity_types"].items():
            nodes.append({
                "id": entity_type,
                "label": entity_type,
                "type": "entity_type",
                "description": type_def.get("description", ""),
                "properties": {
                    "required": type_def.get("required_properties", []),
                    "optional": type_def.get("optional_properties", [])
                }
            })
        
        # Create links for relationship types
        links = []
        for rel_type, rel_def in ontology["relationship_types"].items():
            source_types = rel_def.get("source_types", [])
            target_types = rel_def.get("target_types", [])
            
            # Create links between all valid source and target types
            for source in source_types:
                for target in target_types:
                    links.append({
                        "source": source,
                        "target": target,
                        "label": rel_type,
                        "type": "relationship_type",
                        "description": rel_def.get("description", "")
                    })
        
        # Create visualization data
        visualization = {
            "nodes": nodes,
            "links": links,
            "ontology_name": ontology.get("name", "Unnamed Ontology"),
            "ontology_version": ontology.get("version", "1.0.0")
        }
        
        # Save to file if requested
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(visualization, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save ontology visualization: {str(e)}")
        
        return visualization
    
    def export_ontology(self, output_path: str, format: str = "json") -> bool:
        """
        Export the active ontology to a file.
        
        Args:
            output_path: Path to save the ontology
            format: Format to save in (json, owl, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        ontology = self.get_ontology()
        if not ontology:
            return False
        
        try:
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(ontology, f, indent=2)
                return True
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
        except Exception as e:
            logger.error(f"Failed to export ontology: {str(e)}")
            return False
    
    def get_entity_types(self) -> List[str]:
        """Get list of all entity types in the active ontology"""
        ontology = self.get_ontology()
        if not ontology:
            return []
        
        return list(ontology.get("entity_types", {}).keys())
    
    def get_relationship_types(self) -> List[str]:
        """Get list of all relationship types in the active ontology"""
        ontology = self.get_ontology()
        if not ontology:
            return []
        
        return list(ontology.get("relationship_types", {}).keys())
    
    def get_entity_type_definition(self, entity_type: str) -> Optional[Dict[str, Any]]:
        """Get definition of a specific entity type"""
        ontology = self.get_ontology()
        if not ontology:
            return None
        
        return ontology.get("entity_types", {}).get(entity_type)
    
    def get_relationship_type_definition(self, relationship_type: str) -> Optional[Dict[str, Any]]:
        """Get definition of a specific relationship type"""
        ontology = self.get_ontology()
        if not ontology:
            return None
        
        return ontology.get("relationship_types", {}).get(relationship_type)
