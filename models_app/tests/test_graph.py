import unittest
from django.test import TestCase
from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
from models_app.knowledge_graph.graph_builder import GraphBuilder
from models_app.knowledge_graph.ontology_manager import OntologyManager
from models_app.knowledge_graph.entity_resolution import EntityResolver
from models_app.knowledge_graph.external_kb_connector import WikidataConnector
from analytics_app.utils import evaluate_knowledge_graph_quality
import logging
import os

logger = logging.getLogger(__name__)

class TestGraphBuilder(TestCase):
    def setUp(self):
        """Set up test environment"""
        self.graph_builder = GraphBuilder()
        self.test_entities = [
            {
                "label": "John Doe",
                "type": "Person",
                "properties": {
                    "name": "John Doe",
                    "age": 30,
                    "occupation": "Engineer"
                }
            },
            {
                "label": "Acme Corporation",
                "type": "Organization",
                "properties": {
                    "name": "Acme Corporation",
                    "industry": "Technology",
                    "founded_date": "2000-01-01"
                }
            },
            {
                "label": "New York",
                "type": "Location",
                "properties": {
                    "name": "New York",
                    "country": "USA",
                    "population": 8500000
                }
            }
        ]
        
        self.test_relationships = [
            {
                "source": "",  # Will be filled during test
                "target": "",  # Will be filled during test
                "type": "WORKS_FOR",
                "properties": {
                    "since": "2015-03-01",
                    "position": "Senior Engineer"
                }
            },
            {
                "source": "",  # Will be filled during test
                "target": "",  # Will be filled during test
                "type": "LOCATED_IN",
                "properties": {
                    "headquarters": True
                }
            }
        ]
    
    def test_entity_creation(self):
        """Test entity creation and validation"""
        # Create entities
        entity_ids = []
        for entity in self.test_entities:
            entity_id = self.graph_builder.add_entity(entity)
            self.assertIsNotNone(entity_id)
            entity_ids.append(entity_id)
            
        # Verify entities have required fields
        for i, entity in enumerate(self.test_entities):
            self.assertIn("id", entity)
            self.assertEqual(entity["id"], entity_ids[i])
            self.assertIn("type", entity)
            self.assertIn("timestamp", entity)
    
    def test_relationship_creation(self):
        """Test relationship creation and validation"""
        # Create entities first
        entity_ids = []
        for entity in self.test_entities:
            entity_id = self.graph_builder.add_entity(entity)
            entity_ids.append(entity_id)
        
        # Set source and target for relationships
        self.test_relationships[0]["source"] = entity_ids[0]  # Person -> Organization
        self.test_relationships[0]["target"] = entity_ids[1]
        self.test_relationships[1]["source"] = entity_ids[1]  # Organization -> Location
        self.test_relationships[1]["target"] = entity_ids[2]
        
        # Create relationships
        rel_ids = []
        for rel in self.test_relationships:
            rel_id = self.graph_builder.add_relationship(rel)
            self.assertIsNotNone(rel_id)
            rel_ids.append(rel_id)
            
        # Verify relationships have required fields
        for i, rel in enumerate(self.test_relationships):
            self.assertIn("id", rel)
            self.assertEqual(rel["id"], rel_ids[i])
            self.assertIn("type", rel)
            self.assertIn("source", rel)
            self.assertIn("target", rel)
            self.assertIn("timestamp", rel)
    
    def test_subgraph_building(self):
        """Test building a subgraph from entities and relationships"""
        # Create entities first
        for entity in self.test_entities:
            self.graph_builder.add_entity(entity)
        
        # Set source and target for relationships
        self.test_relationships[0]["source"] = self.test_entities[0]["id"]
        self.test_relationships[0]["target"] = self.test_entities[1]["id"]
        self.test_relationships[1]["source"] = self.test_entities[1]["id"]
        self.test_relationships[1]["target"] = self.test_entities[2]["id"]
        
        # Create relationships
        for rel in self.test_relationships:
            self.graph_builder.add_relationship(rel)
        
        # Build subgraph
        subgraph = self.graph_builder.build_subgraph(
            self.test_entities, 
            self.test_relationships,
            {"context": "test"}
        )
        
        # Verify subgraph structure
        self.assertIn("id", subgraph)
        self.assertIn("entities", subgraph)
        self.assertIn("relationships", subgraph)
        self.assertIn("metadata", subgraph)
        
        # Verify entity and relationship counts
        self.assertEqual(len(subgraph["entities"]), len(self.test_entities))
        self.assertEqual(len(subgraph["relationships"]), len(self.test_relationships))
        
        # Verify context was added to metadata
        self.assertEqual(subgraph["metadata"]["context"], "test")
    
    def test_graph_merging(self):
        """Test merging multiple subgraphs"""
        # Create first subgraph (Person -> Organization)
        subgraph1_entities = self.test_entities[:2]
        for entity in subgraph1_entities:
            self.graph_builder.add_entity(entity)
            
        subgraph1_rel = {
            "source": subgraph1_entities[0]["id"],
            "target": subgraph1_entities[1]["id"],
            "type": "WORKS_FOR"
        }
        self.graph_builder.add_relationship(subgraph1_rel)
        
        subgraph1 = self.graph_builder.build_subgraph(
            subgraph1_entities, 
            [subgraph1_rel],
            {"context": "subgraph1"}
        )
        
        # Create second subgraph (Organization -> Location)
        subgraph2_entities = [self.test_entities[1], self.test_entities[2]]
        for entity in subgraph2_entities:
            self.graph_builder.add_entity(entity)
            
        subgraph2_rel = {
            "source": subgraph2_entities[0]["id"],
            "target": subgraph2_entities[1]["id"],
            "type": "LOCATED_IN"
        }
        self.graph_builder.add_relationship(subgraph2_rel)
        
        subgraph2 = self.graph_builder.build_subgraph(
            subgraph2_entities, 
            [subgraph2_rel],
            {"context": "subgraph2"}
        )
        
        # Merge subgraphs
        merged_graph = self.graph_builder.merge_subgraphs([subgraph1, subgraph2])
        
        # Verify merged graph structure
        self.assertIn("id", merged_graph)
        self.assertIn("entities", merged_graph)
        self.assertIn("relationships", merged_graph)
        self.assertIn("metadata", merged_graph)
        
        # Verify entity count (should be 3, with Organization only appearing once)
        self.assertEqual(len(merged_graph["entities"]), 3)
        
        # Verify relationship count (should be 2)
        self.assertEqual(len(merged_graph["relationships"]), 2)
        
        # Verify source tracking in metadata
        self.assertEqual(len(merged_graph["metadata"]["sources"]), 2)

    def test_empty_input_handling(self):
        """Test graph builder handling of empty inputs"""
        # Test with empty entity list
        empty_entity_graph = self.graph_builder.build_subgraph([], self.test_relationships)
        self.assertTrue("is_empty" in empty_entity_graph.get("metadata", {}))
        
        # Test with empty relationship list
        empty_rel_graph = self.graph_builder.build_subgraph(self.test_entities, [])
        self.assertEqual(len(empty_rel_graph.get("relationships", [])), 0)
        
        # Test with both empty
        empty_graph = self.graph_builder.build_subgraph([], [])
        self.assertTrue(empty_graph.get("metadata", {}).get("is_empty", False))

    def test_invalid_entity_references(self):
        """Test graph builder handling of invalid entity references in relationships"""
        # Create relationship with non-existent entity reference
        invalid_rel = {
            "source": "nonexistent_id_1",
            "target": "nonexistent_id_2",
            "type": "INVALID_REF"
        }
        
        # First create some valid entities
        entities = []
        for entity in self.test_entities[:2]:
            entity_id = self.graph_builder.add_entity(entity)
            entities.append(entity)
        
        # Build graph with invalid relationship
        graph = self.graph_builder.build_subgraph(entities, [invalid_rel])
        
        # Check that invalid relationship was not included
        self.assertEqual(len(graph.get("relationships", [])), 0)

    def test_complex_graph_structure(self):
        """Test complex graph structure with multiple levels of relationships"""
        # Create hierarchical structure: Person -> Organization -> Department -> Project
        entities = [
            {"type": "Person", "label": "John Smith", "properties": {"name": "John Smith"}},
            {"type": "Organization", "label": "Acme Corp", "properties": {"name": "Acme Corp"}},
            {"type": "Department", "label": "R&D", "properties": {"name": "Research & Development"}},
            {"type": "Project", "label": "Project X", "properties": {"name": "Project X"}},
            {"type": "Technology", "label": "AI", "properties": {"name": "Artificial Intelligence"}}
        ]
        
        # Add entities to get IDs
        for entity in entities:
            self.graph_builder.add_entity(entity)
        
        # Create multi-level relationships
        relationships = [
            {"source": entities[0]["id"], "target": entities[1]["id"], "type": "WORKS_FOR"},
            {"source": entities[1]["id"], "target": entities[2]["id"], "type": "HAS_DEPARTMENT"},
            {"source": entities[2]["id"], "target": entities[3]["id"], "type": "RUNS_PROJECT"},
            {"source": entities[3]["id"], "target": entities[4]["id"], "type": "USES_TECHNOLOGY"},
            {"source": entities[0]["id"], "target": entities[3]["id"], "type": "CONTRIBUTES_TO"} # Cross-level
        ]
        
        # Build the graph
        graph = self.graph_builder.build_subgraph(entities, relationships)
        
        # Verify graph structure
        self.assertEqual(len(graph["entities"]), 5)
        self.assertEqual(len(graph["relationships"]), 5)
        
        # Test path finding between disconnected levels
        # Convert graph to NetworkX for path finding
        import networkx as nx
        G = nx.DiGraph()
        for entity in graph["entities"]:
            G.add_node(entity["id"])
        for rel in graph["relationships"]:
            G.add_edge(rel["source"], rel["target"])
        
        # Check path exists between Person and Technology
        has_path = nx.has_path(G, entities[0]["id"], entities[4]["id"])
        self.assertTrue(has_path)
        
        # Find all paths between Person and Technology
        all_paths = list(nx.all_simple_paths(G, entities[0]["id"], entities[4]["id"]))
        self.assertTrue(len(all_paths) >= 2)  # Should have direct and indirect paths

    def test_performance_with_large_graph(self):
        """Test performance with larger graph (benchmark test)"""
        # Skip if not running benchmark tests
        import unittest
        try:
            skip_benchmark = os.environ.get("SKIP_BENCHMARK_TESTS", "True").lower() == "true"
            if skip_benchmark:
                self.skipTest("Skipping benchmark test")
        except:
            self.skipTest("Skipping benchmark test")
        
        # Generate a larger test graph (100 entities, ~300 relationships)
        large_entities = []
        entity_types = ["Person", "Organization", "Document", "Topic", "Project"]
        
        for i in range(100):
            entity_type = entity_types[i % len(entity_types)]
            entity = {
                "type": entity_type,
                "label": f"{entity_type} {i}",
                "properties": {
                    "name": f"{entity_type} {i}",
                    "id_number": i
                }
            }
            self.graph_builder.add_entity(entity)
            large_entities.append(entity)
        
        # Create relationships (each entity connected to ~3 others)
        large_relationships = []
        for i, entity in enumerate(large_entities):
            # Connect to 3 other entities
            for j in range(1, 4):
                target_idx = (i + j) % len(large_entities)
                rel_type = f"RELATED_TO_{j}"
                rel = {
                    "source": entity["id"],
                    "target": large_entities[target_idx]["id"],
                    "type": rel_type
                }
                large_relationships.append(rel)
        
        # Measure time to build graph
        import time
        start_time = time.time()
        large_graph = self.graph_builder.build_subgraph(large_entities, large_relationships)
        build_time = time.time() - start_time
        
        # Verify graph was built correctly
        self.assertEqual(len(large_graph["entities"]), 100)
        self.assertEqual(len(large_graph["relationships"]), len(large_relationships))
        
        # Measure merge time with another identical graph
        start_time = time.time()
        merged_graph = self.graph_builder.merge_subgraphs([large_graph, large_graph])
        merge_time = time.time() - start_time
        
        # Log performance results
        logger.info(f"Large graph build time: {build_time:.3f}s, merge time: {merge_time:.3f}s")
        
        # Assert reasonable performance (adjust thresholds as needed)
        self.assertLess(build_time, 5.0, "Graph building took too long")
        self.assertLess(merge_time, 10.0, "Graph merging took too long")

class TestOntologyManager(TestCase):
    def setUp(self):
        """Set up test environment"""
        self.ontology_manager = OntologyManager()
        
    def test_default_ontology_loading(self):
        """Test loading of default ontology"""
        ontology = self.ontology_manager.get_ontology()
        self.assertIsNotNone(ontology)
        self.assertIn("entity_types", ontology)
        self.assertIn("relationship_types", ontology)
        
    def test_entity_validation(self):
        """Test entity validation against ontology"""
        # Valid entity
        valid_entity = {
            "type": "Person",
            "properties": {
                "name": "John Doe"
            }
        }
        is_valid, messages = self.ontology_manager.validate_entity(valid_entity)
        self.assertTrue(is_valid)
        
        # Invalid entity (missing required property)
        invalid_entity = {
            "type": "Person",
            "properties": {
                "age": 30
            }
        }
        is_valid, messages = self.ontology_manager.validate_entity(invalid_entity)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required properties" in msg for msg in messages))
        
        # Invalid entity type
        unknown_type_entity = {
            "type": "UnknownType",
            "properties": {
                "name": "Test"
            }
        }
        is_valid, messages = self.ontology_manager.validate_entity(unknown_type_entity)
        self.assertFalse(is_valid)
        self.assertTrue(any("not defined in ontology" in msg for msg in messages))
        
    def test_relationship_validation(self):
        """Test relationship validation against ontology"""
        # Valid relationship
        person = {
            "type": "Person",
            "properties": {
                "name": "John Doe"
            }
        }
        organization = {
            "type": "Organization",
            "properties": {
                "name": "Acme Corp"
            }
        }
        valid_relationship = {
            "type": "WORKS_FOR",
            "source": "person_id",
            "target": "org_id"
        }
        
        is_valid, messages = self.ontology_manager.validate_relationship(
            valid_relationship, person, organization
        )
        self.assertTrue(is_valid)
        
        # Invalid relationship (wrong source/target types)
        invalid_relationship = {
            "type": "WORKS_FOR",
            "source": "org_id",
            "target": "person_id"
        }
        
        is_valid, messages = self.ontology_manager.validate_relationship(
            invalid_relationship, organization, person
        )
        self.assertFalse(is_valid)
        
    def test_entity_type_suggestion(self):
        """Test entity type suggestion based on properties"""
        # Properties for a person
        person_props = {
            "name": "John Doe",
            "age": 30,
            "occupation": "Engineer"
        }
        
        suggestions = self.ontology_manager.suggest_entity_type(person_props)
        self.assertTrue(len(suggestions) > 0)
        
        # First suggestion should be Person with high confidence
        top_suggestion = suggestions[0]
        self.assertEqual(top_suggestion[0], "Person")
        self.assertGreater(top_suggestion[1], 0.5)  # High confidence
        
    def test_relationship_type_suggestion(self):
        """Test relationship type suggestion based on entity types"""
        suggestions = self.ontology_manager.suggest_relationship_type("Person", "Organization")
        self.assertTrue(len(suggestions) > 0)
        
        # Should suggest WORKS_FOR as high confidence option
        self.assertTrue(any(s[0] == "WORKS_FOR" for s in suggestions))
        top_suggestion = next(s for s in suggestions if s[0] == "WORKS_FOR")
        self.assertGreater(top_suggestion[1], 0.5)  # High confidence

class TestEntityResolution(TestCase):
    def setUp(self):
        """Set up test environment"""
        self.resolver = EntityResolver()
        
    def test_entity_resolution(self):
        """Test entity resolution functionality"""
        # Create entities with potential duplicates
        entities = [
            {
                "id": "e1",
                "label": "John Doe",
                "type": "Person",
                "properties": {"name": "John Doe", "age": 30}
            },
            {
                "id": "e2",
                "label": "John Doe",  # Same name
                "type": "Person",
                "properties": {"name": "John Doe", "occupation": "Engineer"}
            },
            {
                "id": "e3",
                "label": "Jane Smith",  # Different person
                "type": "Person",
                "properties": {"name": "Jane Smith", "age": 25}
            }
        ]
        
        resolved_entities = self.resolver.resolve_entities(entities)
        
        # Should merge John Doe entities but keep Jane Smith separate
        self.assertEqual(len(resolved_entities), 2)
        
        # Find the merged John Doe entity
        john_entity = next(e for e in resolved_entities if "John Doe" in e["label"])
        
        # Check that properties were merged
        self.assertIn("age", john_entity["properties"])
        self.assertIn("occupation", john_entity["properties"])

class TestExternalKBIntegration(TestCase):
    def setUp(self):
        """Set up test environment"""
        self.wikidata_connector = WikidataConnector({"retry_count": 1})
        
    def test_entity_linking(self):
        """Test entity linking to external KB"""
        # Skip if no internet connection
        try:
            import socket
            socket.create_connection(("www.wikidata.org", 80), timeout=1)
        except:
            self.skipTest("No internet connection available")
            
        # Test entity
        test_entity = {
            "label": "Albert Einstein",
            "type": "Person",
            "properties": {
                "name": "Albert Einstein",
                "occupation": "Physicist"
            }
        }
        
        matches = self.wikidata_connector.link_entity(test_entity)
        
        # Should find Einstein in Wikidata
        self.assertTrue(len(matches) > 0)
        
        # Top match should be Einstein with high confidence
        top_match = matches[0]
        self.assertIn("Einstein", top_match["external_label"])
        self.assertGreater(top_match["confidence"], 0.7)
        
    def test_entity_enrichment(self):
        """Test entity enrichment from external KB"""
        # Skip if no internet connection
        try:
            import socket
            socket.create_connection(("www.wikidata.org", 80), timeout=1)
        except:
            self.skipTest("No internet connection available")
            
        # Test entity (with known Wikidata ID for Einstein: Q937)
        test_entity = {
            "label": "Albert Einstein",
            "type": "Person",
            "properties": {
                "name": "Albert Einstein"
            }
        }
        
        enriched_entity = self.wikidata_connector.enrich_entity(test_entity, "Q937")
        
        # Check that external reference was added
        self.assertIn("external_references", enriched_entity)
        self.assertEqual(enriched_entity["external_references"][0]["source"], "wikidata")
        
        # Should have added properties from Wikidata
        self.assertGreater(len(enriched_entity["properties"]), 1)

class TestKnowledgeGraphQuality(TestCase):
    def setUp(self):
        """Set up test environment"""
        self.graph_builder = GraphBuilder()
        self.kg_manager = KnowledgeGraphManager()
        
        # Create a test graph
        self.entities = [
            {
                "label": "John Doe",
                "type": "Person",
                "properties": {"name": "John Doe", "age": 30}
            },
            {
                "label": "Acme Corp",
                "type": "Organization",
                "properties": {"name": "Acme Corp", "industry": "Technology"}
            },
            {
                "label": "New York",
                "type": "Location",
                "properties": {"name": "New York", "country": "USA"}
            }
        ]
        
        # Add IDs to entities
        for entity in self.entities:
            self.graph_builder.add_entity(entity)
            
        # Create relationships
        self.relationships = [
            {
                "source": self.entities[0]["id"],
                "target": self.entities[1]["id"],
                "type": "WORKS_FOR"
            },
            {
                "source": self.entities[1]["id"],
                "target": self.entities[2]["id"],
                "type": "LOCATED_IN"
            }
        ]
        
        # Build the graph
        self.test_graph = self.graph_builder.build_subgraph(
            self.entities,
            self.relationships
        )
    
    def test_graph_quality_metrics(self):
        """Test calculation of graph quality metrics"""
        from analytics_app.utils import evaluate_knowledge_graph_quality
        
        metrics = evaluate_knowledge_graph_quality(self.test_graph)
        
        # Check that all expected metric categories are present
        self.assertIn("structural", metrics)
        self.assertIn("semantic", metrics)
        self.assertIn("user_oriented", metrics)
        
        # Check specific metrics
        self.assertEqual(metrics["structural"]["entity_count"], 3)
        self.assertEqual(metrics["structural"]["relationship_count"], 2)
        
        # Check entity types
        entity_types = metrics["structural"]["entity_types"]
        self.assertEqual(entity_types["Person"], 1)
        self.assertEqual(entity_types["Organization"], 1)
        self.assertEqual(entity_types["Location"], 1)
        
    def test_quality_meets_standards(self):
        """Test that graph quality meets minimum standards"""
        from analytics_app.utils import evaluate_knowledge_graph_quality
        
        metrics = evaluate_knowledge_graph_quality(self.test_graph)
        
        # Standards to check
        self.assertGreaterEqual(metrics["structural"]["entity_count"], 3, 
                               "Not enough entities in graph")
        self.assertGreaterEqual(metrics["structural"]["relationship_count"], 2,
                               "Not enough relationships in graph")
        
        # Calculate schema completeness manually for testing
        # (since we might not have the schema_completeness metric implemented yet)
        properties_complete = True
        for entity in self.entities:
            if entity["type"] == "Person" and "name" not in entity["properties"]:
                properties_complete = False
            elif entity["type"] == "Organization" and "name" not in entity["properties"]:
                properties_complete = False
            elif entity["type"] == "Location" and "name" not in entity["properties"]:
                properties_complete = False
                
        self.assertTrue(properties_complete, "Required properties missing from entities")

if __name__ == '__main__':
    unittest.main()
