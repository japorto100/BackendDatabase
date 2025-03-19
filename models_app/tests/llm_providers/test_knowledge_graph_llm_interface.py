"""
Test suite for the KnowledgeGraphLLMInterface.

This module tests the integration between knowledge graphs and LLMs,
including graph-augmented prompting and response validation.
"""

import unittest
from unittest.mock import MagicMock, patch
import json

from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface
from models_app.knowledge_graph.graph_storage import GraphStorage
from models_app.knowledge_graph.external_kb_connector import CascadingKBConnector
from models_app.llm_providers.llm_factory import LLMFactory

class TestKnowledgeGraphLLMInterface(unittest.TestCase):
    """Test cases for the KnowledgeGraphLLMInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_factory = MagicMock(spec=LLMFactory)
        self.graph_storage = MagicMock(spec=GraphStorage)
        self.kb_connector = MagicMock(spec=CascadingKBConnector)
        
        # Create mock LLM provider
        self.mock_llm_provider = MagicMock()
        self.llm_factory.get_provider.return_value = self.mock_llm_provider
        
        # Initialize interface with mocks
        self.interface = KnowledgeGraphLLMInterface(
            llm_factory=self.llm_factory,
            graph_storage=self.graph_storage,
            kb_connector=self.kb_connector
        )
        
        # Sample test graph
        self.test_graph = {
            "entities": [
                {
                    "id": "E001",
                    "type": "Person",
                    "label": "John Smith",
                    "properties": {
                        "age": 42,
                        "occupation": "Software Engineer",
                        "email": "john.smith@example.com"
                    }
                },
                {
                    "id": "E002",
                    "type": "Organization",
                    "label": "Acme Corporation",
                    "properties": {
                        "founded": "1990",
                        "industry": "Technology",
                        "location": "Zurich, Switzerland"
                    }
                },
                {
                    "id": "E003",
                    "type": "Project",
                    "label": "Knowledge Graph System",
                    "properties": {
                        "start_date": "2023-01-15",
                        "status": "In Progress",
                        "budget": 500000
                    }
                }
            ],
            "relationships": [
                {
                    "id": "R001",
                    "type": "WORKS_FOR",
                    "source": "E001",
                    "target": "E002",
                    "properties": {
                        "since": "2015-03-01",
                        "position": "Senior Engineer"
                    }
                },
                {
                    "id": "R002",
                    "type": "MANAGES",
                    "source": "E001",
                    "target": "E003",
                    "properties": {
                        "role": "Project Manager",
                        "appointed": "2023-01-15"
                    }
                },
                {
                    "id": "R003",
                    "type": "FUNDS",
                    "source": "E002",
                    "target": "E003",
                    "properties": {
                        "amount": 500000,
                        "approved_date": "2022-12-10"
                    }
                }
            ]
        }
    
    def test_extract_relevant_graph_data(self):
        """Test extracting relevant graph data from a query."""
        # Set up mock for analyze_query
        self.interface.analyze_query = MagicMock(return_value={
            "key_terms": ["John", "Smith", "work"],
            "entities": [{"text": "John Smith", "type": "person", "confidence": 0.9}]
        })
        
        # Extract relevant data
        result = self.interface._extract_relevant_graph_data(
            "Who does John Smith work for?", 
            self.test_graph
        )
        
        # Verify results
        self.assertIn("entities", result)
        self.assertIn("relationships", result)
        self.assertIn("entity_relevance", result)
        
        # Check if John Smith entity was found
        john_entity = next((e for e in result["entities"] if e["label"] == "John Smith"), None)
        self.assertIsNotNone(john_entity)
        
        # Check if WORKS_FOR relationship was found
        works_for_rel = next((r for r in result["relationships"] if r["type"] == "WORKS_FOR"), None)
        self.assertIsNotNone(works_for_rel)
    
    def test_calculate_entity_relevance(self):
        """Test calculating entity relevance scores."""
        # Test data
        entity = {
            "id": "E001",
            "type": "Person",
            "label": "John Smith",
            "properties": {
                "occupation": "Software Engineer",
                "age": 42
            }
        }
        key_terms = ["John", "engineer", "work"]
        entity_mentions = [{"text": "John", "type": "person", "confidence": 0.8}]
        
        # Calculate relevance
        score = self.interface._calculate_entity_relevance(entity, key_terms, entity_mentions)
        
        # Verify score is reasonable
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
    
    def test_construct_graph_augmented_prompt(self):
        """Test constructing a graph-augmented prompt."""
        query = "Who does John Smith work for?"
        graph_data = {
            "entities": [
                {
                    "id": "E001",
                    "label": "John Smith",
                    "type": "Person",
                    "properties": {"occupation": "Engineer"}
                },
                {
                    "id": "E002",
                    "label": "Acme Corp",
                    "type": "Organization",
                    "properties": {"industry": "Technology"}
                }
            ],
            "relationships": [
                {
                    "source": "E001",
                    "target": "E002",
                    "type": "WORKS_FOR"
                }
            ]
        }
        
        # Construct prompt
        prompt = self.interface._construct_graph_augmented_prompt(query, graph_data)
        
        # Verify prompt
        self.assertIn(query, prompt)
        self.assertIn("John Smith", prompt)
        self.assertIn("Acme Corp", prompt)
        self.assertIn("WORKS_FOR", prompt)
    
    def test_generate_graph_augmented_response(self):
        """Test generating a graph-augmented response."""
        # Configure mocks
        self.graph_storage.retrieve_graph.return_value = self.test_graph
        self.mock_llm_provider.generate_text.return_value = "John Smith works for Acme Corporation as a Senior Engineer since 2015."
        
        # Call method
        result = self.interface.generate_graph_augmented_response(
            "Who does John Smith work for?",
            "graph-123"
        )
        
        # Verify result
        self.assertIn("response", result)
        self.assertIn("validation", result)
        self.assertIn("metadata", result)
        self.assertEqual(result["response"], "John Smith works for Acme Corporation as a Senior Engineer since 2015.")
    
    def test_validate_response_against_graph(self):
        """Test validating a response against graph data."""
        # Mock extract_claims_from_response
        self.interface._extract_claims_from_response = MagicMock(return_value=[
            "John Smith works for Acme Corporation",
            "John Smith is 42 years old",
            "John Smith lives in New York"  # This claim is not in the graph
        ])
        
        # Test validation
        result = self.interface._validate_response_against_graph(
            "John Smith works for Acme Corporation. He is 42 years old and lives in New York.",
            self.test_graph
        )
        
        # Verify validation results
        self.assertIn("status", result)
        self.assertIn("verified_claims", result)
        self.assertIn("unverified_claims", result)
        self.assertIn("contradicted_claims", result)
        
        # Should have 2 verified claims and 1 unverified
        self.assertEqual(len(result["verified_claims"]), 2)
        self.assertEqual(len(result["unverified_claims"]), 1)
    
    def test_validate_claim(self):
        """Test validating a single claim against graph data."""
        # Test with a claim that should be verified
        verified_result = self.interface._validate_claim(
            "John Smith is a Software Engineer",
            self.test_graph
        )
        
        # Test with a claim that should be unverified
        unverified_result = self.interface._validate_claim(
            "John Smith lives in Berlin",
            self.test_graph
        )
        
        # Test with a claim that should be contradicted
        contradicted_result = self.interface._validate_claim(
            "John Smith is 35 years old",
            self.test_graph
        )
        
        # Verify results
        self.assertEqual(verified_result["status"], "verified")
        self.assertEqual(unverified_result["status"], "unverified")
        self.assertEqual(contradicted_result["status"], "contradicted")
    
    def test_extract_claims_from_response(self):
        """Test extracting claims from a response."""
        # Configure mock
        self.mock_llm_provider.generate_text.return_value = """
        Here are the claims:
        - John Smith works for Acme Corporation
        - Acme Corporation is located in Zurich
        - John Smith manages the Knowledge Graph System project
        """
        
        # Extract claims
        claims = self.interface._extract_claims_from_response(
            "John Smith works for Acme Corporation, which is located in Zurich. He manages the Knowledge Graph System project."
        )
        
        # Verify claims
        self.assertEqual(len(claims), 3)
        self.assertIn("John Smith works for Acme Corporation", claims)
        self.assertIn("Acme Corporation is located in Zurich", claims)
    
    def test_analyze_query(self):
        """Test analyzing a user query."""
        # Configure mock
        self.mock_llm_provider.generate_text.return_value = json.dumps({
            "key_terms": ["John", "Smith", "work"],
            "entities": [{"text": "John Smith", "type": "person", "confidence": 0.9}]
        })
        
        # Analyze query
        result = self.interface.analyze_query("Who does John Smith work for?")
        
        # Verify result
        self.assertIn("key_terms", result)
        self.assertIn("entities", result)
        self.assertIn("John", result["key_terms"])
        self.assertEqual(result["entities"][0]["text"], "John Smith")
    
    def test_extract_key_terms(self):
        """Test extracting key terms from text."""
        terms = self.interface._extract_key_terms("Who does John Smith work for at Acme Corporation?")
        
        # Verify terms
        self.assertIn("John", terms)
        self.assertIn("Smith", terms)
        self.assertIn("work", terms)
        self.assertIn("Acme", terms)
        self.assertIn("Corporation", terms)
        
        # Should not include stop words
        self.assertNotIn("who", terms)
        self.assertNotIn("does", terms)
        self.assertNotIn("for", terms)
        self.assertNotIn("at", terms)
    
    def test_extract_entity_mentions(self):
        """Test extracting entity mentions from text."""
        mentions = self.interface._extract_entity_mentions("Who does John Smith work for at Acme Corporation?")
        
        # Verify mentions
        self.assertEqual(len(mentions), 2)
        
        # Check for John Smith entity
        john_mention = next((m for m in mentions if m["text"] == "John Smith"), None)
        self.assertIsNotNone(john_mention)
        
        # Check for Acme Corporation entity
        acme_mention = next((m for m in mentions if m["text"] == "Acme Corporation"), None)
        self.assertIsNotNone(acme_mention)

if __name__ == '__main__':
    unittest.main()
