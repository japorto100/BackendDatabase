"""
Integrated testing for document processing, knowledge graph, and knowledge base components.

This module provides comprehensive tests for the entire data processing pipeline from
document extraction to knowledge graph generation to knowledge base integration.
"""

import unittest
import os
import json
from unittest.mock import MagicMock, patch
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile

from models_app.document_processing.document_processor import DocumentProcessor
from models_app.document_processing.extraction_pipeline import ExtractionPipeline
from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
from models_app.knowledge_graph.graph_builder import GraphBuilder
from models_app.knowledge_graph.entity_extractor import EntityExtractor
from models_app.knowledge_graph.relationship_detector import RelationshipDetector
from models_app.knowledge_graph.external_kb_connector import CascadingKBConnector
from models_app.llm_providers.knowledge_graph_llm_interface import KnowledgeGraphLLMInterface


class IntegratedPipelineTest(TestCase):
    """Test the integrated processing pipeline from document to knowledge graph to LLM."""
    
    def setUp(self):
        """Set up test environment with mock documents and components."""
        # Create test document content
        self.test_text = """
        Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976.
        The company is headquartered in Cupertino, California and produces popular products
        like the iPhone, iPad, and MacBook. Tim Cook is the current CEO of Apple.
        """
        
        # Create a mock uploaded file
        self.test_file = SimpleUploadedFile(
            "test_document.txt",
            self.test_text.encode('utf-8'),
            content_type="text/plain"
        )
        
        # Test configuration
        self.config = {
            "run_mode": "test",
            "test_mode": True,
            "document_types": ["text", "pdf"],
            "extraction_models": {
                "entity": "test_entity_model",
                "relationship": "test_relation_model"
            }
        }
        
        # Mock the KB connector to avoid external calls
        self.kb_connector_patch = patch('models_app.knowledge_graph.external_kb_connector.CascadingKBConnector')
        self.mock_kb_connector = self.kb_connector_patch.start()
        self.mock_kb_connector.return_value.enrich_entity.return_value = {"enriched": True}
        
        # Mock LLM Factory to avoid actual LLM calls
        self.llm_factory_patch = patch('models_app.llm_providers.llm_factory.LLMFactory')
        self.mock_llm_factory = self.llm_factory_patch.start()
        self.mock_llm_provider = MagicMock()
        self.mock_llm_provider.generate_text.return_value = "This is a test response about Apple Inc."
        self.mock_llm_factory.return_value.get_provider.return_value = self.mock_llm_provider
        
    def tearDown(self):
        """Clean up after tests."""
        self.kb_connector_patch.stop()
        self.llm_factory_patch.stop()
        
        # Clean up test files
        if os.path.exists("test_document.txt"):
            os.remove("test_document.txt")
    
    @patch('models_app.document_processing.document_processor.DocumentProcessor')
    def test_document_to_graph_pipeline(self, mock_doc_processor):
        """Test the pipeline from document processing to knowledge graph creation."""
        # Configure the document processor mock
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_document.return_value = {
            "text": self.test_text,
            "metadata": {"filename": "test_document.txt", "mime_type": "text/plain"},
            "extracted_text": self.test_text,
            "pages": [{"page_num": 1, "text": self.test_text}]
        }
        mock_doc_processor.return_value = mock_processor_instance
        
        # Create pipeline components
        extraction_pipeline = ExtractionPipeline(config=self.config)
        kg_manager = KnowledgeGraphManager()
        
        # STEP 1: Process document
        processed_doc = extraction_pipeline.process_document(self.test_file)
        self.assertIsNotNone(processed_doc)
        self.assertIn("text", processed_doc)
        
        # STEP 2: Extract entities
        with patch.object(EntityExtractor, 'extract_entities') as mock_extract:
            # Mock entity extraction
            mock_extract.return_value = [
                {"id": "e1", "type": "Person", "label": "Steve Jobs", "span": [29, 39]},
                {"id": "e2", "type": "Person", "label": "Steve Wozniak", "span": [44, 57]},
                {"id": "e3", "type": "Organization", "label": "Apple Inc.", "span": [0, 10]},
                {"id": "e4", "type": "Location", "label": "Cupertino", "span": [93, 102]},
                {"id": "e5", "type": "Location", "label": "California", "span": [104, 114]},
                {"id": "e6", "type": "Product", "label": "iPhone", "span": [145, 151]},
                {"id": "e7", "type": "Product", "label": "iPad", "span": [153, 157]},
                {"id": "e8", "type": "Product", "label": "MacBook", "span": [163, 170]},
                {"id": "e9", "type": "Person", "label": "Tim Cook", "span": [172, 180]}
            ]
            
            entities = extraction_pipeline.extract_entities(processed_doc)
            self.assertEqual(len(entities), 9)
            self.assertEqual(entities[0]["label"], "Steve Jobs")
        
        # STEP 3: Extract relationships
        with patch.object(RelationshipDetector, 'detect_relationships') as mock_detect:
            # Mock relationship detection
            mock_detect.return_value = [
                {"id": "r1", "type": "FOUNDED", "source": "e1", "target": "e3", "confidence": 0.9},
                {"id": "r2", "type": "FOUNDED", "source": "e2", "target": "e3", "confidence": 0.9},
                {"id": "r3", "type": "HEADQUARTERED_IN", "source": "e3", "target": "e4", "confidence": 0.85},
                {"id": "r4", "type": "IN", "source": "e4", "target": "e5", "confidence": 0.95},
                {"id": "r5", "type": "PRODUCES", "source": "e3", "target": "e6", "confidence": 0.8},
                {"id": "r6", "type": "PRODUCES", "source": "e3", "target": "e7", "confidence": 0.8},
                {"id": "r7", "type": "PRODUCES", "source": "e3", "target": "e8", "confidence": 0.8},
                {"id": "r8", "type": "CEO_OF", "source": "e9", "target": "e3", "confidence": 0.9}
            ]
            
            relationships = extraction_pipeline.extract_relationships(processed_doc, entities)
            self.assertEqual(len(relationships), 8)
            self.assertEqual(relationships[0]["type"], "FOUNDED")
        
        # STEP 4: Build knowledge graph
        with patch.object(GraphBuilder, 'build_graph') as mock_build:
            # Mock graph building
            test_graph = {
                "id": "test_graph_id",
                "metadata": {"source": "test_document.txt", "created": "2023-03-01T12:00:00Z"},
                "entities": entities,
                "relationships": relationships
            }
            mock_build.return_value = test_graph
            
            graph = kg_manager.build_graph_from_extracted_data(entities, relationships, 
                                                              document_metadata={"filename": "test_document.txt"})
            self.assertIsNotNone(graph)
            self.assertIn("entities", graph)
            self.assertIn("relationships", graph)
            self.assertEqual(len(graph["entities"]), 9)
            self.assertEqual(len(graph["relationships"]), 8)
        
        # STEP 5: Verify the complete pipeline works
        self.assertIsNotNone(graph)
        return graph
    
    def test_knowledge_base_integration(self):
        """Test the integration with external knowledge bases."""
        # First get a graph from the pipeline
        with patch('models_app.document_processing.extraction_pipeline.ExtractionPipeline.process_document'), \
             patch('models_app.document_processing.extraction_pipeline.ExtractionPipeline.extract_entities'), \
             patch('models_app.document_processing.extraction_pipeline.ExtractionPipeline.extract_relationships'), \
             patch('models_app.knowledge_graph.knowledge_graph_manager.KnowledgeGraphManager.build_graph_from_extracted_data'):
            
            # Create test graph
            test_graph = {
                "id": "test_graph_id",
                "entities": [
                    {"id": "e1", "type": "Organization", "label": "Apple Inc."},
                    {"id": "e2", "type": "Person", "label": "Tim Cook"},
                ],
                "relationships": [
                    {"id": "r1", "type": "CEO_OF", "source": "e2", "target": "e1"}
                ]
            }
            
            # Mock the graph creation
            KnowledgeGraphManager.build_graph_from_extracted_data = MagicMock(return_value=test_graph)
            
            # Create KB connector
            kb_connector = CascadingKBConnector()
            
            # Mock the KB methods
            kb_connector.get_external_entity = MagicMock(return_value={
                "id": "wikidata:Q312", 
                "name": "Apple Inc.",
                "description": "American technology company",
                "properties": {
                    "founded": "1976-04-01",
                    "headquarters": "Cupertino, California",
                    "industry": "Technology"
                }
            })
            
            # Test entity enrichment
            for entity in test_graph["entities"]:
                enriched = kb_connector.enrich_entity(entity)
                self.assertTrue("enriched" in enriched)
            
            # Verify the graph was enriched
            self.assertIsNotNone(test_graph)
    
    def test_kg_llm_integration(self):
        """Test the knowledge graph integration with LLMs."""
        # Create a test graph
        test_graph = {
            "id": "test_graph_id",
            "entities": [
                {"id": "e1", "type": "Organization", "label": "Apple Inc.", 
                 "properties": {"founded": "1976", "industry": "Technology"}},
                {"id": "e2", "type": "Person", "label": "Tim Cook",
                 "properties": {"role": "CEO", "joined": "1998"}},
            ],
            "relationships": [
                {"id": "r1", "type": "CEO_OF", "source": "e2", "target": "e1"}
            ]
        }
        
        # Mock the graph storage
        with patch('models_app.knowledge_graph.graph_storage.GraphStorage') as mock_storage:
            mock_storage_instance = MagicMock()
            mock_storage_instance.retrieve_graph.return_value = test_graph
            mock_storage.return_value = mock_storage_instance
            
            # Create KG-LLM interface
            kg_llm = KnowledgeGraphLLMInterface()
            
            # Test generating a response
            query = "Tell me about Apple's CEO"
            response = kg_llm.generate_graph_augmented_response(query, "test_graph_id")
            
            # Verify the response
            self.assertIsNotNone(response)
            self.assertTrue(isinstance(response, dict))
            
            # Check if LLM was called with graph data
            self.mock_llm_provider.generate_text.assert_called_once()
            call_args = self.mock_llm_provider.generate_text.call_args[1]
            prompt = call_args.get('prompt', '')
            
            # The prompt should contain information about Tim Cook and Apple
            self.assertIn('Tim Cook', prompt) 
            self.assertIn('Apple Inc.', prompt)
    
    def test_run_all_components(self):
        """Test switching between different components in a single test."""
        # Create component flags
        test_components = {
            'document_processing': True,
            'knowledge_graph': True,
            'knowledge_base': True,
            'llm_integration': True
        }
        
        results = {}
        
        # Run document processing if enabled
        if test_components['document_processing']:
            with patch('models_app.document_processing.document_processor.DocumentProcessor.process_document'):
                processor = DocumentProcessor()
                processor.process_document.return_value = {
                    "text": self.test_text,
                    "metadata": {"filename": "test_document.txt"}
                }
                
                result = processor.process_document(self.test_file)
                self.assertIsNotNone(result)
                results['document_processing'] = result
        
        # Run knowledge graph building if enabled
        if test_components['knowledge_graph']:
            with patch('models_app.knowledge_graph.knowledge_graph_manager.KnowledgeGraphManager.build_graph_from_extracted_data'):
                kg_manager = KnowledgeGraphManager()
                
                # Mock entities and relationships
                entities = [{"id": "e1", "label": "Apple Inc.", "type": "Organization"}]
                relationships = [{"id": "r1", "source": "e1", "target": "e2", "type": "FOUNDED"}]
                
                kg_manager.build_graph_from_extracted_data.return_value = {
                    "id": "test_graph",
                    "entities": entities,
                    "relationships": relationships
                }
                
                graph = kg_manager.build_graph_from_extracted_data(
                    entities=entities,
                    relationships=relationships,
                    document_metadata={"filename": "test.txt"}
                )
                
                self.assertIsNotNone(graph)
                results['knowledge_graph'] = graph
        
        # Run knowledge base integration if enabled
        if test_components['knowledge_base'] and 'knowledge_graph' in results:
            kb_connector = CascadingKBConnector()
            enriched_graph = results['knowledge_graph'].copy()
            
            # Mock the enrichment
            for entity in enriched_graph["entities"]:
                entity["external_kb"] = {"source": "wikidata", "id": "Q123"}
            
            results['knowledge_base'] = enriched_graph
        
        # Run LLM integration if enabled
        if test_components['llm_integration'] and 'knowledge_graph' in results:
            kg_llm = KnowledgeGraphLLMInterface()
            
            # Mock the response generation
            with patch.object(kg_llm, 'generate_graph_augmented_response') as mock_generate:
                mock_generate.return_value = {
                    "response": "This is a test response about Apple Inc.",
                    "validation": {"status": "verified"}
                }
                
                response = kg_llm.generate_graph_augmented_response(
                    query="Tell me about Apple",
                    graph_id="test_graph_id"
                )
                
                self.assertIsNotNone(response)
                results['llm_integration'] = response
        
        # Verify all requested components were tested
        for component in test_components:
            if test_components[component]:
                self.assertIn(component, results)


if __name__ == '__main__':
    unittest.main() 