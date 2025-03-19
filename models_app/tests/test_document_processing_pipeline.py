import unittest
import os
import tempfile
from PIL import Image
import numpy as np
from django.test import TestCase
from django.conf import settings

from models_app.vision.document.factory.document_adapter_registry import DocumentAdapterRegistry
from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
from models_app.knowledge_graph.graph_visualization import GraphVisualization

class DocumentProcessingPipelineTest(TestCase):
    """Test the complete document processing pipeline from document to knowledge graph"""
    
    def setUp(self):
        """Set up test environment with sample documents"""
        self.registry = DocumentAdapterRegistry()
        self.kg_manager = KnowledgeGraphManager()
        
        # Create sample documents for testing
        self.test_files = {}
        
        # Create a text document
        self.test_doc = os.path.join(settings.BASE_DIR, 'models_app/tests/test_data/test_doc.txt')
        if not os.path.exists(os.path.dirname(self.test_doc)):
            os.makedirs(os.path.dirname(self.test_doc))
        if not os.path.exists(self.test_doc):
            with open(self.test_doc, 'w') as f:
                f.write("Dies ist ein Testdokument für die Dokumentenverarbeitungspipeline.")
        self.test_files['text'] = self.test_doc
            
        # Create a simple image with text
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
            img = Image.new('RGB', (300, 100), color='white')
            self.test_files['image'] = img_file.name
            img.save(img_file.name)
            
        # Create a PDF with mixed content (if possible in test environment)
        # This part might require external libraries or mock objects
        self.test_files['mixed'] = None  # Placeholder
    
    def tearDown(self):
        """Clean up test files"""
        for file_path in self.test_files.values():
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
    
    def test_document_adapter_selection(self):
        """Test that correct adapters are selected for different document types"""
        text_adapter = self.registry.get_adapter_for_document(self.test_files['text'])
        self.assertEqual(text_adapter.__class__.__name__, "UniversalDocumentAdapter")
        
        if self.test_files['image']:
            image_adapter = self.registry.get_adapter_for_document(self.test_files['image'])
            self.assertEqual(image_adapter.__class__.__name__, "ImageDocumentAdapter")
    
    def test_document_extraction_preparation(self):
        """Test preparation for extraction"""
        text_adapter = self.registry.get_adapter_for_document(self.test_files['text'])
        extraction_data = text_adapter.prepare_for_extraction(self.test_files['text'])
        
        # Check expected structure
        self.assertIn("document_id", extraction_data)
        self.assertIn("content", extraction_data)
        self.assertIn("metadata", extraction_data)
    
    def test_end_to_end_processing(self):
        """Test complete pipeline from document to knowledge graph"""
        # Process text document to graph
        result = self.kg_manager.process_document_to_graph(self.test_files['text'])
        
        # Verify expected output
        self.assertIn("graph_id", result)
        self.assertIn("entity_count", result)
        self.assertIn("relationship_count", result)
        
        # Retrieve graph and verify entities
        graph = self.kg_manager.graph_storage.retrieve_graph(result["graph_id"])
        
        # A proper text document should extract at least some entities
        self.assertGreater(len(graph.get("entities", [])), 0)
        
        # Test visualization generation
        visualizer = GraphVisualization()
        html_viz = visualizer.create_html_visualization(graph)
        self.assertIsNotNone(html_viz)
        
    def test_bidirectional_indexing(self):
        """Test bidirectional indexing between entities and documents"""
        # Process document to graph
        result = self.kg_manager.process_document_to_graph(self.test_files['text'])
        graph_id = result["graph_id"]
        
        # Test entity to document lookup
        from models_app.document_indexer import BidirectionalIndexer
        bidirectional_indexer = BidirectionalIndexer(
            self.kg_manager.graph_storage, 
            self.kg_manager._get_vector_db()
        )
        
        # Get an entity from the graph
        graph = self.kg_manager.graph_storage.retrieve_graph(graph_id)
        if graph.get("entities"):
            entity_id = graph["entities"][0]["id"]
            related_docs = bidirectional_indexer.find_documents_for_entity(entity_id)
            # We should find at least the document we just processed
            self.assertGreaterEqual(len(related_docs), 1)
