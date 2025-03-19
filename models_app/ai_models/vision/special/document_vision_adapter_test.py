"""
Tests for Document Vision Adapter KG Integration

This module tests the Knowledge Graph integration capabilities of the DocumentVisionAdapter.
"""

import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from models_app.ai_models.vision.special.document_vision_adapter import DocumentVisionAdapter


class DocumentVisionAdapterKGTest(unittest.TestCase):
    """Test suite for DocumentVisionAdapter KG integration functionality."""

    def setUp(self):
        """Set up test environment."""
        self.adapter = DocumentVisionAdapter()
        # Create a mock PDF file
        self.test_pdf_path = os.path.join(tempfile.gettempdir(), "test_document.pdf")
        
        # Mock the successful document processing result
        self.mock_processed_doc = {
            "success": True,
            "text": "This is a test document with key information.",
            "metadata": {
                "creation_date": "2023-01-01",
                "author": "Test Author"
            },
            "images": [
                {"image_data": b"mock_image_data", "page_num": 1, "position": (10, 10, 100, 100)}
            ]
        }

    @patch('models_app.ai_models.vision.document_vision_adapter.KAGBuilder')
    def test_process_for_knowledge_graph(self, mock_kag_builder):
        """Test knowledge graph processing."""
        # Configure the mock KAGBuilder
        mock_builder_instance = MagicMock()
        mock_kag_builder.return_value = mock_builder_instance
        
        # Mock the build_knowledge_base method
        mock_builder_instance.build_knowledge_base.return_value = {
            "graph_id": "test_graph_123",
            "entities": 10,
            "relationships": 5
        }
        
        # Mock the process_document method
        self.adapter.process_document = MagicMock(return_value=self.mock_processed_doc)
        
        # Call the method under test
        result = self.adapter.process_for_knowledge_graph(self.test_pdf_path)
        
        # Assert the expected outcomes
        self.assertTrue(result["success"])
        self.assertEqual(result["graph_id"], "test_graph_123")
        self.assertEqual(result["document"], self.mock_processed_doc)
        
        # Verify KAGBuilder was called correctly
        mock_builder_instance.build_knowledge_base.assert_called_once()
        # Verify the document data passed to KAGBuilder
        call_args = mock_builder_instance.build_knowledge_base.call_args[1]
        self.assertEqual(len(call_args["documents"]), 1)
        self.assertEqual(call_args["documents"][0]["content"], "This is a test document with key information.")

    @patch('models_app.ai_models.vision.document_vision_adapter.KAGBuilder')
    @patch('models_app.ai_models.vision.document_vision_adapter.RAGManager')
    @patch('models_app.ai_models.vision.document_vision_adapter.KnowledgeGraphManager')
    @patch('models_app.ai_models.vision.document_vision_adapter.BidirectionalIndexer')
    def test_process_hybrid(self, mock_bi_indexer, mock_kg_manager, mock_rag_manager, mock_kag_builder):
        """Test hybrid processing with KG and RAG."""
        # Configure mock KAGBuilder
        mock_builder_instance = MagicMock()
        mock_kag_builder.return_value = mock_builder_instance
        mock_builder_instance.build_knowledge_base.return_value = {
            "graph_id": "test_graph_123",
            "entities": 10,
            "relationships": 5
        }
        
        # Configure mock RAG components
        mock_rag_instance = MagicMock()
        mock_rag_manager.return_value = mock_rag_instance
        mock_rag_model = MagicMock()
        mock_rag_instance.get_model.return_value = mock_rag_model
        mock_rag_model.add_document.return_value = {"doc_id": "test_doc_456"}
        mock_rag_model.vectorstore = MagicMock()
        
        # Configure KG manager and bidirectional indexer
        mock_kg_instance = MagicMock()
        mock_kg_manager.return_value = mock_kg_instance
        mock_indexer_instance = MagicMock()
        mock_bi_indexer.return_value = mock_indexer_instance
        mock_indexer_instance.link_document_to_graph.return_value = {"linked": True}
        
        # Mock the process document methods
        self.adapter.process_document = MagicMock(return_value=self.mock_processed_doc)
        # Create a modified version of the process_for_knowledge_graph method result
        kg_result = {
            "success": True,
            "document": self.mock_processed_doc,
            "knowledge_graph": {
                "graph_id": "test_graph_123",
                "entities": 10,
                "relationships": 5
            },
            "graph_id": "test_graph_123"
        }
        self.adapter.process_for_knowledge_graph = MagicMock(return_value=kg_result)
        
        # Call the method under test
        result = self.adapter.process_hybrid(self.test_pdf_path, "test_session")
        
        # Assert the expected outcomes
        self.assertTrue(result["success"])
        self.assertEqual(result["document"], self.mock_processed_doc)
        self.assertEqual(result["knowledge_graph"]["graph_id"], "test_graph_123")
        self.assertEqual(result["rag_index"]["session_id"], "test_session")
        self.assertTrue(result["rag_index"]["indexed"])
        self.assertEqual(result["rag_index"]["index_info"]["doc_id"], "test_doc_456")
        self.assertEqual(result["linking"]["linked"], True)
        
        # Verify RAG model was called correctly
        mock_rag_model.add_document.assert_called_once()
        # Verify bidirectional indexer was called correctly
        mock_indexer_instance.link_document_to_graph.assert_called_once_with(
            self.test_pdf_path, "test_graph_123"
        )

    @patch('models_app.ai_models.vision.document_vision_adapter.KAGBuilder', side_effect=ImportError("KAGBuilder not available"))
    def test_kg_fallback(self, mock_kag_builder):
        """Test fallback to vision module KG extraction when KAGBuilder is not available."""
        # Mock the process_document method
        self.adapter.process_document = MagicMock(return_value=self.mock_processed_doc)
        
        # Mock the DocumentKnowledgeExtractor
        with patch('models_app.vision.knowledge_graph.DocumentKnowledgeExtractor') as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_from_document.return_value = [
                {"entity": "key information", "type": "concept"}
            ]
            
            # Call the method under test
            result = self.adapter.process_for_knowledge_graph(self.test_pdf_path)
            
            # Assert the expected outcomes
            self.assertTrue(result["success"])
            self.assertEqual(result["document"], self.mock_processed_doc)
            self.assertIn("knowledge_elements", result)
            self.assertEqual(len(result["knowledge_elements"]), 1)
            self.assertEqual(result["knowledge_elements"][0]["entity"], "key information")
            
            # Verify extractor was called correctly
            mock_extractor.extract_from_document.assert_called_once()


if __name__ == '__main__':
    unittest.main() 