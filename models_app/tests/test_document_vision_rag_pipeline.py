"""
Test script for the complete Document Vision RAG Pipeline.

This script tests the integration between document processing, vision analysis,
indexing, and retrieval in a complete end-to-end workflow.
"""

import os
import sys
import logging
import unittest
import tempfile
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models_app.vision.document.factory.document_adapter_registry import DocumentAdapterRegistry
from models_app.vision.document.adapters.universal_document_adapter import UniversalDocumentAdapter
from models_app.indexing.document_indexer import DocumentIndexer
from models_app.indexing.rag_manager import RAGModelManager
from models_app.multimodal.multimodal_responder import MultimodalResponder
from models_app.multimodal.m_retriever import MultimodalRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestDocumentVisionRAGPipeline(unittest.TestCase):
    """Test the complete Document Vision RAG pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name
        
        # Create test documents
        self.doc_paths = self._create_test_documents()
        
        # Initialize session ID for test
        self.session_id = "test_session_123"
        
        # Clean up any previous test data
        rag_manager = RAGModelManager()
        if rag_manager.model_exists(self.session_id):
            rag_manager.delete_model(self.session_id)
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up test RAG model
        rag_manager = RAGModelManager()
        if rag_manager.model_exists(self.session_id):
            rag_manager.delete_model(self.session_id)
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def _create_test_documents(self):
        """Create test documents for the pipeline"""
        doc_paths = []
        
        # Create a text file
        text_path = os.path.join(self.test_dir, "test_document.txt")
        with open(text_path, "w") as f:
            f.write("This is a test document about artificial intelligence.\n")
            f.write("AI technologies are transforming the world.\n")
            f.write("Machine learning is a subset of AI that focuses on data-driven algorithms.\n")
        doc_paths.append(text_path)
        
        # Create a simple "PDF-like" image with text
        img_path = os.path.join(self.test_dir, "test_invoice.png")
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to get a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw text on the image
        draw.text((50, 50), "INVOICE", fill="black", font=font)
        draw.text((50, 100), "Invoice #: INV-2023-001", fill="black", font=font)
        draw.text((50, 140), "Date: 2023-03-15", fill="black", font=font)
        draw.text((50, 180), "Customer: Acme Corporation", fill="black", font=font)
        draw.text((50, 240), "Description: AI Software License", fill="black", font=font)
        draw.text((50, 280), "Amount: $1,500.00", fill="black", font=font)
        draw.text((50, 400), "Thank you for your business!", fill="black", font=font)
        
        img.save(img_path)
        doc_paths.append(img_path)
        
        return doc_paths
    
    def test_full_pipeline(self):
        """Test the complete pipeline from document processing to retrieval"""
        logger.info("Starting full pipeline test")
        
        # Step 1: Process documents using the appropriate adapters
        processed_docs = []
        registry = DocumentAdapterRegistry()
        
        for doc_path in self.doc_paths:
            logger.info(f"Processing document: {doc_path}")
            adapter = registry.get_adapter_for_document(doc_path)
            self.assertIsNotNone(adapter, f"No adapter found for {doc_path}")
            
            processed_doc = adapter.process_document(doc_path)
            self.assertIsNotNone(processed_doc, f"Document processing failed for {doc_path}")
            
            processed_docs.append(processed_doc)
            
            logger.info(f"Document processed. Content length: {len(processed_doc.content)}")
            logger.info(f"Metadata: {processed_doc.metadata}")
        
        # Step 2: Index the processed documents
        logger.info("Indexing documents")
        indexer = DocumentIndexer()
        
        # Use the index_documents_for_session method to index documents
        success, message, indexed_files = indexer.index_documents_for_session(
            self.session_id, 
            self.doc_paths
        )
        
        self.assertTrue(success, f"Indexing failed: {message}")
        self.assertEqual(len(self.doc_paths), len(indexed_files), "Not all files were indexed")
        logger.info(f"Indexing result: {message}, Files indexed: {indexed_files}")
        
        # Step 3: Test the RAG model retrieval
        logger.info("Testing RAG retrieval")
        rag_manager = RAGModelManager()
        rag_model = rag_manager.get_model(self.session_id)
        
        self.assertIsNotNone(rag_model, "Failed to retrieve RAG model")
        
        # Test a basic query
        query = "What is mentioned about AI in the documents?"
        results = rag_model.similarity_search(query, k=3)
        
        self.assertGreater(len(results), 0, "No results returned from RAG query")
        logger.info(f"RAG query returned {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result.page_content[:100]}...")
        
        # Step 4: Test multimodal search
        logger.info("Testing multimodal search")
        retriever = MultimodalRetriever()
        
        # Test with text query
        text_results = retriever.search_text(
            query="AI technologies", 
            max_results=3
        )
        
        self.assertGreater(len(text_results), 0, "No results from text search")
        logger.info(f"Text search returned {len(text_results)} results")
        
        # Test with image (if we have one)
        if any(p.endswith(('.png', '.jpg', '.jpeg', '.gif')) for p in self.doc_paths):
            image_path = next(p for p in self.doc_paths if p.endswith(('.png', '.jpg', '.jpeg', '.gif')))
            
            # Only test this if we have both image and text
            image_results = retriever.search_with_image(
                text_query="invoice", 
                image_path=image_path,
                max_results=3
            )
            
            logger.info(f"Image + text search returned {len(image_results)} results")
        
        # Step 5: Test multimodal responding
        logger.info("Testing multimodal responding")
        responder = MultimodalResponder()
        
        # Only test with images if we have them
        if any(p.endswith(('.png', '.jpg', '.jpeg', '.gif')) for p in self.doc_paths):
            image_path = next(p for p in self.doc_paths if p.endswith(('.png', '.jpg', '.jpeg', '.gif')))
            
            # Test the lightweight model to avoid external API calls
            response, used_images = responder.generate_response(
                [image_path],
                "What information do you see in this image?",
                model_choice="lightweight"
            )
            
            self.assertIsNotNone(response, "No response from multimodal responder")
            self.assertGreater(len(response), 0, "Empty response from multimodal responder")
            logger.info(f"Multimodal response: {response[:100]}...")
            
            # Test image analysis
            analysis = responder.analyze_image_content(
                image_path,
                analysis_type="document"
            )
            
            self.assertIn('analysis', analysis, "No analysis result returned")
            logger.info(f"Image analysis: {analysis['analysis'][:100]}...")
        
        logger.info("Full pipeline test completed successfully")


if __name__ == "__main__":
    unittest.main()
