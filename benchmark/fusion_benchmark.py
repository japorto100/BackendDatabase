"""
Fusion Benchmark Service

Provides specialized benchmarking capabilities for fusion strategies.
"""

import time
import logging
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import json
from django.conf import settings

from models_app.vision.fusion.hybrid_fusion import HybridFusion
from models_app.vision.fusion.base import EarlyFusion, LateFusion, AttentionFusion
from models_app.vision.colpali.processor import ColPaliProcessor
from models_app.vision.ocr.ocr_model_selector import OCRModelSelector

logger = logging.getLogger(__name__)

class FusionBenchmarkRunner:
    """Service for benchmarking fusion strategies"""
    
    def __init__(self, user=None):
        self.user = user
        self.hybrid_fusion = HybridFusion()
        self.colpali_processor = ColPaliProcessor()
        self.ocr_selector = OCRModelSelector()
        
        # Initialize test data
        self.test_documents = self._load_test_documents()
    
    def _load_test_documents(self) -> Dict[str, List[str]]:
        """Load test document paths by type"""
        # In a real implementation, this would load actual test documents
        # For now, we'll use a simple dictionary with placeholder paths
        
        # Check if test documents directory exists
        test_docs_dir = getattr(settings, 'TEST_DOCUMENTS_DIR', 'test_documents')
        if not os.path.exists(test_docs_dir):
            os.makedirs(test_docs_dir, exist_ok=True)
            logger.warning(f"Created test documents directory: {test_docs_dir}")
        
        # Return dictionary of document types and their paths
        # In a real implementation, this would scan the directory for actual files
        return {
            "academic": [os.path.join(test_docs_dir, "academic_sample.pdf")],
            "business": [os.path.join(test_docs_dir, "business_sample.pdf")],
            "general": [os.path.join(test_docs_dir, "general_sample.pdf")],
            "invoice": [os.path.join(test_docs_dir, "invoice_sample.pdf")],
            "scientific": [os.path.join(test_docs_dir, "scientific_sample.pdf")],
            "legal": [os.path.join(test_docs_dir, "legal_sample.pdf")],
            "technical": [os.path.join(test_docs_dir, "technical_sample.pdf")],
            "presentation": [os.path.join(test_docs_dir, "presentation_sample.pdf")],
        }
    
    def run_fusion_benchmark(self, strategy_name: str, document_type: str) -> Dict[str, Any]:
        """
        Run a benchmark for a specific fusion strategy on a specific document type
        
        Args:
            strategy_name: Name of the fusion strategy ('early', 'late', 'attention', 'hybrid')
            document_type: Type of document to test ('academic', 'business', 'general', etc.)
            
        Returns:
            Dict: Benchmark results
        """
        # Get test document for the specified type
        if document_type not in self.test_documents or not self.test_documents[document_type]:
            # Use a mock document path if no real document is available
            document_path = f"mock_{document_type}_document.pdf"
            logger.warning(f"No test document found for type {document_type}. Using mock path.")
        else:
            document_path = self.test_documents[document_type][0]
        
        # Create document metadata
        metadata = {
            "document_type": document_type,
            "benchmark": True
        }
        
        # Process document with ColPali (or use mock data if document doesn't exist)
        try:
            if os.path.exists(document_path):
                visual_features = self.colpali_processor.process_image(document_path)
            else:
                # Generate mock visual features for testing
                visual_features = {
                    "features": np.random.rand(10, 512).astype(np.float32),
                    "attention_map": np.random.rand(7, 7).astype(np.float32)
                }
        except Exception as e:
            logger.error(f"Error processing document with ColPali: {str(e)}")
            visual_features = {
                "features": np.random.rand(10, 512).astype(np.float32),
                "attention_map": np.random.rand(7, 7).astype(np.float32)
            }
        
        # Process document with OCR (or use mock data if document doesn't exist)
        try:
            if os.path.exists(document_path):
                ocr_model = self.ocr_selector.select_model(document_path)
                text_features = ocr_model.process_image(document_path)
            else:
                # Generate mock text features for testing
                text_features = {
                    "text": f"This is a mock {document_type} document for testing.",
                    "features": np.random.rand(5, 768).astype(np.float32),
                    "confidence": 0.85
                }
        except Exception as e:
            logger.error(f"Error processing document with OCR: {str(e)}")
            text_features = {
                "text": f"This is a mock {document_type} document for testing.",
                "features": np.random.rand(5, 768).astype(np.float32),
                "confidence": 0.85
            }
        
        # Run the benchmark based on the strategy
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        if strategy_name == "early":
            strategy = self.hybrid_fusion.fusion_strategies["early"]
            fused_features = strategy.fuse(visual_features, text_features, metadata)
            confidence = strategy.get_confidence(visual_features, text_features)
            best_strategy = "early"
        elif strategy_name == "late":
            strategy = self.hybrid_fusion.fusion_strategies["late"]
            fused_features = strategy.fuse(visual_features, text_features, metadata)
            confidence = strategy.get_confidence(visual_features, text_features)
            best_strategy = "late"
        elif strategy_name == "attention":
            strategy = self.hybrid_fusion.fusion_strategies["attention"]
            fused_features = strategy.fuse(visual_features, text_features, metadata)
            confidence = strategy.get_confidence(visual_features, text_features)
            best_strategy = "attention"
        elif strategy_name == "hybrid":
            fused_features, best_strategy, confidence = self.hybrid_fusion.fuse_with_best_strategy(
                visual_features, text_features, metadata
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy_name}")
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        memory_usage = end_memory - start_memory
        
        # Calculate quality score (in a real implementation, this would be more sophisticated)
        quality_score = self.hybrid_fusion._calculate_quality_metric(
            fused_features, visual_features, text_features
        )
        
        # Return benchmark results
        return {
            "strategy": strategy_name,
            "document_type": document_type,
            "processing_time_ms": processing_time,
            "memory_usage_mb": memory_usage,
            "quality_score": quality_score,
            "confidence_score": confidence,
            "best_strategy": best_strategy,
            "strategy_confidence": confidence
        }
    
    def run_document_type_benchmark(self, document_type: str) -> Dict[str, Any]:
        """
        Run a benchmark for hybrid fusion on a specific document type
        
        Args:
            document_type: Type of document to test
            
        Returns:
            Dict: Benchmark results
        """
        return self.run_fusion_benchmark("hybrid", document_type)
    
    def get_strategy_comparison(self, document_type: str = None) -> Dict[str, Dict[str, float]]:
        """
        Get comparison data for all fusion strategies
        
        Args:
            document_type: Optional document type to filter results
            
        Returns:
            Dict: Comparison data for visualization
        """
        # Get performance statistics from hybrid fusion
        stats = self.hybrid_fusion.get_performance_statistics()
        
        # If document type is specified, get document-specific stats
        if document_type:
            doc_stats = self.hybrid_fusion.get_performance_by_document_type()
            if document_type in doc_stats:
                return doc_stats[document_type]
        
        return stats
    
    def get_document_type_comparison(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get comparison data for document types
        
        Returns:
            Dict: Document type comparison data for visualization
        """
        return self.hybrid_fusion.get_performance_by_document_type()
    
    def get_strategy_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get strategy recommendations for different document types
        
        Returns:
            Dict: Strategy recommendations
        """
        return self.hybrid_fusion.analyze_document_type_performance() 