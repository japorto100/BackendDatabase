"""
DocumentVisionAdapter Knowledge Graph Integration Demo

This script demonstrates how to use the DocumentVisionAdapter for processing
documents and extracting knowledge graph information.

Usage:
    python document_kg_demo.py <path_to_document>

# KG-Extraktion
python document_kg_demo.py path/to/document.pdf --output-dir=results

# Hybridverarbeitung
python document_kg_demo.py path/to/document.pdf --mode=hybrid --session-id=test-session --output-dir=results
"""

import os
import sys
import json
import logging
from typing import Dict, Any
import argparse

from models_app.ai_models.special.vision.document_vision_adapter import DocumentVisionAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_document_for_kg(document_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Process a document using the DocumentVisionAdapter for knowledge graph extraction.
    
    Args:
        document_path: Path to the document to process
        output_dir: Directory to save outputs (optional)
        
    Returns:
        Dict containing the processing results
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    logger.info(f"Processing document: {document_path}")
    
    # Initialize the adapter
    adapter = DocumentVisionAdapter()
    
    # Process the document for KG extraction
    result = adapter.process_for_knowledge_graph(document_path)
    
    # Save results if output directory is provided
    if output_dir and result.get("success", False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save KG data
        output_file = os.path.join(output_dir, f"{os.path.basename(document_path)}_kg.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved KG extraction results to: {output_file}")
    
    return result


def process_document_hybrid(document_path: str, session_id: str = None, output_dir: str = None) -> Dict[str, Any]:
    """
    Process a document using the hybrid approach (KG + RAG).
    
    Args:
        document_path: Path to the document to process
        session_id: Optional session ID for RAG processing
        output_dir: Directory to save outputs (optional)
        
    Returns:
        Dict containing the processing results
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    logger.info(f"Processing document with hybrid approach: {document_path}")
    
    # Initialize the adapter
    adapter = DocumentVisionAdapter()
    
    # Process with hybrid approach
    result = adapter.process_hybrid(document_path, session_id)
    
    # Save results if output directory is provided
    if output_dir and result.get("success", False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save hybrid processing data
        output_file = os.path.join(output_dir, f"{os.path.basename(document_path)}_hybrid.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved hybrid processing results to: {output_file}")
    
    return result


def display_results(result: Dict[str, Any]):
    """
    Display the processing results in a human-readable format.
    
    Args:
        result: The processing result to display
    """
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    print("\n=== Document Processing Results ===")
    print(f"Document: {result.get('document', {}).get('metadata', {})}")
    
    # Display KG information if available
    if "knowledge_graph" in result:
        kg_data = result["knowledge_graph"]
        print("\n--- Knowledge Graph Information ---")
        print(f"Graph ID: {kg_data.get('graph_id', 'Not available')}")
        print(f"Entities: {kg_data.get('entities', 'Not available')}")
        print(f"Relationships: {kg_data.get('relationships', 'Not available')}")
    
    # Display RAG information if available
    if "rag_index" in result:
        rag_data = result["rag_index"]
        print("\n--- RAG Information ---")
        print(f"Session ID: {rag_data.get('session_id', 'Not available')}")
        print(f"Indexed: {rag_data.get('indexed', False)}")
        if "index_info" in rag_data:
            print(f"Index Info: {rag_data['index_info']}")
    
    # Display linking information if available
    if "linking" in result:
        linking_data = result["linking"]
        print("\n--- Linking Information ---")
        print(f"Linking Status: {linking_data}")
        

def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Document Vision Adapter KG Demo")
    parser.add_argument("document_path", help="Path to the document to process")
    parser.add_argument("--output-dir", help="Directory to save outputs")
    parser.add_argument("--mode", choices=["kg", "hybrid"], default="kg", 
                        help="Processing mode: 'kg' for KG only, 'hybrid' for KG+RAG")
    parser.add_argument("--session-id", help="Session ID for RAG processing (hybrid mode only)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "kg":
            result = process_document_for_kg(args.document_path, args.output_dir)
        else:  # hybrid mode
            result = process_document_hybrid(args.document_path, args.session_id, args.output_dir)
        
        display_results(result)
        
        if result.get("success", False):
            print("\nDocument processed successfully!")
        else:
            print(f"\nDocument processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.exception("Error processing document")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 