# models/indexer.py

import os
import logging
from byaldi import RAGMultiModalModel
from models.converters import convert_docs_to_pdfs
from error_handlers.models_app_errors import handle_indexing_error

logger = logging.getLogger(__name__)

def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali'):
    """
    DEPRECATED: Use index_processed_documents instead.
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update.
        index_path (str): The path where the index should be saved.
        indexer_model (str): The name of the indexer model to use.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    import warnings
    warnings.warn(
        "index_documents is deprecated; use index_processed_documents with document adapters instead",
        DeprecationWarning, 
        stacklevel=2
    )
    
    try:
        logger.info(f"Starting document indexing in folder: {folder_path}")
        # Convert non-PDF documents to PDFs
        convert_docs_to_pdfs(folder_path)
        logger.info("Conversion of non-PDF documents to PDFs completed.")

        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
        logger.info(f"RAG model initialized with {indexer_model}.")

        # Index the documents in the folder
        RAG.index(
            input_path=folder_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )

        logger.info(f"Indexing completed. Index saved at '{index_path}'.")

        return RAG
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise

def index_processed_documents(processed_results, session_id='document_index', index_path=None, 
                              indexer_model='vidore/colpali', embeddings=None):
    """
    Indexes documents that have already been processed by vision adapters.
    
    Args:
        processed_results: List of processed document results from vision adapters
        session_id: The session ID or index name
        index_path: Path to save the index
        indexer_model: The model to use for indexing
        embeddings: Optional pre-configured embeddings model
        
    Returns:
        RAGMultiModalModel: The initialized RAG model
    """
    try:
        logger.info(f"Starting advanced document indexing for session: {session_id}")
        
        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
        
        # Configure embeddings if provided
        if embeddings:
            RAG.set_embeddings(embeddings)
            
        # Process each document result
        for doc_result in processed_results:
            # Extract text content
            text = doc_result.get("text", "")
            
            # Extract metadata and structure
            metadata = doc_result.get("metadata", {})
            structure = doc_result.get("structure", {})
            
            # Add document to index with preserved metadata
            RAG.add_document(
                text=text,
                metadata={
                    **metadata,
                    "document_type": doc_result.get("document_type", "unknown"),
                    "processed_by": doc_result.get("processor", "unknown"),
                    "structure_preserved": True,
                    "has_knowledge_graph": "knowledge_graph" in doc_result
                }
            )
            
            # If document has images, index them as well
            if "images" in doc_result:
                for img_path in doc_result["images"]:
                    RAG.add_image(img_path, metadata=metadata)
        
        # Save the index
        if index_path:
            RAG.save_index(index_path)
            
        logger.info(f"Advanced indexing completed for session {session_id}")
        return RAG
        
    except Exception as e:
        handle_indexing_error(processed_results, indexer_model, e)
        logger.error(f"Error during advanced indexing: {str(e)}")
        raise