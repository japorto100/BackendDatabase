import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from .size_based import ChunkMetadata, SizeBasedChunker

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of processing a chunk."""
    chunk_metadata: ChunkMetadata
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ChunkingOrchestrator:
    """Orchestrates document chunking and processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize orchestrator with configuration."""
        self.config = config or {}
        self.chunker = SizeBasedChunker(config)
        self.max_workers = self.config.get("max_workers", 4)
    
    def process_document(
        self,
        document_path: str,
        processor_func: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        processor_config: Optional[Dict[str, Any]] = None,
        merge_results: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process document with optional chunking.
        
        Args:
            document_path: Path to document
            processor_func: Function to process each chunk
            processor_config: Configuration for processor
            merge_results: Whether to merge processed chunks
            
        Returns:
            Tuple of (success, results)
        """
        try:
            # Check if chunking is needed
            should_chunk, reason = self.chunker.should_chunk(document_path)
            logger.info(f"Chunking decision for {document_path}: {should_chunk} ({reason})")
            
            if not should_chunk:
                # Process document directly
                try:
                    result = processor_func(document_path, processor_config or {})
                    return True, {"results": result, "chunked": False}
                except Exception as e:
                    logger.error(f"Direct processing failed: {str(e)}")
                    return False, {"error": str(e), "chunked": False}
            
            # Create chunks
            chunks = self.chunker.create_chunks(document_path)
            logger.info(f"Created {len(chunks)} chunks for {document_path}")
            
            # Process chunks in parallel
            processing_results = self._process_chunks_parallel(
                chunks, processor_func, processor_config or {}
            )
            
            # Check for processing failures
            failed_chunks = [r for r in processing_results if not r.success]
            if failed_chunks:
                error_msg = f"{len(failed_chunks)} chunks failed processing"
                logger.error(error_msg)
                return False, {
                    "error": error_msg,
                    "failed_chunks": [
                        {"index": r.chunk_metadata.chunk_index, "error": r.error}
                        for r in failed_chunks
                    ],
                    "chunked": True
                }
            
            # Merge results if requested
            if merge_results:
                merged_results = self._merge_processing_results(processing_results)
                
                # Merge document chunks if needed
                if self.config.get("merge_chunks", True):
                    output_path = self.config.get(
                        "output_path",
                        f"merged_{document_path}"
                    )
                    merge_success = self.chunker.merge_chunks(chunks, output_path)
                    if not merge_success:
                        logger.warning("Document chunk merging failed")
                
                return True, {
                    "results": merged_results,
                    "chunked": True,
                    "chunk_count": len(chunks)
                }
            else:
                # Return individual chunk results
                return True, {
                    "results": [
                        {
                            "chunk_index": r.chunk_metadata.chunk_index,
                            "result": r.result
                        }
                        for r in processing_results
                    ],
                    "chunked": True,
                    "chunk_count": len(chunks)
                }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return False, {"error": str(e)}
            
        finally:
            # Cleanup chunks
            try:
                self.chunker.cleanup_chunks(chunks)
            except Exception as e:
                logger.error(f"Chunk cleanup failed: {str(e)}")
    
    def _process_chunks_parallel(
        self,
        chunks: List[ChunkMetadata],
        processor_func: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        processor_config: Dict[str, Any]
    ) -> List[ProcessingResult]:
        """Process chunks in parallel using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(
                    self._process_single_chunk,
                    chunk,
                    processor_func,
                    processor_config
                ): chunk
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Chunk {chunk.chunk_index} processing failed: {str(e)}"
                    )
                    results.append(ProcessingResult(
                        chunk_metadata=chunk,
                        success=False,
                        error=str(e)
                    ))
        
        # Sort results by chunk index
        return sorted(results, key=lambda x: x.chunk_metadata.chunk_index)
    
    def _process_single_chunk(
        self,
        chunk: ChunkMetadata,
        processor_func: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        processor_config: Dict[str, Any]
    ) -> ProcessingResult:
        """Process a single chunk and return result."""
        try:
            # Add chunk metadata to processor config
            chunk_config = processor_config.copy()
            chunk_config.update({
                "chunk_metadata": {
                    "index": chunk.chunk_index,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "format": chunk.format
                }
            })
            
            # Process chunk
            result = processor_func(chunk.path, chunk_config)
            
            return ProcessingResult(
                chunk_metadata=chunk,
                success=True,
                result=result
            )
            
        except Exception as e:
            logger.error(f"Chunk processing error: {str(e)}")
            return ProcessingResult(
                chunk_metadata=chunk,
                success=False,
                error=str(e)
            )
    
    def _merge_processing_results(
        self,
        results: List[ProcessingResult]
    ) -> Dict[str, Any]:
        """
        Merge processing results from multiple chunks.
        Override this method for custom merging logic.
        """
        merged = {
            "text": [],
            "metadata": {
                "chunk_count": len(results),
                "total_size": sum(r.chunk_metadata.size_bytes for r in results)
            }
        }
        
        for result in results:
            if result.result:
                # Merge text content
                if "text" in result.result:
                    merged["text"].extend(result.result["text"])
                
                # Merge other fields
                for key, value in result.result.items():
                    if key != "text":
                        if key not in merged:
                            merged[key] = []
                        if isinstance(value, list):
                            merged[key].extend(value)
                        else:
                            merged[key].append(value)
        
        return merged 