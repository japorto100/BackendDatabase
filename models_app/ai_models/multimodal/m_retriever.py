"""
M-Retriever: Multimodal Retrieval for handling images, text, and other modalities
"""

import logging
import numpy as np
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MultimodalRetriever:
    """
    Implements the M-Retriever approach for multimodal retrieval.
    
    This retriever can search across text and images using a unified embedding space.

    => This would be particularly valuable for our construction-related use cases where 
    it might be beneficial to find out errors made by the construction workers.
    """
    
    def __init__(self):
        # Initialize CLIP for image-text embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize text encoder for text-only queries
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def embed_text(self, text):
        """Embed text using CLIP"""
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        outputs = self.clip_model.get_text_features(**inputs)
        return outputs.detach().numpy()
    
    def embed_image(self, image_path):
        """Embed image using CLIP"""
        try:
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            outputs = self.clip_model.get_image_features(**inputs)
            return outputs.detach().numpy()
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return None
    
    def similarity_search(self, query, items, top_k=5):
        """
        Search for similar items to the query
        
        Args:
            query: Text query or image path
            items: List of items (text or image paths)
            top_k: Number of top results to return
            
        Returns:
            List of top_k most similar items with scores
        """
        # Determine query type and embed accordingly
        if query.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            query_embedding = self.embed_image(query)
            query_type = 'image'
        else:
            query_embedding = self.embed_text(query)
            query_type = 'text'
        
        if query_embedding is None:
            return []
            
        # Get embeddings for all items
        item_embeddings = []
        for i, item in enumerate(items):
            # Check cache
            cache_key = str(item)
            if cache_key in self.embedding_cache:
                embedding = self.embedding_cache[cache_key]
            else:
                # Determine item type and embed accordingly
                if isinstance(item, str) and item.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    embedding = self.embed_image(item)
                    item_type = 'image'
                else:
                    embedding = self.embed_text(str(item))
                    item_type = 'text'
                
                if embedding is not None:
                    self.embedding_cache[cache_key] = embedding
            
            if embedding is not None:
                item_embeddings.append((i, embedding, item_type))
        
        # Calculate similarities
        similarities = []
        for i, embedding, item_type in item_embeddings:
            # Calculate cosine similarity
            similarity = np.dot(query_embedding.flatten(), embedding.flatten()) / (
                np.linalg.norm(query_embedding.flatten()) * np.linalg.norm(embedding.flatten())
            )
            
            # Apply cross-modal discount
            if query_type != item_type:
                similarity *= 0.9  # Slight penalty for cross-modal results
                
            similarities.append((i, float(similarity)))
        
        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for i, score in top_results:
            item = items[i]
            result = {
                'item': item,
                'score': score,
                'type': 'image' if isinstance(item, str) and item.endswith(('.jpg', '.jpeg', '.png', '.gif')) else 'text'
            }
            results.append(result)
        
        return results
        
    def search_with_image(self, text_query, image_path, filters=None, max_results=10):
        """
        Search for documents using a combination of image and text query
        
        Args:
            text_query: Text query string
            image_path: Path to the image file
            filters: Optional filters to apply to search
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return []
            
            # Get image embedding
            image_embedding = self.embed_image(image_path)
            if image_embedding is None:
                logger.error(f"Failed to encode image: {image_path}")
                return []
            
            # Get text embedding
            text_embedding = self.embed_text(text_query)
            
            # Combine embeddings (weighted average)
            combined_embedding = 0.7 * text_embedding + 0.3 * image_embedding
            normalized_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            # Get documents to search (this would be integrated with RAG or document store)
            from ..indexing.rag_manager import RAGModelManager
            rag_manager = RAGModelManager()
            
            # Get all sessions with RAG models
            sessions = rag_manager.get_all_sessions()
            
            all_results = []
            
            # Search across all indexed sessions
            for session_id in sessions:
                rag_model = rag_manager.get_model(session_id)
                if rag_model:
                    # Extract documents from RAG model
                    try:
                        # This would depend on how your RAG stores documents
                        # Typically you'd use similarity search directly on the vector store
                        if hasattr(rag_model, 'vectorstore'):
                            # For vector stores that support embedding search
                            results = rag_model.vectorstore.similarity_search_by_vector(
                                normalized_embedding.flatten().tolist(),
                                k=max_results
                            )
                            
                            # Process and format results
                            for result in results:
                                all_results.append({
                                    'content': result.page_content,
                                    'source': result.metadata.get('source', 'Unknown'),
                                    'score': result.metadata.get('score', 0.5),
                                    'session_id': session_id
                                })
                    except Exception as e:
                        logger.error(f"Error searching session {session_id}: {str(e)}")
            
            # Sort results by score
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Apply filters if provided
            if filters:
                filtered_results = []
                for result in all_results:
                    include = True
                    for key, value in filters.items():
                        if key in result and result[key] != value:
                            include = False
                            break
                    if include:
                        filtered_results.append(result)
                all_results = filtered_results
            
            # Limit results
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error in multimodal search: {str(e)}")
            return []
    
    def search_text(self, query, filters=None, max_results=10):
        """
        Search for documents using text query only
        
        Args:
            query: Text query string
            filters: Optional filters to apply to search
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Get text embedding
            text_embedding = self.embed_text(query)
            
            # Get documents to search
            from ..indexing.rag_manager import RAGModelManager
            rag_manager = RAGModelManager()
            
            # Get all sessions with RAG models
            sessions = rag_manager.get_all_sessions()
            
            all_results = []
            
            # Search across all indexed sessions
            for session_id in sessions:
                rag_model = rag_manager.get_model(session_id)
                if rag_model:
                    # Search using the RAG model
                    try:
                        # This would use the built-in search capabilities of your RAG model
                        results = rag_model.similarity_search(query, k=max_results)
                        
                        # Process and format results
                        for result in results:
                            all_results.append({
                                'content': result.page_content,
                                'source': result.metadata.get('source', 'Unknown'),
                                'score': result.metadata.get('score', 0.5),
                                'session_id': session_id
                            })
                    except Exception as e:
                        logger.error(f"Error searching session {session_id}: {str(e)}")
            
            # Sort results by score
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Apply filters if provided
            if filters:
                filtered_results = []
                for result in all_results:
                    include = True
                    for key, value in filters.items():
                        if key in result and result[key] != value:
                            include = False
                            break
                    if include:
                        filtered_results.append(result)
                all_results = filtered_results
            
            # Limit results
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return []
