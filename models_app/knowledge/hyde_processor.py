"""
HyDE Processor Implementation

HyDE (Hypothetical Document Embeddings) is a technique that improves retrieval by:
1. Generating a hypothetical answer to a query
2. Using this hypothetical answer to enhance search
3. Finding more relevant documents that might be missed with the original query

This implementation provides:
- A base HyDEProcessor class
- Methods for generating and using hypotheses
- Integration with existing search providers
- Caching for performance optimization
- Evidenz-Tracking for source citations
"""

import logging
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from django.conf import settings
from django.core.cache import cache

from .mention_providers import get_mention_provider, MentionProcessor
from search_app.providers import get_search_provider
from search_app.providers.universal import UniversalSearchProvider
from users_app.model_provider import ModelProviderManager

logger = logging.getLogger(__name__)

class SimpleLLMProvider:
    """
    Einfacher LLM Provider für Fallback-Zwecke.
    """
    
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """Generiert Text basierend auf einem Prompt."""
        return f"Ich konnte keine Antwort generieren für: {prompt}", 0.1
    
    def generate_batch(self, prompts: List[str], max_tokens: Optional[int] = None) -> List[Tuple[str, float]]:
        """Generiert Text für mehrere Prompts im Batch."""
        return [(f"Ich konnte keine Antwort generieren für: {prompt}", 0.1) for prompt in prompts]

class HyDEProcessor:
    """
    Base class for Hypothetical Document Embeddings processing.
    
    HyDE improves search relevance by generating a hypothetical answer
    to a query, then using that answer to find better matching documents.
    """
    
    def __init__(self, user=None):
        """Initialize the HyDE processor with default settings."""
        self.cache_duration = getattr(settings, 'HYDE_CACHE_DURATION', 3600)  # 1 hour default
        self.confidence_threshold = getattr(settings, 'HYDE_CONFIDENCE_THRESHOLD', 0.6)
        self.mention_processor = MentionProcessor()
        
        # Speichere den Benutzer für die Modellauswahl
        self.user = user
        
        # Get the LLM provider for hypothesis generation
        self.llm_provider = self._get_llm_provider()
        
        self.evidence_store = {}  # Speichert Evidenz für Quellenangaben
    
    def _get_llm_provider(self):
        """
        Get the LLM provider for hypothesis generation.
        This could be customized based on settings.
        """
        # Verwende den ModelProviderManager, um den Provider basierend auf Benutzereinstellungen zu erhalten
        provider_manager = ModelProviderManager(self.user)
        
        # Wenn der Benutzer HyDE-spezifische Einstellungen hat, verwende diese
        if self.user and hasattr(self.user, 'settings'):
            user_settings = self.user.settings
            
            if user_settings.use_hyde:
                # Verwende das spezifische HyDE-Modell
                return provider_manager.get_provider_instance(
                    provider_name='local',  # Für HyDE verwenden wir lokale Modelle
                    model_name=user_settings.hyde_model
                )
            elif user_settings.use_custom_model and user_settings.custom_model_url:
                # Verwende das benutzerdefinierte Modell, wenn es aktiviert ist und eine URL angegeben wurde
                config = provider_manager.get_provider_config()
                config['provider'] = 'template'
                config['model_url'] = user_settings.custom_model_url
                
                from models_app.llm_providers.template_provider import TemplateProviderFactory
                return TemplateProviderFactory.create_provider(
                    user_settings.custom_model_url, 
                    config
                )
        
        # Ansonsten verwende den Standard-Provider des Benutzers
        return provider_manager.get_provider_instance()
    
    def _get_cache_key(self, query: str) -> str:
        """
        Generiert einen Cache-Schlüssel für eine Anfrage.
        
        Args:
            query: Die Anfrage
            
        Returns:
            Cache-Schlüssel
        """
        return f"hyde_hypothesis_{hashlib.md5(query.encode()).hexdigest()}"
    
    def _store_evidence(self, query: str, evidence: Dict[str, Any]):
        """
        Speichert Evidenz für eine Anfrage.
        
        Args:
            query: Die Anfrage
            evidence: Die Evidenz-Daten
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash not in self.evidence_store:
            self.evidence_store[query_hash] = []
        self.evidence_store[query_hash].append(evidence)
    
    def get_evidence(self, query: str) -> List[Dict[str, Any]]:
        """
        Holt die gespeicherte Evidenz für eine Anfrage.
        
        Args:
            query: Die Anfrage
            
        Returns:
            Liste von Evidenz-Daten
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.evidence_store.get(query_hash, [])
    
    def generate_hypothesis(self, query: str) -> Tuple[str, float]:
        """
        Generate a hypothetical document based on the query.
        
        Args:
            query: The search query
            
        Returns:
            Tuple[str, float]: The generated hypothesis and a confidence score
        """
        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached hypothesis for query: {query[:50]}...")
            hypothesis, confidence = cached_result
            
            # Speichere Evidenz für Quellenangaben
            evidence = {
                "source_type": "cache",
                "content": hypothesis,
                "highlights": [{
                    "start": 0,
                    "end": len(hypothesis),
                    "confidence": confidence,
                    "citation_id": f"citation-{query[:10]}-cached"
                }]
            }
            self._store_evidence(query, evidence)
            
            return hypothesis, confidence
        
        # Process mentions in the query
        processed_query = self.mention_processor.process_mentions(query)
        
        # Create a prompt for hypothesis generation
        prompt = f"""Based on the following query, generate a detailed, factual document that would be a perfect search result.
        
Query: {processed_query}

Hypothetical Document:"""
        
        # Generate the hypothesis using the LLM provider
        try:
            hypothesis, confidence = self.llm_provider.generate_text(prompt)
            
            # Cache the result
            cache.set(cache_key, (hypothesis, confidence), self.cache_duration)
            
            # Store evidence for source attribution
            evidence = {
                "source_type": "hypothesis",
                "content": hypothesis,
                "highlights": [{
                    "start": 0,
                    "end": len(hypothesis),
                    "confidence": confidence,
                    "citation_id": f"citation-{query[:10]}-1"
                }]
            }
            self._store_evidence(query, evidence)
            
            return hypothesis, confidence
            
        except Exception as e:
            logger.error(f"Error generating hypothesis: {str(e)}")
            return "", 0.0
    
    def combine_with_original(self, query: str, hypothesis: str, confidence: float = 0.7, strategy: str = "weighted") -> str:
        """
        Combine the original query with the generated hypothesis.
        
        Args:
            query: The original query
            hypothesis: The generated hypothesis
            confidence: The confidence score for the hypothesis
            strategy: The combination strategy ("weighted", "append", "extract_keywords")
            
        Returns:
            str: The combined query for search
        """
        # If confidence is too low, just use the original query
        if confidence < self.confidence_threshold:
            logger.info(f"Confidence too low ({confidence:.2f}), using original query")
            return query
        
        # Store evidence for source attribution
        evidence = {
            "source_type": "processing",
            "content": f"Query: {query}\nHypothesis: {hypothesis}",
            "highlights": [{
                "start": 7,  # After "Query: "
                "end": 7 + len(query),
                "confidence": 1.0,
                "citation_id": f"citation-{query[:10]}-2"
            }]
        }
        self._store_evidence(query, evidence)
        
        # Combine based on the chosen strategy
        if strategy == "weighted":
            # Weighted combination: Original query has more weight
            enhanced_query = f"{query} {hypothesis}"
        elif strategy == "append":
            # Simple appending
            enhanced_query = f"{query}. Context: {hypothesis}"
        elif strategy == "extract_keywords":
            # Extract keywords from the hypothesis
            # In production, a more complex logic would be used here
            words = hypothesis.split()
            keywords = [word for word in words if len(word) > 5][:5]  # Simple heuristic
            enhanced_query = f"{query} {' '.join(keywords)}"
        else:
            # Default strategy
            enhanced_query = f"{query} {hypothesis}"
        
        # Store evidence for the enhanced query
        evidence = {
            "source_type": "enhanced_query",
            "content": enhanced_query,
            "highlights": [{
                "start": 0,
                "end": len(enhanced_query),
                "confidence": 0.9,
                "citation_id": f"citation-{query[:10]}-3"
            }]
        }
        self._store_evidence(query, evidence)
        
        return enhanced_query
    
    def process_query(self, query: str, strategy: str = "weighted") -> str:
        """
        Process a query using HyDE.
        
        Args:
            query: The search query
            strategy: The combination strategy
            
        Returns:
            str: The enhanced query for search
        """
        # Generate a hypothesis
        hypothesis, confidence = self.generate_hypothesis(query)
        
        # Combine with the original query
        enhanced_query = self.combine_with_original(query, hypothesis, confidence, strategy)
        
        return enhanced_query
    
    def enhance_search(self, query: str, category: str = "web", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Enhance search results using HyDE.
        
        Args:
            query: The original search query
            category: The search category
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Enhanced search results
        """
        # Generate hypothesis
        hypothesis, confidence = self.generate_hypothesis(query)
        
        # If hypothesis generation failed, fall back to regular search
        if not hypothesis or confidence < 0.1:
            logger.warning("Hypothesis generation failed, falling back to regular search")
            return self._perform_regular_search(query, category, limit)
        
        # Combine query with hypothesis
        enhanced_query = self.combine_with_original(query, hypothesis)
        
        # Perform search with enhanced query
        results = self._perform_search(enhanced_query, category, limit)
        
        # Add metadata about the enhancement
        for result in results:
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["enhanced_by_hyde"] = True
            result["metadata"]["hypothesis_confidence"] = confidence
        
        return results
    
    def _perform_search(self, query: str, category: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform search using the appropriate provider.
        
        Args:
            query: The search query
            category: The search category
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        try:
            # Get the appropriate provider based on category
            if category == "web":
                provider = self.mention_processor.get_provider("web")
            else:
                # For other categories, use the local provider
                provider = self.mention_processor.get_provider("local")
            
            if not provider:
                logger.error(f"Provider not found for category: {category}")
                return []
            
            # Perform the search
            results = provider.search(category, query, limit=limit)
            return results
            
        except Exception as e:
            logger.error(f"Error in HyDE search: {str(e)}")
            return []
    
    def _perform_regular_search(self, query: str, category: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform regular search without enhancement.
        
        Args:
            query: The search query
            category: The search category
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        return self._perform_search(query, category, limit) 