"""
Knowledge Graph LLM Interface: Connects knowledge graphs to LLM interactions.

This module provides the interface between the knowledge graph and language models,
enabling knowledge-augmented generation and fact validation capabilities.
"""

import logging
from typing import Dict, List, Any, Optional
import re
import json
from datetime import datetime
import numpy as np

from analytics_app.utils import monitor_kg_llm_integration
from models_app.knowledge_graph.graph_storage import GraphStorage
from models_app.knowledge_graph.external_kb_connector import CascadingKBConnector
from models_app.ai_models.llm_providers.provider_factory import ProviderFactory
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

class KnowledgeGraphLLMInterface:
    """
    Interface between knowledge graphs and language models.
    
    This class enables:
    1. Graph-augmented prompt generation
    2. Fact-checking responses against graph data
    3. Entity and relationship extraction from user queries
    4. Knowledge enrichment of LLM responses
    """
    
    def __init__(self, llm_factory=None, graph_storage=None, kb_connector=None):
        """
        Initialize the KG-LLM interface.
        
        Args:
            llm_factory: Factory for getting LLM providers
            graph_storage: Storage for knowledge graphs
            kb_connector: External knowledge base connector
        """
        self.provider_factory = ProviderFactory()
        self.graph_storage = graph_storage or GraphStorage()
        self.kb_connector = kb_connector or CascadingKBConnector()
        
        # Response cache
        self._response_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    @monitor_kg_llm_integration
    def generate_graph_augmented_response(self, query: str, graph_id: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response augmented with knowledge graph data.
        
        This method:
        1. Retrieves the relevant graph
        2. Extracts relevant graph data based on the query
        3. Constructs a graph-augmented prompt
        4. Generates a response with the LLM
        5. Validates the response against the graph
        
        Args:
            query: User query
            graph_id: Knowledge graph ID
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Dict with response text and metadata
        """
        # Check cache
        cache_key = f"{graph_id}:{query}"
        if cache_key in self._response_cache:
            cache_entry = self._response_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.cache_timeout:
                logger.info(f"Cache hit for query: {query}")
                return cache_entry['data']
        
        # Retrieve the graph
        graph = self.graph_storage.retrieve_graph(graph_id)
        if not graph:
            logger.warning(f"No graph found with ID: {graph_id}")
            return {
                "response": "I don't have access to the requested knowledge graph.",
                "validation": {
                    "status": "failed",
                    "reason": "graph_not_found"
                }
            }
        
        # Extract relevant graph data for this query
        relevant_graph_data = self._extract_relevant_graph_data(query, graph)
        
        # Construct graph-augmented prompt
        augmented_prompt = self._construct_graph_augmented_prompt(
            query, 
            relevant_graph_data
        )
        
        # Get LLM provider
        llm_provider = self.provider_factory.get_provider(
            kwargs.get("provider_name", "default")
        )
        
        # Generate response
        response_text = llm_provider.generate_text(
            prompt=augmented_prompt,
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        # Validate response against graph
        validation_result = self._validate_response_against_graph(
            response_text,
            relevant_graph_data
        )
        
        # Prepare result
        result = {
            "response": response_text,
            "validation": validation_result,
            "metadata": {
                "graph_id": graph_id,
                "entities_used": len(relevant_graph_data.get("entities", [])),
                "relationships_used": len(relevant_graph_data.get("relationships", [])),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Cache the result
        self._response_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now()
        }
        
        return result
    
    def _extract_relevant_graph_data(self, query: str, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant entities and relationships from the graph based on the query.
        
        This method:
        1. Analyzes the query to identify key terms
        2. Matches these terms against graph entities
        3. Selects relevant relationships between matched entities
        4. Calculates relevance scores for entities and relationships
        
        Args:
            query: User query
            graph: Knowledge graph data
            
        Returns:
            Dict with relevant entities and relationships
        """
        # 1. Analyze query for key terms
        query_analysis = self.analyze_query(query)
        key_terms = query_analysis["key_terms"]
        entity_mentions = query_analysis["entities"]
        
        # 2. Find relevant entities
        relevant_entities = []
        entity_relevance = {}
        
        for entity in graph.get("entities", []):
            # Calculate entity relevance score
            entity_score = self._calculate_entity_relevance(entity, key_terms, entity_mentions)
            
            if entity_score > 0.3:  # Relevance threshold
                relevant_entities.append(entity)
                entity_relevance[entity["id"]] = entity_score
        
        # Sort entities by relevance
        relevant_entities.sort(key=lambda e: entity_relevance[e["id"]], reverse=True)
        
        # Limit to top N most relevant entities
        top_n = 10
        relevant_entities = relevant_entities[:top_n]
        
        # 3. Find relationships between relevant entities
        relevant_entity_ids = [entity["id"] for entity in relevant_entities]
        relevant_relationships = []
        
        for relationship in graph.get("relationships", []):
            source_id = relationship.get("source")
            target_id = relationship.get("target")
            
            if source_id in relevant_entity_ids and target_id in relevant_entity_ids:
                # Calculate relationship score based on connected entities
                source_score = entity_relevance.get(source_id, 0)
                target_score = entity_relevance.get(target_id, 0)
                relationship_score = (source_score + target_score) / 2
                
                if relationship_score > 0.4:  # Relationship relevance threshold
                    relationship["relevance_score"] = relationship_score
                    relevant_relationships.append(relationship)
        
        # Sort relationships by relevance score
        relevant_relationships.sort(key=lambda r: r.get("relevance_score", 0), reverse=True)
        
        return {
            "entities": relevant_entities,
            "relationships": relevant_relationships,
            "entity_relevance": entity_relevance
        }
    
    def _calculate_entity_relevance(self, entity: Dict[str, Any], 
                                  key_terms: List[str], 
                                  entity_mentions: List[Dict[str, Any]]) -> float:
        """
        Calculate relevance score for an entity based on query terms.
        
        Args:
            entity: Entity from the knowledge graph
            key_terms: Key terms extracted from the query
            entity_mentions: Entity mentions extracted from the query
            
        Returns:
            Relevance score between 0 and 1
        """
        relevance = 0.0
        
        # 1. Check for direct mention by label
        entity_label = entity.get("label", "").lower()
        for mention in entity_mentions:
            mention_text = mention.get("text", "").lower()
            if mention_text in entity_label or entity_label in mention_text:
                # Direct mention is high relevance
                mention_confidence = mention.get("confidence", 0.8)
                relevance += 0.7 * mention_confidence
        
        # 2. Check for property matches with key terms
        for term in key_terms:
            term = term.lower()
            
            # Check label
            if term in entity_label:
                relevance += 0.5
            
            # Check properties
            for prop_key, prop_value in entity.get("properties", {}).items():
                if prop_value is None:
                    continue
                    
                # Convert to string for comparison
                prop_str = str(prop_value).lower()
                if term in prop_str:
                    relevance += 0.3
        
        # 3. Apply entity type weights
        entity_type = entity.get("type", "").lower()
        
        # Some entity types are more likely to be relevant for questions
        type_weights = {
            "person": 1.2,
            "organization": 1.1,
            "location": 1.0,
            "event": 1.1,
            "concept": 0.9,
            "document": 0.8
        }
        
        type_weight = type_weights.get(entity_type, 1.0)
        relevance *= type_weight
        
        # Cap at 1.0
        return min(relevance, 1.0)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query to extract key terms and entity mentions.
        
        This method uses a combination of rule-based and model-based approaches
        to extract important information from the query.
        
        Args:
            query: User query string
            
        Returns:
            Dict with key terms and entity mentions
        """
        # Get LLM for analysis
        llm = self.provider_factory.get_provider("default")
        
        # Simple query analysis prompt
        analysis_prompt = f"""
        Analyze the following query and extract:
        1. Key terms (important words or phrases)
        2. Entity mentions (references to specific entities)
        
        Query: "{query}"
        
        Format your response as a JSON object with "key_terms" (array of strings) and
        "entities" (array of objects with "text", "type", and "confidence" properties).
        """
        
        # Generate analysis
        analysis_text = llm.generate_text(analysis_prompt, max_tokens=300)
        
        # Extract JSON from response
        try:
            # Find JSON block in response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(0))
            else:
                # Fallback to simple extraction
                analysis_json = {
                    "key_terms": self._extract_key_terms(query),
                    "entities": self._extract_entity_mentions(query)
                }
        except json.JSONDecodeError:
            # Fallback to simple extraction on error
            analysis_json = {
                "key_terms": self._extract_key_terms(query),
                "entities": self._extract_entity_mentions(query)
            }
        
        return analysis_json
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Simple rule-based extraction of key terms"""
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'to', 'of', 'for', 'in', 'on', 'by', 'at', 
                     'that', 'this', 'with', 'which', 'what', 'who', 'where', 'when', 'how'}
        
        # Tokenize by whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _extract_entity_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Simple rule-based extraction of potential entity mentions"""
        # Look for capitalized phrases
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
        
        entities = []
        for match in capitalized:
            entities.append({
                "text": match,
                "type": "unknown",
                "confidence": 0.7
            })
        
        return entities
    
    def _construct_graph_augmented_prompt(self, query: str, graph_data: Dict[str, Any]) -> str:
        """
        Construct a prompt augmented with knowledge graph data.
        
        Args:
            query: User query
            graph_data: Relevant graph data
            
        Returns:
            Augmented prompt string
        """
        # Format relevant entities
        entity_sections = []
        
        for entity in graph_data.get("entities", []):
            properties_text = ""
            for key, value in entity.get("properties", {}).items():
                if value is not None:
                    properties_text += f"- {key}: {value}\n"
            
            entity_section = f"""
            Entity: {entity.get('label', '')}
            Type: {entity.get('type', '')}
            Properties:
            {properties_text}
            """
            entity_sections.append(entity_section)
        
        # Format relevant relationships
        relationship_sections = []
        
        for rel in graph_data.get("relationships", []):
            # Find source and target entity labels
            source_id = rel.get("source")
            target_id = rel.get("target")
            
            source_label = "Unknown"
            target_label = "Unknown"
            
            for entity in graph_data.get("entities", []):
                if entity.get("id") == source_id:
                    source_label = entity.get("label", "Unknown")
                if entity.get("id") == target_id:
                    target_label = entity.get("label", "Unknown")
            
            rel_section = f"{source_label} {rel.get('type', '')} {target_label}"
            relationship_sections.append(rel_section)
        
        # Construct the augmented prompt
        augmented_prompt = f"""
        Answer the following question based on the knowledge graph information provided.
        
        Question: {query}
        
        Knowledge Graph Information:
        
        Entities:
        {''.join(entity_sections)}
        
        Relationships:
        {' | '.join(relationship_sections)}
        
        Please provide a detailed answer based on this information. If the information is insufficient to answer the question completely, please indicate what is missing.
        """
        
        return augmented_prompt
    
    def _validate_response_against_graph(self, response: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a response against knowledge graph data.
        
        This method checks if the response contains information that contradicts
        or isn't supported by the knowledge graph.
        
        Args:
            response: LLM response text
            graph_data: Relevant graph data
            
        Returns:
            Validation result dict
        """
        # 1. Extract claims from response
        claims = self._extract_claims_from_response(response)
        
        # 2. Check each claim against graph data
        verified_claims = []
        unverified_claims = []
        contradicted_claims = []
        
        for claim in claims:
            validation = self._validate_claim(claim, graph_data)
            
            if validation["status"] == "verified":
                verified_claims.append({
                    "claim": claim,
                    "evidence": validation["evidence"]
                })
            elif validation["status"] == "contradicted":
                contradicted_claims.append({
                    "claim": claim,
                    "contradiction": validation["contradiction"]
                })
            else:
                unverified_claims.append({
                    "claim": claim,
                    "reason": validation["reason"]
                })
        
        # 3. Calculate overall validation score
        total_claims = len(claims)
        verified_count = len(verified_claims)
        contradicted_count = len(contradicted_claims)
        
        if total_claims > 0:
            verification_score = verified_count / total_claims
            contradiction_score = contradicted_count / total_claims
        else:
            verification_score = 0.0
            contradiction_score = 0.0
        
        # 4. Determine overall status
        if contradiction_score > 0.2:
            status = "invalid"
            reason = "high_contradiction"
        elif verification_score < 0.3 and total_claims > 0:
            status = "uncertain"
            reason = "low_verification"
        else:
            status = "valid"
            reason = "sufficient_verification"
        
        return {
            "status": status,
            "reason": reason,
            "verification_score": verification_score,
            "contradiction_score": contradiction_score,
            "verified_claims": verified_claims,
            "unverified_claims": unverified_claims,
            "contradicted_claims": contradicted_claims
        }
    
    def _extract_claims_from_response(self, response: str) -> List[str]:
        """
        Extract factual claims from a response.
        
        Args:
            response: Response text
            
        Returns:
            List of claim strings
        """
        # Get LLM for claim extraction
        llm = self.provider_factory.get_provider("default")
        
        # Simple claim extraction prompt
        extraction_prompt = f"""
        Extract factual claims from the following response.
        A factual claim is a statement that asserts something to be true that could be verified.
        
        Response: "{response}"
        
        List each factual claim on a separate line, starting with "- ".
        """
        
        # Generate extraction
        extraction_text = llm.generate_text(extraction_prompt, max_tokens=300)
        
        # Parse claims from the response
        claims = []
        for line in extraction_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                claim = line[2:].strip()
                if claim:
                    claims.append(claim)
        
        return claims
    
    def _validate_claim(self, claim: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single claim against graph data.
        
        Args:
            claim: Claim text
            graph_data: Relevant graph data
            
        Returns:
            Validation result dict
        """
        # Compare claim to entities and their properties
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])
        
        # Convert claim to lowercase for matching
        claim_lower = claim.lower()
        
        # Check if claim mentions entities
        mentioned_entities = []
        for entity in entities:
            entity_label = entity.get("label", "").lower()
            if entity_label and entity_label in claim_lower:
                mentioned_entities.append(entity)
        
        if not mentioned_entities:
            return {
                "status": "unverified",
                "reason": "no_entity_match"
            }
        
        # Check properties of mentioned entities
        supporting_evidence = []
        contradicting_evidence = []
        
        for entity in mentioned_entities:
            # Check each property against the claim
            for prop_key, prop_value in entity.get("properties", {}).items():
                if prop_value is None:
                    continue
                
                prop_str = str(prop_value).lower()
                
                # If property value is in claim, consider it supporting evidence
                if prop_str and prop_str in claim_lower:
                    supporting_evidence.append({
                        "entity": entity.get("label"),
                        "property": prop_key,
                        "value": prop_value
                    })
                
                # Check for contradictions (simple check - can be improved)
                # e.g., if claim says "founded in 1990" but property says "founded in 1980"
                if prop_key.lower() in claim_lower:
                    # Identify specific pattern where property is mentioned but with different value
                    for term in self._extract_key_terms(claim_lower):
                        if term.isdigit() and term != prop_str and prop_str.isdigit():
                            contradicting_evidence.append({
                                "entity": entity.get("label"),
                                "property": prop_key,
                                "claimed_value": term,
                                "actual_value": prop_value
                            })
        
        # Check relationships
        for rel in relationships:
            source_id = rel.get("source")
            target_id = rel.get("target")
            rel_type = rel.get("type", "").lower()
            
            # Find source and target entity labels
            source_entity = next((e for e in entities if e.get("id") == source_id), None)
            target_entity = next((e for e in entities if e.get("id") == target_id), None)
            
            if source_entity and target_entity:
                source_label = source_entity.get("label", "").lower()
                target_label = target_entity.get("label", "").lower()
                
                # Check if relationship is mentioned in claim
                if (source_label in claim_lower and 
                    target_label in claim_lower and 
                    rel_type in claim_lower):
                    supporting_evidence.append({
                        "relationship": f"{source_entity.get('label')} {rel_type} {target_entity.get('label')}"
                    })
        
        # Determine validation status
        if contradicting_evidence:
            return {
                "status": "contradicted",
                "contradiction": contradicting_evidence[0]  # Return first contradiction
            }
        elif supporting_evidence:
            return {
                "status": "verified",
                "evidence": supporting_evidence
            }
        else:
            return {
                "status": "unverified",
                "reason": "no_supporting_evidence"
            }
    
    def generate_follow_up_query(self, original_query, findings):
        """
        Generate a follow-up query based on initial findings to explore deeper
        
        Args:
            original_query: The original query
            findings: Text findings from the initial search
            
        Returns:
            A follow-up query to explore deeper
        """
        prompt = f"""Based on the original query and the findings below, generate a follow-up query 
that will explore the topic deeper. Focus on filling in knowledge gaps or exploring interesting aspects.

ORIGINAL QUERY: {original_query}

INITIAL FINDINGS:
{findings}

The follow-up query should:
1. Be more specific than the original query
2. Focus on unexplored aspects or contradictions in the findings
3. Be phrased as a search query (not a question to a person)

FOLLOW-UP QUERY:"""
        
        # Generate the follow-up query
        follow_up_query, _ = self.provider_factory.get_provider("default").generate_text(prompt, max_tokens=100)
        
        return follow_up_query.strip()
    
    def generate_research_summary(self, query, findings):
        """
        Generate a comprehensive research summary based on exploration findings
        
        Args:
            query: The original research query
            findings: Formatted text with all key findings
            
        Returns:
            Comprehensive research summary
        """
        prompt = f"""Synthesize the findings below into a comprehensive research summary that answers the original query.

ORIGINAL QUERY: {query}

KEY FINDINGS:
{findings}

Your research summary should:
1. Provide a complete answer to the original query
2. Organize information by concept/topic
3. Highlight key insights and patterns
4. Note any contradictions or areas needing further research
5. Be comprehensive yet concise

COMPREHENSIVE RESEARCH SUMMARY:"""
        
        # Generate the summary
        summary, _ = self.provider_factory.get_provider("default").generate_text(prompt, max_tokens=1000)
        
        return summary.strip()

    def generate_with_self_consistency(self, query, n_samples=3):
        """Generate answers with self-consistency checking"""
        logger.info(f"Generating with self-consistency for: {query}")
        
        # Generate multiple answers with temperature variation
        responses = []
        confidence_scores = []
        
        llm_provider = self.provider_factory.get_provider("default")
        
        # Create prompt
        base_prompt = f"""Answer the following question accurately:
        
        Question: {query}
        
        Answer:"""
        
        # Generate multiple responses with different temperatures
        for i in range(n_samples):
            temperature = 0.3 + (i * 0.2)  # Vary from 0.3 to 0.7
            response, confidence = llm_provider.generate_text(
                base_prompt, 
                temperature=temperature,
                max_tokens=300
            )
            responses.append(response)
            confidence_scores.append(confidence)
        
        # Extract claims from each response
        all_claims = []
        for response in responses:
            claims = self._extract_claims_from_response(response)
            all_claims.extend(claims)
        
        # Find common claims
        from collections import Counter
        claim_counter = Counter(all_claims)
        
        # Get claims with consensus (appear in multiple responses)
        consensus_claims = [claim for claim, count in claim_counter.items() 
                            if count > 1]
        
        # Generate final answer using consensus claims
        if consensus_claims:
            consensus_prompt = f"""Based on the following verified facts, provide a comprehensive answer to the question.
            
            Question: {query}
            
            Verified facts:
            {' '.join(['- ' + claim for claim in consensus_claims])}
            
            Provide a coherent answer using only these verified facts:"""
            
            final_answer, confidence = llm_provider.generate_text(
                consensus_prompt,
                temperature=0.3,  # Lower temperature for factual response
                max_tokens=400
            )
            
            return final_answer, 0.9  # High confidence due to consensus
        else:
            # No consensus, return most confident original response
            best_index = confidence_scores.index(max(confidence_scores))
            return responses[best_index], 0.5  # Lower confidence

    def perform_r3_reasoning(self, query, search_results):
        """
        Implement the Retrieval-Reading-Reasoning (R^3) approach.
        
        Args:
            query: Original query
            search_results: Retrieved documents
            
        Returns:
            Reasoned answer based on retrieved information
        """
        logger.info(f"Performing R^3 reasoning for query: {query}")
        
        # 1. RETRIEVAL (already done - results passed in)
        
        # 2. READING: Extract key information
        extracted_info = []
        for result_type, items in search_results.items():
            for item in items:
                if 'content' in item:
                    # Extract entities and relationships
                    from models_app.knowledge_graph.entity_extractor import EntityExtractor
                    extractor = EntityExtractor()
                    entities = extractor.extract_entities(item['content'])
                    
                    # Extract key passages
                    passages = self._extract_key_passages(item['content'], query)
                    
                    extracted_info.append({
                        'entities': entities,
                        'passages': passages,
                        'source': f"{result_type}:{item.get('id', '')}"
                    })
        
        # Format extracted information
        formatted_info = ""
        for info in extracted_info:
            formatted_info += f"SOURCE: {info['source']}\n"
            formatted_info += "ENTITIES: " + ", ".join([e.get('label', '') for e in info['entities']]) + "\n"
            formatted_info += "PASSAGES:\n" + "\n".join([f"- {p}" for p in info['passages']]) + "\n\n"
        
        # 3. REASONING: Apply logical reasoning to draw conclusions
        reasoning_prompt = f"""Based on the following information, answer the question by reasoning through the evidence step by step.

QUESTION: {query}

INFORMATION:
{formatted_info}

REASONING:
1. First, identify the key facts relevant to the question.
2. Consider how these facts relate to each other.
3. Identify any contradictions or gaps in the information.
4. Draw logical conclusions based on the evidence.

ANSWER:"""
        
        # Get LLM provider
        llm_provider = self.provider_factory.get_provider("default")
        
        # Generate reasoned answer
        reasoned_answer, confidence = llm_provider.generate_text(
            reasoning_prompt,
            temperature=0.3,  # Low temperature for logical reasoning
            max_tokens=500
        )
        
        return {
            'answer': reasoned_answer,
            'confidence': confidence,
            'extracted_info': extracted_info
        }

    def _extract_key_passages(self, text, query, max_passages=3):
        """Extract key passages relevant to the query"""
        # Split into sentences or paragraphs
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score each sentence based on relevance to query
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            # Simple term overlap scoring
            query_terms = set(query.lower().split())
            sentence_terms = set(sentence.lower().split())
            overlap = len(query_terms.intersection(sentence_terms))
            
            score = overlap / max(len(query_terms), 1)
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top passages
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sentences[:max_passages]]

    def semantic_graph_query(self, query_text, graph_data):
        """Enhance graph queries with semantic similarity"""
        # Import here to avoid circular imports
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Generate query embedding
        query_embedding = self._create_embedding(query_text)
        if not query_embedding:
            return []
        
        # Calculate similarity to entity descriptions
        entity_similarities = []
        for entity in graph_data.get("entities", []):
            entity_text = self._get_entity_text(entity)
            entity_embedding = self._create_embedding(entity_text)
            if entity_embedding:
                # Convert to 2D arrays for cosine_similarity
                q_emb = np.array(query_embedding).reshape(1, -1)
                e_emb = np.array(entity_embedding).reshape(1, -1)
                similarity = cosine_similarity(q_emb, e_emb)[0][0]
                entity_similarities.append((entity, similarity))
        
        # Sort by similarity and return top entities
        entity_similarities.sort(key=lambda x: x[1], reverse=True)
        return [e[0] for e in entity_similarities[:10]]

    def _create_embedding(self, text):
        """Create text embedding"""
        if not text or not isinstance(text, str):
            return None
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
            return model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            from error_handlers.models_app_errors import handle_embedding_error
            handle_embedding_error(text, "sentence-transformers/all-MiniLM-L12-v2", e)
            return None
