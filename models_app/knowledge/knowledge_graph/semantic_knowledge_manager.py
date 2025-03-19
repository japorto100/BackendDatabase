class SemanticKnowledgeManager:
    """
    Manages the semantic representation of knowledge for LLM consumption.
    Implements the "LLM-friendly semantic knowledge management" component of KAG.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def format_knowledge_for_llm(self, graph, query=None):
        """Format knowledge graph in LLM-friendly format"""
        # Extract relevant subgraph if query is provided
        if query:
            # Perform relevance filtering
            relevant_entities, relevant_relationships = self._extract_relevant_subgraph(graph, query)
        else:
            relevant_entities = graph.get("entities", [])
            relevant_relationships = graph.get("relationships", [])
        
        # Format in natural language
        knowledge_text = "# Knowledge Graph Information\n\n"
        
        # Add entities
        knowledge_text += "## Entities\n\n"
        for entity in relevant_entities:
            knowledge_text += f"- {entity.get('label')} (Type: {entity.get('type')})\n"
            if entity.get("properties"):
                knowledge_text += "  Properties:\n"
                for key, value in entity.get("properties", {}).items():
                    knowledge_text += f"  - {key}: {value}\n"
        
        # Add relationships
        knowledge_text += "\n## Relationships\n\n"
        for rel in relevant_relationships:
            # Find source and target entity labels
            source_label = next((e.get("label") for e in relevant_entities if e.get("id") == rel.get("source")), rel.get("source"))
            target_label = next((e.get("label") for e in relevant_entities if e.get("id") == rel.get("target")), rel.get("target"))
            knowledge_text += f"- {source_label} --> {rel.get('type')} --> {target_label}\n"
        
        return knowledge_text
    
    def _extract_relevant_subgraph(self, graph, query):
        """Extract a query-relevant subgraph"""
        # Implement relevance scoring based on query
        # This is a simplified version - production would use embeddings
        relevant_entity_ids = set()
        
        # Simple keyword matching (would be replaced with semantic matching)
        query_terms = query.lower().split()
        for entity in graph.get("entities", []):
            entity_text = f"{entity.get('label', '')} {entity.get('type', '')}"
            for prop_val in entity.get("properties", {}).values():
                if isinstance(prop_val, str):
                    entity_text += f" {prop_val}"
            
            entity_text = entity_text.lower()
            if any(term in entity_text for term in query_terms):
                relevant_entity_ids.add(entity.get("id"))
        
        # Include 1-hop neighbors
        for rel in graph.get("relationships", []):
            if rel.get("source") in relevant_entity_ids:
                relevant_entity_ids.add(rel.get("target"))
            elif rel.get("target") in relevant_entity_ids:
                relevant_entity_ids.add(rel.get("source"))
        
        # Filter entities and relationships
        relevant_entities = [e for e in graph.get("entities", []) if e.get("id") in relevant_entity_ids]
        relevant_relationships = [
            r for r in graph.get("relationships", []) 
            if r.get("source") in relevant_entity_ids and r.get("target") in relevant_entity_ids
        ]
        
        return relevant_entities, relevant_relationships
