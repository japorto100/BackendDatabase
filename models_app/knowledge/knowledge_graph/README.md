# Knowledge Graph Core Components

This module contains the core components for building, storing, and querying knowledge graphs. It provides a comprehensive framework for knowledge extraction and representation.

## Architecture Overview

The knowledge graph system follows a 3-layer architecture:

1. **Data Ingestion and Integration Layer**
   - Document processing (`models_app/vision/document/`)
   - Domain-specific entity extraction (`models_app/vision/knowledge_graph/`)

2. **Knowledge Graph Core Layer** (this module)
   - Entity management (base `EntityExtractor`)
   - Relationship detection (`RelationshipDetector`)
   - Graph construction (`GraphBuilder`)
   - Graph storage and retrieval (`GraphStorage`)

3. **Knowledge Consumption Layer**
   - Graph visualization (`GraphVisualization`)
   - Query interfaces
   - Integration with RAG systems

## Core Components

### KnowledgeGraphManager

Central orchestrator for the entire knowledge graph pipeline:
- Coordinates document processing, entity extraction, and graph building
- Provides a unified interface for creating and querying knowledge graphs
- Manages batch processing of multiple documents

### EntityExtractor

Base class for entity extraction:
- Provides core NLP functionality for entity recognition
- Defines the interface for domain-specific extractors
- Handles entity normalization and classification

### RelationshipDetector

Identifies relationships between entities:
- Detects semantic relationships from text
- Identifies co-occurrence patterns
- Analyzes spatial and hierarchical relationships
- Merges relationships from different sources

### GraphBuilder

Constructs the knowledge graph structure:
- Creates entity and relationship nodes
- Builds subgraphs from document extractions
- Merges subgraphs from multiple sources
- Ensures graph integrity and consistency

### GraphStorage

Manages persistence and retrieval of knowledge graphs:
- Stores graphs in a triple store and/or vector database
- Provides query interfaces for graph retrieval
- Handles updates and versioning
- Supports graph merging and cross-referencing

### GraphVisualization

Renders knowledge graphs for visual exploration:
- Creates interactive graph visualizations
- Supports filtering and focusing on subgraphs
- Provides customizable layouts and styling

## Integration with Document Processing

The knowledge graph system integrates with the document processing pipeline:

1. Document processors extract structured data via `prepare_for_extraction()`
2. Domain-specific entity extractors process the structured data
3. `KnowledgeGraphManager` orchestrates the flow from documents to knowledge graphs

## Usage Example

```python
# Create and query a knowledge graph
from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager

# Initialize the manager
kg_manager = KnowledgeGraphManager()

# Process multiple documents to create a combined knowledge graph
result = kg_manager.batch_process_documents([
    "path/to/document1.pdf",
    "path/to/document2.docx",
    "path/to/image.jpg"
])

# Store the graph
graph_id = result["graph_id"]

# Query the knowledge graph
query_result = kg_manager.query_graph("Who is the CEO?", graph_id=graph_id)
```

## Interfaces

The module provides interfaces (`interfaces.py`) that define the contract for each component:
- `EntityExtractorInterface`
- `RelationshipDetectorInterface`
- `GraphBuilderInterface`
- `GraphStorageInterface`

These interfaces ensure consistency and interoperability between components. 