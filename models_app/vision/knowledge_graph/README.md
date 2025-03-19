# Knowledge Graph Components

This module contains domain-specific entity extractors that work with the document processing pipeline to extract entities for knowledge graph construction.

## Architecture Overview

The knowledge graph architecture follows a layered approach:

1. **Document Processing Layer**: Handles document parsing and structure extraction
   - Located in `models_app/vision/document/`
   - Document adapters implement `prepare_for_extraction()` to produce standardized structured data

2. **Entity Extraction Layer**: Extracts entities from structured document data
   - Domain-specific extractors for different document types (text, image, hybrid)
   - This folder contains the specialized extractors for vision-based content

3. **Knowledge Graph Core Layer**: Builds and manages the knowledge graph
   - Located in `models_app/knowledge_graph/`
   - Includes relationship detection, graph building, and storage components

## Components

### DocumentEntityExtractor

Extracts entities from text-based documents:
- Text entities (people, organizations, locations, etc.)
- Document structure entities (headings, paragraphs, etc.)
- Metadata entities (title, author, date, etc.)

### VisualEntityExtractor

Extracts entities from image-based documents:
- Visual elements (images, charts, diagrams)
- OCR text regions
- Spatial relationships between visual elements

### HybridEntityExtractor

Extracts entities from mixed-content documents:
- Combines text and visual extraction
- Correlates entities across different modalities
- Merges related entities to create rich, multimodal entities

## Integration with Document Processing

The knowledge graph components integrate with the document processing pipeline through the standardized `prepare_for_extraction()` method:

1. `DocumentProcessorFactory.prepare_document_for_extraction()` processes documents and returns structured data
2. Domain-specific extractors consume this structured data through their `extract_from_document()` methods
3. `KnowledgeGraphManager` orchestrates the entire process from document processing to graph construction

## Usage Example

```python
# Get a knowledge graph from a document
from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager

# Initialize the manager
kg_manager = KnowledgeGraphManager()

# Process a document to create a knowledge graph
result = kg_manager.process_document_to_graph("path/to/document.pdf")

# Access the knowledge graph
graph = result["graph"]
```

## Future Enhancements

- Enhanced entity disambiguation
- Cross-document entity resolution
- Integration with vector databases for efficient retrieval
# Knowledge Graph Component Usage Guide

## Architecture Overview

The knowledge graph system is designed with a layered architecture:

1. **Base Services** - Provide raw/structured data from different sources:
   - OCR adapters (`BaseOCRAdapter`)
   - ColPali processor
   - Document processing

2. **Domain-Specific Extractors** - Extract entities from specific data types:
   - `DocumentEntityExtractor` - For text content
   - `VisualEntityExtractor` - For visual content
   - `HybridEntityExtractor` - Combines both approaches

3. **Core Knowledge Graph Framework** - Builds and manages the graph:
   - `EntityExtractor` - Base extraction functionality
   - `RelationshipDetector` - Detects relationships between entities
   - `GraphBuilder` - Constructs the graph structure
   - `GraphStorage` - Stores and retrieves graphs
   - `GraphVisualization` - Creates visualizations

## Proper Usage Flow

To create knowledge graphs from multimodal data:

```python
# 1. Get structured data from base services
ocr_result = ocr_adapter.process_image(image_path)
colpali_result = colpali_processor.process_image(image_path)

# 2. Use domain-specific extractors
doc_extractor = DocumentEntityExtractor()
visual_extractor = VisualEntityExtractor()
hybrid_extractor = HybridEntityExtractor()

# Extract entities from OCR text
text_entities = doc_extractor.extract_from_text(ocr_result["text"], ocr_result["metadata"])

# Extract entities from visual data
visual_entities = visual_extractor.extract_from_colpali(colpali_result)

# Or use the hybrid extractor for combined approach
all_entities = hybrid_extractor.extract_from_hybrid_source(
    text=ocr_result["text"], 
    colpali_result=colpali_result,
    metadata=document_metadata
)

# 3. Build and visualize the graph
relationship_detector = RelationshipDetector()
relationships = relationship_detector.detect_relationships(all_entities)

graph_builder = GraphBuilder()
knowledge_graph = graph_builder.build_subgraph(all_entities, relationships)

visualizer = GraphVisualization()
html_viz = visualizer.create_visualization(knowledge_graph, output_format="html")
```

## Anti-Pattern: Direct Knowledge Graph Creation in Base Services

Avoid implementing knowledge graph extraction directly in OCR adapters or ColPali processor. 
This approach duplicates functionality and violates separation of concerns. 