import logging
import time
import functools
from functools import wraps
import os
import json
import psutil
from typing import Dict, Any, List, Callable, Optional, Tuple, Set
import inspect
import platform
import threading
import numpy as np
import torch
from django.utils import timezone
from .models import AnalyticsEvent
import traceback

def log_model_event(event_type, model_id, model_version, model_provider, user=None, duration=None, error=None):
    """
    Log a model-related event to the analytics system
    
    Args:
        event_type: Type of event (model_inference, model_load, model_error, model_update)
        model_id: Identifier for the model
        model_version: Version of the model
        model_provider: Provider of the model (OpenAI, Anthropic, local, etc.)
        user: User who triggered the event (optional)
        duration: Duration of the operation in seconds (optional)
        error: Error message if applicable (optional)
    """
    # Get system resource usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss
    cpu_percent = process.cpu_percent(interval=0.1)
    
    # Create event data
    data = {
        'model_details': {
            'id': model_id,
            'version': model_version,
            'provider': model_provider,
        },
        'performance': {
            'duration_ms': duration * 1000 if duration else None,
            'memory_used_bytes': memory_usage,
            'cpu_percent': cpu_percent,
        }
    }
    
    # Add error information if present
    if error:
        data['error'] = {
            'message': str(error),
            'type': type(error).__name__
        }
    
    # Create the analytics event
    AnalyticsEvent.objects.create(
        user=user,
        event_type=event_type,
        model_id=model_id,
        model_version=model_version,
        model_provider=model_provider,
        response_time=duration,
        data=data,
        resource_usage={
            'memory_bytes': memory_usage,
            'cpu_percent': cpu_percent,
            'disk_usage': dict(psutil.disk_usage('/')),
        }
    )

def monitor_performance(category: str):
    """
    Generischer Performance-Monitoring-Decorator mit Kategorie.
    
    Args:
        category: Kategorie der zu überwachenden Funktion.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    gpu_memory_before = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                
                result = func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_used = memory_after - memory_before
                
                log_data = {
                    "category": category,
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "memory_usage_mb": memory_used,
                }
                
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    gpu_memory_used = gpu_memory_after - gpu_memory_before
                    log_data["gpu_memory_usage_mb"] = gpu_memory_used
                
                # Zusätzliche Informationen sammeln
                context_info = _extract_context_info(args, kwargs, category)
                if context_info:
                    log_data.update(context_info)
                
                logger.info(f"Performance [{category}]: {log_data}")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"Error in {category}.{func.__name__}: {str(e)} after {execution_time:.2f}s")
                raise
                
        return wrapper
    return decorator

def _extract_context_info(args, kwargs, category: str) -> Dict[str, Any]:
    """
    Extrahiert kontextspezifische Informationen aus den Argumenten basierend auf der Kategorie.
    """
    context = {}
    
    try:
        # Für Dokument-Verarbeitung
        if category == "document":
            if len(args) > 1 and isinstance(args[1], str):
                context["document_path"] = args[1]
            elif "document_path" in kwargs:
                context["document_path"] = kwargs["document_path"]
        
        # Für OCR-Verarbeitung
        elif category == "ocr":
            if len(args) > 1 and isinstance(args[1], str):
                context["image_path"] = args[1]
            elif "image_path" in kwargs:
                context["image_path"] = kwargs["image_path"]
        
        # Für ColPali-Verarbeitung
        elif category == "colpali":
            if len(args) > 1:
                if isinstance(args[1], str):
                    context["image_source"] = "path"
                    if len(args[1]) < 100:  # Vermeiden von zu langen Pfaden
                        context["image_path"] = args[1]
                else:
                    context["image_source"] = type(args[1]).__name__
        
        # Für Fusion-Verarbeitung
        elif category == "fusion":
            if len(args) > 2 and isinstance(args[3], str):
                context["document_type"] = args[3]
            elif "document_type" in kwargs:
                context["document_type"] = kwargs["document_type"]
                
        # Für Auswahl/Selektion
        elif category == "selector":
            # Für document_processor_factory
            if len(args) > 1 and isinstance(args[1], str):
                context["file_path"] = os.path.basename(args[1])
                context["file_type"] = os.path.splitext(args[1])[1]
            
            # Für model_selector
            if hasattr(args[0], "available_models") and len(args) > 1:
                context["available_models"] = len(getattr(args[0], "available_models", []))
                if "model_name" in kwargs:
                    context["selected_model"] = kwargs["model_name"]
                    
        # Für erste Auswahl (Dokument-Prozessor-Auswahl)
        elif category == "first_selector":
            document_path = kwargs.get('document_path', kwargs.get('file_path', 'unknown'))
            if isinstance(document_path, str):
                context["doc_type"] = os.path.splitext(document_path)[1].lower() if '.' in document_path else 'unknown'
                context["file_path"] = os.path.basename(document_path)
                
        # Für zweite Auswahl (OCR-Modell-Auswahl)
        elif category == "second_selector":
            image_path = kwargs.get('image_path', 'unknown')
            if isinstance(image_path, str):
                context["img_type"] = os.path.splitext(image_path)[1].lower() if '.' in image_path else 'unknown'
                context["file_path"] = os.path.basename(image_path)
                
            # Zusätzliche Metriken für OCR-Modell-Auswahl
            if hasattr(args[0], 'analyze_document') and isinstance(image_path, str):
                try:
                    analysis = args[0].analyze_document(image_path)
                    context["doc_complexity"] = f"tables:{analysis.get('tables', 0):.1f},equations:{analysis.get('equations', 0):.1f}"
                except:
                    pass
                    
        # Füge bei Bedarf weitere Kategorien hinzu
        
    except Exception as e:
        logger.debug(f"Error extracting context info: {str(e)}")
    
    return context

# Spezifische Monitoring-Decorators für verschiedene Kategorien
def monitor_document_performance(func: Callable) -> Callable:
    """Überwacht die Performance von Dokumentenverarbeitungs-Funktionen."""
    return monitor_performance("document")(func)

def monitor_ocr_performance(func: Callable) -> Callable:
    """Überwacht die Performance von OCR-Funktionen."""
    return monitor_performance("ocr")(func)

def monitor_colpali_performance(func: Callable) -> Callable:
    """Überwacht die Performance von ColPali-Funktionen."""
    return monitor_performance("colpali")(func)

def monitor_fusion_performance(func: Callable) -> Callable:
    """Überwacht die Performance von Fusions-Funktionen."""
    return monitor_performance("fusion")(func)

def monitor_selector_performance(func: Callable) -> Callable:
    """Überwacht die Performance von Auswahl-Funktionen."""
    return monitor_performance("selector")(func)

def monitor_first_selector_performance(func: Callable) -> Callable:
    """Überwacht die Performance von Prozessor-Auswahl-Funktionen."""
    return monitor_performance("first_selector")(func)

def monitor_second_selector_performance(func: Callable) -> Callable:
    """Überwacht die Performance von OCR-Modell-Auswahl-Funktionen."""
    return monitor_performance("second_selector")(func)

# Erweiterter Performance-Decorator mit detaillierten Metriken
def monitor_detailed_performance(category: str, 
                              track_input_sizes: bool = False, 
                              track_output_sizes: bool = False,
                              track_memory_timeline: bool = False):
    """
    Erweiterter Performance-Decorator mit zusätzlichen Detail-Metriken.
    
    Args:
        category: Kategorie der zu überwachenden Funktion
        track_input_sizes: Ob Eingangsgrößen aufgezeichnet werden sollen
        track_output_sizes: Ob Ausgabegrößen aufgezeichnet werden sollen
        track_memory_timeline: Ob ein detailliertes Memory-Profil erstellt werden soll
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Eingangsgrößen messen
            input_sizes = {}
            if track_input_sizes:
                input_sizes = _measure_sizes(list(args) + list(kwargs.values()), 
                                          prefix="input")
            
            # Memory-Timeline initialisieren
            memory_timeline = []
            memory_tracker = None
            
            if track_memory_timeline:
                memory_tracker = _MemoryTracker(interval=0.1)
                memory_tracker.start()
            
            try:
                result = func(*args, **kwargs)
                
                # Ausgabegrößen messen
                output_sizes = {}
                if track_output_sizes:
                    output_sizes = _measure_sizes([result], prefix="output")
                
                end_time = time.time()
                execution_time = end_time - start_time
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_used = memory_after - memory_before
                
                # Memory-Timeline stoppen
                if track_memory_timeline and memory_tracker:
                    memory_tracker.stop()
                    memory_timeline = memory_tracker.get_timeline()
                
                log_data = {
                    "category": category,
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "memory_usage_mb": memory_used,
                }
                
                # Eingangs-/Ausgabegrößen hinzufügen
                if track_input_sizes:
                    log_data.update(input_sizes)
                    
                if track_output_sizes:
                    log_data.update(output_sizes)
                    
                if track_memory_timeline:
                    log_data["memory_timeline"] = {
                        "times": [t[0] for t in memory_timeline],
                        "values": [t[1] for t in memory_timeline],
                        "peak": max([t[1] for t in memory_timeline]) if memory_timeline else 0
                    }
                
                # Kontextinformationen extrahieren
                context_info = _extract_context_info(args, kwargs, category)
                if context_info:
                    log_data.update(context_info)
                
                logger.info(f"Detailed Performance [{category}]: {log_data}")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"Error in {category}.{func.__name__}: {str(e)} after {execution_time:.2f}s")
                logger.error(traceback.format_exc())
                
                if track_memory_timeline and memory_tracker:
                    memory_tracker.stop()
                
                raise
                
        return wrapper
    return decorator

def _measure_sizes(objects, prefix: str) -> Dict[str, Any]:
    """Misst die Größen verschiedener Objekte."""
    sizes = {}
    
    for i, obj in enumerate(objects):
        key = f"{prefix}_{i}"
        
        if isinstance(obj, (str, bytes)):
            sizes[f"{key}_size"] = len(obj)
        elif isinstance(obj, (list, tuple)):
            sizes[f"{key}_length"] = len(obj)
        elif isinstance(obj, dict):
            sizes[f"{key}_items"] = len(obj)
        elif isinstance(obj, np.ndarray):
            sizes[f"{key}_shape"] = list(obj.shape)
            sizes[f"{key}_size_mb"] = obj.nbytes / (1024 * 1024)
        elif hasattr(obj, "__len__"):
            try:
                sizes[f"{key}_length"] = len(obj)
            except:
                pass
    
    return sizes

class _MemoryTracker:
    """Hilfsklasse zum Tracken des Speicherverbrauchs über Zeit."""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.timeline = []
        self.thread = None
        
    def _track(self):
        import threading
        import time
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        
        while self.running:
            current_time = time.time() - start_time
            memory = process.memory_info().rss / (1024 * 1024)  # MB
            self.timeline.append((current_time, memory))
            time.sleep(self.interval)
    
    def start(self):
        import threading
        
        self.running = True
        self.thread = threading.Thread(target=self._track)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def get_timeline(self):
        return self.timeline

def handle_fusion_errors(func):
    """Decorator to handle fusion processing errors."""
    import functools
    import logging
    
    logger = logging.getLogger(__name__)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fusion error in {func.__name__}: {str(e)}")
            # Structure the error response based on function name
            if "best_strategy" in func.__name__:
                return None, "fallback", 0.0
            elif "ensemble" in func.__name__:
                return None, {"error": str(e)}
            else:
                return None
    
    return wrapper

def monitor_orchestration_performance(func):
    """
    Monitor components that coordinate between multiple processors.
    
    This decorator tracks the performance of orchestration functions that coordinate
    multiple processing components. It measures execution time and logs which components
    were used and how many times each was called.
    
    Args:
        func: The function to monitor
        
    Returns:
        wrapper: A wrapped function that includes performance monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Track which components get used and how many times
        components_used = {}
        
        # Create component tracking method if needed
        if not hasattr(args[0], '_track_component'):
            def _track_component(component_name):
                components_used[component_name] = components_used.get(component_name, 0) + 1
            args[0]._track_component = _track_component
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Log orchestration metrics
        execution_time = time.time() - start_time
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Orchestration: {func.__name__}, time={execution_time:.2f}s, "
            f"components_used={len(components_used)}, details={components_used}"
        )
        
        return result
    return wrapper

def calculate_ontological_consistency(graph, ontology=None):
    """
    Measure how well entities and relationships conform to a defined ontology
    
    Args:
        graph: The knowledge graph to evaluate
        ontology: Optional ontology to check against (if None, uses default)
        
    Returns:
        Dict containing ontological consistency metrics
    """
    if not ontology:
        # Try to load default ontology if available
        try:
            from models_app.knowledge_graph.ontology_manager import OntologyManager
            ontology_manager = OntologyManager()
            ontology = ontology_manager.get_ontology()
        except (ImportError, AttributeError):
            # No ontology available, return basic metrics
            return {
                "consistency_score": 0.0,
                "valid_entity_types": 0,
                "valid_relationship_types": 0,
                "unknown_entity_types": 0,
                "unknown_relationship_types": 0
            }
    
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])
    
    # Count valid and invalid entity types
    valid_entity_types = 0
    unknown_entity_types = 0
    entity_type_counts = {}
    
    for entity in entities:
        entity_type = entity.get("type", "unknown")
        entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        # Check if entity type exists in ontology
        if ontology and "entity_types" in ontology and entity_type in ontology["entity_types"]:
            valid_entity_types += 1
        else:
            unknown_entity_types += 1
    
    # Count valid and invalid relationship types
    valid_relationship_types = 0
    unknown_relationship_types = 0
    relationship_type_counts = {}
    
    for rel in relationships:
        rel_type = rel.get("type", "unknown")
        relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
        
        # Check if relationship type exists in ontology
        if ontology and "relationship_types" in ontology and rel_type in ontology["relationship_types"]:
            valid_relationship_types += 1
        else:
            unknown_relationship_types += 1
    
    # Calculate overall consistency score
    total_elements = len(entities) + len(relationships)
    valid_elements = valid_entity_types + valid_relationship_types
    
    consistency_score = valid_elements / total_elements if total_elements > 0 else 0.0
    
    return {
        "consistency_score": consistency_score,
        "valid_entity_types": valid_entity_types,
        "valid_relationship_types": valid_relationship_types,
        "unknown_entity_types": unknown_entity_types,
        "unknown_relationship_types": unknown_relationship_types,
        "entity_type_distribution": entity_type_counts,
        "relationship_type_distribution": relationship_type_counts
    }

def calculate_interlinking_degree(graph):
    """
    Measure percentage of entities connected to external knowledge sources
    
    Args:
        graph: The knowledge graph to evaluate
        
    Returns:
        Dict containing interlinking metrics
    """
    entities = graph.get("entities", [])
    
    # Count entities with external links
    entities_with_external_links = 0
    external_sources = {}
    
    for entity in entities:
        # Check for external links in entity properties
        has_external_link = False
        
        # External links might be in different formats depending on implementation
        # Common patterns include: sameAs, owl:sameAs, externalIds, links, etc.
        for link_property in ["sameAs", "owl:sameAs", "externalIds", "links", "external_references"]:
            external_links = entity.get(link_property, [])
            
            if isinstance(external_links, list) and external_links:
                has_external_link = True
                # Track sources
                for link in external_links:
                    if isinstance(link, dict) and "source" in link:
                        source = link["source"]
                        external_sources[source] = external_sources.get(source, 0) + 1
                    elif isinstance(link, str) and ":" in link:
                        source = link.split(":")[0]
                        external_sources[source] = external_sources.get(source, 0) + 1
            elif isinstance(external_links, dict) and external_links:
                has_external_link = True
                # Track sources from dict
                for source, link in external_links.items():
                    external_sources[source] = external_sources.get(source, 0) + 1
        
        if has_external_link:
            entities_with_external_links += 1
    
    # Calculate interlinking degree
    total_entities = len(entities)
    interlinking_degree = entities_with_external_links / total_entities if total_entities > 0 else 0.0
    
    return {
        "interlinking_degree": interlinking_degree,
        "entities_with_external_links": entities_with_external_links,
        "total_entities": total_entities,
        "external_sources": external_sources
    }

def calculate_schema_completeness(graph, required_properties=None):
    """
    Measure the coverage of required schema properties across entity types
    
    Args:
        graph: The knowledge graph to evaluate
        required_properties: Optional dict mapping entity types to required properties
        
    Returns:
        Dict containing schema completeness metrics
    """
    entities = graph.get("entities", [])
    
    # If no required properties provided, try to load from ontology
    if not required_properties:
        try:
            from models_app.knowledge_graph.ontology_manager import OntologyManager
            ontology_manager = OntologyManager()
            ontology = ontology_manager.get_ontology()
            if ontology and "entity_types" in ontology:
                required_properties = {
                    entity_type: entity_def.get("required_properties", [])
                    for entity_type, entity_def in ontology["entity_types"].items()
                }
        except (ImportError, AttributeError):
            # Create default required properties based on what's in the graph
            required_properties = {}
            entity_types = set(entity.get("type", "unknown") for entity in entities)
            for entity_type in entity_types:
                required_properties[entity_type] = []
    
    # Calculate completeness by entity type
    entity_type_completeness = {}
    
    for entity in entities:
        entity_type = entity.get("type", "unknown")
        properties = entity.get("properties", {})
        
        # Skip if no required properties for this type
        if entity_type not in required_properties:
            continue
        
        # Get required properties for this entity type
        type_required_props = required_properties[entity_type]
        
        # Calculate completeness for this entity
        if not type_required_props:
            # No required properties defined
            entity_completeness = 1.0
        else:
            # Count required properties that are present
            present_required_props = sum(1 for prop in type_required_props if prop in properties)
            entity_completeness = present_required_props / len(type_required_props)
        
        # Update entity type completeness
        if entity_type not in entity_type_completeness:
            entity_type_completeness[entity_type] = {
                "total_entities": 0,
                "completeness_sum": 0.0
            }
        
        entity_type_completeness[entity_type]["total_entities"] += 1
        entity_type_completeness[entity_type]["completeness_sum"] += entity_completeness
    
    # Calculate average completeness for each entity type
    type_completeness = {
        entity_type: {
            "average_completeness": data["completeness_sum"] / data["total_entities"] if data["total_entities"] > 0 else 0.0,
            "entity_count": data["total_entities"]
        }
        for entity_type, data in entity_type_completeness.items()
    }
    
    # Calculate overall schema completeness
    total_entities = sum(data["total_entities"] for data in entity_type_completeness.values())
    overall_completeness = sum(data["completeness_sum"] for data in entity_type_completeness.values())
    
    return {
        "overall_completeness": overall_completeness / total_entities if total_entities > 0 else 0.0,
        "total_entities": total_entities,
        "type_completeness": type_completeness
    }

# Performance metrics
def calculate_performance_metrics(graph_id=None, graph=None, query_count=5):
    """
    Measure knowledge graph performance metrics
    
    Args:
        graph_id: ID of the graph to measure (optional)
        graph: Graph object to measure (optional)
        query_count: Number of test queries to run
        
    Returns:
        Dict containing performance metrics
    """
    import time
    
    # Get graph if only ID provided
    if not graph and graph_id:
        from models_app.knowledge_graph.graph_storage import GraphStorage
        storage = GraphStorage()
        # Time graph retrieval
        start_time = time.time()
        graph = storage.retrieve_graph(graph_id)
        retrieval_time = time.time() - start_time
    else:
        retrieval_time = 0.0
    
    if not graph:
        return {
            "error": "No graph provided or found",
            "retrieval_time": retrieval_time
        }
    
    # Basic graph metrics
    entity_count = len(graph.get("entities", []))
    relationship_count = len(graph.get("relationships", []))
    
    # Create test queries
    test_queries = []
    
    # Pick random pairs of entities for path queries
    import random
    entities = graph.get("entities", [])
    
    if len(entities) >= 2:
        for _ in range(min(query_count, len(entities))):
            src_entity = random.choice(entities)
            tgt_entity = random.choice(entities)
            while src_entity == tgt_entity:
                tgt_entity = random.choice(entities)
            
            test_queries.append({
                "type": "path",
                "source": src_entity["id"],
                "target": tgt_entity["id"]
            })
    
    # Execute test queries and measure time
    query_times = []
    
    for query in test_queries:
        if query["type"] == "path":
            # Measure time to find path between entities
            start_time = time.time()
            path = find_path_between_entities(graph, query["source"], query["target"])
            query_time = time.time() - start_time
            query_times.append(query_time)
    
    # Calculate average query time
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0.0
    
    return {
        "entity_count": entity_count,
        "relationship_count": relationship_count,
        "retrieval_time_ms": retrieval_time * 1000,
        "average_query_time_ms": avg_query_time * 1000,
        "queries_executed": len(query_times)
    }

def find_path_between_entities(graph, source_id, target_id, max_depth=3):
    """Helper function to find path between entities"""
    # Create networkx graph
    import networkx as nx
    G = nx.DiGraph()
    
    # Add nodes and edges
    for entity in graph.get("entities", []):
        G.add_node(entity["id"])
    
    for rel in graph.get("relationships", []):
        if "source" in rel and "target" in rel:
            G.add_edge(rel["source"], rel["target"])
    
    # Try to find shortest path
    try:
        path = nx.shortest_path(G, source=source_id, target=target_id)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

# Update the main evaluation function to include new metrics
def evaluate_knowledge_graph_quality(graph, reference_data=None):
    """
    Evaluate the quality of a knowledge graph
    
    Args:
        graph: The knowledge graph to evaluate
        reference_data: Optional reference data for comparison (gold standard)
        
    Returns:
        Dict containing quality metrics
    """
    metrics = {
        "structural": calculate_structural_metrics(graph),
        "semantic": calculate_semantic_metrics(graph),
        "user_oriented": calculate_user_oriented_metrics(graph),
        "ontological": calculate_ontological_consistency(graph),
        "interlinking": calculate_interlinking_degree(graph),
        "schema_completeness": calculate_schema_completeness(graph),
        "performance": calculate_performance_metrics(graph=graph)
    }
    
    if reference_data:
        metrics["accuracy"] = calculate_accuracy_metrics(graph, reference_data)
    
    return metrics

def calculate_structural_metrics(graph):
    """Calculate structural metrics of the knowledge graph"""
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])
    
    # Count entity types
    entity_types = {}
    for entity in entities:
        entity_type = entity.get("type", "unknown")
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    # Count relationship types
    relationship_types = {}
    for rel in relationships:
        rel_type = rel.get("type", "unknown")
        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
    
    # Calculate network metrics
    from networkx import Graph
    G = Graph()
    
    # Add nodes and edges
    for entity in entities:
        G.add_node(entity["id"])
    
    for rel in relationships:
        if rel["source"] in G and rel["target"] in G:
            G.add_edge(rel["source"], rel["target"])
    
    # Calculate metrics
    try:
        import networkx as nx
        
        # Average degree - average number of connections per entity
        avg_degree = sum(dict(G.degree()).values()) / len(G) if len(G) > 0 else 0
        
        # Density - ratio of actual connections to possible connections
        density = nx.density(G)
        
        # Try to calculate connected components
        connected_components = list(nx.connected_components(G))
        largest_component_size = max([len(c) for c in connected_components]) if connected_components else 0
        
        # Centrality - identify important nodes
        centrality = {}
        if len(G) > 0:
            degree_centrality = nx.degree_centrality(G)
            centrality = {
                "highest_centrality_node": max(degree_centrality.items(), key=lambda x: x[1])[0] if degree_centrality else None,
                "highest_centrality_value": max(degree_centrality.values()) if degree_centrality else 0
            }
    except Exception as e:
        avg_degree = 0
        density = 0
        connected_components = []
        largest_component_size = 0
        centrality = {}
    
    return {
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "entity_types": entity_types,
        "relationship_types": relationship_types,
        "avg_connections_per_entity": avg_degree,
        "graph_density": density,
        "component_count": len(connected_components) if 'connected_components' in locals() else 0,
        "largest_component_size": largest_component_size,
        "centrality": centrality
    }

def calculate_semantic_metrics(graph):
    """Calculate semantic quality metrics"""
    entities = graph.get("entities", [])
    
    # Check entity property completeness
    property_completeness = {}
    for entity in entities:
        entity_type = entity.get("type", "unknown")
        properties = entity.get("properties", {})
        
        if entity_type not in property_completeness:
            property_completeness[entity_type] = {
                "total_entities": 0,
                "property_counts": {}
            }
        
        property_completeness[entity_type]["total_entities"] += 1
        
        for prop_name in properties:
            if prop_name not in property_completeness[entity_type]["property_counts"]:
                property_completeness[entity_type]["property_counts"][prop_name] = 0
            property_completeness[entity_type]["property_counts"][prop_name] += 1
    
    # Calculate property coverage percentages
    property_coverage = {}
    for entity_type, data in property_completeness.items():
        total = data["total_entities"]
        property_coverage[entity_type] = {
            prop: count / total 
            for prop, count in data["property_counts"].items()
        }
    
    return {
        "property_completeness": property_completeness,
        "property_coverage": property_coverage
    }

def calculate_user_oriented_metrics(graph, query_log=None):
    """
    Calculate user-oriented metrics - how useful the graph is for actual user queries
    
    This is what makes these metrics "user-oriented" - they measure how well
    the knowledge graph serves actual user information needs rather than just
    measuring structural properties.
    
    Args:
        graph: The knowledge graph
        query_log: Optional log of user queries to test against the graph
        
    Returns:
        Dict of user-oriented metrics
    """
    # If we don't have a query log, we can only estimate potential usefulness
    if not query_log:
        entities = graph.get("entities", [])
        relationships = graph.get("relationships", [])
        
        # Estimate query potential based on entity and relationship diversity
        entity_types = set(entity.get("type", "unknown") for entity in entities)
        relationship_types = set(rel.get("type", "unknown") for rel in relationships)
        
        return {
            "entity_type_diversity": len(entity_types),
            "relationship_type_diversity": len(relationship_types),
            "estimated_query_potential": len(entity_types) * len(relationship_types)
        }
    
    # With a query log, we can measure actual query answering capability
    answered_queries = 0
    partial_answers = 0
    
    for query in query_log:
        # Simulate query answering logic
        # This would use the knowledge graph to attempt to answer the query
        # and then score how well it did
        
        # Simplified example:
        if contains_entities_for_query(graph, query):
            if contains_relationships_for_query(graph, query):
                answered_queries += 1
            else:
                partial_answers += 1
    
    return {
        "query_coverage": answered_queries / len(query_log) if query_log else 0,
        "partial_query_coverage": partial_answers / len(query_log) if query_log else 0,
        "total_query_coverage": (answered_queries + partial_answers) / len(query_log) if query_log else 0
    }

def contains_entities_for_query(graph, query):
    """Check if graph contains entities mentioned in query"""
    # Implementation would depend on query format
    # Simplified example:
    return True

def contains_relationships_for_query(graph, query):
    """Check if graph contains relationships needed for query"""
    # Implementation would depend on query format
    # Simplified example:
    return True

def calculate_accuracy_metrics(graph, reference_data):
    """
    Calculate accuracy by comparing to reference data
    
    Args:
        graph: The graph to evaluate
        reference_data: Gold standard data for comparison
        
    Returns:
        Dict of accuracy metrics
    """
    # Implement comparison logic
    # This would typically involve entity matching and relationship validation
    
    # Simplified metrics
    return {
        "entity_precision": 0.9,  # Placeholder values
        "entity_recall": 0.8,
        "entity_f1": 0.85,
        "relationship_precision": 0.7,
        "relationship_recall": 0.65,
        "relationship_f1": 0.67
    }

# Add to existing monitor_performance decorator to track KG metrics
def monitor_kg_performance(func):
    """
    Monitor knowledge graph generation performance.
    
    This decorator tracks performance and quality metrics for knowledge graph operations
    and saves them to the analytics database for reporting and visualization.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Capture function context
        context = _extract_context_info(args, kwargs, 'kg_operation')
        
        try:
            # Run the function
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log error
            logger.error(f"KG Operation failed: {func.__name__}, error={str(e)}, time={execution_time:.2f}s")
            
            # Save failed operation metrics
            try:
                event_data = {
                    'event_type': 'kg_error',
                    'operation': func.__name__,
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e),
                    'context': context,
                    'time': timezone.now()
                }
                
                # Add event to database
                AnalyticsEvent.objects.create(
                    event_type='kg_error',
                    operation=func.__name__,
                    response_time=execution_time,
                    status=False,
                    data=event_data
                )
            except Exception as log_error:
                logger.error(f"Failed to log KG error: {str(log_error)}")
                
            # Re-raise the original exception
            raise
        
        # Calculate execution metrics
        end_time = time.time()
        execution_time = end_time - start_time
        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Get graph from result
        graph = None
        if result:
            if isinstance(result, dict):
                if "entities" in result and "relationships" in result:
                    # Result is the graph itself
                    graph = result
                elif "graph" in result:
                    # Result has graph as a property
                    graph = result["graph"]
                elif "graph_id" in result:
                    # Result has a graph ID, try to retrieve it
                    try:
                        from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
                        kg_manager = KnowledgeGraphManager()
                        graph = kg_manager.graph_storage.retrieve_graph(result["graph_id"])
                    except Exception as e:
                        logger.warning(f"Could not retrieve graph with ID {result.get('graph_id')}: {str(e)}")
        
        # Calculate quality metrics if we have a graph
        quality_metrics = {}
        if graph:
            try:
                quality_metrics = evaluate_knowledge_graph_quality(graph)
                
                # Extract key metrics for logging
                entity_count = quality_metrics['structural']['entity_count']
                relationship_count = quality_metrics['structural']['relationship_count']
                completeness = quality_metrics.get('schema_completeness', {}).get('overall_completeness', 0)
                density = quality_metrics['structural'].get('graph_density', 0)
                
                # Log metrics
                logger.info(f"KG Operation: {func.__name__}, time={execution_time:.2f}s, "
                           f"entities={entity_count}, relationships={relationship_count}, "
                           f"completeness={completeness:.1f}%, density={density:.3f}")
                
                # Add quality metrics to analytics database
                event_data = {
                    'event_type': 'kg_metrics',
                    'operation': func.__name__,
                    'execution_time': execution_time,
                    'memory_used_mb': memory_used,
                    'success': success,
                    'metrics': quality_metrics,
                    'context': context,
                    'time': timezone.now().isoformat()
                }
                
                # Include graph ID if available
                if isinstance(result, dict) and "graph_id" in result:
                    event_data['graph_id'] = result["graph_id"]
                
                # Add event to database
                try:
                    AnalyticsEvent.objects.create(
                        event_type='kg_metrics',
                        operation=func.__name__,
                        model_id=event_data.get('graph_id', ''),
                        response_time=execution_time,
                        status=True,
                        data=event_data
                    )
                except Exception as db_error:
                    logger.error(f"Failed to save KG metrics to database: {str(db_error)}")
                
            except Exception as metrics_error:
                logger.error(f"Error calculating quality metrics: {str(metrics_error)}")
        
        # Attach metrics to result if it's a dictionary
        if isinstance(result, dict) and quality_metrics:
            # Only attach metrics if they're not already there
            if "metrics" not in result:
                result["metrics"] = quality_metrics
        
        return result
    
    return wrapper

def monitor_kg_llm_integration(func):
    """
    Monitor knowledge graph LLM integration performance.
    
    This decorator tracks performance metrics for operations that involve
    both knowledge graphs and language models, such as graph-augmented prompting.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Capture function context
        context = _extract_context_info(args, kwargs, 'kg_llm_integration')
        
        # Extract graph_id if present in kwargs
        graph_id = kwargs.get('graph_id', None)
        if not graph_id and len(args) > 1:
            # Try to get graph_id from positional args (assuming second arg is graph_id in most methods)
            graph_id = args[1]
        
        # Extract query if present
        query = kwargs.get('query', None)
        if not query and len(args) > 0:
            # Try to get query from positional args (assuming first arg is query in most methods)
            query = args[0]
            
        try:
            # Run the function
            result = func(*args, **kwargs)
            success = True
            
            # Calculate execution metrics
            end_time = time.time()
            execution_time = end_time - start_time
            memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Extract metrics from result if available
            response_metrics = {}
            validation_result = None
            
            if isinstance(result, dict):
                if 'validation' in result:
                    validation_result = result['validation']
                if 'metadata' in result:
                    response_metrics = result['metadata']
            
            # Create event data
            event_data = {
                'event_type': 'kg_llm_integration',
                'operation': func.__name__,
                'execution_time': execution_time,
                'memory_used_mb': memory_used,
                'success': success,
                'context': context,
                'time': timezone.now().isoformat(),
                'graph_id': graph_id,
                'query_length': len(query) if isinstance(query, str) else 0
            }
            
            # Add validation metrics if available
            if validation_result:
                event_data['validation'] = {
                    'status': validation_result.get('status', 'unknown'),
                    'verification_score': validation_result.get('verification_score', 0),
                    'contradiction_score': validation_result.get('contradiction_score', 0),
                    'verified_claims_count': len(validation_result.get('verified_claims', [])),
                    'unverified_claims_count': len(validation_result.get('unverified_claims', [])),
                    'contradicted_claims_count': len(validation_result.get('contradicted_claims', []))
                }
            
            # Add response metrics if available
            if response_metrics:
                event_data['response_metrics'] = response_metrics
            
            # Log the successful operation
            logger.info(f"KG-LLM Integration: {func.__name__}, time={execution_time:.2f}s, "
                       f"graph_id={graph_id}, query_len={event_data['query_length']}")
            
            if validation_result:
                logger.info(f"  Validation: status={validation_result.get('status', 'unknown')}, "
                          f"verification={validation_result.get('verification_score', 0):.2f}, "
                          f"contradiction={validation_result.get('contradiction_score', 0):.2f}")
            
            # Save metrics to database
            try:
                AnalyticsEvent.objects.create(
                    event_type='kg_llm_integration',
                    operation=func.__name__,
                    model_id=str(graph_id) if graph_id else '',
                    response_time=execution_time,
                    status=True,
                    data=event_data
                )
            except Exception as db_error:
                logger.error(f"Failed to save KG-LLM metrics to database: {str(db_error)}")
            
            return result
            
        except Exception as e:
            # Calculate metrics for failed operation
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log error
            logger.error(f"KG-LLM Integration failed: {func.__name__}, error={str(e)}, time={execution_time:.2f}s")
            
            # Save failed operation metrics
            try:
                event_data = {
                    'event_type': 'kg_llm_error',
                    'operation': func.__name__,
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e),
                    'context': context,
                    'time': timezone.now().isoformat(),
                    'graph_id': graph_id,
                    'query_length': len(query) if isinstance(query, str) else 0
                }
                
                # Add event to database
                AnalyticsEvent.objects.create(
                    event_type='kg_llm_error',
                    operation=func.__name__,
                    model_id=str(graph_id) if graph_id else '',
                    response_time=execution_time,
                    status=False,
                    data=event_data
                )
            except Exception as log_error:
                logger.error(f"Failed to log KG-LLM error: {str(log_error)}")
            
            # Re-raise the original exception
            raise
    
    return wrapper

    
    