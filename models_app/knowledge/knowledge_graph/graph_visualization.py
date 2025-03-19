"""
Knowledge graph visualization utilities.

This module provides functionality to visualize knowledge graphs in various formats
including interactive HTML, static image visualizations, and network diagrams.
"""

import logging
import json
import os
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from django.conf import settings
from django.template.loader import render_to_string

logger = logging.getLogger(__name__)

class GraphVisualization:
    """
    Creates visualizations for knowledge graphs.
    
    This class provides methods to visualize knowledge graphs in different formats:
    - Interactive HTML visualizations (using D3.js or vis.js)
    - Static image visualizations (using matplotlib)
    - Network diagrams (using networkx)
    
    Configuration options:
    - DEFAULT_NODE_COLOR: Default color for nodes
    - DEFAULT_EDGE_COLOR: Default color for edges
    - MAX_NODES_INTERACTIVE: Maximum number of nodes for interactive visualizations
    - DEFAULT_VISUALIZATION_TYPE: Default visualization type (html, static, network)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the graph visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load config from settings if not provided
        if not self.config:
            self.config = {
                'node_color': getattr(settings, 'GRAPH_VIZ_NODE_COLOR', '#1f77b4'),
                'edge_color': getattr(settings, 'GRAPH_VIZ_EDGE_COLOR', '#7f7f7f'),
                'max_nodes_interactive': getattr(settings, 'GRAPH_VIZ_MAX_NODES', 500),
                'default_type': getattr(settings, 'GRAPH_VIZ_DEFAULT_TYPE', 'html'),
                'font_size': getattr(settings, 'GRAPH_VIZ_FONT_SIZE', 10),
                'width': getattr(settings, 'GRAPH_VIZ_WIDTH', 800),
                'height': getattr(settings, 'GRAPH_VIZ_HEIGHT', 600),
                'template_dir': getattr(settings, 'GRAPH_VIZ_TEMPLATE_DIR', 'knowledge_graph/templates/')
            }
            
        logger.info(f"GraphVisualization initialized with config: {self.config}")
        
        # Color maps for different entity and relationship types
        self.entity_colors = {
            'person': '#e41a1c',
            'organization': '#377eb8',
            'location': '#4daf4a',
            'date': '#984ea3',
            'product': '#ff7f00',
            'visual_element': '#a65628',
            'text_block': '#f781bf',
            'unknown': '#999999'
        }
        
        self.relationship_colors = {
            'part_of': '#1f77b4',
            'belongs_to': '#ff7f0e',
            'contains': '#2ca02c',
            'works_for': '#d62728',
            'located_in': '#9467bd',
            'semantically_related_to': '#8c564b',
            'visually_similar_to': '#e377c2',
            'above': '#7f7f7f',
            'below': '#bcbd22',
            'left_of': '#17becf',
            'right_of': '#aec7e8',
            'co_occurs_with': '#ffbb78',
            'unknown': '#999999'
        }
    
    def create_visualization(self, graph: Dict[str, Any], 
                           output_format: str = None, 
                           output_path: Optional[str] = None,
                           options: Dict[str, Any] = None) -> Union[str, bytes, None]:
        """
        Create a visualization of the knowledge graph.
        
        This is the main entry point for creating visualizations. It determines
        the appropriate visualization type based on the graph size and options.
        
        Args:
            graph: The knowledge graph to visualize
            output_format: Format of the visualization ('html', 'png', 'svg', 'json')
            output_path: Optional path to save the visualization
            options: Additional options for customization
            
        Returns:
            Visualization output (HTML string, image bytes, or None if saved to file)
        """
        options = options or {}
        
        # Determine output format if not specified
        if not output_format:
            output_format = self.config.get('default_type', 'html')
            
        # Check if graph is valid
        if not self._validate_graph(graph):
            logger.error("Invalid graph structure for visualization")
            return None
            
        # Check if graph is too large for interactive visualization
        num_nodes = len(graph.get('entities', []))
        max_nodes = self.config.get('max_nodes_interactive', 500)
        
        if num_nodes > max_nodes and output_format == 'html':
            logger.warning(f"Graph too large for interactive visualization ({num_nodes} nodes). Using static visualization.")
            output_format = 'png'
            
        # Create visualization based on format
        if output_format == 'html':
            return self.create_html_visualization(graph, output_path, options)
        elif output_format == 'png':
            return self.create_static_visualization(graph, 'png', output_path, options)
        elif output_format == 'svg':
            return self.create_static_visualization(graph, 'svg', output_path, options)
        elif output_format == 'json':
            return self.create_json_visualization(graph, output_path, options)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return None
    
    def create_html_visualization(self, graph: Dict[str, Any], 
                                output_path: Optional[str] = None,
                                options: Dict[str, Any] = None) -> Optional[str]:
        """
        Create an interactive HTML visualization of the knowledge graph.
        
        This method creates an interactive HTML visualization using D3.js or vis.js.
        
        Args:
            graph: The knowledge graph to visualize
            output_path: Optional path to save the HTML file
            options: Additional options for customization
            
        Returns:
            HTML string or None if saved to file
        """
        options = options or {}
        
        # Convert graph to visualization format
        viz_data = self._prepare_graph_for_visualization(graph, options)
        
        # Determine visualization library (d3 or vis.js)
        viz_library = options.get('viz_library', 'vis')
        
        # Create HTML using template
        if viz_library == 'd3':
            template_name = 'd3_graph_template.html'
        else:
            template_name = 'visjs_graph_template.html'
            
        try:
            template_dir = self.config.get('template_dir', 'knowledge_graph/templates/')
            context = {
                'graph_data': json.dumps(viz_data),
                'graph_id': f"graph_{uuid.uuid4().hex[:8]}",
                'width': options.get('width', self.config.get('width', 800)),
                'height': options.get('height', self.config.get('height', 600)),
                'title': options.get('title', graph.get('metadata', {}).get('title', 'Knowledge Graph')),
                'show_legend': options.get('show_legend', True),
                'timestamp': datetime.now().isoformat()
            }
            
            html_content = render_to_string(f"{template_dir}/{template_name}", context)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML visualization saved to {output_path}")
                return None
                
            return html_content
            
        except Exception as e:
            logger.error(f"Error creating HTML visualization: {e}")
            return None
    
    def create_static_visualization(self, graph: Dict[str, Any], 
                                  format_type: str = 'png',
                                  output_path: Optional[str] = None,
                                  options: Dict[str, Any] = None) -> Optional[bytes]:
        """
        Create a static image visualization of the knowledge graph.
        
        This method creates a static image (PNG or SVG) using matplotlib and networkx.
        
        Args:
            graph: The knowledge graph to visualize
            format_type: Image format ('png' or 'svg')
            output_path: Optional path to save the image file
            options: Additional options for customization
            
        Returns:
            Image bytes or None if saved to file
        """
        options = options or {}
        
        # Create networkx graph
        G = self._convert_to_networkx(graph)
        
        # Set up figure
        fig_width = options.get('width', self.config.get('width', 800)) / 100
        fig_height = options.get('height', self.config.get('height', 600)) / 100
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # Get node positions
        layout_type = options.get('layout', 'spring')
        pos = self._get_graph_layout(G, layout_type)
        
        # Get node colors based on entity types
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'unknown')
            node_colors.append(self.entity_colors.get(node_type, self.config.get('node_color')))
            
        # Get edge colors based on relationship types
        edge_colors = []
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_colors.append(self.relationship_colors.get(edge_type, self.config.get('edge_color')))
            
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, 
                             node_size=options.get('node_size', 300))
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6,
                             width=options.get('edge_width', 1.0),
                             arrowsize=options.get('arrow_size', 10))
        
        # Add labels if requested
        if options.get('show_labels', True):
            font_size = options.get('font_size', self.config.get('font_size', 10))
            nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
            
        # Add title
        title = options.get('title', graph.get('metadata', {}).get('title', 'Knowledge Graph'))
        plt.title(title)
        
        # Remove axes
        plt.axis('off')
        
        # Add legend if requested
        if options.get('show_legend', True):
            # Create legend for node types
            node_types = set()
            for entity in graph.get('entities', []):
                node_types.add(entity.get('type', 'unknown'))
                
            # Create legend for edge types
            edge_types = set()
            for relationship in graph.get('relationships', []):
                edge_types.add(relationship.get('type', 'unknown'))
                
            # Add legend elements
            legend_elements = []
            for node_type in node_types:
                color = self.entity_colors.get(node_type, self.config.get('node_color'))
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color, markersize=10, 
                                              label=f"Entity: {node_type}"))
                                              
            for edge_type in edge_types:
                color = self.relationship_colors.get(edge_type, self.config.get('edge_color'))
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                              label=f"Relation: {edge_type}"))
                                              
            ax.legend(handles=legend_elements, loc='best')
            
        # Save or return result
        if output_path:
            plt.savefig(output_path, format=format_type, bbox_inches='tight')
            plt.close()
            logger.info(f"Static visualization saved to {output_path}")
            return None
        else:
            buf = BytesIO()
            plt.savefig(buf, format=format_type, bbox_inches='tight')
            plt.close()
            return buf.getvalue()
    
    def create_json_visualization(self, graph: Dict[str, Any], 
                                output_path: Optional[str] = None,
                                options: Dict[str, Any] = None) -> Optional[str]:
        """
        Create a JSON representation of the knowledge graph for visualization.
        
        This method creates a JSON representation that can be used by various
        visualization libraries.
        
        Args:
            graph: The knowledge graph to visualize
            output_path: Optional path to save the JSON file
            options: Additional options for customization
            
        Returns:
            JSON string or None if saved to file
        """
        options = options or {}
        
        # Convert graph to visualization format
        viz_data = self._prepare_graph_for_visualization(graph, options)
        
        # Convert to JSON
        json_data = json.dumps(viz_data, indent=2)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            logger.info(f"JSON visualization saved to {output_path}")
            return None
            
        return json_data
    
    def create_embedded_visualization(self, graph: Dict[str, Any], 
                                    options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create an embeddable visualization for web applications.
        
        This method creates an embeddable visualization that can be inserted
        into a web application (e.g., as part of a Django view).
        
        Args:
            graph: The knowledge graph to visualize
            options: Additional options for customization
            
        Returns:
            Dictionary with HTML, CSS, and JavaScript components
        """
        options = options or {}
        
        # Convert graph to visualization format
        viz_data = self._prepare_graph_for_visualization(graph, options)
        
        # Determine visualization library (d3 or vis.js)
        viz_library = options.get('viz_library', 'vis')
        
        # Create embedded components
        if viz_library == 'd3':
            template_name = 'd3_embedded_template.html'
            css_template = 'd3_embedded_style.css'
            js_template = 'd3_embedded_script.js'
        else:
            template_name = 'visjs_embedded_template.html'
            css_template = 'visjs_embedded_style.css'
            js_template = 'visjs_embedded_script.js'
            
        try:
            template_dir = self.config.get('template_dir', 'knowledge_graph/templates/')
            context = {
                'graph_data': json.dumps(viz_data),
                'graph_id': f"graph_{uuid.uuid4().hex[:8]}",
                'width': options.get('width', self.config.get('width', 800)),
                'height': options.get('height', self.config.get('height', 600)),
                'title': options.get('title', graph.get('metadata', {}).get('title', 'Knowledge Graph')),
                'show_legend': options.get('show_legend', True),
                'timestamp': datetime.now().isoformat()
            }
            
            html_content = render_to_string(f"{template_dir}/{template_name}", context)
            css_content = render_to_string(f"{template_dir}/{css_template}", context)
            js_content = render_to_string(f"{template_dir}/{js_template}", context)
            
            return {
                'html': html_content,
                'css': css_content,
                'js': js_content,
                'graph_id': context['graph_id']
            }
            
        except Exception as e:
            logger.error(f"Error creating embedded visualization: {e}")
            return {
                'html': f"<div>Error creating visualization: {e}</div>",
                'css': '',
                'js': '',
                'graph_id': f"error_{uuid.uuid4().hex[:8]}"
            }
    
    def create_thumbnail(self, graph: Dict[str, Any], 
                        width: int = 200,
                        height: int = 150,
                        output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Create a thumbnail image of the knowledge graph.
        
        This method creates a small thumbnail image of the knowledge graph,
        useful for previews in interfaces.
        
        Args:
            graph: The knowledge graph to visualize
            width: Thumbnail width in pixels
            height: Thumbnail height in pixels
            output_path: Optional path to save the thumbnail file
            
        Returns:
            Thumbnail image bytes or None if saved to file
        """
        # Create static visualization with appropriate options
        options = {
            'width': width * 2,  # Create larger than needed for better quality
            'height': height * 2,
            'show_labels': False,
            'show_legend': False,
            'node_size': 100,
            'edge_width': 0.5
        }
        
        image_bytes = self.create_static_visualization(graph, 'png', None, options)
        
        if not image_bytes:
            return None
            
        # Resize to thumbnail dimensions
        try:
            image = Image.open(BytesIO(image_bytes))
            thumbnail = image.resize((width, height), Image.LANCZOS)
            
            # Save to file if path provided
            if output_path:
                thumbnail.save(output_path, format='PNG')
                logger.info(f"Thumbnail saved to {output_path}")
                return None
                
            # Return thumbnail bytes
            buf = BytesIO()
            thumbnail.save(buf, format='PNG')
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return None
    
    def _validate_graph(self, graph: Dict[str, Any]) -> bool:
        """
        Validate that the graph has the required structure for visualization.
        
        Args:
            graph: The knowledge graph to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if required fields are present
            if 'entities' not in graph or 'relationships' not in graph:
                logger.error("Graph missing required fields: 'entities' and 'relationships'")
                return False
                
            # Check if entities have required fields
            for entity in graph.get('entities', []):
                if 'id' not in entity:
                    logger.error("Entity missing required field: 'id'")
                    return False
                    
            # Check if relationships have required fields
            for relationship in graph.get('relationships', []):
                if 'source' not in relationship or 'target' not in relationship:
                    logger.error("Relationship missing required fields: 'source' and 'target'")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating graph: {e}")
            return False
    
    def _prepare_graph_for_visualization(self, graph: Dict[str, Any], 
                                       options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare the graph data for visualization.
        
        This method converts the knowledge graph data into a format suitable
        for visualization libraries (D3.js, vis.js).
        
        Args:
            graph: The knowledge graph to prepare
            options: Additional options for customization
            
        Returns:
            Prepared graph data for visualization
        """
        options = options or {}
        
        # Get entities and relationships
        entities = graph.get('entities', [])
        relationships = graph.get('relationships', [])
        
        # Filter out entities and relationships with low confidence
        min_confidence = options.get('min_confidence', 0.5)
        
        filtered_entities = [
            entity for entity in entities
            if entity.get('confidence', 1.0) >= min_confidence
        ]
        
        filtered_relationships = [
            rel for rel in relationships
            if rel.get('confidence', 1.0) >= min_confidence
        ]
        
        # Create visualization data structure
        viz_data = {
            'nodes': [],
            'edges': [],
            'metadata': graph.get('metadata', {})
        }
        
        # Process entities
        entity_map = {}  # Map entity IDs to indices
        
        for i, entity in enumerate(filtered_entities):
            entity_id = entity.get('id')
            entity_type = entity.get('type', 'unknown')
            entity_text = entity.get('text', '')
            
            # Determine node color based on entity type
            color = self.entity_colors.get(entity_type, self.config.get('node_color'))
            
            # Create node data
            node_data = {
                'id': entity_id,
                'label': entity_text[:20] + ('...' if len(entity_text) > 20 else ''),
                'title': self._create_entity_tooltip(entity),
                'color': color,
                'group': entity_type,
                'value': entity.get('confidence', 1.0) * 10  # Node size based on confidence
            }
            
            # Add custom node properties
            if 'attributes' in entity:
                for key, value in entity.get('attributes', {}).items():
                    if key not in node_data and isinstance(value, (str, int, float, bool)):
                        node_data[key] = value
                        
            viz_data['nodes'].append(node_data)
            entity_map[entity_id] = i
            
        # Process relationships
        for relationship in filtered_relationships:
            source_id = relationship.get('source')
            target_id = relationship.get('target')
            
            # Skip if source or target entity not in filtered entities
            if source_id not in entity_map or target_id not in entity_map:
                continue
                
            relationship_type = relationship.get('type', 'unknown')
            
            # Determine edge color based on relationship type
            color = self.relationship_colors.get(relationship_type, self.config.get('edge_color'))
            
            # Create edge data
            edge_data = {
                'id': relationship.get('id', f"edge_{uuid.uuid4().hex[:8]}"),
                'from': source_id,
                'to': target_id,
                'label': relationship_type,
                'title': self._create_relationship_tooltip(relationship),
                'color': color,
                'width': relationship.get('confidence', 1.0) * 3  # Edge width based on confidence
            }
            
            # Add custom edge properties
            if 'attributes' in relationship:
                for key, value in relationship.get('attributes', {}).items():
                    if key not in edge_data and isinstance(value, (str, int, float, bool)):
                        edge_data[key] = value
                        
            viz_data['edges'].append(edge_data)
            
        return viz_data
    
    def _convert_to_networkx(self, graph: Dict[str, Any]) -> nx.Graph:
        """
        Convert the knowledge graph to a NetworkX graph.
        
        Args:
            graph: The knowledge graph to convert
            
        Returns:
            NetworkX graph object
        """
        # Create directed or undirected graph
        if graph.get('type') == 'directed':
            G = nx.DiGraph()
        else:
            G = nx.Graph()
            
        # Add nodes (entities)
        for entity in graph.get('entities', []):
            entity_id = entity.get('id')
            if entity_id:
                G.add_node(entity_id, **entity)
                
        # Add edges (relationships)
        for relationship in graph.get('relationships', []):
            source_id = relationship.get('source')
            target_id = relationship.get('target')
            
            if source_id and target_id:
                G.add_edge(source_id, target_id, **relationship)
                
        return G
    
    def _get_graph_layout(self, G: nx.Graph, layout_type: str = 'spring') -> Dict[Any, Tuple[float, float]]:
        """
        Get node positions for the graph layout.
        
        Args:
            G: NetworkX graph
            layout_type: Type of layout ('spring', 'circular', 'kamada_kawai', etc.)
            
        Returns:
            Dictionary mapping nodes to positions
        """
        if layout_type == 'spring':
            return nx.spring_layout(G, seed=42)
        elif layout_type == 'circular':
            return nx.circular_layout(G)
        elif layout_type == 'kamada_kawai':
            return nx.kamada_kawai_layout(G)
        elif layout_type == 'spectral':
            return nx.spectral_layout(G)
        elif layout_type == 'shell':
            return nx.shell_layout(G)
        elif layout_type == 'planar':
            try:
                return nx.planar_layout(G)
            except nx.NetworkXException:
                logger.warning("Graph is not planar, falling back to spring layout")
                return nx.spring_layout(G, seed=42)
        else:
            return nx.spring_layout(G, seed=42)
    
    def _create_entity_tooltip(self, entity: Dict[str, Any]) -> str:
        """
        Create a tooltip HTML for an entity.
        
        Args:
            entity: Entity data
            
        Returns:
            HTML tooltip string
        """
        entity_type = entity.get('type', 'Unknown')
        entity_text = entity.get('text', '')
        confidence = entity.get('confidence', 0.0)
        
        tooltip = f"<div class='tooltip'>"
        tooltip += f"<strong>Type:</strong> {entity_type}<br>"
        
        if entity_text:
            tooltip += f"<strong>Text:</strong> {entity_text}<br>"
            
        tooltip += f"<strong>Confidence:</strong> {confidence:.2f}<br>"
        
        # Add metadata if available
        if 'metadata' in entity:
            tooltip += "<hr>"
            for key, value in entity.get('metadata', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    tooltip += f"<strong>{key}:</strong> {value}<br>"
                    
        # Add attributes if available
        if 'attributes' in entity:
            tooltip += "<hr>"
            for key, value in entity.get('attributes', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    tooltip += f"<strong>{key}:</strong> {value}<br>"
                    
        tooltip += "</div>"
        return tooltip
    
    def _create_relationship_tooltip(self, relationship: Dict[str, Any]) -> str:
        """
        Create a tooltip HTML for a relationship.
        
        Args:
            relationship: Relationship data
            
        Returns:
            HTML tooltip string
        """
        relationship_type = relationship.get('type', 'Unknown')
        confidence = relationship.get('confidence', 0.0)
        
        tooltip = f"<div class='tooltip'>"
        tooltip += f"<strong>Type:</strong> {relationship_type}<br>"
        tooltip += f"<strong>Confidence:</strong> {confidence:.2f}<br>"
        
        # Add metadata if available
        if 'metadata' in relationship:
            tooltip += "<hr>"
            for key, value in relationship.get('metadata', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    tooltip += f"<strong>{key}:</strong> {value}<br>"
                    
        # Add attributes if available
        if 'attributes' in relationship:
            tooltip += "<hr>"
            for key, value in relationship.get('attributes', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    tooltip += f"<strong>{key}:</strong> {value}<br>"
                    
        tooltip += "</div>"
        return tooltip
