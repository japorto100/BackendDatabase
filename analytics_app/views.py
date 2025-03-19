from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import AnalyticsEvent
from .serializers import AnalyticsEventSerializer
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.db.models import Avg, Count
from django.utils import timezone
from datetime import timedelta
from chat_app.models import ChatSession, Message
from models_app.models import UploadedFile
from models_app.knowledge_graph.graph_visualization import GraphVisualization
from models_app.knowledge_graph.knowledge_graph_manager import KnowledgeGraphManager
from django.http import JsonResponse
import json
import logging
from models_app.ai_models.utils.common.metrics import get_vision_metrics, export_all_metrics
from models_app.ai_models.vision.vision_factory import VisionProviderFactory

logger = logging.getLogger(__name__)

class AnalyticsListView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get analytics data (admin only)"""
        if not request.user.is_staff:
            return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
            
        events = AnalyticsEvent.objects.all()[:100]  # Limit to 100 most recent events
        serializer = AnalyticsEventSerializer(events, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        """Log an analytics event"""
        serializer = AnalyticsEventSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@login_required
@user_passes_test(lambda u: u.is_staff)
def dashboard_view(request):
    """Render the admin dashboard"""
    return render(request, 'analytics_app/dashboard.html')

class DashboardDataView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get dashboard data for admin"""
        if not request.user.is_staff:
            return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
        
        # Get basic stats
        stats = {
            'total_users': User.objects.count(),
            'total_chats': ChatSession.objects.count(),
            'total_messages': Message.objects.count(),
            'total_files': UploadedFile.objects.count()
        }
        
        # Get API requests for the last 7 days
        end_date = timezone.now()
        start_date = end_date - timedelta(days=7)
        
        # Prepare data for API requests chart
        api_requests = {
            'labels': [],
            'values': []
        }
        
        for i in range(7):
            day = end_date - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            count = AnalyticsEvent.objects.filter(
                event_type='api_request',
                timestamp__gte=day_start,
                timestamp__lte=day_end
            ).count()
            
            api_requests['labels'].append(day_start.strftime('%Y-%m-%d'))
            api_requests['values'].append(count)
        
        # Reverse the lists to show chronological order
        api_requests['labels'].reverse()
        api_requests['values'].reverse()
        
        # Get average response times by endpoint
        response_times = {
            'endpoints': [],
            'times': []
        }
        
        endpoint_times = AnalyticsEvent.objects.filter(
            event_type='api_request',
            timestamp__gte=start_date
        ).values('endpoint').annotate(
            avg_time=Avg('response_time')
        ).order_by('-avg_time')[:5]
        
        for item in endpoint_times:
            response_times['endpoints'].append(item['endpoint'])
            response_times['times'].append(item['avg_time'] * 1000)  # Convert to ms
        
        # Get recent requests with performance data
        recent_requests = AnalyticsEvent.objects.filter(
            event_type='api_request'
        ).order_by('-timestamp')[:10]
        
        recent_requests_data = []
        for req in recent_requests:
            performance_data = req.data.get('performance', {})
            recent_requests_data.append({
                'timestamp': req.timestamp,
                'user': req.user.username if req.user else None,
                'endpoint': req.endpoint,
                'method': req.method,
                'status_code': req.status_code,
                'response_time': req.response_time,
                'memory_used': performance_data.get('memory_used_bytes', 0) / (1024 * 1024),  # Convert to MB
                'cpu_percent': performance_data.get('cpu_percent', 0)
            })
        
        # Get performance metrics over time
        performance_metrics = {
            'labels': [],
            'response_times': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        
        # Group by hour for the last 24 hours
        for i in range(24):
            hour = end_date - timedelta(hours=i)
            hour_start = hour.replace(minute=0, second=0, microsecond=0)
            hour_end = hour.replace(minute=59, second=59, microsecond=999999)
            
            events = AnalyticsEvent.objects.filter(
                event_type='api_request',
                timestamp__gte=hour_start,
                timestamp__lte=hour_end
            )
            
            avg_response_time = events.aggregate(Avg('response_time'))['response_time__avg'] or 0
            
            # Calculate average memory and CPU usage
            memory_usage = 0
            cpu_usage = 0
            count = 0
            
            for event in events:
                performance_data = event.data.get('performance', {})
                if 'memory_used_bytes' in performance_data:
                    memory_usage += performance_data['memory_used_bytes']
                    count += 1
                if 'cpu_percent' in performance_data:
                    cpu_usage += performance_data['cpu_percent']
            
            if count > 0:
                memory_usage = memory_usage / count / (1024 * 1024)  # Convert to MB
                cpu_usage = cpu_usage / count
            
            performance_metrics['labels'].append(hour_start.strftime('%H:%M'))
            performance_metrics['response_times'].append(avg_response_time * 1000)  # Convert to ms
            performance_metrics['memory_usage'].append(memory_usage)
            performance_metrics['cpu_usage'].append(cpu_usage)
        
        # Reverse the lists to show chronological order
        performance_metrics['labels'].reverse()
        performance_metrics['response_times'].reverse()
        performance_metrics['memory_usage'].reverse()
        performance_metrics['cpu_usage'].reverse()
        
        # Get model usage statistics
        model_usage = {}
        model_events = AnalyticsEvent.objects.filter(
            event_type='model_inference',
            timestamp__gte=start_date
        )
        
        # Group by model_id and count
        model_counts = model_events.values('model_id', 'model_provider').annotate(
            count=Count('id'),
            avg_time=Avg('response_time')
        ).order_by('-count')
        
        for item in model_counts:
            model_id = item['model_id']
            model_usage[model_id] = {
                'count': item['count'],
                'provider': item['model_provider'],
                'avg_response_time': item['avg_time'] * 1000 if item['avg_time'] else 0,  # Convert to ms
            }
        
        # Get model errors
        model_errors = AnalyticsEvent.objects.filter(
            event_type='model_error',
            timestamp__gte=start_date
        ).values('model_id').annotate(
            error_count=Count('id')
        )
        
        for item in model_errors:
            model_id = item['model_id']
            if model_id in model_usage:
                model_usage[model_id]['error_count'] = item['error_count']
            else:
                model_usage[model_id] = {'error_count': item['error_count']}
        
        # Format for response
        model_stats = {
            'models': [],
            'usage_counts': [],
            'avg_response_times': [],
            'error_counts': []
        }
        
        for model_id, stats in model_usage.items():
            model_stats['models'].append(model_id)
            model_stats['usage_counts'].append(stats.get('count', 0))
            model_stats['avg_response_times'].append(stats.get('avg_response_time', 0))
            model_stats['error_counts'].append(stats.get('error_count', 0))
        
        return Response({
            'stats': stats,
            'api_requests': api_requests,
            'response_times': response_times,
            'recent_requests': recent_requests_data,
            'performance_metrics': performance_metrics,
            'model_stats': model_stats
        })

@login_required
@user_passes_test(lambda u: u.is_staff)
def knowledge_graph_view(request, graph_id=None):
    """View for displaying knowledge graphs"""
    context = {"page_title": "Knowledge Graph Visualization"}
    
    if graph_id:
        # Get the graph by ID
        kg_manager = KnowledgeGraphManager()
        graph = kg_manager.graph_storage.retrieve_graph(graph_id)
        
        # Create visualization
        visualizer = GraphVisualization()
        viz_data = visualizer.create_embedded_visualization(graph)
        
        context["graph_data"] = viz_data
        context["graph_id"] = graph_id
        context["graph_summary"] = kg_manager._generate_graph_summary(graph)
    
    # Get list of available graphs for selection
    kg_manager = KnowledgeGraphManager()
    available_graphs = kg_manager.list_available_graphs()
    context["available_graphs"] = available_graphs
    
    return render(request, 'analytics_app/knowledge_graph.html', context)

@login_required
@user_passes_test(lambda u: u.is_staff)
def graph_data_api(request, graph_id):
    """API endpoint for graph data"""
    kg_manager = KnowledgeGraphManager()
    graph = kg_manager.graph_storage.retrieve_graph(graph_id)
    
    # Convert to D3.js format
    nodes = []
    links = []
    
    # Map entity IDs to indices for D3
    entity_indices = {}
    for i, entity in enumerate(graph.get("entities", [])):
        entity_indices[entity["id"]] = i
        nodes.append({
            "id": i,
            "label": entity.get("label", ""),
            "type": entity.get("type", ""),
            "properties": entity.get("properties", {})
        })
    
    # Create links
    for rel in graph.get("relationships", []):
        if rel["source"] in entity_indices and rel["target"] in entity_indices:
            links.append({
                "source": entity_indices[rel["source"]],
                "target": entity_indices[rel["target"]],
                "type": rel.get("type", "")
            })
    
    return JsonResponse({"nodes": nodes, "links": links})

@login_required
@user_passes_test(lambda u: u.is_staff)
def graph_quality_report(request, graph_id=None):
    """View for examining knowledge graph quality metrics"""
    context = {
        'title': 'Knowledge Graph Quality Report'
    }
    
    # Handle the selection redirect
    if graph_id == 'select' and 'graph_id' in request.GET:
        selected_id = request.GET.get('graph_id')
        if selected_id:
            return redirect('graph_quality_report', graph_id=selected_id)
    
    if graph_id and graph_id != 'select':
        kg_manager = KnowledgeGraphManager()
        graph = kg_manager.graph_storage.retrieve_graph(graph_id)
        
        if graph:
            # Calculate quality metrics
            from analytics_app.utils import evaluate_knowledge_graph_quality
            metrics = evaluate_knowledge_graph_quality(graph)
            
            # Add performance metrics
            from analytics_app.utils import calculate_performance_metrics
            performance_metrics = calculate_performance_metrics(graph_id=graph_id, graph=graph)
            metrics['performance'] = performance_metrics
            
            # Ensure that the ontological metrics exist
            if 'ontological' not in metrics:
                metrics['ontological'] = {
                    'type_validity_percentage': 100.0,
                    'relationship_validity_percentage': 100.0,
                    'property_conformance': 100.0,
                    'violation_count': 0
                }
            
            # Ensure structural metrics are complete
            structural_defaults = {
                'entity_count': len(graph.get('entities', [])),
                'relationship_count': len(graph.get('relationships', [])),
                'graph_density': 0.0,
                'average_degree': 0.0,
                'connected_components_count': 1,
                'largest_component_size': len(graph.get('entities', [])),
                'isolated_entities_count': 0,
                'entity_types': {},
                'relationship_types': {}
            }
            
            for key, value in structural_defaults.items():
                if key not in metrics['structural']:
                    metrics['structural'][key] = value
            
            # Count entity types if not already done
            if not metrics['structural']['entity_types']:
                entity_types = {}
                for entity in graph.get('entities', []):
                    entity_type = entity.get('type', 'Unknown')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                metrics['structural']['entity_types'] = entity_types
            
            # Count relationship types if not already done
            if not metrics['structural']['relationship_types']:
                relationship_types = {}
                for rel in graph.get('relationships', []):
                    rel_type = rel.get('type', 'Unknown')
                    relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                metrics['structural']['relationship_types'] = relationship_types
            
            # Ensure schema completeness metrics exist
            if 'schema_completeness' not in metrics:
                metrics['schema_completeness'] = {
                    'overall_completeness': 100.0,
                    'type_completeness': {}
                }
                
                for entity_type in metrics['structural']['entity_types'].keys():
                    metrics['schema_completeness']['type_completeness'][entity_type] = {
                        'average_completeness': 100.0
                    }
            
            context["graph"] = graph
            context["graph_id"] = graph_id
            context["metrics"] = metrics
            
            # Format metrics for visualization
            metrics_json = {
                'labels': list(metrics['structural']['entity_types'].keys()),
                'entity_counts': list(metrics['structural']['entity_types'].values()),
                'completeness': []
            }
            
            # Add completeness data for each entity type
            for entity_type in metrics_json['labels']:
                type_completeness = metrics['schema_completeness']['type_completeness'].get(entity_type, {})
                completeness_value = type_completeness.get('average_completeness', 0)
                metrics_json['completeness'].append(completeness_value)
            
            context["metrics_json"] = json.dumps(metrics_json)
    
    # Get list of available graphs
    kg_manager = KnowledgeGraphManager()
    available_graphs = []
    
    # Get more detailed info for each graph
    for graph_info in kg_manager.list_available_graphs():
        graph_id = graph_info.get('id')
        if graph_id:
            try:
                graph = kg_manager.graph_storage.retrieve_graph(graph_id)
                available_graphs.append({
                    'id': graph_id,
                    'name': graph_info.get('name', f"Graph {graph_id}"),
                    'entity_count': len(graph.get('entities', [])),
                    'relationship_count': len(graph.get('relationships', []))
                })
            except Exception as e:
                logger.error(f"Error retrieving graph {graph_id}: {str(e)}")
                # Include basic info if retrieval fails
                available_graphs.append({
                    'id': graph_id,
                    'name': graph_info.get('name', f"Graph {graph_id}"),
                    'entity_count': 0,
                    'relationship_count': 0
                })
    
    context["available_graphs"] = available_graphs
    
    return render(request, 'analytics_app/graph_quality_report.html', context)

@login_required
@user_passes_test(lambda u: u.is_staff)
def vision_analytics_view(request):
    """Render the vision models analytics dashboard"""
    
    # Get available vision providers
    factory = VisionProviderFactory()
    providers = factory.list_providers()
    
    # Get metrics data for all vision providers
    provider_metrics = {}
    total_requests = 0
    total_errors = 0
    total_processing_time = 0
    provider_usage_counts = {}
    
    for provider in providers:
        try:
            # Get models for this provider
            models = factory.list_models(provider)
            
            # Get metrics for each model
            model_metrics = {}
            for model in models:
                try:
                    # Get metrics collector for this provider+model
                    metrics_collector = get_vision_metrics(f"{provider}_{model}")
                    
                    # Get statistics for this model
                    stats = metrics_collector.get_stats()
                    
                    # Add to model metrics
                    model_metrics[model] = {
                        'requests': stats.get('total_operations', 0),
                        'errors': stats.get('total_errors', 0),
                        'avg_response_time_ms': stats.get('avg_operation_time_ms', {}).get('process_image', 0),
                        'avg_confidence': stats.get('avg_confidence', 0),
                        'avg_image_size': stats.get('avg_image_size', 0),
                        'most_common_errors': stats.get('common_errors', [])[:3],
                        'last_used': stats.get('last_operation_time', 'Never')
                    }
                    
                    # Update totals
                    total_requests += stats.get('total_operations', 0)
                    total_errors += stats.get('total_errors', 0)
                    
                    # Calculate average processing time weighted by number of requests
                    model_requests = stats.get('total_operations', 0)
                    if model_requests > 0:
                        model_avg_time = stats.get('avg_operation_time_ms', {}).get('process_image', 0)
                        total_processing_time += model_avg_time * model_requests
                    
                    # Track provider usage
                    provider_usage_counts[provider] = provider_usage_counts.get(provider, 0) + model_requests
                    
                except Exception as e:
                    model_metrics[model] = {
                        'error': str(e)
                    }
            
            # Add to provider metrics
            provider_metrics[provider] = model_metrics
            
        except Exception as e:
            provider_metrics[provider] = {
                'error': str(e)
            }
    
    # Calculate overall average processing time
    avg_processing_time = total_processing_time / total_requests if total_requests > 0 else 0
    
    # Get provider usage distribution
    provider_usage = [
        {
            'name': provider,
            'count': count,
            'percentage': (count / total_requests * 100) if total_requests > 0 else 0
        }
        for provider, count in provider_usage_counts.items()
    ]
    provider_usage.sort(key=lambda x: x['count'], reverse=True)
    
    # Generate data for charts
    provider_chart_data = {
        'labels': [p['name'] for p in provider_usage],
        'values': [p['count'] for p in provider_usage]
    }
    
    # Get all models' response times for comparison
    model_comparison = []
    for provider, models in provider_metrics.items():
        for model_name, metrics in models.items():
            if 'requests' in metrics and metrics['requests'] > 0:
                model_comparison.append({
                    'name': f"{provider} - {model_name}",
                    'avg_response_time_ms': metrics.get('avg_response_time_ms', 0),
                    'avg_confidence': metrics.get('avg_confidence', 0),
                    'requests': metrics.get('requests', 0)
                })
    
    # Sort by response time
    model_comparison.sort(key=lambda x: x['avg_response_time_ms'])
    
    # Chart data for model comparison
    model_comparison_chart = {
        'labels': [m['name'] for m in model_comparison],
        'response_times': [m['avg_response_time_ms'] for m in model_comparison],
        'confidence': [m['avg_confidence'] for m in model_comparison]
    }
    
    context = {
        'providers': providers,
        'provider_metrics': provider_metrics,
        'total_requests': total_requests,
        'total_errors': total_errors,
        'error_rate': (total_errors / total_requests * 100) if total_requests > 0 else 0,
        'avg_processing_time': avg_processing_time,
        'provider_usage': provider_usage,
        'provider_chart_data': provider_chart_data,
        'model_comparison': model_comparison,
        'model_comparison_chart': model_comparison_chart
    }
    
    return render(request, 'analytics_app/vision_analytics.html', context)
