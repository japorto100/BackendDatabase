from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.urls import reverse

from .models import BenchmarkTask, BenchmarkRun, BenchmarkResult, BenchmarkSummary
from .services import BenchmarkRunner
from .fusion_benchmark import FusionBenchmarkRunner
from .vision_benchmark import VisionBenchmarkRunner
from models_app.ai_models.vision.vision_factory import VisionProviderFactory

@login_required
def benchmark_dashboard(request):
    """Dashboard view for benchmarks"""
    tasks = BenchmarkTask.objects.filter(is_public=True) | BenchmarkTask.objects.filter(created_by=request.user)
    recent_runs = BenchmarkRun.objects.filter(user=request.user).order_by('-start_time')[:5]
    
    context = {
        'tasks': tasks,
        'recent_runs': recent_runs,
    }
    
    return render(request, 'benchmark/dashboard.html', context)

@login_required
def create_benchmark(request):
    """View for creating a new benchmark run"""
    from models_app.models import AIModel
    
    tasks = BenchmarkTask.objects.filter(is_public=True) | BenchmarkTask.objects.filter(created_by=request.user)
    models = AIModel.objects.filter(is_active=True)
    
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        task_ids = request.POST.getlist('tasks')
        model_ids = request.POST.getlist('models')
        iterations = int(request.POST.get('iterations', 1))
        
        if name and task_ids and model_ids:
            runner = BenchmarkRunner(request.user)
            try:
                benchmark_run = runner.create_benchmark_run(
                    name=name,
                    description=description,
                    task_ids=task_ids,
                    model_ids=model_ids,
                    iterations=iterations
                )
                
                # In a real app, you'd queue this as a background task
                # For simplicity, we'll run it synchronously here
                runner.run_benchmark(benchmark_run.id)
                
                return redirect('benchmark_results', run_id=benchmark_run.id)
            except ValueError as e:
                context = {
                    'tasks': tasks,
                    'models': models,
                    'error': str(e),
                }
                return render(request, 'benchmark/create_benchmark.html', context)
    
    context = {
        'tasks': tasks,
        'models': models,
    }
    
    return render(request, 'benchmark/create_benchmark.html', context)

@login_required
def benchmark_results(request, run_id):
    """View for displaying benchmark results"""
    benchmark_run = get_object_or_404(BenchmarkRun, id=run_id, user=request.user)
    
    try:
        summary = BenchmarkSummary.objects.get(benchmark_run=benchmark_run)
    except BenchmarkSummary.DoesNotExist:
        return redirect('benchmark_dashboard')
    
    context = {
        'benchmark_run': benchmark_run,
        'summary': summary,
    }
    
    return render(request, 'benchmark/results.html', context)

@login_required
@require_POST
def run_benchmark(request, run_id):
    """API endpoint for running a benchmark"""
    benchmark_run = get_object_or_404(BenchmarkRun, id=run_id, user=request.user)
    
    if benchmark_run.status in ['completed', 'running']:
        return JsonResponse({
            'success': False,
            'message': f"Benchmark is already {benchmark_run.status}."
        })
    
    # In a real app, you'd queue this as a background task
    runner = BenchmarkRunner(request.user)
    try:
        runner.run_benchmark(run_id)
        return JsonResponse({
            'success': True,
            'redirect': reverse('benchmark_results', args=[run_id])
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        })

@login_required
def fusion_benchmark_results(request):
    """View for displaying fusion benchmark results"""
    # Create fusion benchmark runner
    runner = FusionBenchmarkRunner(request.user)
    
    # Get performance statistics
    strategy_stats = runner.get_strategy_comparison()
    document_stats = runner.get_document_type_comparison()
    recommendations = runner.get_strategy_recommendations()
    
    # Get performance history from hybrid fusion
    performance_history = runner.hybrid_fusion.performance_history
    
    context = {
        'strategy_stats': strategy_stats,
        'document_stats': document_stats,
        'recommendations': recommendations,
        'performance_history': performance_history,
    }
    
    return render(request, 'benchmark/fusion_results.html', context)

@login_required
def vision_benchmark_dashboard(request):
    """Dashboard view for vision benchmarks"""
    # Get latest vision benchmark runs
    vision_runs = BenchmarkRun.objects.filter(
        user=request.user,
        tasks__name__startswith='Vision'  # Simple way to identify vision benchmarks
    ).order_by('-start_time').distinct()[:10]
    
    # Get available vision providers and models
    factory = VisionProviderFactory()
    providers = factory.list_providers()
    
    provider_models = {}
    for provider in providers:
        try:
            provider_models[provider] = factory.list_models(provider)
        except Exception as e:
            provider_models[provider] = []
    
    context = {
        'vision_runs': vision_runs,
        'providers': providers,
        'provider_models': provider_models
    }
    
    return render(request, 'benchmark/vision_dashboard.html', context)

@login_required
def create_vision_benchmark(request):
    """View for creating a new vision benchmark run"""
    factory = VisionProviderFactory()
    providers = factory.list_providers()
    
    provider_models = {}
    for provider in providers:
        try:
            provider_models[provider] = factory.list_models(provider)
        except Exception as e:
            provider_models[provider] = []
    
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        provider_names = request.POST.getlist('providers')
        
        # Get model names for each provider
        model_names = {}
        for provider in provider_names:
            model_names[provider] = request.POST.getlist(f'models_{provider}')
        
        image_categories = request.POST.getlist('image_categories')
        task_types = request.POST.getlist('task_types')
        iterations = int(request.POST.get('iterations', 1))
        
        if name and provider_names and task_types and image_categories:
            runner = VisionBenchmarkRunner(request.user)
            try:
                benchmark_run = runner.create_vision_benchmark_run(
                    name=name,
                    description=description,
                    provider_names=provider_names,
                    model_names=model_names,
                    image_categories=image_categories,
                    task_types=task_types,
                    iterations=iterations
                )
                
                # Redirect to dashboard with success message
                return redirect('vision_benchmark_dashboard')
            except ValueError as e:
                context = {
                    'providers': providers,
                    'provider_models': provider_models,
                    'error': str(e),
                }
                return render(request, 'benchmark/create_vision_benchmark.html', context)
    
    # Default image categories and task types
    image_categories = [
        ('natural', 'Natural Images'),
        ('people', 'People'),
        ('artwork', 'Artwork'),
        ('documents', 'Documents'),
        ('charts', 'Charts and Graphs'),
        ('diagrams', 'Diagrams'),
        ('objects', 'Objects'),
        ('mixed', 'Mixed Content')
    ]
    
    task_types = [
        ('caption', 'Image Captioning'),
        ('vqa', 'Visual Question Answering'),
        ('classify', 'Image Classification'),
        ('count', 'Object Counting'),
        ('ocr', 'Text Extraction (OCR)'),
        ('detail', 'Detailed Description')
    ]
    
    context = {
        'providers': providers,
        'provider_models': provider_models,
        'image_categories': image_categories,
        'task_types': task_types,
    }
    
    return render(request, 'benchmark/create_vision_benchmark.html', context)

@login_required
@require_POST
def run_vision_benchmark(request, run_id):
    """API endpoint for running a vision benchmark"""
    benchmark_run = get_object_or_404(BenchmarkRun, id=run_id, user=request.user)
    
    if benchmark_run.status in ['completed', 'running']:
        return JsonResponse({
            'success': False,
            'message': f"Benchmark is already {benchmark_run.status}."
        })
    
    # In a real app, you'd queue this as a background task
    runner = VisionBenchmarkRunner(request.user)
    try:
        runner.run_benchmark(benchmark_run.id)
        return JsonResponse({
            'success': True,
            'message': "Benchmark completed successfully."
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f"Error running benchmark: {str(e)}"
        })

@login_required
def vision_benchmark_results(request, run_id):
    """View for displaying vision benchmark results"""
    benchmark_run = get_object_or_404(BenchmarkRun, id=run_id, user=request.user)
    
    try:
        summary = BenchmarkSummary.objects.get(benchmark_run=benchmark_run)
    except BenchmarkSummary.DoesNotExist:
        return redirect('vision_benchmark_dashboard')
    
    # Get detailed results as needed
    results = BenchmarkResult.objects.filter(benchmark_run=benchmark_run)
    
    # Create visualizations and tables for the template
    # This is a simplified version - your actual implementation would have more
    # detailed visualizations
    
    # Parse provider names from model_id
    providers = set()
    for result in results:
        model_id = result.model_id
        if ':' in model_id:
            provider = model_id.split(':')[0]
            providers.add(provider)
    
    # Get performance by provider
    provider_performance = {}
    for provider in providers:
        provider_results = results.filter(model_id__startswith=f"{provider}:")
        if provider_results.exists():
            # Simple average response time
            total_time = 0
            count = 0
            for result in provider_results:
                if 'response_time_ms' in result.metrics:
                    total_time += result.metrics['response_time_ms']
                    count += 1
            
            avg_time = total_time / count if count > 0 else 0
            provider_performance[provider] = {
                'avg_response_time_ms': avg_time,
                'result_count': count
            }
    
    context = {
        'benchmark_run': benchmark_run,
        'summary': summary,
        'results': results[:100],  # Limit to avoid overwhelming the page
        'provider_performance': provider_performance,
        'providers': list(providers)
    }
    
    return render(request, 'benchmark/vision_results.html', context)

@login_required
def compare_vision_providers(request):
    """View for comparing vision providers across benchmarks"""
    # Get recent benchmark runs with summary data
    runs = BenchmarkRun.objects.filter(
        user=request.user,
        tasks__name__startswith='Vision',
        status='completed'
    ).order_by('-start_time').distinct()[:10]
    
    run_data = []
    for run in runs:
        try:
            summary = BenchmarkSummary.objects.get(benchmark_run=run)
            # Extract rankings from the summary
            if 'rankings' in summary.rankings:
                response_time_ranking = summary.rankings.get('response_time', [])
                confidence_ranking = summary.rankings.get('confidence', [])
                
                run_data.append({
                    'id': run.id,
                    'name': run.name,
                    'date': run.start_time,
                    'response_time_ranking': response_time_ranking,
                    'confidence_ranking': confidence_ranking
                })
        except BenchmarkSummary.DoesNotExist:
            continue
    
    context = {
        'runs': run_data
    }
    
    return render(request, 'benchmark/compare_vision.html', context)
