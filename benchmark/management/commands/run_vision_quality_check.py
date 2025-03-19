"""
Management command to run vision quality checks and alerts.

Runs benchmarks on vision models and generates quality alerts.
"""

import logging
import os
import sys
import json
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from benchmark.models import BenchmarkRun, BenchmarkTask, BenchmarkResult
from models_app.ai_models.utils.common.metrics_alerts import connect_benchmark_to_alerts
from benchmark.vision_benchmark import VisionBenchmarkRunner

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run quality checks on vision models and generate alerts'

    def add_arguments(self, parser):
        parser.add_argument(
            '--run-id',
            type=int,
            help='The ID of an existing benchmark run to analyze'
        )
        parser.add_argument(
            '--providers',
            nargs='+',
            help='List of providers to test (e.g., "qwen lightweight gemini")'
        )
        parser.add_argument(
            '--models',
            nargs='+',
            help='List of models to test (format: "provider:model" e.g., "qwen:vl gemini:vision")'
        )
        parser.add_argument(
            '--categories',
            nargs='+',
            default=['natural', 'artwork', 'diagrams', 'documents', 'charts'],
            help='Image categories to test'
        )
        parser.add_argument(
            '--task-types',
            nargs='+',
            default=['caption', 'vqa', 'classify', 'count', 'ocr'],
            help='Task types to benchmark'
        )
        parser.add_argument(
            '--iterations',
            type=int,
            default=1,
            help='Number of iterations for each benchmark test'
        )
        parser.add_argument(
            '--alerts-only',
            action='store_true',
            help='Only generate alerts without running benchmarks'
        )
        parser.add_argument(
            '--export-path',
            type=str,
            help='Path to export the benchmark results'
        )

    def handle(self, *args, **options):
        # Process command line options
        run_id = options.get('run_id')
        providers = options.get('providers')
        models_list = options.get('models')
        categories = options.get('categories')
        task_types = options.get('task_types')
        iterations = options.get('iterations')
        alerts_only = options.get('alerts_only')
        export_path = options.get('export_path')
        
        # Initialize the benchmark runner
        benchmark_runner = VisionBenchmarkRunner()
        
        # If alerts-only mode, just run alerts on an existing benchmark
        if alerts_only:
            if not run_id:
                raise CommandError("You must specify a run-id when using --alerts-only")
            
            self.generate_alerts(run_id)
            self.stdout.write(self.style.SUCCESS(f"Generated alerts for benchmark run #{run_id}"))
            return
            
        # If run_id is provided, analyze existing benchmark
        if run_id:
            try:
                benchmark_run = BenchmarkRun.objects.get(id=run_id)
                self.stdout.write(f"Analyzing existing benchmark run: {benchmark_run.name}")
                
                # Generate quality alerts
                self.generate_alerts(run_id)
                
                # Export if needed
                if export_path:
                    self.export_results(benchmark_run, export_path)
                    
                self.stdout.write(self.style.SUCCESS(f"Successfully analyzed benchmark run #{run_id}"))
                return
                
            except BenchmarkRun.DoesNotExist:
                raise CommandError(f"Benchmark run #{run_id} does not exist")
                
        # Otherwise, create and run a new benchmark
        if not providers and not models_list:
            raise CommandError("You must provide either --providers or --models")
            
        # Process providers list
        provider_names = []
        model_names = {}
        
        if providers:
            provider_names = providers
            # Use default model for each provider
            for provider in provider_names:
                available_models = benchmark_runner.factory.list_models(provider)
                if available_models:
                    model_names[provider] = [available_models[0]]
                else:
                    self.stdout.write(self.style.WARNING(f"No models available for provider {provider}"))
                    
        # Process models list (format: "provider:model")
        if models_list:
            for model_spec in models_list:
                if ':' in model_spec:
                    provider, model = model_spec.split(':', 1)
                    if provider not in provider_names:
                        provider_names.append(provider)
                    if provider not in model_names:
                        model_names[provider] = []
                    model_names[provider].append(model)
                else:
                    self.stdout.write(self.style.WARNING(f"Invalid model format: {model_spec}, expected 'provider:model'"))
        
        if not provider_names or not any(model_names.values()):
            raise CommandError("No valid providers or models specified")
            
        # Create and run benchmark
        benchmark_name = f"Vision Quality Benchmark ({', '.join(provider_names)})"
        benchmark_description = f"Quality benchmark for vision providers: {', '.join(provider_names)}"
        
        try:
            # Create the benchmark run
            benchmark_run = benchmark_runner.create_vision_benchmark_run(
                name=benchmark_name,
                description=benchmark_description,
                provider_names=provider_names,
                model_names=model_names,
                image_categories=categories,
                task_types=task_types,
                iterations=iterations
            )
            
            self.stdout.write(f"Created benchmark run #{benchmark_run.id}: {benchmark_name}")
            self.stdout.write("Running benchmark tests (this may take a while)...")
            
            # Run the benchmark
            benchmark_run = benchmark_runner.run_benchmark(benchmark_run.id)
            
            if benchmark_run.status == 'completed':
                self.stdout.write(self.style.SUCCESS(f"Benchmark run #{benchmark_run.id} completed successfully"))
                
                # Export if needed
                if export_path:
                    self.export_results(benchmark_run, export_path)
            else:
                self.stdout.write(self.style.ERROR(f"Benchmark run #{benchmark_run.id} failed"))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error running benchmark: {str(e)}"))
            raise CommandError(str(e))
    
    def generate_alerts(self, run_id):
        """Generate alerts from benchmark results"""
        try:
            benchmark_run = BenchmarkRun.objects.get(id=run_id)
            
            # Get all results for this benchmark run
            benchmark_results = []
            for result in BenchmarkResult.objects.filter(benchmark_run=benchmark_run):
                # Convert Django model to dictionary for the alert system
                benchmark_results.append({
                    'model_id': result.model_id,
                    'metrics': result.metrics
                })
            
            # Get baseline results if a previous benchmark run exists to compare against
            baseline_results = None
            try:
                # Look for a previous successful benchmark run with similar tasks and models
                previous_run = BenchmarkRun.objects.filter(
                    status='completed',
                    id__lt=benchmark_run.id
                ).order_by('-id').first()
                
                if previous_run:
                    self.stdout.write(f"Using benchmark run #{previous_run.id} as baseline for comparison")
                    baseline_results = []
                    for result in BenchmarkResult.objects.filter(benchmark_run=previous_run):
                        baseline_results.append({
                            'model_id': result.model_id,
                            'metrics': result.metrics
                        })
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Failed to load baseline benchmark results: {e}"))
            
            # Connect to alert system
            from models_app.ai_models.utils.common.metrics_alerts import connect_benchmark_to_alerts
            connect_benchmark_to_alerts(benchmark_results, baseline_results)
            self.stdout.write("Connected benchmark results to alert system")
            
        except BenchmarkRun.DoesNotExist:
            raise CommandError(f"Benchmark run #{run_id} does not exist")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error generating alerts: {str(e)}"))
            raise CommandError(str(e))
    
    def export_results(self, benchmark_run, export_path):
        """Export benchmark results to a file"""
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Get all results
            results = []
            for result in BenchmarkResult.objects.filter(benchmark_run=benchmark_run):
                results.append({
                    'model_id': result.model_id,
                    'task_name': result.task.name,
                    'prompt_index': result.prompt_index,
                    'metrics': result.metrics,
                    'response_text': result.response_text
                })
                
            # Get summary if available
            summary = None
            if hasattr(benchmark_run, 'benchmarksummary'):
                summary = benchmark_run.benchmarksummary.rankings
                
            # Create export data
            export_data = {
                'run_id': benchmark_run.id,
                'name': benchmark_run.name,
                'description': benchmark_run.description,
                'status': benchmark_run.status,
                'start_time': benchmark_run.start_time.isoformat() if benchmark_run.start_time else None,
                'end_time': benchmark_run.end_time.isoformat() if benchmark_run.end_time else None,
                'models_tested': benchmark_run.models_tested,
                'iterations': benchmark_run.iterations,
                'results': results,
                'summary': summary
            }
            
            # Write to file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.stdout.write(self.style.SUCCESS(f"Exported benchmark results to {export_path}"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error exporting results: {str(e)}")) 