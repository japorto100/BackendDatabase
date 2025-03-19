import time
import json
import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple
from django.utils import timezone
from django.db import transaction
from django.conf import settings

from models_app.models import AIModel
from .models import BenchmarkTask, BenchmarkRun, BenchmarkResult, BenchmarkSummary

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Service for running benchmarks against models"""
    
    def __init__(self, user=None):
        self.user = user
    
    def create_benchmark_run(self, name: str, description: str, 
                           task_ids: List[int], model_ids: List[str], 
                           iterations: int = 1) -> BenchmarkRun:
        """Create a new benchmark run"""
        tasks = BenchmarkTask.objects.filter(id__in=task_ids)
        if not tasks.exists():
            raise ValueError("No valid tasks provided")
        
        # Validate models exist
        valid_models = []
        for model_id in model_ids:
            try:
                model = AIModel.objects.get(model_id=model_id, is_active=True)
                valid_models.append(model_id)
            except AIModel.DoesNotExist:
                logger.warning(f"Model {model_id} not found or not active")
        
        if not valid_models:
            raise ValueError("No valid models provided")
        
        benchmark_run = BenchmarkRun.objects.create(
            name=name,
            description=description,
            user=self.user,
            models_tested=valid_models,
            iterations=iterations,
            status='pending'
        )
        benchmark_run.tasks.set(tasks)
        
        return benchmark_run
    
    def run_benchmark(self, benchmark_run_id: int) -> BenchmarkRun:
        """Run a benchmark and collect results"""
        try:
            benchmark_run = BenchmarkRun.objects.get(id=benchmark_run_id)
        except BenchmarkRun.DoesNotExist:
            raise ValueError(f"Benchmark run {benchmark_run_id} not found")
        
        # Update status to running
        benchmark_run.status = 'running'
        benchmark_run.save()
        
        try:
            # Get models
            model_ids = benchmark_run.models_tested
            models = {}
            for model_id in model_ids:
                try:
                    model = AIModel.objects.get(model_id=model_id, is_active=True)
                    # Get the appropriate provider for this model
                    provider = self._get_provider_for_model(model)
                    if provider:
                        models[model_id] = {
                            'model': model,
                            'provider': provider
                        }
                except AIModel.DoesNotExist:
                    logger.warning(f"Model {model_id} not found or not active")
            
            if not models:
                raise ValueError("No valid models available")
            
            # Run benchmarks for each task
            for task in benchmark_run.tasks.all():
                self._run_task_benchmark(benchmark_run, task, models)
            
            # Generate summary
            self._generate_summary(benchmark_run)
            
            # Update status to completed
            benchmark_run.status = 'completed'
            benchmark_run.end_time = timezone.now()
            benchmark_run.save()
            
        except Exception as e:
            logger.exception(f"Error running benchmark: {e}")
            benchmark_run.status = 'failed'
            benchmark_run.end_time = timezone.now()
            benchmark_run.save()
        
        return benchmark_run
    
    def _get_provider_for_model(self, model):
        """Get the appropriate provider for a model"""
        # This would connect to your existing provider system
        # For now, we'll return a dummy provider
        from models_app.llm_providers import get_provider_for_model
        return get_provider_for_model(model)
    
    def _run_task_benchmark(self, benchmark_run, task, models):
        """Run benchmarks for a specific task against all models"""
        prompts = task.prompts
        if isinstance(prompts, str):
            prompts = json.loads(prompts)
        
        for model_id, model_data in models.items():
            model = model_data['model']
            provider = model_data['provider']
            
            for prompt_index, prompt in enumerate(prompts):
                for iteration in range(benchmark_run.iterations):
                    try:
                        # Measure performance
                        start_time = time.time()
                        response = provider.generate_text(prompt)
                        end_time = time.time()
                        
                        # Calculate metrics
                        response_time = (end_time - start_time) * 1000  # ms
                        token_count = self._estimate_tokens(response)
                        input_token_count = self._estimate_tokens(prompt)
                        
                        # Get electricity cost if available
                        electricity_cost = self._get_electricity_cost(model, input_token_count, token_count)
                        
                        # Store result
                        metrics = {
                            'response_time_ms': response_time,
                            'output_tokens': token_count,
                            'input_tokens': input_token_count,
                            'token_efficiency': token_count / max(1, input_token_count),
                            'electricity_cost': electricity_cost
                        }
                        
                        # Add quality metrics if reference answers available
                        if task.reference_answers:
                            quality_metrics = self._evaluate_quality(
                                response, 
                                task.reference_answers[prompt_index] if prompt_index < len(task.reference_answers) else None
                            )
                            metrics.update(quality_metrics)
                        
                        BenchmarkResult.objects.create(
                            benchmark_run=benchmark_run,
                            task=task,
                            model_id=model_id,
                            prompt_index=prompt_index,
                            prompt_text=prompt,
                            response_text=response,
                            metrics=metrics,
                            iteration=iteration
                        )
                    
                    except Exception as e:
                        logger.exception(f"Error benchmarking {model_id} with prompt {prompt_index}: {e}")
                        # Create a failed result
                        BenchmarkResult.objects.create(
                            benchmark_run=benchmark_run,
                            task=task,
                            model_id=model_id,
                            prompt_index=prompt_index,
                            prompt_text=prompt,
                            response_text=f"ERROR: {str(e)}",
                            metrics={'error': str(e)},
                            iteration=iteration
                        )
    
    def _estimate_tokens(self, text):
        """Estimate token count for a text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _get_electricity_cost(self, model, input_tokens, output_tokens):
        """Get electricity cost for a model run if available"""
        try:
            from models_app.electricity_cost import ElectricityCostTracker
            tracker = ElectricityCostTracker(self.user)
            return tracker.estimate_cost(model.model_id, input_tokens, output_tokens)
        except Exception as e:
            logger.warning(f"Could not get electricity cost: {e}")
            return None
    
    def _evaluate_quality(self, response, reference=None):
        """Evaluate response quality against reference answer"""
        # Simple implementation - in production, you'd use more sophisticated metrics
        metrics = {}
        
        if reference:
            # Simple lexical similarity
            response_words = set(response.lower().split())
            reference_words = set(reference.lower().split())
            
            if reference_words:
                overlap = len(response_words.intersection(reference_words))
                precision = overlap / max(1, len(response_words))
                recall = overlap / max(1, len(reference_words))
                f1 = 2 * precision * recall / max(0.001, precision + recall)
                
                metrics.update({
                    'lexical_precision': precision,
                    'lexical_recall': recall,
                    'lexical_f1': f1
                })
        
        # Add length metrics
        metrics['response_length'] = len(response)
        
        return metrics
    
    def _generate_summary(self, benchmark_run):
        """Generate summary statistics for a benchmark run"""
        results = BenchmarkResult.objects.filter(benchmark_run=benchmark_run)
        
        if not results.exists():
            logger.warning(f"No results found for benchmark run {benchmark_run.id}")
            return
        
        # Group results by model and task
        summary_data = {}
        rankings = {}
        
        # Process all models
        for model_id in benchmark_run.models_tested:
            summary_data[model_id] = {}
            
            # Process all tasks for this model
            for task in benchmark_run.tasks.all():
                task_results = results.filter(model_id=model_id, task=task)
                
                if not task_results.exists():
                    continue
                
                # Calculate average metrics
                metrics_sum = {}
                metrics_list = {}
                
                for result in task_results:
                    for metric, value in result.metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_sum[metric] = metrics_sum.get(metric, 0) + value
                            if metric not in metrics_list:
                                metrics_list[metric] = []
                            metrics_list[metric].append(value)
                
                # Calculate averages and standard deviations
                avg_metrics = {}
                for metric, total in metrics_sum.items():
                    avg_metrics[f"avg_{metric}"] = total / task_results.count()
                    if len(metrics_list[metric]) > 1:
                        avg_metrics[f"std_{metric}"] = statistics.stdev(metrics_list[metric])
                    else:
                        avg_metrics[f"std_{metric}"] = 0
                
                summary_data[model_id][task.id] = {
                    'task_name': task.name,
                    'metrics': avg_metrics,
                    'result_count': task_results.count()
                }
        
        # Calculate rankings for each metric
        all_metrics = set()
        for model_data in summary_data.values():
            for task_data in model_data.values():
                all_metrics.update(task_data['metrics'].keys())
        
        # Filter for average metrics only
        ranking_metrics = [m for m in all_metrics if m.startswith('avg_')]
        
        for metric in ranking_metrics:
            rankings[metric] = []
            
            # Collect values for all models
            model_values = []
            for model_id in summary_data:
                # Calculate average across all tasks
                values = []
                for task_id in summary_data[model_id]:
                    if metric in summary_data[model_id][task_id]['metrics']:
                        values.append(summary_data[model_id][task_id]['metrics'][metric])
                
                if values:
                    avg_value = sum(values) / len(values)
                    model_values.append((model_id, avg_value))
            
            # Sort by value (ascending or descending depending on metric)
            descending = not (metric.endswith('_ms') or metric.endswith('_cost'))
            model_values.sort(key=lambda x: x[1], reverse=descending)
            
            # Create rankings
            rankings[metric] = [{'model_id': m[0], 'value': m[1], 'rank': i+1} 
                              for i, m in enumerate(model_values)]
        
        # Create summary
        with transaction.atomic():
            # Delete existing summary if it exists
            BenchmarkSummary.objects.filter(benchmark_run=benchmark_run).delete()
            
            BenchmarkSummary.objects.create(
                benchmark_run=benchmark_run,
                results=summary_data,
                rankings=rankings
            ) 