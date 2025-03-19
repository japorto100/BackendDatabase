"""
Management command to run edge case tests on vision models.

This command tests vision providers against challenging edge case images
to evaluate robustness and error handling.
"""

import logging
import os
import sys
import json
import time
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone

from benchmark.vision_benchmark import VisionBenchmarkRunner
from models_app.ai_models.vision.vision_factory import VisionProviderFactory

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run edge case tests on vision models to evaluate robustness'

    def add_arguments(self, parser):
        parser.add_argument(
            '--providers',
            nargs='+',
            help='List of providers to test (e.g., "qwen lightweight gemini gpt4v")'
        )
        parser.add_argument(
            '--models',
            nargs='+',
            help='List of models to test (format: "provider:model" e.g., "qwen:vl gemini:vision")'
        )
        parser.add_argument(
            '--generate',
            action='store_true',
            help='Force generation of synthetic edge case images'
        )
        parser.add_argument(
            '--export-path',
            type=str,
            help='Path to export the test results (JSON format)'
        )
        parser.add_argument(
            '--edge-types',
            nargs='+',
            help='Specific edge case types to test (e.g., "corrupt extreme_resolution")'
        )

    def handle(self, *args, **options):
        # Process command line options
        providers_list = options.get('providers')
        models_list = options.get('models')
        generate_synthetic = options.get('generate')
        export_path = options.get('export_path')
        edge_types = options.get('edge_types')
        
        if not providers_list and not models_list:
            raise CommandError("You must provide either --providers or --models")
        
        # Initialize benchmark runner and provider factory
        benchmark_runner = VisionBenchmarkRunner()
        factory = VisionProviderFactory()
        
        # If edge case images should be regenerated
        if generate_synthetic:
            self.stdout.write("Generating synthetic edge case images...")
            benchmark_runner._generate_synthetic_edge_cases()
        
        # If specific edge types are requested, filter the available types
        if edge_types:
            filtered_edge_cases = {}
            for edge_type in edge_types:
                if edge_type in benchmark_runner.edge_case_images:
                    filtered_edge_cases[edge_type] = benchmark_runner.edge_case_images[edge_type]
                else:
                    self.stdout.write(self.style.WARNING(f"Edge case type '{edge_type}' not found"))
            benchmark_runner.edge_case_images = filtered_edge_cases
        
        # Process providers and models to test
        test_configs = []
        
        # If providers are specified, use all available models
        if providers_list:
            for provider_name in providers_list:
                available_models = factory.list_models(provider_name)
                if available_models:
                    for model_name in available_models:
                        test_configs.append((provider_name, model_name))
                else:
                    self.stdout.write(self.style.WARNING(f"No models available for provider {provider_name}"))
        
        # If models are specified in provider:model format
        if models_list:
            for model_spec in models_list:
                if ':' in model_spec:
                    provider, model = model_spec.split(':', 1)
                    test_configs.append((provider, model))
                else:
                    self.stdout.write(self.style.WARNING(f"Invalid model format: {model_spec}, expected 'provider:model'"))
        
        if not test_configs:
            raise CommandError("No valid providers or models specified")
        
        # Run edge case tests
        start_time = time.time()
        results = {}
        
        for provider_name, model_name in test_configs:
            self.stdout.write(f"Running edge case tests for {provider_name}/{model_name}...")
            try:
                result = benchmark_runner.run_edge_case_tests(provider_name, model_name)
                results[f"{provider_name}:{model_name}"] = result
                
                # Display summary
                if "error" in result:
                    self.stdout.write(self.style.ERROR(f"  Error: {result['error']}"))
                else:
                    success_rate = (1 - result["error_rate"]) * 100 if result["total_tests"] > 0 else 0
                    self.stdout.write(
                        f"  Tests: {result['successful_tests']}/{result['total_tests']} successful " +
                        f"({success_rate:.1f}% success rate)"
                    )
                    
                    # Display results by edge case type
                    for edge_type, type_result in result["edge_case_types"].items():
                        type_success_rate = (type_result["successful"] / type_result["total"]) * 100 if type_result["total"] > 0 else 0
                        self.stdout.write(
                            f"    {edge_type}: {type_result['successful']}/{type_result['total']} " +
                            f"({type_success_rate:.1f}% success)"
                        )
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error testing {provider_name}/{model_name}: {str(e)}"))
                results[f"{provider_name}:{model_name}"] = {"error": str(e)}
        
        # Calculate overall stats
        total_time = time.time() - start_time
        total_tests = sum(
            result.get("total_tests", 0) 
            for result in results.values() 
            if "error" not in result
        )
        successful_tests = sum(
            result.get("successful_tests", 0) 
            for result in results.values() 
            if "error" not in result
        )
        
        self.stdout.write(self.style.SUCCESS(
            f"Edge case testing completed in {total_time:.1f} seconds. " +
            f"Overall: {successful_tests}/{total_tests} tests passed " +
            f"({(successful_tests/total_tests)*100:.1f}% success rate)" if total_tests > 0 else "No tests were run."
        ))
        
        # Export results if requested
        if export_path:
            self.export_results(results, export_path)
    
    def export_results(self, results, export_path):
        """Export test results to a file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
            
            # Add metadata to results
            export_data = {
                "meta": {
                    "timestamp": timezone.now().isoformat(),
                    "total_providers_tested": len(results),
                },
                "results": results
            }
            
            # Write to file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.stdout.write(self.style.SUCCESS(f"Results exported to {export_path}"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error exporting results: {str(e)}")) 