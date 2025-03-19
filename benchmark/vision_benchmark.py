"""
Vision Benchmark System

Provides specialized benchmarking capabilities for vision models and providers.

======================> benchmark/management/commands/create_vision_benchmark.py needs to get updated
"""

import time
import logging
import psutil
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
from django.conf import settings
from django.utils import timezone
import io
import random

# Need to add these imports for text quality evaluation
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import re
from difflib import SequenceMatcher
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from models_app.ai_models.utils.common.config import VisionConfig, get_vision_config
from models_app.ai_models.utils.common.metrics import get_vision_metrics, export_all_metrics
from models_app.ai_models.vision.vision_factory import VisionProviderFactory
from models_app.ai_models.utils.vision.image_processing import encode_image_to_base64
from models_app.ai_models.utils.common.errors import ImageProcessingError, VisionModelError
from .models import BenchmarkTask, BenchmarkRun, BenchmarkResult, BenchmarkSummary

logger = logging.getLogger(__name__)

class VisionBenchmarkRunner:
    """Service for benchmarking vision models and providers"""
    
    def __init__(self, user=None):
        """Initialize the vision benchmark runner"""
        self.user = user
        self.factory = VisionProviderFactory()
        self.metrics = get_vision_metrics("vision_benchmark")
        
        # Initialize test data
        self.test_images = self._load_test_images()
        
        # Define standard prompt templates
        self.prompt_templates = {
            "caption": "Describe this image in detail.",
            "vqa": "What {question} in this image?",
            "classify": "What category best describes this image? Choose from: {categories}",
            "count": "Count the number of {object} in this image.",
            "ocr": "Extract all text visible in this image.",
            "detail": "Describe this image with extreme detail, noting all objects, colors, and spatial relationships."
        }
        
        # Load reference answers if available
        self.reference_answers = self._load_reference_answers()
        
        # Initialize smoothing function for BLEU score calculation
        self.smoother = SmoothingFunction().method1
        
        # Edge case test data
        self.edge_case_images = self._load_edge_case_images()
    
    def _load_test_images(self) -> Dict[str, List[str]]:
        """Load test image paths by category"""
        # Check if test images directory exists
        test_imgs_dir = getattr(settings, 'TEST_IMAGES_DIR', 'test_images')
        if not os.path.exists(test_imgs_dir):
            os.makedirs(test_imgs_dir, exist_ok=True)
            logger.warning(f"Created test images directory: {test_imgs_dir}")
        
        # Return dictionary of image categories and their paths
        return {
            "natural": self._get_images_by_category(test_imgs_dir, "natural"),
            "artwork": self._get_images_by_category(test_imgs_dir, "artwork"),
            "diagrams": self._get_images_by_category(test_imgs_dir, "diagrams"),
            "documents": self._get_images_by_category(test_imgs_dir, "documents"),
            "charts": self._get_images_by_category(test_imgs_dir, "charts"),
            "people": self._get_images_by_category(test_imgs_dir, "people"),
            "objects": self._get_images_by_category(test_imgs_dir, "objects"),
            "mixed": self._get_images_by_category(test_imgs_dir, "mixed")
        }
    
    def _get_images_by_category(self, base_dir: str, category: str) -> List[str]:
        """Get image paths for a specific category"""
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir, exist_ok=True)
            logger.info(f"Created category directory: {category_dir}")
            return []
            
        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
        image_paths = []
        
        for file in os.listdir(category_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(category_dir, file))
        
        return image_paths
    
    def _load_reference_answers(self) -> Dict[str, Dict[str, str]]:
        """
        Load reference answers for benchmark tasks.
        
        Returns:
            Dict: Dictionary mapping image paths to reference answers by task type
        """
        ref_file = getattr(settings, 'VISION_REFERENCE_ANSWERS_FILE', 
                         os.path.join(settings.MEDIA_ROOT, 'benchmark', 'vision_reference_answers.json'))
        
        references = {}
        
        try:
            if os.path.exists(ref_file):
                with open(ref_file, 'r') as f:
                    references = json.load(f)
                logger.info(f"Loaded {len(references)} reference answers from {ref_file}")
            else:
                logger.warning(f"Reference answers file not found: {ref_file}")
        except Exception as e:
            logger.error(f"Error loading reference answers: {e}")
        
        return references
    
    def create_vision_benchmark_run(self, 
                                   name: str, 
                                   description: str,
                                   provider_names: List[str],
                                   model_names: Dict[str, List[str]],
                                   image_categories: List[str],
                                   task_types: List[str],
                                   iterations: int = 1) -> BenchmarkRun:
        """
        Create a new vision benchmark run
        
        Args:
            name: Name of the benchmark run
            description: Description of the benchmark run
            provider_names: List of provider names to benchmark
            model_names: Dictionary mapping provider names to lists of model names
            image_categories: List of image categories to include
            task_types: List of task types to benchmark
            iterations: Number of iterations to run for each test
            
        Returns:
            BenchmarkRun: The created benchmark run
        """
        # Validate provider and model combinations
        valid_models = {}
        for provider in provider_names:
            valid_models[provider] = []
            for model in model_names.get(provider, []):
                # Check if this provider/model combo is available
                try:
                    # Get available models for this provider
                    available_models = self.factory.list_models(provider)
                    if model in available_models:
                        valid_models[provider].append(model)
                    else:
                        logger.warning(f"Model {model} not available for provider {provider}")
                except Exception as e:
                    logger.warning(f"Could not validate model {model} for provider {provider}: {e}")
        
        # Filter out providers with no valid models
        valid_models = {p: m for p, m in valid_models.items() if m}
        
        if not valid_models:
            raise ValueError("No valid provider/model combinations found")
            
        # Create benchmark tasks for each task type
        task_ids = []
        for task_type in task_types:
            benchmark_task = self._create_vision_task(task_type, image_categories)
            task_ids.append(benchmark_task.id)
            
        # Format models for storage
        models_to_test = []
        for provider, models in valid_models.items():
            for model in models:
                models_to_test.append(f"{provider}:{model}")
                
        # Create the benchmark run
        benchmark_run = BenchmarkRun.objects.create(
            name=name,
            description=description,
            user=self.user,
            models_tested=models_to_test,
            iterations=iterations,
            status='pending'
        )
        
        # Associate tasks with the run
        tasks = BenchmarkTask.objects.filter(id__in=task_ids)
        benchmark_run.tasks.set(tasks)
        
        return benchmark_run
    
    def _create_vision_task(self, task_type: str, image_categories: List[str]) -> BenchmarkTask:
        """
        Create a benchmark task for a specific vision task type
        
        Args:
            task_type: Type of vision task (caption, vqa, classify, etc.)
            image_categories: Categories of images to include
            
        Returns:
            BenchmarkTask: The created benchmark task
        """
        # Get prompt template for this task type
        prompt_template = self.prompt_templates.get(task_type, "Describe this image.")
        
        # Generate task name and description
        task_name = f"Vision {task_type.capitalize()} Task"
        description = f"Benchmark for vision {task_type} capabilities across different models and image types."
        
        # Prepare prompts - combination of images and prompts
        prompts = []
        reference_answers = []
        
        for category in image_categories:
            # Get images for this category
            images = self.test_images.get(category, [])
            if not images:
                logger.warning(f"No test images found for category {category}")
                continue
                
            # For each image, create a prompt
            for image_path in images[:5]:  # Limit to 5 images per category to keep benchmarks manageable
                if task_type == "caption":
                    prompt = {
                        "image": image_path,
                        "text": prompt_template
                    }
                    # Get reference caption if available
                    reference = self._get_reference_answer(image_path, "caption")
                    
                elif task_type == "vqa":
                    # Generate different questions for different image types
                    if category == "natural":
                        question = "objects can you see"
                    elif category == "people":
                        question = "people are there"
                    elif category == "charts":
                        question = "is the trend shown"
                    else:
                        question = "is depicted"
                        
                    prompt_text = prompt_template.format(question=question)
                    prompt = {
                        "image": image_path,
                        "text": prompt_text
                    }
                    # Get reference VQA answer if available
                    reference = self._get_reference_answer(image_path, "vqa", question=question)
                    
                elif task_type == "classify":
                    # Common categories for classification
                    categories = "landscape, portrait, abstract, food, animal, architecture, technology"
                    prompt = {
                        "image": image_path,
                        "text": prompt_template.format(categories=categories)
                    }
                    # Get reference classification if available
                    reference = self._get_reference_answer(image_path, "classify")
                    
                elif task_type == "count":
                    # Object depends on category
                    if category == "people":
                        object_type = "people"
                    elif category == "objects":
                        object_type = "distinct objects"
                    else:
                        object_type = "elements"
                        
                    prompt = {
                        "image": image_path,
                        "text": prompt_template.format(object=object_type)
                    }
                    # Get reference count if available
                    reference = self._get_reference_answer(image_path, "count", object=object_type)
                    
                else:  # Default to detail or OCR
                    prompt = {
                        "image": image_path,
                        "text": prompt_template
                    }
                    # Get reference answer if available
                    reference = self._get_reference_answer(image_path, task_type)
                
                prompts.append(prompt)
                reference_answers.append(reference)
        
        # Define metrics for this task
        metrics = {
            "response_time": "Time to generate response in milliseconds",
            "token_count": "Number of tokens in the response",
            "response_length": "Character length of the response"
        }
        
        # Add task-specific metrics
        if task_type == "caption":
            metrics["bleu_score"] = "BLEU score against reference caption"
            metrics["semantic_similarity"] = "Semantic similarity to reference caption"
        elif task_type == "classify":
            metrics["category_accuracy"] = "Accuracy of category prediction"
            metrics["category_confidence"] = "Confidence in the category prediction"
        elif task_type == "count":
            metrics["count_accuracy"] = "Accuracy of counting objects"
            metrics["count_error"] = "Absolute error in counting"
        elif task_type == "ocr":
            metrics["text_accuracy"] = "Accuracy of extracted text"
            metrics["character_error_rate"] = "Character error rate compared to reference"
        
        # Create and return the task
        task = BenchmarkTask.objects.create(
            name=task_name,
            description=description,
            category='custom',  # Use custom category for vision tasks
            prompts=prompts,
            reference_answers=reference_answers,
            metrics=metrics,
            created_by=self.user,
            is_public=True  # Make vision benchmark tasks public by default
        )
        
        return task
    
    def _get_reference_answer(self, image_path: str, task_type: str, **kwargs) -> Optional[str]:
        """
        Get reference answer for an image and task type if available.
        
        Args:
            image_path: Path to the image
            task_type: Type of task (caption, vqa, etc.)
            **kwargs: Additional parameters for specific task types
            
        Returns:
            Optional[str]: Reference answer or None if not available
        """
        # Normalize path to handle differences in path representation
        norm_path = os.path.normpath(image_path)
        
        # Check if we have reference answers for this image
        if norm_path not in self.reference_answers:
            return None
            
        # Check if we have reference answers for this task type
        image_refs = self.reference_answers[norm_path]
        if task_type not in image_refs:
            return None
            
        # Handle specific task types with additional parameters
        if task_type == "vqa":
            # For VQA, the reference answer depends on the question
            question = kwargs.get('question', '')
            question_key = question.lower().strip()
            
            # Try to find closest matching question
            if isinstance(image_refs[task_type], dict):
                if question_key in image_refs[task_type]:
                    return image_refs[task_type][question_key]
                    
                # Try to find a similar question
                for ref_question, answer in image_refs[task_type].items():
                    if self._is_similar_text(question_key, ref_question, threshold=0.7):
                        return answer
            
            return None
            
        elif task_type == "count":
            # For counting, the reference answer depends on the object
            object_type = kwargs.get('object', '')
            object_key = object_type.lower().strip()
            
            # Try to find closest matching object
            if isinstance(image_refs[task_type], dict):
                if object_key in image_refs[task_type]:
                    return image_refs[task_type][object_key]
                    
                # Try to find a similar object
                for ref_object, answer in image_refs[task_type].items():
                    if self._is_similar_text(object_key, ref_object, threshold=0.7):
                        return answer
            
            return None
        
        # For other task types, return the reference answer directly
        return image_refs[task_type]
    
    def run_benchmark(self, benchmark_run_id: int) -> BenchmarkRun:
        """
        Run a vision benchmark and collect results
        
        Args:
            benchmark_run_id: ID of the benchmark run to execute
            
        Returns:
            BenchmarkRun: The updated benchmark run
        """
        try:
            benchmark_run = BenchmarkRun.objects.get(id=benchmark_run_id)
        except BenchmarkRun.DoesNotExist:
            raise ValueError(f"Benchmark run {benchmark_run_id} not found")
        
        # Update status to running
        benchmark_run.status = 'running'
        benchmark_run.save()
        
        try:
            # Parse models to test
            models_to_test = []
            for model_str in benchmark_run.models_tested:
                parts = model_str.split(':')
                if len(parts) == 2:
                    provider, model = parts
                    models_to_test.append((provider, model))
                else:
                    logger.warning(f"Invalid model format: {model_str}")
            
            if not models_to_test:
                raise ValueError("No valid models to test")
            
            # Run benchmarks for each task
            for task in benchmark_run.tasks.all():
                self._run_vision_task_benchmark(benchmark_run, task, models_to_test)
            
            # Generate summary
            self._generate_vision_summary(benchmark_run)
            
            # Update status to completed
            benchmark_run.status = 'completed'
            benchmark_run.end_time = timezone.now()
            benchmark_run.save()
            
            # Export metrics for this benchmark run
            export_dir = os.path.join(settings.MEDIA_ROOT, 'benchmarks', f'run_{benchmark_run_id}')
            export_all_metrics(export_dir)
            
            # Connect benchmark results to the alert system
            try:
                # Import here to avoid circular imports
                from models_app.ai_models.utils.common.metrics_alerts import connect_benchmark_to_alerts
                
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
                        logger.info(f"Using benchmark run #{previous_run.id} as baseline for comparison")
                        baseline_results = []
                        for result in BenchmarkResult.objects.filter(benchmark_run=previous_run):
                            baseline_results.append({
                                'model_id': result.model_id,
                                'metrics': result.metrics
                            })
                except Exception as be:
                    logger.warning(f"Failed to load baseline benchmark results: {be}")
                
                # Connect to alert system
                connect_benchmark_to_alerts(benchmark_results, baseline_results)
                logger.info("Connected benchmark results to alert system")
                
            except Exception as ae:
                logger.warning(f"Failed to connect benchmark results to alert system: {ae}")
            
        except Exception as e:
            logger.exception(f"Error running vision benchmark: {e}")
            benchmark_run.status = 'failed'
            benchmark_run.end_time = timezone.now()
            benchmark_run.save()
        
        return benchmark_run
    
    def _run_vision_task_benchmark(self, 
                                  benchmark_run: BenchmarkRun, 
                                  task: BenchmarkTask, 
                                  models: List[Tuple[str, str]]):
        """
        Run benchmarks for a specific vision task
        
        Args:
            benchmark_run: The benchmark run
            task: The benchmark task to run
            models: List of (provider_name, model_name) tuples to benchmark
        """
        # Get prompts and reference answers for this task
        prompts = task.prompts
        if isinstance(prompts, str):
            prompts = json.loads(prompts)
            
        reference_answers = task.reference_answers
        if isinstance(reference_answers, str):
            reference_answers = json.loads(reference_answers)
        
        # For each model
        for provider_name, model_name in models:
            try:
                # Get provider instance with metrics enabled
                config = {
                    "model_name": model_name,
                    "enable_metrics": True
                }
                
                provider = self.factory.get_provider(provider_name, config)
                
                # Initialize the provider
                provider.initialize()
                
                # For each prompt
                for prompt_index, prompt_data in enumerate(prompts):
                    # Process prompt data
                    image_path = prompt_data.get("image")
                    prompt_text = prompt_data.get("text", "Describe this image.")
                    
                    # Get reference answer if available
                    reference_answer = None
                    if reference_answers and len(reference_answers) > prompt_index:
                        reference_answer = reference_answers[prompt_index]
                    
                    # Run iterations
                    for iteration in range(benchmark_run.iterations):
                        try:
                            # Measure performance
                            start_time = time.time()
                            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                            
                            # Process image and get response
                            result_text, confidence = provider.process_image(image_path, prompt_text)
                            
                            # Calculate metrics
                            end_time = time.time()
                            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                            
                            response_time = (end_time - start_time) * 1000  # Convert to ms
                            memory_used = end_memory - start_memory
                            token_count = self._estimate_tokens(result_text)
                            
                            # Get GPU metrics if available
                            gpu_metrics = self._get_gpu_metrics(provider)
                            
                            # Build metrics dictionary
                            metrics = {
                                "response_time_ms": response_time,
                                "memory_usage_mb": memory_used,
                                "output_tokens": token_count,
                                "confidence": confidence,
                                "response_length": len(result_text)
                            }
                            
                            # Add GPU metrics if available
                            if gpu_metrics:
                                metrics.update(gpu_metrics)
                            
                            # Evaluate text quality if reference answer is available
                            if reference_answer:
                                quality_metrics = self._evaluate_text_quality(
                                    task.name,
                                    result_text, 
                                    reference_answer
                                )
                                metrics.update(quality_metrics)
                            
                            # Create the benchmark result
                            BenchmarkResult.objects.create(
                                benchmark_run=benchmark_run,
                                task=task,
                                model_id=f"{provider_name}:{model_name}",
                                prompt_index=prompt_index,
                                prompt_text=json.dumps(prompt_data),
                                response_text=result_text,
                                metrics=metrics,
                                iteration=iteration
                            )
                            
                            # Record in the benchmark metrics
                            self.metrics.record_vision_operation(
                                operation="benchmark",
                                details={
                                    "provider": provider_name,
                                    "model": model_name,
                                    "task": task.name,
                                    "response_time_ms": response_time,
                                    "memory_usage_mb": memory_used,
                                    "confidence": confidence
                                }
                            )
                            
                        except Exception as e:
                            logger.exception(f"Error benchmarking {provider_name}:{model_name} with prompt {prompt_index}: {e}")
                            
                            # Create a failed result
                            BenchmarkResult.objects.create(
                                benchmark_run=benchmark_run,
                                task=task,
                                model_id=f"{provider_name}:{model_name}",
                                prompt_index=prompt_index,
                                prompt_text=json.dumps(prompt_data),
                                response_text=f"ERROR: {str(e)}",
                                metrics={"error": str(e)},
                                iteration=iteration
                            )
                            
                            # Record error in metrics
                            self.metrics.record_vision_error(
                                error_type="benchmark_error",
                                details={
                                    "provider": provider_name,
                                    "model": model_name,
                                    "task": task.name,
                                    "error": str(e)
                                }
                            )
                
            except Exception as e:
                logger.exception(f"Error with provider {provider_name}:{model_name}: {e}")
                self.metrics.record_vision_error(
                    error_type="provider_error",
                    details={
                        "provider": provider_name,
                        "model": model_name,
                        "error": str(e)
                    }
                )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _get_gpu_metrics(self, provider) -> Dict[str, float]:
        """Get GPU metrics from the provider's metrics collector"""
        metrics = {}
        
        try:
            # Try to get GPU metrics from the provider's metrics collector
            if hasattr(provider, 'metrics') and provider.metrics:
                # Extract GPU utilization if available
                if hasattr(provider.metrics, 'resources') and 'gpu_utilization' in provider.metrics.resources:
                    gpu_util = provider.metrics.resources['gpu_utilization']
                    if gpu_util and len(gpu_util) > 0:
                        metrics['gpu_utilization'] = sum(gpu_util) / len(gpu_util)
                
                # Extract GPU memory if available
                if hasattr(provider.metrics, 'resources') and 'gpu_memory' in provider.metrics.resources:
                    gpu_mem = provider.metrics.resources['gpu_memory']
                    if gpu_mem and len(gpu_mem) > 0:
                        metrics['gpu_memory_percent'] = sum(gpu_mem) / len(gpu_mem)
                        
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def _evaluate_text_quality(self, task_name: str, generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        Evaluate the quality of generated text compared to reference text.
        Uses different metrics based on the task type.
        
        Args:
            task_name: Name of the task
            generated_text: Generated text
            reference_text: Reference text
            
        Returns:
            Dict: Dictionary of quality metrics
        """
        metrics = {}
        
        # Extract task type from task name
        task_type = task_name.lower().split()[1] if len(task_name.split()) > 1 else "caption"
        
        # Calculate BLEU score for text generation tasks
        if task_type in ["caption", "detail"]:
            try:
                # Tokenize texts
                generated_tokens = word_tokenize(generated_text.lower())
                reference_tokens = word_tokenize(reference_text.lower())
                
                # Calculate BLEU score
                bleu_score = sentence_bleu([reference_tokens], generated_tokens, 
                                       smoothing_function=self.smoother)
                metrics["bleu_score"] = bleu_score
                
                # Calculate semantic similarity (using simple approach)
                semantic_sim = self._calculate_semantic_similarity(generated_text, reference_text)
                metrics["semantic_similarity"] = semantic_sim
            except Exception as e:
                logger.warning(f"Error calculating text quality metrics: {e}")
        
        # Calculate classification accuracy for classify task
        elif task_type == "classify":
            try:
                # Extract category from reference and generated texts
                # This assumes the format "The category is: X" or similar
                ref_category = self._extract_category(reference_text)
                gen_category = self._extract_category(generated_text)
                
                # Calculate accuracy (1.0 if match, 0.0 if not)
                accuracy = 1.0 if ref_category and gen_category and ref_category.lower() == gen_category.lower() else 0.0
                metrics["category_accuracy"] = accuracy
            except Exception as e:
                logger.warning(f"Error calculating classification accuracy: {e}")
        
        # Calculate counting accuracy for count task
        elif task_type == "count":
            try:
                # Extract numbers from reference and generated texts
                ref_count = self._extract_number(reference_text)
                gen_count = self._extract_number(generated_text)
                
                if ref_count is not None and gen_count is not None:
                    # Calculate absolute error
                    abs_error = abs(ref_count - gen_count)
                    metrics["count_error"] = abs_error
                    
                    # Calculate accuracy (scaled between 0 and 1)
                    # Use max(1, ref_count) to avoid division by zero
                    accuracy = max(0, 1.0 - (abs_error / max(1, ref_count)))
                    metrics["count_accuracy"] = accuracy
            except Exception as e:
                logger.warning(f"Error calculating counting accuracy: {e}")
        
        # Calculate OCR accuracy for OCR task
        elif task_type == "ocr":
            try:
                # Calculate character error rate (CER)
                cer = self._calculate_character_error_rate(generated_text, reference_text)
                metrics["character_error_rate"] = cer
                
                # Calculate accuracy as 1 - CER
                metrics["text_accuracy"] = max(0, 1.0 - cer)
            except Exception as e:
                logger.warning(f"Error calculating OCR accuracy: {e}")
                
        # Add general text similarity metrics for all tasks
        metrics["text_similarity"] = self._calculate_text_similarity(generated_text, reference_text)
        
        return metrics
    
    def _extract_category(self, text: str) -> Optional[str]:
        """
        Extract a category name from text.
        
        Args:
            text: Text to extract category from
            
        Returns:
            Optional[str]: Extracted category or None if not found
        """
        # Common patterns for category extraction
        patterns = [
            r"category\s+is\s*:?\s*([\w\s]+)",
            r"categorized\s+as\s*:?\s*([\w\s]+)",
            r"classified\s+as\s*:?\s*([\w\s]+)",
            r"belongs\s+to\s*:?\s*([\w\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                category = match.group(1).strip()
                return category
        
        # If no pattern matches, return the first word after removing common stop words
        words = text.lower().replace('.', ' ').replace(',', ' ').split()
        stop_words = ["the", "a", "an", "is", "are", "this", "that", "image", "shows", "depicts", "contains", "i", "see"]
        
        for word in words:
            if word not in stop_words:
                return word
        
        return None
    
    def _extract_number(self, text: str) -> Optional[int]:
        """
        Extract a number from text.
        
        Args:
            text: Text to extract number from
            
        Returns:
            Optional[int]: Extracted number or None if not found
        """
        # Look for digits
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))
        
        # Look for number words
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        
        for word, value in number_words.items():
            if word in text.lower():
                return value
        
        return None
    
    def _calculate_character_error_rate(self, generated_text: str, reference_text: str) -> float:
        """
        Calculate character error rate (CER) between generated and reference text.
        
        Args:
            generated_text: Generated text
            reference_text: Reference text
            
        Returns:
            float: Character error rate (lower is better)
        """
        # Normalize texts
        generated_norm = ''.join(generated_text.lower().split())
        reference_norm = ''.join(reference_text.lower().split())
        
        # Calculate edit distance
        edit_distance = self._levenshtein_distance(generated_norm, reference_norm)
        
        # Calculate CER
        cer = edit_distance / max(1, len(reference_norm))
        
        return cer
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein (edit) distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            int: Edit distance
        """
        m, n = len(s1), len(s2)
        
        # Create distance matrix
        d = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        
        # Fill the matrix
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,  # Deletion
                        d[i][j - 1] + 1,  # Insertion
                        d[i - 1][j - 1] + 1  # Substitution
                    )
        
        return d[m][n]
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        This is a simple implementation that could be enhanced with embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Semantic similarity (0-1)
        """
        # For a simple implementation, use word overlap
        # In a real system, you'd use embeddings or a semantic model
        
        # Tokenize and normalize
        words1 = set(word_tokenize(text1.lower()))
        words2 = set(word_tokenize(text2.lower()))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "and", "in", "on", "at", "to", "of", "for", "with", "by"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Text similarity (0-1)
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _is_similar_text(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """
        Check if two texts are similar.
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (0-1)
            
        Returns:
            bool: True if texts are similar, False otherwise
        """
        similarity = self._calculate_text_similarity(text1, text2)
        return similarity >= threshold
    
    def _generate_vision_summary(self, benchmark_run: BenchmarkRun):
        """
        Generate summary statistics for a vision benchmark run
        
        Args:
            benchmark_run: The benchmark run to summarize
        """
        results = BenchmarkResult.objects.filter(benchmark_run=benchmark_run)
        
        if not results.exists():
            logger.warning(f"No results found for benchmark run {benchmark_run.id}")
            return
        
        # Group results by model and task
        summary_data = {}
        rankings = {}
        
        # Process all models
        for model_str in benchmark_run.models_tested:
            summary_data[model_str] = {}
            
            # Process all tasks for this model
            for task in benchmark_run.tasks.all():
                task_results = results.filter(model_id=model_str, task=task)
                
                if not task_results.exists():
                    continue
                
                # Calculate metrics
                metrics_data = self._calculate_vision_metrics(task_results)
                
                summary_data[model_str][task.id] = {
                    'task_name': task.name,
                    'metrics': metrics_data,
                    'result_count': task_results.count()
                }
        
        # Calculate rankings for key metrics
        key_metrics = ['avg_response_time_ms', 'avg_confidence', 'avg_memory_usage_mb']
        
        for metric in key_metrics:
            rankings[metric] = []
            
            # Collect values for all models
            model_values = []
            for model_str in summary_data:
                # Calculate average across all tasks
                values = []
                for task_id in summary_data[model_str]:
                    metrics = summary_data[model_str][task_id]['metrics']
                    if metric in metrics:
                        values.append(metrics[metric])
                
                if values:
                    avg_value = sum(values) / len(values)
                    model_values.append((model_str, avg_value))
            
            # Sort by value (ascending for time and memory, descending for confidence)
            descending = metric == 'avg_confidence'
            model_values.sort(key=lambda x: x[1], reverse=descending)
            
            # Create rankings with rank, model, and value
            for i, (model, value) in enumerate(model_values):
                rankings[metric].append({
                    'rank': i + 1,
                    'model': model,
                    'value': value
                })
        
        # Create model profiles (strengths and weaknesses)
        model_profiles = {}
        for model_str in summary_data:
            # Get provider and model name
            parts = model_str.split(':')
            provider = parts[0] if len(parts) > 1 else model_str
            
            # Find best and worst tasks for this model
            best_task = None
            worst_task = None
            best_score = -float('inf')
            worst_score = float('inf')
            
            for task_id, task_data in summary_data[model_str].items():
                task_name = task_data['task_name']
                
                # Use response time (lower is better) and confidence (higher is better)
                response_time = task_data['metrics'].get('avg_response_time_ms', float('inf'))
                confidence = task_data['metrics'].get('avg_confidence', 0)
                
                # Calculate a composite score (normalize response time and weight confidence more heavily)
                time_score = 1000 / max(1, response_time)  # Invert so higher is better
                composite_score = (time_score * 0.3) + (confidence * 0.7)  # Weight confidence more
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_task = task_name
                
                if composite_score < worst_score:
                    worst_score = composite_score
                    worst_task = task_name
            
            # Build model profile
            model_profiles[model_str] = {
                'provider': provider,
                'best_task': best_task,
                'worst_task': worst_task,
                'avg_response_time': self._get_model_average(model_str, summary_data, 'avg_response_time_ms'),
                'avg_confidence': self._get_model_average(model_str, summary_data, 'avg_confidence'),
                'avg_memory_usage': self._get_model_average(model_str, summary_data, 'avg_memory_usage_mb')
            }
        
        # Create the benchmark summary
        BenchmarkSummary.objects.create(
            benchmark_run=benchmark_run,
            results=summary_data,
            rankings={
                'response_time': rankings.get('avg_response_time_ms', []),
                'confidence': rankings.get('avg_confidence', []),
                'memory_usage': rankings.get('avg_memory_usage_mb', []),
                'model_profiles': model_profiles
            }
        )
    
    def _calculate_vision_metrics(self, results) -> Dict[str, float]:
        """
        Calculate metrics from benchmark results
        
        Args:
            results: Queryset of BenchmarkResult objects
            
        Returns:
            Dict: Dictionary of calculated metrics
        """
        metrics_sum = {}
        metrics_list = {}
        
        for result in results:
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    metrics_sum[metric] = metrics_sum.get(metric, 0) + value
                    if metric not in metrics_list:
                        metrics_list[metric] = []
                    metrics_list[metric].append(value)
        
        # Calculate averages and standard deviations
        metrics_data = {}
        for metric, total in metrics_sum.items():
            metrics_data[f"avg_{metric}"] = total / results.count()
            if len(metrics_list[metric]) > 1:
                import statistics
                metrics_data[f"std_{metric}"] = statistics.stdev(metrics_list[metric])
            else:
                metrics_data[f"std_{metric}"] = 0
        
        return metrics_data
    
    def _get_model_average(self, model_str: str, summary_data: Dict, metric: str) -> float:
        """Get the average value of a metric across all tasks for a model"""
        values = []
        for task_id in summary_data[model_str]:
            if metric in summary_data[model_str][task_id]['metrics']:
                values.append(summary_data[model_str][task_id]['metrics'][metric])
        
        if not values:
            return 0
            
        return sum(values) / len(values)
    
    def _load_edge_case_images(self) -> Dict[str, List[str]]:
        """
        Load edge case test images for testing model robustness.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping edge case type to image paths
        """
        edge_case_dir = Path(settings.MEDIA_ROOT) / 'benchmark' / 'edge_cases'
        if not edge_case_dir.exists():
            logger.warning(f"Edge case directory {edge_case_dir} not found. Creating it.")
            edge_case_dir.mkdir(parents=True, exist_ok=True)
            # Since we don't have edge cases yet, we'll return an empty dict
            return {}
        
        edge_case_types = {
            'corrupt': [],           # Corrupt/malformed image files
            'extreme_resolution': [], # Very small or large images
            'unusual_content': [],   # Images with unusual/edge case content
            'unusual_format': [],    # Unusual image formats (TIFF, HDR, etc.)
            'no_content': [],        # Blank or near-blank images
            'text_dense': [],        # Images with dense text content
            'low_contrast': [],      # Low contrast images
            'high_noise': [],        # Images with excessive noise
            'artifacts': [],         # Images with compression artifacts
            'rotated': [],           # Rotated or inverted images
        }
        
        # Populate each edge case type with images from corresponding directories
        for edge_type in edge_case_types:
            type_dir = edge_case_dir / edge_type
            if type_dir.exists():
                edge_case_types[edge_type] = [
                    str(file) for file in type_dir.glob('*')
                    if file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff')
                ]
                logger.info(f"Loaded {len(edge_case_types[edge_type])} {edge_type} edge case images")
            else:
                logger.warning(f"Edge case type directory {type_dir} not found")
        
        return edge_case_types
    
    def run_edge_case_tests(self, provider_name: str, model_name: str) -> Dict[str, Any]:
        """
        Run edge case tests on a specific provider and model to evaluate robustness.
        
        Args:
            provider_name: The name of the provider to test
            model_name: The name of the model to test
            
        Returns:
            Dict[str, Any]: Results of edge case testing with metrics
        """
        logger.info(f"Running edge case tests for {provider_name}/{model_name}")
        start_time = time.time()
        
        # Get provider instance
        provider = self.factory.get_provider(provider_name, model_name)
        if not provider:
            logger.error(f"Could not get provider {provider_name}/{model_name}")
            return {"error": f"Provider {provider_name}/{model_name} not available"}
        
        # Initialize results
        results = {
            "provider": provider_name,
            "model": model_name,
            "total_tests": 0,
            "successful_tests": 0,
            "failure_tests": 0,
            "error_rate": 0.0,
            "error_details": {},
            "performance_details": {},
            "edge_case_types": {},
        }
        
        # Generate edge cases if none exist
        if not self.edge_case_images:
            logger.warning("No edge case images found, generating synthetic ones")
            self._generate_synthetic_edge_cases()
        
        # Test each edge case type
        for edge_type, images in self.edge_case_images.items():
            if not images:
                logger.warning(f"No images for edge case type {edge_type}, skipping")
                continue
                
            type_results = {
                "total": len(images),
                "successful": 0,
                "errors": 0,
                "error_types": {},
                "avg_confidence": 0.0,
                "avg_processing_time": 0.0,
            }
            
            # Process images of this type
            confidences = []
            processing_times = []
            
            for image_path in images:
                try:
                    # Basic prompt for testing
                    prompt = "Describe this image briefly."
                    
                    # Time the processing
                    process_start = time.time()
                    response, confidence = provider.process_image(image_path, prompt)
                    process_time = time.time() - process_start
                    
                    # Record successful processing
                    type_results["successful"] += 1
                    confidences.append(confidence)
                    processing_times.append(process_time)
                    
                except Exception as e:
                    # Record error details
                    error_type = e.__class__.__name__
                    if error_type not in type_results["error_types"]:
                        type_results["error_types"][error_type] = 0
                    type_results["error_types"][error_type] += 1
                    type_results["errors"] += 1
                    logger.warning(f"Edge case error ({edge_type}): {str(e)}")
            
            # Calculate statistics
            if confidences:
                type_results["avg_confidence"] = sum(confidences) / len(confidences)
            if processing_times:
                type_results["avg_processing_time"] = sum(processing_times) / len(processing_times)
            
            # Add to overall results
            results["total_tests"] += type_results["total"]
            results["successful_tests"] += type_results["successful"]
            results["failure_tests"] += type_results["errors"]
            results["edge_case_types"][edge_type] = type_results
        
        # Calculate overall error rate
        if results["total_tests"] > 0:
            results["error_rate"] = results["failure_tests"] / results["total_tests"]
        
        # Record total processing time
        results["total_time"] = time.time() - start_time
        
        # Record metrics
        if self.metrics:
            self.metrics.record_custom_metric(
                "edge_case_testing", 
                "total_tests", 
                results["total_tests"],
                {"provider": provider_name, "model": model_name}
            )
            self.metrics.record_custom_metric(
                "edge_case_testing", 
                "error_rate", 
                results["error_rate"],
                {"provider": provider_name, "model": model_name}
            )
        
        logger.info(f"Edge case testing completed for {provider_name}/{model_name}: "
                   f"{results['successful_tests']}/{results['total_tests']} tests passed "
                   f"({(1-results['error_rate'])*100:.1f}% success rate)")
        
        return results
    
    def _generate_synthetic_edge_cases(self) -> None:
        """
        Generate synthetic edge case images for testing when real ones are not available.
        
        This creates a variety of problematic images to test model robustness.
        """
        edge_case_dir = Path(settings.MEDIA_ROOT) / 'benchmark' / 'edge_cases'
        edge_case_dir.mkdir(parents=True, exist_ok=True)
        
        # Get a base image to work with
        base_images = []
        for category, images in self.test_images.items():
            if images:
                base_images.extend(images)
        
        if not base_images:
            logger.error("No base images available to generate synthetic edge cases")
            return
        
        # Define edge case types and creation functions
        edge_cases = {
            'corrupt': self._create_corrupt_image,
            'extreme_resolution': self._create_extreme_resolution_image,
            'no_content': self._create_blank_image,
            'low_contrast': self._create_low_contrast_image,
            'high_noise': self._create_noisy_image,
            'rotated': self._create_rotated_image,
        }
        
        # Generate edge cases
        self.edge_case_images = {}
        
        for edge_type, creator_function in edge_cases.items():
            type_dir = edge_case_dir / edge_type
            type_dir.mkdir(exist_ok=True)
            
            # Generate 5 images of each type
            generated_paths = []
            for i in range(5):
                base_image_path = random.choice(base_images)
                try:
                    # Load base image
                    with Image.open(base_image_path) as img:
                        # Create edge case
                        edge_case_img = creator_function(img)
                        # Save to file
                        output_path = type_dir / f"synthetic_{edge_type}_{i}.png"
                        edge_case_img.save(str(output_path))
                        generated_paths.append(str(output_path))
                except Exception as e:
                    logger.error(f"Failed to create {edge_type} edge case from {base_image_path}: {e}")
            
            self.edge_case_images[edge_type] = generated_paths
            logger.info(f"Generated {len(generated_paths)} synthetic {edge_type} images")
    
    def _create_corrupt_image(self, img: Image.Image) -> Image.Image:
        """Create a corrupt image that is still loadable but problematic"""
        # Convert to JPEG with low quality, then add random corruption
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=10)
        buffer.seek(0)
        corrupt_data = bytearray(buffer.getvalue())
        
        # Corrupt some bytes, but avoid headers
        for i in range(20):
            idx = random.randint(100, len(corrupt_data) - 100)
            corrupt_data[idx] = random.randint(0, 255)
        
        # Try to load the corrupted image, if it fails return original with artifacts
        try:
            return Image.open(io.BytesIO(corrupt_data))
        except:
            # If loading fails, return a heavily compressed version instead
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=1)
            buffer.seek(0)
            return Image.open(buffer)
    
    def _create_extreme_resolution_image(self, img: Image.Image) -> Image.Image:
        """Create an image with extreme dimensions"""
        # Randomly choose between very small or very large
        if random.choice([True, False]):
            # Very small image
            return img.resize((8, 8), Image.NEAREST)
        else:
            # Very large/disproportionate image
            return img.resize((img.width * 10, img.height), Image.NEAREST)
    
    def _create_blank_image(self, img: Image.Image) -> Image.Image:
        """Create a nearly blank image"""
        # Create a new blank image with slight noise
        blank = Image.new('RGB', img.size, (240, 240, 240))
        # Add minimal noise
        for _ in range(100):
            x = random.randint(0, img.width - 1)
            y = random.randint(0, img.height - 1)
            blank.putpixel((x, y), (random.randint(230, 250), random.randint(230, 250), random.randint(230, 250)))
        return blank
    
    def _create_low_contrast_image(self, img: Image.Image) -> Image.Image:
        """Create a low contrast version of the image"""
        # Convert to numpy array for manipulation
        np_img = np.array(img)
        # Reduce contrast
        np_img = np_img.astype(float)
        mean = np_img.mean(axis=(0, 1))
        np_img = (np_img - mean) * 0.3 + mean
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    
    def _create_noisy_image(self, img: Image.Image) -> Image.Image:
        """Create a noisy version of the image"""
        # Convert to numpy array for manipulation
        np_img = np.array(img).astype(float)
        # Add noise
        noise = np.random.normal(0, 50, np_img.shape)
        np_img = np_img + noise
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    
    def _create_rotated_image(self, img: Image.Image) -> Image.Image:
        """Create a rotated or inverted version of the image"""
        # Randomly choose rotation or inversion
        choice = random.randint(0, 3)
        if choice == 0:
            # Rotate 90 degrees
            return img.rotate(90, expand=True)
        elif choice == 1:
            # Rotate 180 degrees
            return img.rotate(180)
        elif choice == 2:
            # Flip horizontally
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # Flip vertically
            return img.transpose(Image.FLIP_TOP_BOTTOM) 