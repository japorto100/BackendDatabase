from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import json

User = get_user_model()

class BenchmarkTask(models.Model):
    """Model for storing benchmark task definitions"""
    name = models.CharField(max_length=100)
    description = models.TextField()
    category = models.CharField(max_length=50, choices=[
        ('reasoning', 'Reasoning & Problem Solving'),
        ('factual', 'Factual Knowledge'),
        ('instruction', 'Instruction Following'),
        ('creativity', 'Creativity & Generation'),
        ('code', 'Code Understanding & Generation'),
        ('conversation', 'Conversational Ability'),
        ('custom', 'Custom Task'),
    ])
    prompts = models.JSONField(help_text="List of prompts for this task")
    reference_answers = models.JSONField(null=True, blank=True, 
                                        help_text="Optional reference answers for evaluation")
    metrics = models.JSONField(help_text="Metrics to evaluate for this task")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    is_public = models.BooleanField(default=False)
    
    def __str__(self):
        return self.name
    
    @property
    def prompt_count(self):
        """Return the number of prompts in this task"""
        if isinstance(self.prompts, str):
            return len(json.loads(self.prompts))
        return len(self.prompts)


class BenchmarkRun(models.Model):
    """Model for storing benchmark run information"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    tasks = models.ManyToManyField(BenchmarkTask)
    models_tested = models.JSONField(help_text="List of model IDs that were benchmarked")
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    iterations = models.IntegerField(default=1, 
                                    help_text="Number of times each prompt was run")
    
    def __str__(self):
        return f"{self.name} ({self.status})"
    
    @property
    def duration(self):
        """Return the duration of the benchmark run"""
        if not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()


class BenchmarkResult(models.Model):
    """Model for storing individual benchmark results"""
    benchmark_run = models.ForeignKey(BenchmarkRun, on_delete=models.CASCADE, 
                                     related_name='results')
    task = models.ForeignKey(BenchmarkTask, on_delete=models.CASCADE)
    model_id = models.CharField(max_length=100)
    prompt_index = models.IntegerField(help_text="Index of the prompt in the task")
    prompt_text = models.TextField()
    response_text = models.TextField()
    metrics = models.JSONField(help_text="Measured metrics for this result")
    iteration = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['benchmark_run', 'model_id', 'task']),
        ]
    
    def __str__(self):
        return f"Result for {self.model_id} on {self.task.name} (prompt {self.prompt_index})"


class BenchmarkSummary(models.Model):
    """Model for storing aggregated benchmark results"""
    benchmark_run = models.OneToOneField(BenchmarkRun, on_delete=models.CASCADE, 
                                        related_name='summary')
    results = models.JSONField(help_text="Aggregated results by model and task")
    rankings = models.JSONField(help_text="Rankings of models by different metrics")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Summary for {self.benchmark_run.name}"
