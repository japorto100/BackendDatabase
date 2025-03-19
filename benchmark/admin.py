from django.contrib import admin
from django.urls import path, reverse
from django.utils.html import format_html
from django.shortcuts import render, redirect
from django.http import JsonResponse

from .models import BenchmarkTask, BenchmarkRun, BenchmarkResult, BenchmarkSummary

@admin.register(BenchmarkTask)
class BenchmarkTaskAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'prompt_count', 'created_at', 'is_public', 'created_by')
    list_filter = ('category', 'is_public', 'created_at')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'category', 'is_public')
        }),
        ('Prompts and Evaluation', {
            'fields': ('prompts', 'reference_answers', 'metrics'),
            'description': 'Define the prompts and evaluation criteria for this benchmark task.'
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
    
    def save_model(self, request, obj, form, change):
        if not obj.created_by:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(BenchmarkRun)
class BenchmarkRunAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'model_count', 'task_count', 'start_time', 'duration_display', 'user')
    list_filter = ('status', 'start_time', 'user')
    search_fields = ('name', 'description')
    readonly_fields = ('start_time', 'end_time', 'status', 'duration_display')
    
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'user')
        }),
        ('Configuration', {
            'fields': ('tasks', 'models_tested', 'iterations')
        }),
        ('Status', {
            'fields': ('status', 'start_time', 'end_time', 'duration_display'),
        }),
    )
    
    def duration_display(self, obj):
        if obj.duration is not None:
            seconds = obj.duration
            if seconds < 60:
                return f"{seconds:.1f} seconds"
            elif seconds < 3600:
                return f"{seconds/60:.1f} minutes"
            else:
                return f"{seconds/3600:.1f} hours"
        return "N/A"
    duration_display.short_description = "Duration"
    
    def model_count(self, obj):
        return len(obj.models_tested)
    model_count.short_description = "Models"
    
    def task_count(self, obj):
        return obj.tasks.count()
    task_count.short_description = "Tasks"
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('<int:run_id>/view-results/',
                 self.admin_site.admin_view(self.view_results),
                 name='benchmark_run_results'),
            path('<int:run_id>/run-benchmark/',
                 self.admin_site.admin_view(self.run_benchmark),
                 name='run_benchmark'),
        ]
        return custom_urls + urls
    
    def view_results(self, request, run_id):
        """View for displaying benchmark results"""
        from .services import BenchmarkRunner
        
        benchmark_run = self.get_object(request, run_id)
        
        try:
            summary = BenchmarkSummary.objects.get(benchmark_run=benchmark_run)
            context = {
                'title': f'Results for {benchmark_run.name}',
                'benchmark_run': benchmark_run,
                'summary': summary,
                'opts': self.model._meta,
            }
            return render(request, 'admin/benchmark/benchmark_results.html', context)
        except BenchmarkSummary.DoesNotExist:
            self.message_user(request, "No results available for this benchmark run.")
            return redirect('admin:benchmark_benchmarkrun_change', run_id)
    
    def run_benchmark(self, request, run_id):
        """View for running a benchmark"""
        from .services import BenchmarkRunner
        
        benchmark_run = self.get_object(request, run_id)
        
        if benchmark_run.status in ['completed', 'running']:
            self.message_user(request, f"Benchmark is already {benchmark_run.status}.")
            return redirect('admin:benchmark_benchmarkrun_change', run_id)
        
        # Run the benchmark (in a real app, this would be a background task)
        runner = BenchmarkRunner(request.user)
        try:
            runner.run_benchmark(run_id)
            self.message_user(request, "Benchmark completed successfully.")
            return redirect('admin:benchmark_run_results', run_id)
        except Exception as e:
            self.message_user(request, f"Error running benchmark: {str(e)}")
            return redirect('admin:benchmark_benchmarkrun_change', run_id)


@admin.register(BenchmarkResult)
class BenchmarkResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'benchmark_run', 'task', 'model_id', 'prompt_index', 'iteration', 'timestamp')
    list_filter = ('benchmark_run', 'task', 'model_id', 'timestamp')
    search_fields = ('prompt_text', 'response_text')
    readonly_fields = ('benchmark_run', 'task', 'model_id', 'prompt_index', 'prompt_text', 
                      'response_text', 'metrics', 'iteration', 'timestamp')


@admin.register(BenchmarkSummary)
class BenchmarkSummaryAdmin(admin.ModelAdmin):
    list_display = ('id', 'benchmark_run', 'created_at')
    readonly_fields = ('benchmark_run', 'results', 'rankings', 'created_at')
