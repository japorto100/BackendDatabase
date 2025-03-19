from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from benchmark.models import BenchmarkTask
import os
import json

User = get_user_model()

class Command(BaseCommand):
    help = 'Creates benchmark tasks for fusion strategies'
    
    def handle(self, *args, **options):
        # Get or create admin user
        admin_user = User.objects.filter(is_superuser=True).first()
        if not admin_user:
            self.stdout.write(self.style.WARNING('No admin user found. Tasks will have no creator.'))
        
        # Create fusion strategy benchmark task
        fusion_task, created = BenchmarkTask.objects.get_or_create(
            name='Fusion Strategy Evaluation',
            defaults={
                'description': 'Tests the performance of different fusion strategies (early, late, attention, hybrid) on various document types.',
                'category': 'custom',
                'prompts': [
                    "Evaluate early fusion strategy on academic document",
                    "Evaluate late fusion strategy on academic document",
                    "Evaluate attention fusion strategy on academic document",
                    "Evaluate hybrid fusion strategy on academic document",
                    "Evaluate early fusion strategy on business document",
                    "Evaluate late fusion strategy on business document",
                    "Evaluate attention fusion strategy on business document",
                    "Evaluate hybrid fusion strategy on business document",
                    "Evaluate early fusion strategy on general document",
                    "Evaluate late fusion strategy on general document",
                    "Evaluate attention fusion strategy on general document",
                    "Evaluate hybrid fusion strategy on general document",
                ],
                'metrics': ['processing_time_ms', 'memory_usage_mb', 'quality_score', 'confidence_score'],
                'created_by': admin_user,
                'is_public': True
            }
        )
        
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created task: {fusion_task.name}'))
        else:
            self.stdout.write(f'Task already exists: {fusion_task.name}')
        
        # Create document type benchmark task
        document_task, created = BenchmarkTask.objects.get_or_create(
            name='Document Type Fusion Performance',
            defaults={
                'description': 'Tests the performance of fusion strategies on different document types.',
                'category': 'custom',
                'prompts': [
                    "Process academic paper with hybrid fusion",
                    "Process business report with hybrid fusion",
                    "Process invoice with hybrid fusion",
                    "Process scientific article with hybrid fusion",
                    "Process news article with hybrid fusion",
                    "Process legal document with hybrid fusion",
                    "Process technical manual with hybrid fusion",
                    "Process presentation slides with hybrid fusion",
                ],
                'metrics': ['processing_time_ms', 'memory_usage_mb', 'quality_score', 'best_strategy', 'strategy_confidence'],
                'created_by': admin_user,
                'is_public': True
            }
        )
        
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created task: {document_task.name}'))
        else:
            self.stdout.write(f'Task already exists: {document_task.name}')
        
        self.stdout.write(self.style.SUCCESS('Fusion benchmark tasks created successfully!')) 