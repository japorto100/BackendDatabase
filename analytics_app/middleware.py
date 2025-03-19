import time
import json
import psutil
import os
from django.utils import timezone
from .models import AnalyticsEvent

class RequestLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Code to be executed for each request before the view is called
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        
        # Process the request
        response = self.get_response(request)
        
        # Code to be executed for each request after the view is called
        duration = time.time() - start_time
        end_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_used = end_memory - start_memory
        
        # Only log API requests
        if request.path.startswith('/api/'):
            # Don't log sensitive endpoints
            if not request.path.startswith('/api/users/'):
                self.log_request(request, response, duration, memory_used)
        
        return response
    
    def log_request(self, request, response, duration, memory_used):
        """Log the request to the database"""
        # Get user if authenticated
        user = request.user if request.user.is_authenticated else None
        
        # Get request data
        try:
            if request.method in ['POST', 'PUT', 'PATCH']:
                if request.content_type and 'application/json' in request.content_type:
                    request_data = json.loads(request.body) if request.body else {}
                else:
                    request_data = request.POST.dict()
                
                # Remove sensitive data
                if 'password' in request_data:
                    request_data['password'] = '[REDACTED]'
                if 'api_key' in request_data:
                    request_data['api_key'] = '[REDACTED]'
            else:
                request_data = request.GET.dict()
        except:
            request_data = {}
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Get disk I/O metrics
        disk_io = psutil.disk_io_counters()
        
        # Get network metrics
        net_io = psutil.net_io_counters()
        
        # Create analytics event
        AnalyticsEvent.objects.create(
            user=user,
            event_type='api_request',
            endpoint=request.path,
            method=request.method,
            status_code=response.status_code,
            response_time=duration,
            ip_address=self.get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            data={
                'request': request_data,
                'response_status': response.status_code,
                'performance': {
                    'response_time_ms': duration * 1000,  # Convert to milliseconds
                    'memory_used_bytes': memory_used,
                    'cpu_percent': cpu_percent,
                    'disk_io': {
                        'read_count': disk_io.read_count,
                        'write_count': disk_io.write_count,
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    },
                    'network_io': {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv,
                        'packets_sent': net_io.packets_sent,
                        'packets_recv': net_io.packets_recv
                    }
                }
            }
        )
    
    def get_client_ip(self, request):
        """Get the client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

class SecurityHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Add security headers
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
        
        return response

class PerformanceMonitoringMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.process = psutil.Process(os.getpid())

    def __call__(self, request):
        # Start monitoring
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        start_cpu_time = self.process.cpu_times()
        
        # Process the request
        response = self.get_response(request)
        
        # End monitoring
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        end_cpu_time = self.process.cpu_times()
        
        # Calculate metrics
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        cpu_user_delta = end_cpu_time.user - start_cpu_time.user
        cpu_system_delta = end_cpu_time.system - start_cpu_time.system
        
        # Add performance headers for debugging
        response['X-Response-Time-ms'] = str(int(duration * 1000))
        response['X-Memory-Usage-KB'] = str(int(memory_delta / 1024))
        
        # Store detailed metrics in request for potential logging
        request.performance_metrics = {
            'response_time': duration,
            'memory_usage': memory_delta,
            'cpu_user_time': cpu_user_delta,
            'cpu_system_time': cpu_system_delta,
            'timestamp': timezone.now()
        }
        
        return response 