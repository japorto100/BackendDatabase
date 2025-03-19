from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
import time
import psutil
from .models import AnalyticsEvent

# Create your tests here.

class PerformanceTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        self.client.login(username='testuser', password='testpassword')
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss
        
    def tearDown(self):
        end_time = time.time()
        memory_end = psutil.Process().memory_info().rss
        print(f"Test execution time: {end_time - self.start_time:.4f} seconds")
        print(f"Memory usage: {(memory_end - self.memory_start) / 1024 / 1024:.2f} MB")
    
    def test_api_endpoint_performance(self):
        """Test the performance of an API endpoint"""
        # Make a request to an API endpoint
        start_time = time.time()
        response = self.client.get(reverse('analytics:dashboard-data'))
        duration = time.time() - start_time
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Check performance
        self.assertLess(duration, 1.0, "API response took too long")
        
        # Check if event was logged
        events = AnalyticsEvent.objects.filter(event_type='api_request')
        self.assertTrue(events.exists())
        
        # Check performance metrics in the event
        event = events.first()
        self.assertIn('performance', event.data)
        self.assertIn('response_time_ms', event.data['performance'])
        self.assertIn('memory_used_bytes', event.data['performance'])
        self.assertIn('cpu_percent', event.data['performance'])
