"""
Tests for Multimodal API Endpoints
"""
import json
import os
import pytest
from django.test import Client, TestCase
from django.urls import reverse
from django.contrib.auth.models import User

@pytest.mark.django_db
class TestMultimodalEndpoints(TestCase):
    """Test class for multimodal API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        
        # Create client
        self.client = Client()
        self.client.login(username='testuser', password='testpassword')
        
        # Test image path (update this to a test image in your test directory)
        self.test_image_path = os.path.join(
            os.path.dirname(__file__), 
            'test_images', 
            'test_image.jpg'
        )
        
        # Create test directory if it doesn't exist
        os.makedirs(os.path.dirname(self.test_image_path), exist_ok=True)
        
        # Create a simple test image if it doesn't exist
        if not os.path.exists(self.test_image_path):
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(self.test_image_path)
    
    def test_multimodal_chat(self):
        """Test the multimodal chat endpoint"""
        # Skip if the test image doesn't exist
        if not os.path.exists(self.test_image_path):
            pytest.skip("Test image not available")
        
        url = reverse('models:multimodal_chat')
        data = {
            'message': 'What do you see in this image?',
            'image_paths': [self.test_image_path],
            'session_id': '12345',
            'model': 'lightweight'  # Use lightweight model for testing
        }
        
        response = self.client.post(
            url,
            json.dumps(data),
            content_type='application/json'
        )
        
        # Validate response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'success')
        self.assertIn('response', response_data)
        self.assertIn('used_images', response_data)
    
    def test_multimodal_search(self):
        """Test the multimodal search endpoint"""
        # Skip if the test image doesn't exist
        if not os.path.exists(self.test_image_path):
            pytest.skip("Test image not available")
        
        url = reverse('models:multimodal_search')
        data = {
            'query': 'Find documents with similar content to this image',
            'image_path': self.test_image_path,
            'max_results': 5
        }
        
        response = self.client.post(
            url,
            json.dumps(data),
            content_type='application/json'
        )
        
        # Validate response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'success')
        self.assertIn('results', response_data)
        self.assertIn('count', response_data)
    
    def test_analyze_image(self):
        """Test the image analysis endpoint"""
        # Skip if the test image doesn't exist
        if not os.path.exists(self.test_image_path):
            pytest.skip("Test image not available")
        
        url = reverse('models:analyze_image')
        data = {
            'image_path': self.test_image_path,
            'analysis_type': 'general'
        }
        
        response = self.client.post(
            url,
            json.dumps(data),
            content_type='application/json'
        )
        
        # Validate response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'success')
        self.assertIn('analysis', response_data)
        self.assertIn('type', response_data)
        self.assertIn('image', response_data) 