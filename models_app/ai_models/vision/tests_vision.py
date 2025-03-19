"""
Test suite for vision providers and image processing.

This module tests the functionality, error handling, and edge cases of vision providers.
"""

import unittest
import os
import time
import logging
import tempfile
from PIL import Image
import io
import base64
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import vision modules to test
from models_app.ai_models.vision.vision_factory import VisionProviderFactory
from models_app.ai_models.utils.vision.image_processing import (
    decode_base64_image, 
    encode_image_to_base64, 
    resize_image_for_model,
    validate_image,
    fix_corrupt_image,
    support_multiple_images,
    handle_high_resolution_image,
    clear_image_cache
)
from models_app.ai_models.utils.common.errors import (
    VisionModelError, 
    ImageProcessingError,
    ModelUnavailableError
)

# Configure logging
logging.basicConfig(level=logging.ERROR)

class ImageProcessingTests(unittest.TestCase):
    """Test image processing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a small test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Save test image to a file
        self.test_image_path = os.path.join(self.temp_dir.name, 'test_image.jpg')
        self.test_image.save(self.test_image_path)
        
        # Create a base64 representation
        buffer = io.BytesIO()
        self.test_image.save(buffer, format='JPEG')
        self.base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.base64_image_with_header = f"data:image/jpeg;base64,{self.base64_image}"
        
        # Clear image cache
        clear_image_cache()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_encode_decode_base64(self):
        """Test encoding and decoding base64 images."""
        # Test encode
        encoded = encode_image_to_base64(self.test_image)
        self.assertIsInstance(encoded, str)
        
        # Test decode
        decoded = decode_base64_image(encoded)
        self.assertIsInstance(decoded, Image.Image)
        
        # Test with header
        decoded_with_header = decode_base64_image(self.base64_image_with_header)
        self.assertIsInstance(decoded_with_header, Image.Image)
        
        # Test with whitespace
        encoded_with_whitespace = f"{encoded[:10]} \n{encoded[10:]}"
        decoded_with_whitespace = decode_base64_image(encoded_with_whitespace)
        self.assertIsInstance(decoded_with_whitespace, Image.Image)
    
    def test_resize_image(self):
        """Test image resizing."""
        # Test standard resize
        resized = resize_image_for_model(self.test_image, (64, 64))
        self.assertEqual(resized.size, (64, 64))
        
        # Test resize with different aspect ratio
        tall_image = Image.new('RGB', (50, 200), color='blue')
        resized_tall = resize_image_for_model(tall_image, (100, 100))
        self.assertEqual(resized_tall.size, (100, 100))
        
        # Test very small image
        tiny_image = Image.new('RGB', (10, 10), color='green')
        resized_tiny = resize_image_for_model(tiny_image, (224, 224))
        self.assertEqual(resized_tiny.size, (224, 224))
    
    def test_validate_image(self):
        """Test image validation."""
        # Valid image
        is_valid, error, processed = validate_image(self.test_image)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        self.assertIsInstance(processed, Image.Image)
        
        # Valid image path
        is_valid, error, processed = validate_image(self.test_image_path)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Valid base64
        is_valid, error, processed = validate_image(self.base64_image_with_header)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Invalid image - too small
        tiny_image = Image.new('RGB', (5, 5), color='black')
        is_valid, error, processed = validate_image(tiny_image)
        self.assertFalse(is_valid)
        self.assertIn("too small", error)
        
        # Invalid image - solid color
        solid_image = Image.new('RGB', (100, 100), color='white')
        is_valid, error, processed = validate_image(solid_image)
        self.assertFalse(is_valid)
        self.assertIn("too few colors", error)
        
        # Non-existent file
        is_valid, error, processed = validate_image("/path/does/not/exist.jpg")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIsNone(processed)
    
    def test_fix_corrupt_image(self):
        """Test corrupt image fixing."""
        # Create a slightly corrupted image (still valid)
        buffer = io.BytesIO()
        self.test_image.save(buffer, format='JPEG', quality=10)
        buffer.seek(0)
        corrupt_data = bytearray(buffer.getvalue())
        
        # Corrupt some non-critical bytes
        corrupt_data[100] = 0
        corrupt_data[101] = 0
        
        # Test fixing bytes
        fixed = fix_corrupt_image(corrupt_data)
        self.assertIsInstance(fixed, Image.Image)
        
        # Test fixing PIL image
        fixed_pil = fix_corrupt_image(self.test_image)
        self.assertIsInstance(fixed_pil, Image.Image)
    
    def test_handle_high_resolution(self):
        """Test handling high resolution images."""
        # Create a high-res image
        high_res = Image.new('RGB', (1024, 1024), color='purple')
        
        # Test tiling
        tiles = handle_high_resolution_image(high_res, method="tile")
        self.assertIsInstance(tiles, list)
        self.assertTrue(len(tiles) > 1)
        
        # Test resizing
        resized = handle_high_resolution_image(high_res, method="resize")
        self.assertIsInstance(resized, Image.Image)
        self.assertEqual(resized.size, (224, 224))
        
        # Test very high resolution
        very_high_res = Image.new('RGB', (10000, 5000), color='yellow')
        result = handle_high_resolution_image(very_high_res)
        # Should force resize for extremely large images
        self.assertIsInstance(result, Image.Image)
    
    def test_multiple_images(self):
        """Test processing multiple images."""
        # Create second test image
        image2 = Image.new('RGB', (150, 150), color='blue')
        image2_path = os.path.join(self.temp_dir.name, 'test_image2.jpg')
        image2.save(image2_path)
        
        # Test list of PIL images
        images_list = [self.test_image, image2]
        processed = support_multiple_images(images_list)
        self.assertEqual(len(processed), 2)
        
        # Test list of mixed types
        mixed_list = [self.test_image_path, image2, self.base64_image_with_header]
        processed = support_multiple_images(mixed_list)
        self.assertEqual(len(processed), 3)
        
        # Test with too many images
        many_images = [self.test_image] * 10
        processed = support_multiple_images(many_images, max_images=5)
        self.assertEqual(len(processed), 5)
        
        # Test with one bad image
        mixed_with_bad = [self.test_image, "not_an_image", image2]
        processed = support_multiple_images(mixed_with_bad)
        self.assertEqual(len(processed), 2)
        
        # Test empty list
        with self.assertRaises(MultiImageProcessingError):
            support_multiple_images([])
        
        # Test all bad images
        with self.assertRaises(MultiImageProcessingError):
            support_multiple_images(["bad1", "bad2"])


class VisionFactoryTests(unittest.TestCase):
    """Test the vision provider factory."""
    
    def setUp(self):
        """Set up test environment."""
        self.factory = VisionProviderFactory()
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = self.factory.list_providers()
        self.assertIsInstance(providers, list)
        self.assertTrue(len(providers) > 0)
    
    def test_list_models(self):
        """Test listing available models for providers."""
        for provider in self.factory.list_providers():
            models = self.factory.list_models(provider)
            self.assertIsInstance(models, list)
    
    def test_find_similar_providers(self):
        """Test finding similar providers for typos."""
        # Test with slight typo
        similar = self.factory._find_similar_providers("gemni")
        self.assertTrue("gemini" in similar)
        
        # Test with another typo
        similar = self.factory._find_similar_providers("gptv")
        self.assertTrue("gpt4v" in similar)
    
    def test_find_fallback_model(self):
        """Test finding fallback models."""
        # Make some plausible fallbacks
        fallbacks = self.factory._find_fallback_model("gemini", "pro-vision")
        self.assertTrue(len(fallbacks) > 0)
    
    def test_provider_not_found(self):
        """Test behavior when provider not found."""
        provider = self.factory.get_provider("nonexistent", "model")
        self.assertIsNone(provider)


class MockVisionProviderTests(unittest.TestCase):
    """Test vision provider with mocks."""
    
    def setUp(self):
        """Set up test environment with mocks."""
        # Mock the factory to avoid actual provider initialization
        self.patcher = patch('models_app.ai_models.vision.vision_factory.VisionProviderFactory')
        self.mock_factory = self.patcher.start()
        
        # Create a mock provider
        self.mock_provider = MagicMock()
        self.mock_provider.process_image.return_value = ("This is a test image", 0.95)
        self.mock_provider.initialize.return_value = True
        
        # Make the factory return our mock
        factory_instance = self.mock_factory.return_value
        factory_instance.get_provider.return_value = self.mock_provider
        
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
    
    def tearDown(self):
        """Clean up mocks."""
        self.patcher.stop()
    
    def test_basic_functionality(self):
        """Test basic functionality with mock provider."""
        # Get a provider from factory
        factory = VisionProviderFactory()
        provider = factory.get_provider("mock", "test")
        
        # Test initialization
        self.assertTrue(provider.initialize())
        
        # Test processing image
        text, confidence = provider.process_image(self.test_image)
        self.assertEqual(text, "This is a test image")
        self.assertEqual(confidence, 0.95)
    
    def test_error_handling(self):
        """Test error handling with mock provider."""
        # Configure mock to raise an error
        self.mock_provider.process_image.side_effect = VisionModelError("Test error")
        
        # Get a provider from factory
        factory = VisionProviderFactory()
        provider = factory.get_provider("mock", "test")
        
        # Test error handling
        with self.assertRaises(VisionModelError):
            provider.process_image(self.test_image)


# Run tests if this file is executed directly
if __name__ == "__main__":
    unittest.main() 