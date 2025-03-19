"""
Tests für den HyDEProcessor.

python manage.py test models_app.tests.test_hyde_processor

Diese Tests überprüfen die Funktionalität des HyDEProcessors, einschließlich:
- Hypothesengenerierung
- Kombination mit der ursprünglichen Anfrage
- Verbesserte Suche
- Caching
- Evidenz-Tracking
"""

import unittest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.core.cache import cache
from django.contrib.auth.models import User

from models_app.hyde_processor import HyDEProcessor, SimpleLLMProvider
from users_app.model_provider import ModelProviderManager


class TestSimpleLLMProvider(TestCase):
    """Tests für den SimpleLLMProvider."""
    
    def setUp(self):
        """Testumgebung einrichten."""
        self.provider = SimpleLLMProvider()
    
    def test_generate_text_how_question(self):
        """Test der Textgenerierung für 'Wie'-Fragen."""
        prompt = "Generate a detailed answer to this question: How do I bake a cake?"
        text = self.provider.generate_text(prompt)[0]
        self.assertIn("process to bake a cake", text)
        self.assertIn("systematic approach", text)
    
    def test_generate_text_what_question(self):
        """Test der Textgenerierung für 'Was'-Fragen."""
        prompt = "Generate a detailed answer to this question: What is machine learning?"
        text = self.provider.generate_text(prompt)[0]
        self.assertIn("Regarding machine learning", text)
        self.assertIn("multiple perspectives", text)
    
    def test_generate_text_why_question(self):
        """Test der Textgenerierung für 'Warum'-Fragen."""
        prompt = "Generate a detailed answer to this question: Why is the sky blue?"
        text = self.provider.generate_text(prompt)[0]
        self.assertIn("reasons for the sky blue", text)
        self.assertIn("multifaceted", text)
    
    def test_generate_text_other_question(self):
        """Test der Textgenerierung für andere Fragen."""
        prompt = "Generate a detailed answer to this question: Explain quantum computing."
        text = self.provider.generate_text(prompt)[0]
        self.assertIn("exploring Explain quantum computing", text)
        self.assertIn("deeper understanding", text)
    
    def test_max_tokens_limit(self):
        """Test der Begrenzung der maximalen Tokens."""
        prompt = "Generate a detailed answer to this question: How does photosynthesis work?"
        text = self.provider.generate_text(prompt, max_tokens=10)[0]
        # Ungefähr 4 Zeichen pro Token, also maximal 40 Zeichen
        self.assertLessEqual(len(text), 50)


class TestHyDEProcessor(TestCase):
    """Test cases for HyDEProcessor"""

    def setUp(self):
        """Set up test environment"""
        # Create a test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        
        # Create user settings
        from users_app.models import UserSettings
        self.user_settings = UserSettings.objects.create(
            user=self.user,
            generation_model='gpt-3.5-turbo',
            hyde_model='deepseek-coder-1.3b-instruct',
            use_hyde=True
        )
        
        # Create a mock provider
        self.mock_provider = MagicMock()
        self.mock_provider.generate_text.return_value = ("This is a hypothesis", 0.8)
        
        # Create a mock provider manager
        self.mock_provider_manager = MagicMock()
        self.mock_provider_manager.get_hyde_provider.return_value = self.mock_provider
        
        # Create the processor with mocked dependencies
        self.processor = HyDEProcessor(
            user=self.user,
            provider_manager=self.mock_provider_manager
        )
        
        # Clear cache before tests
        cache.clear()

    def test_initialization(self):
        """Test processor initialization"""
        # Verify the processor was initialized with the user
        self.assertEqual(self.processor.user, self.user)
        
        # Verify the provider manager was set
        self.assertEqual(self.processor.provider_manager, self.mock_provider_manager)
        
        # Verify the LLM provider was obtained from the manager
        self.mock_provider_manager.get_hyde_provider.assert_called_once()
        self.assertEqual(self.processor.llm_provider, self.mock_provider)

    def test_generate_hypothesis(self):
        """Test hypothesis generation"""
        # Generate a hypothesis
        query = "What is the capital of France?"
        hypothesis, confidence = self.processor.generate_hypothesis(query)
        
        # Verify the hypothesis was generated
        self.assertEqual(hypothesis, "This is a hypothesis")
        self.assertEqual(confidence, 0.8)
        
        # Verify the provider was called with the correct prompt
        self.mock_provider.generate_text.assert_called_once()
        call_args = self.mock_provider.generate_text.call_args[0][0]
        self.assertIn(query, call_args)

    def test_hypothesis_caching(self):
        """Test hypothesis caching"""
        # Generate a hypothesis
        query = "What is the capital of Germany?"
        hypothesis1, confidence1 = self.processor.generate_hypothesis(query)
        
        # Verify the provider was called
        self.mock_provider.generate_text.assert_called_once()
        
        # Reset the mock
        self.mock_provider.generate_text.reset_mock()
        
        # Generate the same hypothesis again
        hypothesis2, confidence2 = self.processor.generate_hypothesis(query)
        
        # Verify the provider was not called again (cache hit)
        self.mock_provider.generate_text.assert_not_called()
        
        # Verify the cached result was returned
        self.assertEqual(hypothesis1, hypothesis2)
        self.assertEqual(confidence1, confidence2)

    def test_combine_with_original(self):
        """Test combining hypothesis with original query"""
        # Original query and hypothesis
        query = "Who was Napoleon?"
        hypothesis = "Napoleon Bonaparte was a French military leader and emperor who conquered much of Europe in the early 19th century."
        
        # Combine with original
        enhanced_query = self.processor.combine_with_original(query, hypothesis)
        
        # Verify the enhanced query contains both the original and the hypothesis
        self.assertIn(query, enhanced_query)
        self.assertIn(hypothesis, enhanced_query)

    def test_enhance_query(self):
        """Test query enhancement"""
        # Set up the mock provider to return a hypothesis
        self.mock_provider.generate_text.return_value = (
            "Paris is the capital and largest city of France, situated on the Seine River.",
            0.9
        )
        
        # Enhance a query
        query = "What is the capital of France?"
        enhanced_query = self.processor.enhance_query(query)
        
        # Verify the query was enhanced
        self.assertIn(query, enhanced_query)
        self.assertIn("Paris", enhanced_query)
        
        # Verify the provider was called
        self.mock_provider.generate_text.assert_called_once()

    def test_evidence_storage(self):
        """Test evidence storage"""
        # Generate a hypothesis and store evidence
        query = "What is the capital of Italy?"
        hypothesis, confidence = self.processor.generate_hypothesis(query)
        
        # Get the evidence
        evidence = self.processor.get_evidence(query)
        
        # Verify evidence was stored
        self.assertIsNotNone(evidence)
        self.assertGreater(len(evidence), 0)
        
        # Verify evidence contains the hypothesis
        self.assertEqual(evidence[0]["content"], "This is a hypothesis")
        self.assertEqual(evidence[0]["highlights"][0]["confidence"], 0.8)

    def test_search_with_hyde(self):
        """Test search with HyDE"""
        # Mock the search method
        with patch.object(self.processor, '_perform_search') as mock_search:
            mock_search.return_value = [
                {"title": "Result 1", "snippet": "This is result 1"},
                {"title": "Result 2", "snippet": "This is result 2"}
            ]
            
            # Perform a search with HyDE
            query = "What is the capital of Spain?"
            results = self.processor.search(query, "web", limit=2)
            
            # Verify the search was performed
            mock_search.assert_called_once()
            
            # Verify the results were returned
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["title"], "Result 1")
            self.assertEqual(results[1]["title"], "Result 2")

    def test_search_without_hyde(self):
        """Test search without HyDE"""
        # Disable HyDE
        self.user_settings.use_hyde = False
        self.user_settings.save()
        
        # Create a new processor with updated settings
        processor = HyDEProcessor(
            user=self.user,
            provider_manager=self.mock_provider_manager
        )
        
        # Mock the regular search method
        with patch.object(processor, '_perform_regular_search') as mock_regular_search:
            mock_regular_search.return_value = [
                {"title": "Regular Result 1", "snippet": "This is regular result 1"},
                {"title": "Regular Result 2", "snippet": "This is regular result 2"}
            ]
            
            # Perform a search without HyDE
            query = "What is the capital of Spain?"
            results = processor.search(query, "web", limit=2)
            
            # Verify the regular search was performed
            mock_regular_search.assert_called_once()
            
            # Verify the results were returned
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["title"], "Regular Result 1")
            self.assertEqual(results[1]["title"], "Regular Result 2")


if __name__ == '__main__':
    unittest.main() 