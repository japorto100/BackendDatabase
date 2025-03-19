"""
Multimodal Responder - Core class for handling multimodal requests
that involve both text and images.
"""

import logging
import os
from typing import List, Tuple, Dict, Any, Optional

from models_app.ai_models.vision import VisionProviderFactory
from models_app.ai_models.vision.gpt4v.service import GPT4VisionService
from models_app.ai_models.vision.gemini.service import GeminiVisionService
from models_app.ai_models.vision.qwen.service import QwenVisionService
from models_app.ai_models.vision.lightweight.service import LightweightVisionService

logger = logging.getLogger(__name__)

class MultimodalResponder:
    """
    Main class for handling multimodal (text + image) interactions, leveraging
    vision-capable language models to generate responses based on image content.
    """
    
    def __init__(self):
        """Initialize the MultimodalResponder"""
        self.vision_factory = VisionProviderFactory()
    
    def generate_response(self, images: List[str], text_prompt: str, **kwargs) -> Tuple[str, List[str]]:
        """
        Generate a response based on text and one or more images.
        
        Args:
            images: List of image file paths
            text_prompt: Text prompt/question about the images
            **kwargs: Additional parameters including model_choice
            
        Returns:
            Tuple of (response_text, used_images)
        """
        try:
            # Get model choice from kwargs, defaulting to 'qwen'
            model_choice = kwargs.get('model_choice', 'qwen')
            
            # Validate images (check they exist)
            valid_images = []
            for img_path in images:
                if os.path.exists(img_path):
                    valid_images.append(img_path)
                else:
                    logger.warning(f"Image not found: {img_path}")
            
            if not valid_images:
                return "No valid images provided.", []
            
            # Create vision provider configuration
            config = {
                'provider_type': 'vision',
                'vision_type': model_choice,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1024)
            }
            
            # For cloud providers, add API key if specified
            if model_choice in ['gemini', 'gpt4v']:
                api_key = kwargs.get('api_key')
                if api_key:
                    config['api_key'] = api_key
            
            # Create the appropriate vision service based on model_choice
            if model_choice == 'gpt4v':
                service = GPT4VisionService(config)
            elif model_choice == 'gemini':
                service = GeminiVisionService(config)
            elif model_choice == 'qwen':
                service = QwenVisionService(config)
            elif model_choice == 'lightweight':
                service = LightweightVisionService(config)
            else:
                # Use factory for unknown model types
                service = self.vision_factory.create_provider(config)
            
            # Process multiple images if more than one valid image
            if len(valid_images) > 1:
                # Check if the service supports multiple images
                if hasattr(service, 'process_multiple_images') and callable(getattr(service, 'process_multiple_images')):
                    response_text, confidence = service.process_multiple_images(valid_images, text_prompt)
                else:
                    # Fall back to processing the first image only
                    logger.warning(f"Service {model_choice} doesn't support multiple images. Using only the first image.")
                    response_text, confidence = service.process_image(valid_images[0], text_prompt)
            else:
                # Process single image
                response_text, confidence = service.process_image(valid_images[0], text_prompt)
            
            return response_text, valid_images
            
        except Exception as e:
            logger.error(f"Error generating multimodal response: {str(e)}")
            return f"Error generating response: {str(e)}", []
    
    def generate_responses_with_different_models(self, images: List[str], text_prompt: str, 
                                                models: List[str] = None) -> Dict[str, str]:
        """
        Generate responses using multiple different models for comparison.
        
        Args:
            images: List of image file paths
            text_prompt: Text prompt/question about the images
            models: List of model choices to use (defaults to ['qwen', 'gemini', 'gpt4v', 'lightweight'])
            
        Returns:
            Dict mapping model names to their responses
        """
        if models is None:
            models = ['qwen', 'gemini', 'gpt4v', 'lightweight']
        
        results = {}
        
        for model in models:
            try:
                response, _ = self.generate_response(images, text_prompt, model_choice=model)
                results[model] = response
            except Exception as e:
                logger.error(f"Error with {model}: {str(e)}")
                results[model] = f"Error: {str(e)}"
        
        return results
    
    def analyze_image_content(self, image_path: str, analysis_type: str = 'general') -> Dict[str, Any]:
        """
        Perform a specific type of image analysis.
        
        Args:
            image_path: Path to the image
            analysis_type: Type of analysis to perform - 'general', 'document', 'chart', 'object', etc.
            
        Returns:
            Dict with analysis results
        """
        # Determine the best model and prompt for this analysis type
        model_mapping = {
            'general': 'lightweight',
            'document': 'qwen',
            'chart': 'gemini',
            'object': 'gpt4v',
            'text': 'qwen',
            'detailed': 'gpt4v'
        }
        
        prompt_mapping = {
            'general': "Describe what you see in this image.",
            'document': "Extract and structure all important information from this document.",
            'chart': "Analyze this chart/graph and explain what it shows. Include key data points and trends.",
            'object': "Identify all objects in this image and provide details about them.",
            'text': "Extract all text from this image.",
            'detailed': "Provide a very detailed analysis of everything in this image."
        }
        
        # Get the appropriate model and prompt
        model = model_mapping.get(analysis_type, 'lightweight')
        prompt = prompt_mapping.get(analysis_type, prompt_mapping['general'])
        
        # Generate the response
        response, _ = self.generate_response([image_path], prompt, model_choice=model)
        
        return {
            'analysis': response,
            'type': analysis_type,
            'model_used': model
        }
    
    def process_image(self, image_url: str, prompt: str, model_id: str = "default_vision_model") -> str:
        """
        Process an image and generate a response.
        This method is used by the views.process_image endpoint.
        
        Args:
            image_url: URL or path to the image
            prompt: The prompt/question about the image
            model_id: ID of the model to use
            
        Returns:
            Generated response text
        """
        try:
            # Determine model_choice from model_id
            if "gpt" in model_id.lower():
                model_choice = "gpt4v"
            elif "qwen" in model_id.lower():
                model_choice = "qwen"
            elif "gemini" in model_id.lower():
                model_choice = "gemini"
            else:
                model_choice = "lightweight"
            
            # Generate response
            response, _ = self.generate_response([image_url], prompt, model_choice=model_choice)
            return response
            
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            return f"Error processing image: {str(e)}"
    
    def process_document(self, document_path: str, prompt: str, model_id: str = "default_vision_model") -> str:
        """
        Process a document and generate a response.
        This method is used by the views.analyze_document endpoint.
        
        Args:
            document_path: Path to the document
            prompt: The prompt/question about the document
            model_id: ID of the model to use
            
        Returns:
            Generated response text
        """
        try:
            # Check if the document exists
            if not os.path.exists(document_path):
                return f"Document not found: {document_path}"
            
            # Use DocumentVisionAdapter for document processing
            from models_app.ai_models.vision import DocumentVisionAdapter
            
            # Determine the best vision service based on model_id
            if "gpt" in model_id.lower():
                config = {'model': model_id, 'vision_type': 'gpt4v'}
            elif "qwen" in model_id.lower():
                config = {'model': model_id, 'vision_type': 'qwen'}
            elif "gemini" in model_id.lower():
                config = {'model': model_id, 'vision_type': 'gemini'}
            else:
                # Default to a lightweight model
                config = {'model': model_id, 'vision_type': 'lightweight'}
            
            # Create DocumentVisionAdapter with the appropriate config
            document_adapter = DocumentVisionAdapter(config)
            
            # Process the document
            result = document_adapter.process_document(document_path, prompt)
            
            return result.get('text', f"Error: No text response generated from {document_path}")
            
        except Exception as e:
            logger.error(f"Error in process_document: {str(e)}")
            return f"Error processing document: {str(e)}"
