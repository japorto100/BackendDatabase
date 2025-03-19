"""
Test script for LightweightVisionService

This script demonstrates how to use the LightweightVisionService with different models.
"""

import os
import sys
import logging
from pathlib import Path
import argparse
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to run this as a standalone script
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models_app.ai_models.vision.lightweight.service import LightweightVisionService
from models_app.ai_models.vision.lightweight.model_manager import LightweightVisionModelManager
from models_app.ai_models.utils.common.errors import ModelUnavailableError, VisionModelError


def test_clip_model(image_path: str):
    """Test the CLIP model for classification and similarity."""
    logger.info("Testing CLIP model...")
    
    # Configure for CLIP
    config = {
        'model_type': 'clip',
        'variant': 'base',
        'device': 'auto'
    }
    
    # Create service
    service = LightweightVisionService(config)
    try:
        service.initialize()
        
        # Print model info
        logger.info(f"Model info: {service.get_model_info()}")
        
        # Test classification with dedicated method
        candidate_labels = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a landscape",
            "a photo of food",
            "a photo of a building"
        ]
        
        result = service.classify_image(image_path, candidate_labels)
        logger.info("Classification results:")
        for label, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {label}: {score:.4f}")
        
        # Test classification via process_image
        classification_prompt = "Classify this image as: cat, dog, landscape, food, building"
        class_text, confidence = service.process_image(image_path, classification_prompt)
        logger.info(f"Classification via process_image (confidence: {confidence:.2f}):")
        logger.info(class_text)
        
        # Test text similarity
        descriptions = [
            "An animal relaxing",
            "A natural landscape",
            "A busy city street",
            "A delicious meal"
        ]
        
        result = service.compute_text_similarity(image_path, descriptions)
        logger.info("Text similarity results:")
        for text, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {text}: {score:.4f}")
        
        # Test embedding
        embedding = service.get_image_embedding(image_path)
        logger.info(f"Image embedding shape: {embedding.shape}")
        
        return service
    
    except (ModelUnavailableError, VisionModelError) as e:
        logger.error(f"Error with CLIP model: {e}")
        return None


def test_blip_model(image_path: str):
    """Test the BLIP model for captioning and visual QA."""
    logger.info("Testing BLIP model...")
    
    # Configure for BLIP
    config = {
        'model_type': 'blip',
        'variant': 'base',
        'device': 'auto',
        'temperature': 0.7,
        'max_length': 50
    }
    
    # Create service
    service = LightweightVisionService(config)
    try:
        service.initialize()
        
        # Print model info
        logger.info(f"Model info: {service.get_model_info()}")
        
        # Test captioning with dedicated method
        caption = service.generate_caption(image_path)
        logger.info(f"Generated caption: {caption}")
        
        # Test captioning via process_image
        caption_prompt = "Describe this image in detail:"
        caption_text, confidence = service.process_image(image_path, caption_prompt)
        logger.info(f"Caption via process_image (confidence: {confidence:.2f}):")
        logger.info(caption_text)
        
        # Test captioning with prompt
        prompted_caption = service.generate_caption(image_path, prompt="Describe this image in detail:")
        logger.info(f"Prompted caption: {prompted_caption}")
        
        # Test visual QA with dedicated method
        questions = [
            "What is the main subject of this image?",
            "What colors are visible in this image?",
            "Is there a person in this image?",
            "What time of day does this appear to be?"
        ]
        
        logger.info("Visual QA results with dedicated method:")
        for question in questions:
            answer = service.answer_question(image_path, question)
            logger.info(f"  Q: {question}")
            logger.info(f"  A: {answer}")
        
        # Test visual QA via process_image
        logger.info("Visual QA results via process_image:")
        for question in questions:
            answer_text, confidence = service.process_image(image_path, question)
            logger.info(f"  Q: {question}")
            logger.info(f"  A: {answer_text} (confidence: {confidence:.2f})")
        
        return service
    
    except (ModelUnavailableError, VisionModelError) as e:
        logger.error(f"Error with BLIP model: {e}")
        return None


def test_clip_phi_model(image_path: str):
    """Test the CLIP+Phi hybrid model."""
    logger.info("Testing CLIP+Phi hybrid model...")
    
    # Configure for CLIP+Phi
    config = {
        'model_type': 'clip_phi',
        'variant': 'base',
        'device': 'auto',
        'temperature': 0.8,
        'max_length': 100
    }
    
    # Create service
    service = LightweightVisionService(config)
    try:
        service.initialize()
        
        # Print model info
        logger.info(f"Model info: {service.get_model_info()}")
        
        # Test processing with different prompts
        prompts = [
            "Describe this image in detail.",
            "What do you see in this image?",
            "Can you identify the main elements in this image?",
            "What mood does this image convey?"
        ]
        
        for prompt in prompts:
            logger.info(f"Testing prompt: {prompt}")
            response_text, confidence = service.process_image(image_path, prompt)
            logger.info(f"Response (confidence: {confidence:.2f}):")
            logger.info(response_text)
        
        # Test text generation (only supported by clip_phi)
        text_prompts = [
            "Write a short poem about nature.",
            "Explain how neural networks work in simple terms."
        ]
        
        logger.info("Testing text-only generation:")
        for prompt in text_prompts:
            try:
                text, confidence = service.generate_text(prompt)
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Generated text (confidence: {confidence:.2f}):")
                logger.info(text)
            except VisionModelError as e:
                logger.error(f"Error generating text: {e}")
        
        return service
    
    except (ModelUnavailableError, VisionModelError) as e:
        logger.error(f"Error with CLIP+Phi model: {e}")
        return None


def test_all_available_models(image_path: str):
    """Test all available models that can be loaded on this system."""
    logger.info("Testing all available models...")
    
    # Get list of available models
    model_manager = LightweightVisionModelManager()
    available_models = model_manager.list_available_models()
    
    logger.info(f"Found {len(available_models)} possible model types:")
    for model in available_models:
        logger.info(f"  - {model['type']}: {model['description']}")
    
    # Test each model type
    working_models = []
    
    for model_info in available_models:
        model_type = model_info['type']
        logger.info(f"\nTesting model type: {model_type}")
        
        # Configure for this model type
        config = {
            'model_type': model_type,
            'variant': model_info['variants'][0] if model_info['variants'] else 'base',
            'device': 'auto',
            'enable_metrics': True  # Enable metrics collection
        }
        
        # Create service
        service = LightweightVisionService(config)
        
        # Check if it's available on this system
        try:
            # Try to initialize it
            service.initialize()
            logger.info(f"Successfully initialized {model_type}")
                
            # Display supported tasks
            tasks = [task for task, supported in service._supported_tasks.items() if supported]
            logger.info(f"Supported tasks: {', '.join(tasks)}")
                
            # Try one of the supported tasks
            if 'image_classification' in tasks:
                logger.info(f"Testing image classification with {model_type}")
                result, confidence = service.process_image(
                    image_path, 
                    "Classify this image as: person, animal, landscape, food, building"
                )
                logger.info(f"Classification result (confidence: {confidence:.2f}):")
                logger.info(result)
            elif 'image_captioning' in tasks:
                logger.info(f"Testing image captioning with {model_type}")
                caption, confidence = service.process_image(
                    image_path, 
                    "Describe this image"
                )
                logger.info(f"Caption (confidence: {confidence:.2f}):")
                logger.info(caption)
            elif 'image_embedding' in tasks:
                logger.info(f"Testing image embedding with {model_type}")
                embedding = service.get_image_embedding(image_path)
                logger.info(f"Embedding shape: {embedding.shape}")
                
            working_models.append(model_type)
            
        except (ModelUnavailableError, VisionModelError) as e:
            logger.error(f"Error with {model_type}: {e}")
    
    logger.info(f"\nWorking models: {', '.join(working_models)}")
    return working_models


def main():
    """Main function for testing the lightweight vision service."""
    parser = argparse.ArgumentParser(description='Test lightweight vision services')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['clip', 'blip', 'clip_phi', 'all', 'available'],
                        help='Model to test')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return
    
    # Verify it's a valid image
    try:
        img = Image.open(args.image)
        img.verify()
        logger.info(f"Testing with image: {args.image} ({img.format}, {img.size})")
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        return
    
    # Run tests based on selected model
    if args.model == 'clip':
        test_clip_model(args.image)
    elif args.model == 'blip':
        test_blip_model(args.image)
    elif args.model == 'clip_phi':
        test_clip_phi_model(args.image)
    elif args.model == 'available':
        test_all_available_models(args.image)
    else:  # 'all'
        logger.info("Testing all models...")
        clip_service = test_clip_model(args.image)
        blip_service = test_blip_model(args.image)
        clip_phi_service = test_clip_phi_model(args.image)
        
        logger.info("All tests completed!")


if __name__ == "__main__":
    main() 