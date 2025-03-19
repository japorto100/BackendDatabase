# Lightweight Vision Models

This module provides lightweight vision AI models that can run efficiently on consumer hardware, including CPUs and modest GPUs. The implementation follows a manager/service pattern, with a clean separation of concerns.

## Components

- **LightweightVisionModelManager**: Manages model configurations, hardware detection, and model initialization.
- **LightweightVisionService**: Provides a high-level API for vision tasks (classification, captioning, VQA, etc.)

## Supported Models

The module supports various lightweight vision models:

- **CLIP**: For zero-shot image classification and text-image similarity
- **BLIP**: For image captioning and visual question answering
- **PaliGemma**: For multimodal tasks with good performance
- **CLIP+Phi**: A hybrid approach using CLIP for image understanding and Phi for text generation

## Hardware Requirements

Each model has different hardware requirements:

| Model | Min RAM | Min GPU Memory | Preferred Device |
|-------|---------|---------------|------------------|
| CLIP  | 2GB     | 2GB           | CUDA             |
| BLIP  | 4GB     | 4GB           | CUDA             |
| PaliGemma | 8GB | 8GB           | CUDA             |
| CLIP+Phi | 2GB  | 0GB           | CPU              |

The module will automatically select the most appropriate device based on hardware availability.

## Supported Tasks

Different models support different vision tasks:

- **Image Classification**: CLIP, CLIP+Phi
- **Image Captioning**: BLIP, PaliGemma, CLIP+Phi
- **Visual Question Answering**: BLIP, PaliGemma, CLIP+Phi
- **Image Embedding**: All models
- **Text-Image Similarity**: CLIP, PaliGemma

## Usage Examples

### Basic Usage

```python
from models_app.ai_models.vision import LightweightVisionService

# Configure and create service
config = {
    'model_type': 'clip',    # One of: 'clip', 'blip', 'paligemma', 'clip_phi'
    'variant': 'base',       # Model variant
    'device': 'auto',        # 'auto', 'cuda', 'cpu', or 'mps'
    'temperature': 0.7,      # For text generation
    'max_length': 50         # For text generation
}

# Create service
vision_service = LightweightVisionService(config)

# Initialize (loads model)
vision_service.initialize()

# Process an image (auto-selects appropriate task)
result = vision_service.process_image(
    image_input="path/to/image.jpg",  # Can be path, URL, base64, or PIL Image
    task='auto',                      # 'auto', 'classify', 'caption', 'vqa', 'embed', 'similarity'
    prompt="Describe this image",     # Optional prompt for captioning or VQA
    candidate_labels=["cat", "dog"]   # Optional labels for classification or similarity
)

print(result)
```

### Specific Tasks

```python
# Image classification
classifications = vision_service.classify_image(
    "path/to/image.jpg",
    candidate_labels=["cat", "dog", "bird", "landscape"]
)

# Image captioning
caption = vision_service.generate_caption(
    "path/to/image.jpg",
    prompt="Describe what you see in detail:"
)

# Visual question answering
answer = vision_service.answer_question(
    "path/to/image.jpg",
    question="What is the main subject of this image?"
)

# Get image embedding
embedding = vision_service.get_image_embedding("path/to/image.jpg")

# Text-image similarity
similarities = vision_service.compute_text_similarity(
    "path/to/image.jpg",
    texts=["a cat sleeping", "a dog running", "a mountain landscape"]
)
```

## Testing

The module includes a test script (`test_service.py`) that demonstrates how to use the various models and tasks.

Run it with:

```bash
python -m models_app.ai_models.vision.lightweight.test_service --image path/to/test_image.jpg --model clip
```

Options for the `--model` parameter:
- `clip`: Test CLIP models
- `blip`: Test BLIP models
- `clip_phi`: Test CLIP+Phi hybrid models
- `available`: Test all models available on the current hardware
- `all`: Test all model types (default)

## Integration with Django Views

To use these vision models in a Django view:

```python
from models_app.ai_models.vision import LightweightVisionService
from django.http import JsonResponse

def process_image_view(request):
    # Get image from request
    image_file = request.FILES.get('image')
    if not image_file:
        return JsonResponse({'error': 'No image provided'}, status=400)
    
    # Get parameters from request
    task = request.POST.get('task', 'auto')
    prompt = request.POST.get('prompt')
    model_type = request.POST.get('model_type', 'clip')
    
    # Configure service
    config = {
        'model_type': model_type,
        'device': 'auto'
    }
    
    # Create and initialize service
    vision_service = LightweightVisionService(config)
    if not vision_service.initialize():
        return JsonResponse({'error': 'Failed to initialize vision service'}, status=500)
    
    # Process image
    result = vision_service.process_image(
        image_input=image_file,
        task=task,
        prompt=prompt
    )
    
    # Return result
    return JsonResponse(result)
```

## Error Handling

The service includes comprehensive error handling for:
- Invalid image inputs
- Unsupported models/tasks
- Hardware limitations
- Model initialization failures

Errors are logged and appropriate exceptions are raised with descriptive messages.

## Performance Considerations

- Models are loaded on first use and then cached
- The system selects appropriate quantization levels based on hardware
- Device selection is automatic but can be overridden
- RAM and GPU memory requirements are checked before loading models 