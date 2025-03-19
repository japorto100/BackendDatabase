# AI Model Management System

This document provides an overview of the AI model management system in LocalGPT Vision. It explains how to use different models, configure custom models, and extend the system with new model providers.

## Table of Contents

1. [Available Models](#available-models)
2. [Using the Model Selector](#using-the-model-selector)
3. [Custom Models](#custom-models)
4. [HyDE (Hypothetical Document Embeddings)](#hyde)
5. [Model Providers](#model-providers)
6. [Adding New Providers](#adding-new-providers)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)

## Available Models

LocalGPT Vision supports multiple AI models from various providers:

### OpenAI Models
- **GPT-3.5 Turbo**: Fast and cost-effective for most tasks
- **GPT-4**: More powerful reasoning and instruction following
- **GPT-4 Vision**: Supports image analysis and understanding

### Anthropic Models
- **Claude 3 Haiku**: Fast, efficient model for straightforward tasks
- **Claude 3 Sonnet**: Balanced performance and capabilities
- **Claude 3 Opus**: Most powerful Claude model with advanced reasoning

### DeepSeek Models
- **DeepSeek Coder**: Specialized for code understanding and generation
- **DeepSeek Chat**: General purpose chat model

### Local Models
- **Mistral 7B**: Efficient open-source model that can run locally
- **Llama 2**: Meta's open-source large language model
- **Other Hugging Face models**: Support for various open-source models

## Using the Model Selector

The Model Selector component allows you to choose which AI model to use for your interactions:

1. Click the "Model Settings" button in the chat interface
2. Select a model from the dropdown menu
3. Adjust advanced settings if needed:
   - **Temperature**: Controls randomness (0.0-2.0)
   - **Max Tokens**: Controls maximum response length
4. Click "Save as Default" to remember your preferences

Different models have different capabilities and performance characteristics. Vision-capable models are marked with a "Vision" badge and can analyze images.

## Custom Models

You can use custom models by enabling the "Use custom model" checkbox and providing a model URL or path:

### Supported Sources:
- **Hugging Face models**: `https://huggingface.co/mistralai/Mistral-7B-v0.1`
- **GitHub repositories**: `https://github.com/facebookresearch/llama`
- **Local paths**: `/path/to/your/model` or `C:\models\your-model`

Custom models will be loaded according to their source and made available for use. Note that:
- Local models require appropriate hardware (GPU recommended)
- Some models may have specific requirements or limitations
- Loading large models may take time and resources

## HyDE

HyDE (Hypothetical Document Embeddings) is a technique that improves search results by generating a hypothetical answer to your query before searching. This helps find more relevant information.

You can configure which model is used for HyDE separately from your chat model:

1. Go to User Settings
2. In the "Search" section, select a model for HyDE
3. Toggle "Use HyDE for search" on or off

For HyDE, smaller and faster models often work well, as they only need to generate a plausible hypothesis, not a perfect answer.

## Model Providers

The system uses a provider-based architecture to support different AI model services:

### OpenAILLMProvider
Connects to OpenAI's API services for GPT models.

### AnthropicLLMProvider
Connects to Anthropic's API services for Claude models.

### DeepSeekLLMProvider
Connects to DeepSeek's API services for DeepSeek models.

### LocalLLMProvider
Runs models locally using Hugging Face Transformers.

### TemplateProviderFactory
Creates appropriate providers based on model URLs or paths.

## Adding New Providers

Developers can extend the system with new model providers:

1. Create a new provider class that implements the LLMProvider interface
2. Add the provider to the ModelProviderManager
3. Update the available models list

Example of a minimal provider implementation:

```python
from models_app.llm_providers.base_provider import BaseLLMProvider

class MyCustomProvider(BaseLLMProvider):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "default-model")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
        # Initialize your client
        self.client = YourAPIClient(api_key=self.api_key)
    
    def generate_text(self, prompt):
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.text, self._calculate_confidence(response)
        except Exception as e:
            return f"Error generating text: {str(e)}", 0.0
    
    def generate_batch(self, prompts):
        results = []
        for prompt in prompts:
            text, confidence = self.generate_text(prompt)
            results.append((text, confidence))
        return results
```

## Performance Considerations

Different models have different performance characteristics:

- **Cloud-based models** (OpenAI, Anthropic, DeepSeek):
  - Require API keys and internet connection
  - Incur usage costs
  - Generally offer higher quality responses
  - No local hardware requirements

- **Local models**:
  - Run on your own hardware
  - No API costs or internet dependency
  - Require significant RAM and preferably a GPU
  - May have lower quality than cloud models
  - Provide better privacy as data stays local

For optimal performance:
- Use smaller models for HyDE and simpler tasks
- Use vision models only when image analysis is needed
- Consider quantized models for local deployment
- Enable caching for repeated queries

## Troubleshooting

Common issues and solutions:

### API Connection Issues
- Verify API keys are correctly set in your environment variables or settings
- Check internet connection
- Ensure API service is available

### Local Model Issues
- Verify you have sufficient RAM (16GB+ recommended)
- For GPU acceleration, ensure CUDA is properly installed
- Try a smaller or quantized model if you experience out-of-memory errors

### Custom Model Loading Failures
- Verify the model URL or path is correct
- Ensure you have necessary permissions to access the model
- Check that the model format is supported

### Slow Responses
- Local models: Consider using a smaller or quantized model
- Cloud models: Check your internet connection
- Enable response caching for frequently asked questions

For additional help, check the application logs or contact support. 