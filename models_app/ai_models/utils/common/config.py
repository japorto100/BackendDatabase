"""
Configuration management system for AI models.

This module provides a comprehensive configuration system for AI models,
particularly for speech-to-text and text-to-speech services. It manages
configuration loading, validation, and persistence.

Design Rationale:
- Provides centralized configuration management instead of passing arguments
- Allows for configuration from files, environment variables, or code
- Enables validation and default values for configuration parameters
- Supports hierarchical configuration with inheritance
"""

import os
import logging
import json
import yaml
from typing import Dict, Any, List, Optional, Union, Type, ClassVar
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import threading

from django.conf import settings

logger = logging.getLogger(__name__)

@dataclass
class BaseAudioConfig:
    """
    Base configuration for audio services.
    
    This class defines common configuration parameters for all audio
    services, providing default values and validation.
    
    Attributes:
        cache_dir: Directory to cache models
        device: Device to use for inference
        use_gpu: Whether to use GPU for inference
        verbose: Whether to enable verbose logging
        enable_metrics: Whether to collect metrics
        metrics_export_on_cleanup: Whether to export metrics on cleanup
        metrics_reset_on_cleanup: Whether to reset metrics on cleanup
        timeout_seconds: Timeout for operations in seconds
    """
    
    # General configuration
    cache_dir: str = field(default_factory=lambda: os.path.join(settings.MEDIA_ROOT, 'model_cache'))
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    use_gpu: bool = True
    verbose: bool = False
    
    # Performance tuning
    batch_size: int = 1
    num_workers: int = 2
    use_half_precision: bool = True
    compute_type: str = "auto"  # "auto", "float32", "float16", "int8"
    
    # Caching options
    enable_cache: bool = True
    max_cache_size_mb: int = 1024 * 2  # 2 GB
    cache_ttl_seconds: int = 60 * 60 * 24  # 24 hours
    
    # Metrics collection
    enable_metrics: bool = True
    metrics_export_on_cleanup: bool = True
    metrics_reset_on_cleanup: bool = True
    
    # Error handling
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        
        # Ensure cache directory exists
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Resolve device if set to auto
        if self.device == "auto":
            self.device = self._resolve_device()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate cache directory
        if self.cache_dir and not isinstance(self.cache_dir, str):
            raise ValueError("cache_dir must be a string")
        
        # Validate device
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        
        # Validate performance parameters
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        
        # Validate cache parameters
        if self.max_cache_size_mb <= 0:
            raise ValueError("max_cache_size_mb must be positive")
        
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        
        # Validate error handling parameters
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
    
    def _resolve_device(self) -> str:
        """Resolve the device based on available hardware."""
        if not self.use_gpu:
            return "cpu"
            
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseAudioConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            BaseAudioConfig: Configuration object
        """
        # Filter out keys that are not in the dataclass
        valid_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'BaseAudioConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            BaseAudioConfig: Configuration object
        """
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {json_path}: {str(e)}")
            raise
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseAudioConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            BaseAudioConfig: Configuration object
        """
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {yaml_path}: {str(e)}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return asdict(self)
    
    def save_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to JSON file
        """
        try:
            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration to {json_path}: {str(e)}")
            raise
    
    def save_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to YAML file
        """
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving configuration to {yaml_path}: {str(e)}")
            raise
    
    def merge(self, other: 'BaseAudioConfig') -> 'BaseAudioConfig':
        """
        Merge with another configuration.
        
        This method creates a new configuration object by merging
        the current configuration with another configuration.
        Parameters from the other configuration take precedence.
        
        Args:
            other: Configuration to merge with
            
        Returns:
            BaseAudioConfig: Merged configuration
        """
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return type(self).from_dict(merged_dict)


@dataclass
class BaseConfig:
    """
    Base configuration for all AI models.
    
    This class defines common configuration parameters for all AI models,
    providing default values and validation.
    
    Attributes:
        cache_dir: Directory to cache models
        device: Device to use for inference
        use_gpu: Whether to use GPU for inference
        verbose: Whether to enable verbose logging
        enable_metrics: Whether to collect metrics
    """
    
    # General configuration
    cache_dir: str = field(default_factory=lambda: os.path.join(settings.MEDIA_ROOT, 'model_cache'))
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    use_gpu: bool = True
    verbose: bool = False
    
    # Performance tuning
    batch_size: int = 1
    num_workers: int = 2
    use_half_precision: bool = True
    compute_type: str = "auto"  # "auto", "float32", "float16", "int8"
    
    # Caching options
    enable_cache: bool = True
    max_cache_size_mb: int = 1024 * 2  # 2 GB
    cache_ttl_seconds: int = 60 * 60 * 24  # 24 hours
    
    # Metrics collection
    enable_metrics: bool = True
    metrics_export_on_cleanup: bool = True
    metrics_reset_on_cleanup: bool = True
    
    # Error handling
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        
        # Ensure cache directory exists
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Resolve device if set to auto
        if self.device == "auto":
            self.device = self._resolve_device()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate cache directory
        if self.cache_dir and not isinstance(self.cache_dir, str):
            raise ValueError("cache_dir must be a string")
        
        # Validate device
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        
        # Validate performance parameters
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        
        # Validate cache parameters
        if self.max_cache_size_mb <= 0:
            raise ValueError("max_cache_size_mb must be positive")
        
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        
        # Validate error handling parameters
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
    
    def _resolve_device(self) -> str:
        """Resolve the device based on available hardware."""
        if not self.use_gpu:
            return "cpu"
            
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            BaseConfig: Configuration object
        """
        # Filter out keys that are not in the dataclass
        valid_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'BaseConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            BaseConfig: Configuration object
        """
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {json_path}: {str(e)}")
            raise
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            BaseConfig: Configuration object
        """
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration from {yaml_path}: {str(e)}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return asdict(self)
    
    def save_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to JSON file
        """
        try:
            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration to {json_path}: {str(e)}")
            raise
    
    def save_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to YAML file
        """
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving configuration to {yaml_path}: {str(e)}")
            raise
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """
        Merge with another configuration.
        
        This method creates a new configuration object by merging
        the current configuration with another configuration.
        Parameters from the other configuration take precedence.
        
        Args:
            other: Configuration to merge with
            
        Returns:
            BaseConfig: Merged configuration
        """
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return type(self).from_dict(merged_dict)


@dataclass
class LLMConfig(BaseConfig):
    """
    Configuration for Language Model (LLM) services.
    
    This class extends the base configuration with parameters
    specific to LLM services.
    
    Attributes:
        temperature: Temperature for sampling (higher = more random)
        top_p: Top-p sampling parameter (nucleus sampling)
        max_tokens: Maximum number of tokens to generate
        stop_sequences: Sequences that stop generation
        context_window: Maximum context window size
        system_prompt: Default system prompt for the model
        model_format: Format of the model (gguf, onnx, etc.)
    """
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1024
    stop_sequences: List[str] = field(default_factory=list)
    
    # Context handling
    context_window: int = 4096
    system_prompt: str = "You are a helpful AI assistant."
    memory_window_tokens: int = 2048
    
    # Model configuration
    model_format: str = "auto"  # "auto", "gguf", "onnx", "pt", "safetensors"
    quantization: str = "none"  # "none", "4bit", "8bit", "int8", "int4"
    
    # Provider-specific configurations
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    
    # Advanced options
    streaming: bool = True
    use_memory: bool = False
    model_loader: str = "auto"  # "auto", "transformers", "ctransformers", "llama.cpp"
    
    def _validate(self):
        """Validate LLM-specific configuration parameters."""
        super()._validate()
        
        # Validate generation parameters
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Validate context parameters
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        
        if self.memory_window_tokens < 0:
            raise ValueError("memory_window_tokens must be non-negative")
        
        # Validate model format
        valid_formats = {"auto", "gguf", "onnx", "pt", "safetensors"}
        if self.model_format not in valid_formats:
            raise ValueError(f"model_format must be one of {valid_formats}")
        
        # Validate quantization
        valid_quantization = {"none", "4bit", "8bit", "int8", "int4"}
        if self.quantization not in valid_quantization:
            raise ValueError(f"quantization must be one of {valid_quantization}")


@dataclass
class STTConfig(BaseAudioConfig):
    """
    Configuration for speech-to-text services.
    
    This class extends the base audio configuration with parameters
    specific to speech-to-text services.
    
    Attributes:
        language: Default language for transcription
        task: Task type (transcribe or translate)
        beam_size: Beam size for decoding
        temperature: Temperature for sampling
        initial_prompt: Initial prompt for transcription
        without_timestamps: Whether to generate timestamps
        word_timestamps: Whether to generate word timestamps
    """
    
    # STT-specific configuration
    language: str = "auto"
    task: str = "transcribe"  # "transcribe" or "translate"
    beam_size: int = 5
    temperature: float = 0.0
    initial_prompt: Optional[str] = None
    without_timestamps: bool = False
    word_timestamps: bool = False
    
    # Audio preprocessing
    sample_rate: int = 16000
    channels: int = 1
    normalize_audio: bool = True
    remove_silence: bool = False
    
    # Performance metrics
    words_per_minute: Optional[float] = None
    
    def _validate(self):
        """Validate STT-specific configuration parameters."""
        super()._validate()
        
        # Validate task
        valid_tasks = {"transcribe", "translate"}
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}")
        
        # Validate beam size
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")
        
        # Validate temperature
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        # Validate audio parameters
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        
        if self.channels <= 0:
            raise ValueError("channels must be positive")


@dataclass
class TTSConfig(BaseAudioConfig):
    """
    Configuration for text-to-speech services.
    
    This class extends the base audio configuration with parameters
    specific to text-to-speech services.
    
    Attributes:
        language: Default language for synthesis
        voice_id: Default voice ID
        rate: Speech rate multiplier
        pitch: Voice pitch multiplier
        volume: Volume level
    """
    
    # TTS-specific configuration
    language: str = "en"
    voice_id: str = "default"
    rate: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    
    # Audio output configuration
    sample_rate: int = 22050
    output_format: str = "wav"  # "wav", "mp3", "ogg"
    bit_rate: int = 192  # For compressed formats
    
    # Text preprocessing
    normalize_text: bool = True
    remove_punctuation: bool = False
    add_punctuation: bool = False
    
    # Generation parameters
    max_sentence_length: int = 500
    sentence_silence_ms: int = 500
    
    def _validate(self):
        """Validate TTS-specific configuration parameters."""
        super()._validate()
        
        # Validate rate, pitch, and volume
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        
        if self.pitch <= 0:
            raise ValueError("pitch must be positive")
        
        if self.volume <= 0:
            raise ValueError("volume must be positive")
        
        # Validate audio parameters
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        
        valid_formats = {"wav", "mp3", "ogg"}
        if self.output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}")
        
        if self.bit_rate <= 0:
            raise ValueError("bit_rate must be positive")
        
        # Validate generation parameters
        if self.max_sentence_length <= 0:
            raise ValueError("max_sentence_length must be positive")
        
        if self.sentence_silence_ms < 0:
            raise ValueError("sentence_silence_ms must be non-negative")


@dataclass
class VisionConfig(BaseConfig):
    """
    Configuration for Vision Model services.
    
    This class extends the base configuration with parameters
    specific to vision model services.
    
    Attributes:
        max_image_size: Maximum size for input images (width/height in pixels)
        image_format: Output image format (PNG, JPEG, etc.)
        supports_multiple_images: Whether the provider supports multiple images
        temperature: Temperature for sampling (higher = more random)
        max_tokens: Maximum number of tokens to generate
        supports_tiling: Whether the provider supports image tiling
        tile_size: Size of tiles for high-resolution images
        max_tiles: Maximum number of tiles to use
        detail_level: Level of detail for image analysis (low, medium, high)
        ocr_enabled: Whether to enable OCR for document processing
    """
    
    # Image processing parameters
    max_image_size: int = 1024
    image_format: str = "JPEG"
    supports_multiple_images: bool = False
    preserve_aspect_ratio: bool = True
    normalize_images: bool = True
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    
    # High-resolution image handling
    supports_tiling: bool = False
    tile_size: int = 512
    max_tiles: int = 6
    
    # Analysis options
    detail_level: str = "medium"  # "low", "medium", "high"
    ocr_enabled: bool = False
    document_analysis_mode: str = "combined"  # "text_only", "vision_only", "combined"
    
    # Provider-specific configurations
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    
    # Model options
    model_type: str = "auto"  # "vqa", "captioning", "detection", "segmentation", "auto"
    quantization: str = "none"  # "none", "4bit", "8bit", "int8", "int4"
    
    def _validate(self):
        """Validate vision-specific configuration parameters."""
        super()._validate()
        
        # Validate image processing parameters
        if self.max_image_size <= 0:
            raise ValueError("max_image_size must be positive")
        
        valid_formats = {"JPEG", "PNG", "WEBP", "BMP"}
        if self.image_format not in valid_formats:
            raise ValueError(f"image_format must be one of {valid_formats}")
        
        # Validate generation parameters
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Validate high-resolution image parameters
        if self.supports_tiling:
            if self.tile_size <= 0:
                raise ValueError("tile_size must be positive")
            
            if self.max_tiles <= 0:
                raise ValueError("max_tiles must be positive")
        
        # Validate analysis options
        valid_detail_levels = {"low", "medium", "high", "auto"}
        if self.detail_level not in valid_detail_levels:
            raise ValueError(f"detail_level must be one of {valid_detail_levels}")
        
        valid_doc_modes = {"text_only", "vision_only", "combined"}
        if self.document_analysis_mode not in valid_doc_modes:
            raise ValueError(f"document_analysis_mode must be one of {valid_doc_modes}")


class ConfigManager:
    """
    Manager for configurations of multiple services.
    
    This class manages configurations for STT, TTS, LLM, and Vision services,
    providing methods to load, save, and access configurations.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """
        Get singleton instance of ConfigManager.
        
        Returns:
            ConfigManager: Singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.stt_configs = {}
        self.tts_configs = {}
        self.llm_configs = {}
        self.vision_configs = {}
        
        # Default configuration paths
        self.config_dir = os.path.join(settings.MEDIA_ROOT, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Try to load existing configurations
        self._load_existing_configs()
    
    def _load_existing_configs(self):
        """Load existing configurations from config directory."""
        try:
            # Load STT configurations
            stt_config_dir = os.path.join(self.config_dir, 'stt')
            if os.path.exists(stt_config_dir):
                for file_name in os.listdir(stt_config_dir):
                    if file_name.endswith('.json'):
                        service_name = file_name[:-5]  # Remove '.json'
                        try:
                            config = STTConfig.from_json(os.path.join(stt_config_dir, file_name))
                            self.stt_configs[service_name] = config
                        except Exception as e:
                            logger.warning(f"Error loading STT config {file_name}: {str(e)}")
            
            # Load TTS configurations
            tts_config_dir = os.path.join(self.config_dir, 'tts')
            if os.path.exists(tts_config_dir):
                for file_name in os.listdir(tts_config_dir):
                    if file_name.endswith('.json'):
                        service_name = file_name[:-5]  # Remove '.json'
                        try:
                            config = TTSConfig.from_json(os.path.join(tts_config_dir, file_name))
                            self.tts_configs[service_name] = config
                        except Exception as e:
                            logger.warning(f"Error loading TTS config {file_name}: {str(e)}")
                            
            # Load LLM configurations
            llm_config_dir = os.path.join(self.config_dir, 'llm')
            if os.path.exists(llm_config_dir):
                for file_name in os.listdir(llm_config_dir):
                    if file_name.endswith('.json'):
                        service_name = file_name[:-5]  # Remove '.json'
                        try:
                            config = LLMConfig.from_json(os.path.join(llm_config_dir, file_name))
                            self.llm_configs[service_name] = config
                        except Exception as e:
                            logger.warning(f"Error loading LLM config {file_name}: {str(e)}")
                            
            # Load Vision configurations
            vision_config_dir = os.path.join(self.config_dir, 'vision')
            if os.path.exists(vision_config_dir):
                for file_name in os.listdir(vision_config_dir):
                    if file_name.endswith('.json'):
                        service_name = file_name[:-5]  # Remove '.json'
                        try:
                            config = VisionConfig.from_json(os.path.join(vision_config_dir, file_name))
                            self.vision_configs[service_name] = config
                        except Exception as e:
                            logger.warning(f"Error loading Vision config {file_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading existing configurations: {str(e)}")
    
    def get_stt_config(self, service_name: str) -> STTConfig:
        """
        Get STT configuration for a service.
        
        Args:
            service_name: Name of the STT service
            
        Returns:
            STTConfig: Configuration for the service
        """
        if service_name not in self.stt_configs:
            # Create default configuration
            self.stt_configs[service_name] = STTConfig()
        return self.stt_configs[service_name]
    
    def get_tts_config(self, service_name: str) -> TTSConfig:
        """
        Get TTS configuration for a service.
        
        Args:
            service_name: Name of the TTS service
            
        Returns:
            TTSConfig: Configuration for the service
        """
        if service_name not in self.tts_configs:
            # Create default configuration
            self.tts_configs[service_name] = TTSConfig()
        return self.tts_configs[service_name]
        
    def get_llm_config(self, service_name: str) -> LLMConfig:
        """
        Get LLM configuration for a service.
        
        Args:
            service_name: Name of the LLM service
            
        Returns:
            LLMConfig: Configuration for the service
        """
        if service_name not in self.llm_configs:
            # Create default configuration
            self.llm_configs[service_name] = LLMConfig()
        return self.llm_configs[service_name]
        
    def get_vision_config(self, service_name: str) -> VisionConfig:
        """
        Get Vision configuration for a service.
        
        Args:
            service_name: Name of the Vision service
            
        Returns:
            VisionConfig: Configuration for the service
        """
        if service_name not in self.vision_configs:
            # Create default configuration
            self.vision_configs[service_name] = VisionConfig()
        return self.vision_configs[service_name]
    
    def set_stt_config(self, service_name: str, config: STTConfig) -> None:
        """
        Set STT configuration for a service.
        
        Args:
            service_name: Name of the STT service
            config: Configuration for the service
        """
        self.stt_configs[service_name] = config
        self._save_stt_config(service_name)
    
    def set_tts_config(self, service_name: str, config: TTSConfig) -> None:
        """
        Set TTS configuration for a service.
        
        Args:
            service_name: Name of the TTS service
            config: Configuration for the service
        """
        self.tts_configs[service_name] = config
        self._save_tts_config(service_name)
        
    def set_llm_config(self, service_name: str, config: LLMConfig) -> None:
        """
        Set LLM configuration for a service.
        
        Args:
            service_name: Name of the LLM service
            config: Configuration for the service
        """
        self.llm_configs[service_name] = config
        self._save_llm_config(service_name)
        
    def set_vision_config(self, service_name: str, config: VisionConfig) -> None:
        """
        Set Vision configuration for a service.
        
        Args:
            service_name: Name of the Vision service
            config: Configuration for the service
        """
        self.vision_configs[service_name] = config
        self._save_vision_config(service_name)
    
    def _save_stt_config(self, service_name: str) -> None:
        """
        Save STT configuration to file.
        
        Args:
            service_name: Name of the STT service
        """
        config = self.stt_configs[service_name]
        stt_config_dir = os.path.join(self.config_dir, 'stt')
        os.makedirs(stt_config_dir, exist_ok=True)
        config.save_json(os.path.join(stt_config_dir, f"{service_name}.json"))
    
    def _save_tts_config(self, service_name: str) -> None:
        """
        Save TTS configuration to file.
        
        Args:
            service_name: Name of the TTS service
        """
        config = self.tts_configs[service_name]
        tts_config_dir = os.path.join(self.config_dir, 'tts')
        os.makedirs(tts_config_dir, exist_ok=True)
        config.save_json(os.path.join(tts_config_dir, f"{service_name}.json"))
        
    def _save_llm_config(self, service_name: str) -> None:
        """
        Save LLM configuration to file.
        
        Args:
            service_name: Name of the LLM service
        """
        config = self.llm_configs[service_name]
        llm_config_dir = os.path.join(self.config_dir, 'llm')
        os.makedirs(llm_config_dir, exist_ok=True)
        config.save_json(os.path.join(llm_config_dir, f"{service_name}.json"))
        
    def _save_vision_config(self, service_name: str) -> None:
        """
        Save Vision configuration to file.
        
        Args:
            service_name: Name of the Vision service
        """
        config = self.vision_configs[service_name]
        vision_config_dir = os.path.join(self.config_dir, 'vision')
        os.makedirs(vision_config_dir, exist_ok=True)
        config.save_json(os.path.join(vision_config_dir, f"{service_name}.json"))
    
    def save_all_configs(self) -> None:
        """Save all configurations to files."""
        # Save STT configurations
        for service_name in self.stt_configs:
            self._save_stt_config(service_name)
        
        # Save TTS configurations
        for service_name in self.tts_configs:
            self._save_tts_config(service_name)
            
        # Save LLM configurations
        for service_name in self.llm_configs:
            self._save_llm_config(service_name)
            
        # Save Vision configurations
        for service_name in self.vision_configs:
            self._save_vision_config(service_name)
    
    def list_stt_configs(self) -> List[str]:
        """
        List available STT configurations.
        
        Returns:
            List[str]: Names of STT services with configurations
        """
        return list(self.stt_configs.keys())
    
    def list_tts_configs(self) -> List[str]:
        """
        List available TTS configurations.
        
        Returns:
            List[str]: Names of TTS services with configurations
        """
        return list(self.tts_configs.keys())
        
    def list_llm_configs(self) -> List[str]:
        """
        List available LLM configurations.
        
        Returns:
            List[str]: Names of LLM services with configurations
        """
        return list(self.llm_configs.keys())
        
    def list_vision_configs(self) -> List[str]:
        """
        List available Vision configurations.
        
        Returns:
            List[str]: Names of Vision services with configurations
        """
        return list(self.vision_configs.keys())


# Helper functions for easier access
def get_stt_config(service_name: str) -> STTConfig:
    """
    Get STT configuration for a service.
    
    Args:
        service_name: Name of the STT service
        
    Returns:
        STTConfig: Configuration for the service
    """
    return ConfigManager.get_instance().get_stt_config(service_name)

def get_tts_config(service_name: str) -> TTSConfig:
    """
    Get TTS configuration for a service.
    
    Args:
        service_name: Name of the TTS service
        
    Returns:
        TTSConfig: Configuration for the service
    """
    return ConfigManager.get_instance().get_tts_config(service_name)

def get_llm_config(service_name: str) -> LLMConfig:
    """
    Get LLM configuration for a service.
    
    Args:
        service_name: Name of the LLM service
        
    Returns:
        LLMConfig: Configuration for the service
    """
    return ConfigManager.get_instance().get_llm_config(service_name)

def get_vision_config(service_name: str) -> VisionConfig:
    """
    Get Vision configuration for a service.
    
    Args:
        service_name: Name of the Vision service
        
    Returns:
        VisionConfig: Configuration for the service
    """
    return ConfigManager.get_instance().get_vision_config(service_name)

def set_stt_config(service_name: str, config: STTConfig) -> None:
    """
    Set STT configuration for a service.
    
    Args:
        service_name: Name of the STT service
        config: Configuration for the service
    """
    ConfigManager.get_instance().set_stt_config(service_name, config)

def set_tts_config(service_name: str, config: TTSConfig) -> None:
    """
    Set TTS configuration for a service.
    
    Args:
        service_name: Name of the TTS service
        config: Configuration for the service
    """
    ConfigManager.get_instance().set_tts_config(service_name, config)
    
def set_llm_config(service_name: str, config: LLMConfig) -> None:
    """
    Set LLM configuration for a service.
    
    Args:
        service_name: Name of the LLM service
        config: Configuration for the service
    """
    ConfigManager.get_instance().set_llm_config(service_name, config)

def set_vision_config(service_name: str, config: VisionConfig) -> None:
    """
    Set Vision configuration for a service.
    
    Args:
        service_name: Name of the Vision service
        config: Configuration for the service
    """
    ConfigManager.get_instance().set_vision_config(service_name, config) 