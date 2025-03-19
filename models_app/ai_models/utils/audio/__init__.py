"""
Audio utilities for processing, text handling, and caching.

This package provides utilities for audio processing, text normalization,
and caching capabilities for both TTS and STT operations.
"""

# Export key functions from processing module
from .processing import (
    convert_audio_format,
    split_audio_on_silence,
    normalize_audio,
    get_audio_duration,
    convert_to_mono,
    detect_speech_segments,
    trim_silence,
    assess_audio_quality
)

# Export key functions from text_processing
from .text_processing import (
    normalize_text,
    preprocess_text_for_tts,
    detect_language,
    is_ssml,
    validate_ssml,
    generate_ssml,
    segment_long_text,
    extract_spoken_phrases
)

# Export caching utilities
from .caching import (
    TTSCache,
    STTCache,
    get_tts_cache,
    get_stt_cache
)

__all__ = [
    # Processing functions
    'convert_audio_format', 'split_audio_on_silence', 'normalize_audio',
    'get_audio_duration', 'convert_to_mono', 'detect_speech_segments',
    'trim_silence', 'assess_audio_quality',
    
    # Text processing functions
    'normalize_text', 'preprocess_text_for_tts', 'detect_language',
    'is_ssml', 'validate_ssml', 'generate_ssml', 'segment_long_text',
    'extract_spoken_phrases',
    
    # Caching classes and functions
    'TTSCache', 'STTCache', 'get_tts_cache', 'get_stt_cache'
]

# This module will contain audio-specific utilities in the future.
# For now, it's a placeholder for organization purposes. 