"""
Faster Whisper module for speech transcription.

This module provides an optimized implementation of OpenAI's Whisper model
with efficient memory usage for speech-to-text transcription.
"""

from .service import WhisperFasterService
from .model_manager import WhisperFasterModelManager

__all__ = [
    'WhisperFasterService',
    'WhisperFasterModelManager'
] 