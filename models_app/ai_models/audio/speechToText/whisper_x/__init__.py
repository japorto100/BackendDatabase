"""
WhisperX module for speech transcription with speaker diarization.

This module provides an implementation of WhisperX with enhanced features
such as speaker identification/diarization and improved alignment.
"""

from .service import WhisperXService
from .model_manager import WhisperXModelManager

__all__ = [
    'WhisperXService',
    'WhisperXModelManager'
] 