"""
Insanely Fast Whisper module for speech transcription.

This module provides an optimized implementation of OpenAI's Whisper model
for high-performance speech-to-text transcription.
"""

from .service import WhisperInsanelyFastService
from .model_manager import WhisperInsanelyFastModelManager

__all__ = [
    'WhisperInsanelyFastService',
    'WhisperInsanelyFastModelManager'
] 