"""
Text-to-Speech (TTS) module for synthesizing speech from text.

This package provides a factory pattern for accessing different TTS engines,
with a unified API that abstracts away the implementation details.
"""

from .tts_factory import TTSFactory, TTSEngine
from .base_tts_service import BaseTTSService, handle_tts_errors, handle_audio_processing_errors, synthesis_operation
from .spark_tts.service import SparkTTSService
from .coqui_tts.service import CoquiTTSService
from .mozilla_tts.service import MozillaTTSService

__all__ = [
    'TTSFactory',
    'TTSEngine',
    'BaseTTSService', 
    'handle_tts_errors',
    'handle_audio_processing_errors',
    'synthesis_operation',
    'SparkTTSService',
    'CoquiTTSService',
    'MozillaTTSService'
] 