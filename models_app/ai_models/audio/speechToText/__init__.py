"""
Speech-to-Text (STT) module for transcribing audio files and streams.

This package provides a factory pattern for accessing different STT engines,
with a unified API that abstracts away the implementation details.
"""

from .stt_factory import STTFactory, STTEngine
from .base_stt_service import BaseSTTService, handle_stt_errors, handle_audio_processing_errors
from .whisper_insanely_fast.service import WhisperInsanelyFastService
from .whisper_insanely_fast.model_manager import WhisperInsanelyFastModelManager
from .whisper_faster.service import WhisperFasterService
from .whisper_faster.model_manager import WhisperFasterModelManager
from .whisper_x.service import WhisperXService
from .whisper_x.model_manager import WhisperXModelManager

__all__ = [
    'STTFactory',
    'STTEngine',
    'BaseSTTService',
    'handle_stt_errors',
    'handle_audio_processing_errors',
    'WhisperInsanelyFastService',
    'WhisperInsanelyFastModelManager',
    'WhisperFasterService',
    'WhisperFasterModelManager',
    'WhisperXService',
    'WhisperXModelManager'
] 