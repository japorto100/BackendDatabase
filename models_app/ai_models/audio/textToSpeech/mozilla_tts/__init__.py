"""
Mozilla TTS integration module.

This module provides integration with Mozilla's Text-to-Speech framework,
allowing conversion of text to natural-sounding speech.
"""

from .model_manager import MozillaTTSModelManager
from .service import MozillaTTSService

__all__ = ["MozillaTTSModelManager", "MozillaTTSService"] 