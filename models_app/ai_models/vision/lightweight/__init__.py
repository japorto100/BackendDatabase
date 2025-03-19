"""
Lightweight Vision Module

This module provides integration with lightweight local vision models
that can run efficiently even on modest hardware.
"""

from .model_manager import LightweightVisionModelManager
from .service import LightweightVisionService

__all__ = [
    "LightweightVisionModelManager",
    "LightweightVisionService",
] 