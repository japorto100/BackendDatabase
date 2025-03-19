"""
Audio caching system for Text-to-Speech (TTS) and Speech-to-Text (STT) operations.

This module provides functionality to cache audio outputs and transcriptions
to improve performance and reduce redundant processing.
"""

import os
import json
import hashlib
import logging
import time
import shutil
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class TTSCache:
    """
    Cache for Text-to-Speech outputs to avoid regenerating common phrases.
    
    This cache stores the paths to generated audio files along with their metadata
    and provides methods to retrieve and store cache entries.
    """
    
    def __init__(self, cache_dir: str = None, max_size_mb: int = 500, 
                max_age_days: int = 30, cleanup_interval: int = 3600):
        """
        Initialize the TTS cache.
        
        Args:
            cache_dir: Directory to store cached files. If None, uses a default location.
            max_size_mb: Maximum cache size in megabytes
            max_age_days: Maximum age of cache entries in days
            cleanup_interval: Interval between cache cleanup operations in seconds
        """
        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.tts_cache')
        
        self.cache_dir = cache_dir
        self.audio_dir = os.path.join(cache_dir, 'audio')
        self.metadata_file = os.path.join(cache_dir, 'metadata.json')
        self.max_size_mb = max_size_mb
        self.max_age_days = max_age_days
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = 0
        self.lock = threading.RLock()
        
        # Create directories if they don't exist
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Run cleanup if needed
        self._maybe_cleanup()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from file or create empty metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading TTS cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with self.lock:
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f)
            except Exception as e:
                logger.error(f"Error saving TTS cache metadata: {e}")
    
    def _get_cache_key(self, text: str, engine: str, voice_id: Optional[str], 
                     options: Optional[Dict[str, Any]]) -> str:
        """
        Generate a unique cache key for the given parameters.
        
        Args:
            text: Text to synthesize
            engine: TTS engine name
            voice_id: Voice ID
            options: Additional options affecting synthesis
            
        Returns:
            Unique hash string for the parameters
        """
        # Create key components
        components = [text, engine]
        
        if voice_id:
            components.append(str(voice_id))
        
        if options:
            # Sort options to ensure consistent ordering
            sorted_options = sorted((str(k), str(v)) for k, v in options.items())
            for k, v in sorted_options:
                components.append(f"{k}={v}")
        
        # Join and hash
        key_str = "|".join(components)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get_cached_audio(self, text: str, engine: str, voice_id: Optional[str] = None, 
                        options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get cached audio file path if available.
        
        Args:
            text: Text that was synthesized
            engine: TTS engine name
            voice_id: Voice ID
            options: Additional options used for synthesis
            
        Returns:
            Path to cached audio file or None if not found
        """
        self._maybe_cleanup()
        
        with self.lock:
            # Generate cache key
            key = self._get_cache_key(text, engine, voice_id, options)
            
            # Check if key exists in metadata
            if key in self.metadata:
                entry = self.metadata[key]
                audio_path = entry.get('path')
                
                # Verify file exists
                if audio_path and os.path.exists(audio_path):
                    # Update last accessed time
                    entry['last_accessed'] = time.time()
                    self._save_metadata()
                    return audio_path
                else:
                    # File doesn't exist, remove from metadata
                    del self.metadata[key]
                    self._save_metadata()
        
        return None
    
    def cache_audio(self, text: str, audio_path: str, engine: str, 
                   voice_id: Optional[str] = None, options: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Cache an audio file.
        
        Args:
            text: Text that was synthesized
            audio_path: Path to the audio file
            engine: TTS engine name
            voice_id: Voice ID
            options: Additional options used for synthesis
            metadata: Additional metadata to store
            
        Returns:
            Path to the cached audio file
        """
        with self.lock:
            # Generate cache key
            key = self._get_cache_key(text, engine, voice_id, options)
            
            # Create cache entry
            entry = {
                'text': text,
                'engine': engine,
                'created': time.time(),
                'last_accessed': time.time(),
                'size_bytes': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            }
            
            if voice_id:
                entry['voice_id'] = voice_id
            
            if options:
                entry['options'] = options
            
            if metadata:
                entry['metadata'] = metadata
            
            # Copy audio file to cache
            filename = f"{key}{os.path.splitext(audio_path)[1]}"
            cache_path = os.path.join(self.audio_dir, filename)
            
            try:
                shutil.copy2(audio_path, cache_path)
                entry['path'] = cache_path
                
                # Update metadata
                self.metadata[key] = entry
                self._save_metadata()
                
                return cache_path
            except Exception as e:
                logger.error(f"Error caching audio file: {e}")
                return audio_path
    
    def _maybe_cleanup(self):
        """Run cache cleanup if it's time."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up old cache entries and ensure cache doesn't exceed max size.
        """
        with self.lock:
            try:
                current_time = time.time()
                self.last_cleanup = current_time
                
                # Remove entries older than max age
                max_age_seconds = self.max_age_days * 24 * 3600
                to_remove = []
                
                for key, entry in self.metadata.items():
                    age = current_time - entry.get('last_accessed', entry.get('created', 0))
                    if age > max_age_seconds:
                        to_remove.append(key)
                
                for key in to_remove:
                    self._remove_entry(key)
                
                # Check total cache size
                total_size = self._get_cache_size()
                max_size_bytes = self.max_size_mb * 1024 * 1024
                
                if total_size > max_size_bytes:
                    # Sort entries by last accessed time (oldest first)
                    sorted_entries = sorted(
                        self.metadata.items(),
                        key=lambda x: x[1].get('last_accessed', x[1].get('created', 0))
                    )
                    
                    # Remove oldest entries until under max size
                    for key, _ in sorted_entries:
                        self._remove_entry(key)
                        
                        # Recalculate size
                        total_size = self._get_cache_size()
                        if total_size <= max_size_bytes:
                            break
                
                # Save updated metadata
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
    
    def _remove_entry(self, key: str):
        """Remove a cache entry and its file."""
        entry = self.metadata.get(key)
        if not entry:
            return
        
        # Delete file
        audio_path = entry.get('path')
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception as e:
                logger.error(f"Error deleting cached file {audio_path}: {e}")
        
        # Remove from metadata
        del self.metadata[key]
    
    def _get_cache_size(self) -> int:
        """Calculate total cache size in bytes."""
        return sum(entry.get('size_bytes', 0) for entry in self.metadata.values())
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            # Delete all files
            for entry in self.metadata.values():
                audio_path = entry.get('path')
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.unlink(audio_path)
                    except Exception as e:
                        logger.error(f"Error deleting cached file {audio_path}: {e}")
            
            # Clear metadata
            self.metadata = {}
            self._save_metadata()


class STTCache:
    """
    Cache for Speech-to-Text results to avoid redundant transcription of identical audio.
    
    This cache stores transcription results along with their metadata
    and provides methods to retrieve and store cache entries.
    """
    
    def __init__(self, cache_dir: str = None, max_size_mb: int = 100, 
                max_age_days: int = 7, cleanup_interval: int = 3600):
        """
        Initialize the STT cache.
        
        Args:
            cache_dir: Directory to store cached data. If None, uses a default location.
            max_size_mb: Maximum cache size in megabytes
            max_age_days: Maximum age of cache entries in days
            cleanup_interval: Interval between cache cleanup operations in seconds
        """
        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.stt_cache')
        
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, 'metadata.json')
        self.max_size_mb = max_size_mb
        self.max_age_days = max_age_days
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = 0
        self.lock = threading.RLock()
        
        # Create directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Run cleanup if needed
        self._maybe_cleanup()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from file or create empty metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading STT cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with self.lock:
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f)
            except Exception as e:
                logger.error(f"Error saving STT cache metadata: {e}")
    
    def _get_audio_hash(self, audio_path: str) -> str:
        """
        Calculate hash for an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Hash string for the audio file
        """
        if not os.path.exists(audio_path):
            return None
        
        hash_md5 = hashlib.md5()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_cache_key(self, audio_hash: str, engine: str, 
                     options: Optional[Dict[str, Any]]) -> str:
        """
        Generate a unique cache key for the given parameters.
        
        Args:
            audio_hash: Hash of the audio file
            engine: STT engine name
            options: Additional options affecting transcription
            
        Returns:
            Unique hash string for the parameters
        """
        # Create key components
        components = [audio_hash, engine]
        
        if options:
            # Sort options to ensure consistent ordering
            sorted_options = sorted((str(k), str(v)) for k, v in options.items())
            for k, v in sorted_options:
                components.append(f"{k}={v}")
        
        # Join and hash
        key_str = "|".join(components)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get_cached_transcription(self, audio_path: str, engine: str, 
                               options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached transcription result if available.
        
        Args:
            audio_path: Path to the audio file
            engine: STT engine name
            options: Additional options used for transcription
            
        Returns:
            Cached transcription result or None if not found
        """
        self._maybe_cleanup()
        
        with self.lock:
            # Calculate audio hash
            audio_hash = self._get_audio_hash(audio_path)
            if not audio_hash:
                return None
            
            # Generate cache key
            key = self._get_cache_key(audio_hash, engine, options)
            
            # Check if key exists in metadata
            if key in self.metadata:
                entry = self.metadata[key]
                
                # Update last accessed time
                entry['last_accessed'] = time.time()
                self._save_metadata()
                
                return {
                    'text': entry.get('text'),
                    'confidence': entry.get('confidence'),
                    'segments': entry.get('segments', []),
                    'language': entry.get('language'),
                    'cached': True
                }
        
        return None
    
    def cache_transcription(self, audio_path: str, transcription: Dict[str, Any], 
                          engine: str, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Cache a transcription result.
        
        Args:
            audio_path: Path to the audio file
            transcription: Transcription result to cache
            engine: STT engine name
            options: Additional options used for transcription
        """
        with self.lock:
            # Calculate audio hash
            audio_hash = self._get_audio_hash(audio_path)
            if not audio_hash:
                return
            
            # Generate cache key
            key = self._get_cache_key(audio_hash, engine, options)
            
            # Create cache entry
            entry = {
                'audio_hash': audio_hash,
                'engine': engine,
                'created': time.time(),
                'last_accessed': time.time(),
                'text': transcription.get('text', ''),
                'confidence': transcription.get('confidence'),
                'segments': transcription.get('segments', []),
                'language': transcription.get('language'),
                'size_bytes': len(json.dumps(transcription))
            }
            
            if options:
                entry['options'] = options
            
            # Update metadata
            self.metadata[key] = entry
            self._save_metadata()
    
    def _maybe_cleanup(self):
        """Run cache cleanup if it's time."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up old cache entries and ensure cache doesn't exceed max size.
        """
        with self.lock:
            try:
                current_time = time.time()
                self.last_cleanup = current_time
                
                # Remove entries older than max age
                max_age_seconds = self.max_age_days * 24 * 3600
                to_remove = []
                
                for key, entry in self.metadata.items():
                    age = current_time - entry.get('last_accessed', entry.get('created', 0))
                    if age > max_age_seconds:
                        to_remove.append(key)
                
                for key in to_remove:
                    del self.metadata[key]
                
                # Check total cache size
                total_size = sum(entry.get('size_bytes', 0) for entry in self.metadata.values())
                max_size_bytes = self.max_size_mb * 1024 * 1024
                
                if total_size > max_size_bytes:
                    # Sort entries by last accessed time (oldest first)
                    sorted_entries = sorted(
                        self.metadata.items(),
                        key=lambda x: x[1].get('last_accessed', x[1].get('created', 0))
                    )
                    
                    # Remove oldest entries until under max size
                    for key, _ in sorted_entries:
                        del self.metadata[key]
                        
                        # Recalculate size
                        total_size = sum(entry.get('size_bytes', 0) for entry in self.metadata.values())
                        if total_size <= max_size_bytes:
                            break
                
                # Save updated metadata
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            # Clear metadata
            self.metadata = {}
            self._save_metadata()


# Singleton instances for global use
_tts_cache = None
_stt_cache = None

def get_tts_cache() -> TTSCache:
    """Get the global TTS cache instance."""
    global _tts_cache
    if _tts_cache is None:
        _tts_cache = TTSCache()
    return _tts_cache

def get_stt_cache() -> STTCache:
    """Get the global STT cache instance."""
    global _stt_cache
    if _stt_cache is None:
        _stt_cache = STTCache()
    return _stt_cache 