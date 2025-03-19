"""
API views for Speech-to-Text functionality.

This module provides Django REST Framework views for transcribing audio files and streams.
"""

import logging
import json
import os
from typing import Dict, Any
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from utils.config_handler import config

from .stt_factory import STTFactory, STTEngine
from models_app.ai_models.utils.audio.processing import extract_audio_from_video, assess_audio_quality

logger = logging.getLogger(__name__)

# Configure model caching
CACHE_MODELS = config.get('stt', {}).get('cache_models', True)

@api_view(['GET'])
def list_engines(request):
    """List all available STT engines with capabilities"""
    factory = STTFactory(cache_models=CACHE_MODELS)
    engines = factory.list_available_engines()
    
    # Format response with detailed engine information
    result = [{
        'id': engine.value,
        'name': engine.value.capitalize().replace('_', ' '),
        'features': get_engine_features(engine)
    } for engine in engines]
    
    return JsonResponse({'engines': result})

@api_view(['GET'])
def engine_details(request, engine_id):
    """Get detailed information about a specific STT engine"""
    try:
        engine = STTEngine(engine_id)
        factory = STTFactory(cache_models=CACHE_MODELS)
        
        try:
            # Get available models
            models = factory.list_available_models(engine)
            
            # Get available languages
            languages = factory.list_available_languages(engine)
            
            return JsonResponse({
                'id': engine.value,
                'name': engine.value.capitalize().replace('_', ' '),
                'features': get_engine_features(engine),
                'models': models,
                'languages': languages,
                'status': 'available'
            })
        finally:
            # Clean up if not caching models
            if not CACHE_MODELS:
                factory.cleanup()
                
    except Exception as e:
        logger.error(f"Error getting engine details: {str(e)}")
        return JsonResponse({
            'id': engine_id,
            'error': str(e),
            'status': 'error'
        }, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def transcribe_audio(request):
    """Transcribe an audio file"""
    factory = None
    audio_path = None
    
    try:
        if 'audio_file' not in request.FILES:
            return JsonResponse({'error': 'Audio file is required'}, status=400)
        
        audio_file = request.FILES['audio_file']
        engine_id = request.data.get('engine', STTEngine.WHISPER_INSANELY_FAST.value)
        language = request.data.get('language')
        
        # Parse options
        options = {}
        if 'options' in request.data:
            try:
                options = json.loads(request.data['options'])
            except:
                options = {}
        
        # Extract options
        model_size = options.get('model_size', 'base')
        word_timestamps = options.get('word_timestamps', False)
        detect_speakers = options.get('detect_speakers', False)
        remove_noise = options.get('remove_noise', False)
        initial_prompt = options.get('initial_prompt')
        
        # Save audio file to temporary location
        audio_path = handle_uploaded_audio(audio_file)
        
        # Check if it's a video file and extract audio if needed
        if audio_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                audio_path = extract_audio_from_video(audio_path)
            except Exception as e:
                logger.error(f"Error extracting audio from video: {e}")
                return JsonResponse({'error': f"Error extracting audio from video: {str(e)}"}, status=500)
        
        # Assess audio quality
        quality = assess_audio_quality(audio_path)
        
        # Create STT factory
        factory = STTFactory(cache_models=CACHE_MODELS)
        engine = STTEngine(engine_id)
        
        # Transcribe audio
        result = factory.transcribe_audio(
            audio_path,
            engine=engine,
            language=language,
            apply_preprocessing=True,
            remove_noise=remove_noise,
            detect_speakers=detect_speakers,
            word_timestamps=word_timestamps,
            model_size=model_size,
            initial_prompt=initial_prompt
        )
        
        # Include audio quality info in response
        result['audio_quality'] = {
            'duration': quality.get('duration', 0),
            'sample_rate': quality.get('sample_rate', 0),
            'channels': quality.get('channels', 0),
            'estimated_snr_db': quality.get('estimated_snr_db', 0),
            'is_suitable_for_stt': quality.get('is_suitable_for_stt', True)
        }
        
        # Include engine info
        result['engine'] = engine_id
        result['model_size'] = model_size
        
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
        
    finally:
        # Clean up resources
        try:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary audio file: {e}")
            
        try:
            if factory and not CACHE_MODELS:
                factory.cleanup()
        except Exception as e:
            logger.warning(f"Failed to clean up STT factory: {e}")

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def transcribe_url(request):
    """Transcribe audio from a URL"""
    factory = None
    tmp_path = None
    
    try:
        data = request.data
        audio_url = data.get('url')
        
        if not audio_url:
            return JsonResponse({'error': 'Audio URL is required'}, status=400)
        
        engine_id = data.get('engine', STTEngine.WHISPER_INSANELY_FAST.value)
        language = data.get('language')
        options = data.get('options', {})
        
        # Extract options
        model_size = options.get('model_size', 'base')
        word_timestamps = options.get('word_timestamps', False)
        detect_speakers = options.get('detect_speakers', False)
        remove_noise = options.get('remove_noise', False)
        initial_prompt = options.get('initial_prompt')
        
        # Download audio file from URL
        import requests
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=get_suffix_from_url(audio_url), delete=False) as tmp:
            tmp_path = tmp.name
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
        
        # Check if it's a video file and extract audio if needed
        if tmp_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                tmp_path = extract_audio_from_video(tmp_path)
            except Exception as e:
                logger.error(f"Error extracting audio from video: {e}")
                return JsonResponse({'error': f"Error extracting audio from video: {str(e)}"}, status=500)
        
        # Assess audio quality
        quality = assess_audio_quality(tmp_path)
        
        # Create STT factory
        factory = STTFactory(cache_models=CACHE_MODELS)
        engine = STTEngine(engine_id)
        
        # Transcribe audio
        result = factory.transcribe_audio(
            tmp_path,
            engine=engine,
            language=language,
            apply_preprocessing=True,
            remove_noise=remove_noise,
            detect_speakers=detect_speakers,
            word_timestamps=word_timestamps,
            model_size=model_size,
            initial_prompt=initial_prompt
        )
        
        # Include audio quality info in response
        result['audio_quality'] = {
            'duration': quality.get('duration', 0),
            'sample_rate': quality.get('sample_rate', 0),
            'channels': quality.get('channels', 0),
            'estimated_snr_db': quality.get('estimated_snr_db', 0),
            'is_suitable_for_stt': quality.get('is_suitable_for_stt', True)
        }
        
        # Include engine info
        result['engine'] = engine_id
        result['model_size'] = model_size
        result['source_url'] = audio_url
        
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Error in URL transcription: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
        
    finally:
        # Clean up resources
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary audio file: {e}")
            
        try:
            if factory and not CACHE_MODELS:
                factory.cleanup()
        except Exception as e:
            logger.warning(f"Failed to clean up STT factory: {e}")

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def transcribe_stream(request):
    """Transcribe audio from a streaming source with continuous output"""
    factory = None
    audio_path = None
    
    if 'audio_file' not in request.FILES:
        return JsonResponse({'error': 'Audio file is required'}, status=400)
    
    audio_file = request.FILES['audio_file']
    engine_id = request.query_params.get('engine', STTEngine.WHISPER_INSANELY_FAST.value)
    language = request.query_params.get('language')
    model_size = request.query_params.get('model_size', 'base')
    
    # Save audio file to temporary location
    audio_path = handle_uploaded_audio(audio_file)
    
    try:
        # Create STT factory
        factory = STTFactory(cache_models=CACHE_MODELS)
        engine = STTEngine(engine_id)
        
        # Function to stream transcription results
        def generate_transcription():
            try:
                with open(audio_path, 'rb') as audio_stream:
                    for result in factory.transcribe_continuous(
                        audio_stream,
                        engine=engine,
                        language=language,
                        chunk_size_ms=5000,
                        model_size=model_size
                    ):
                        # Convert result to JSON string
                        yield f"data: {json.dumps(result)}\n\n"
                        
                        # If this is the final result, stop streaming
                        if result.get('final', False):
                            break
            except Exception as e:
                logger.error(f"Error in streaming transcription generator: {e}")
                yield f"data: {json.dumps({'error': str(e), 'final': True})}\n\n"
            finally:
                # Clean up temporary file
                try:
                    if audio_path and os.path.exists(audio_path):
                        os.unlink(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary audio file: {e}")
        
        # Return a streaming response
        response = StreamingHttpResponse(
            generate_transcription(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        return response
        
    except Exception as e:
        logger.error(f"Error in streaming transcription: {str(e)}")
        # Clean up temporary file
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
        # Clean up factory
        if factory and not CACHE_MODELS:
            try:
                factory.cleanup()
            except:
                pass
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def optimize_audio_for_stt(request):
    """Optimize an audio file for better STT performance"""
    factory = None
    audio_path = None
    processed_path = None
    
    try:
        if 'audio_file' not in request.FILES:
            return JsonResponse({'error': 'Audio file is required'}, status=400)
        
        audio_file = request.FILES['audio_file']
        remove_noise = request.data.get('remove_noise', 'false').lower() == 'true'
        
        # Save uploaded file
        audio_path = handle_uploaded_audio(audio_file)
        
        # Get quality before optimization
        quality_before = assess_audio_quality(audio_path)
        
        # Create STT factory
        factory = STTFactory(cache_models=CACHE_MODELS)
        
        # Preprocess audio (optimization)
        processed_path = factory._preprocess_audio(audio_path, remove_noise=remove_noise)
        
        # Get quality after optimization
        quality_after = assess_audio_quality(processed_path)
        
        # Return file path and quality metrics
        if os.path.exists(processed_path):
            return JsonResponse({
                'success': True,
                'file_url': f"/media/stt_output/{os.path.basename(processed_path)}",
                'quality_before': {
                    'duration': quality_before.get('duration', 0),
                    'sample_rate': quality_before.get('sample_rate', 0),
                    'estimated_snr_db': quality_before.get('estimated_snr_db', 0),
                    'is_suitable_for_stt': quality_before.get('is_suitable_for_stt', False)
                },
                'quality_after': {
                    'duration': quality_after.get('duration', 0),
                    'sample_rate': quality_after.get('sample_rate', 0),
                    'estimated_snr_db': quality_after.get('estimated_snr_db', 0),
                    'is_suitable_for_stt': quality_after.get('is_suitable_for_stt', True)
                },
                'optimizations_applied': {
                    'converted_to_mono': quality_before.get('channels', 1) > 1,
                    'normalized_volume': True,
                    'removed_noise': remove_noise,
                    'resampled': quality_before.get('sample_rate', 16000) != 16000
                }
            })
        else:
            return JsonResponse({'error': 'Failed to optimize audio'}, status=500)
            
    except Exception as e:
        logger.error(f"Error in audio optimization: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
        
    finally:
        # Clean up temporary files
        try:
            if audio_path and os.path.exists(audio_path) and audio_path != processed_path:
                os.unlink(audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary audio file: {e}")
            
        try:
            if factory and not CACHE_MODELS:
                factory.cleanup()
        except Exception as e:
            logger.warning(f"Failed to clean up STT factory: {e}")

# Helper functions
def get_engine_features(engine):
    """Get features for an STT engine"""
    features = []
    
    if engine == STTEngine.WHISPER_INSANELY_FAST:
        features = ['word_timestamps', 'multilingual', 'translate', 'fast_inference']
    elif engine == STTEngine.WHISPER_FASTER:
        features = ['word_timestamps', 'multilingual', 'translate', 'optimized_memory']
    elif engine == STTEngine.WHISPER_X:
        features = ['word_timestamps', 'multilingual', 'translate', 'speaker_diarization']
    
    return features

def handle_uploaded_audio(audio_file):
    """Save uploaded audio file to temporary location"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.name)[1], delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        return tmp.name

def get_suffix_from_url(url):
    """Get file extension from URL"""
    import urllib.parse
    path = urllib.parse.urlparse(url).path
    extension = os.path.splitext(path)[1]
    return extension if extension else '.wav' 