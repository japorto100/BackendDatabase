import logging
import json
import os
from typing import Dict, Any
from django.http import JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from utils.config_handler import config

from .tts_factory import TTSFactory, TTSEngine
from .coqui_tts.service import CoquiTTSService
from .mozilla_tts.service import MozillaTTSService
from .spark_tts.service import SparkTTSService
from models_app.ai_models.utils.audio.processing import assess_audio_quality

logger = logging.getLogger(__name__)

@api_view(['GET'])
def list_engines(request):
    """List all available TTS engines with capabilities"""
    factory = TTSFactory()
    engines = factory.list_available_engines()
    
    # Format response with detailed engine information
    result = [{
        'id': engine.value,
        'name': engine.value.capitalize(),
        'features': get_engine_features(engine)
    } for engine in engines]
    
    return JsonResponse({'engines': result})

@api_view(['GET'])
def engine_details(request, engine_id):
    """Get detailed information about a specific TTS engine"""
    try:
        engine = TTSEngine(engine_id)
        factory = TTSFactory()
        
        # Get service and model info
        service = factory.get_service(engine)
        voices = service.list_available_voices() if hasattr(service, 'list_available_voices') else []
        
        return JsonResponse({
            'id': engine.value,
            'name': engine.value.capitalize(),
            'features': get_engine_features(engine),
            'voices': voices,
            'status': 'available'
        })
    except Exception as e:
        logger.error(f"Error getting engine details: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def synthesize_speech(request):
    """Synthesize speech from text"""
    try:
        data = request.data
        text = data.get('text')
        engine_id = data.get('engine', TTSEngine.SPARK.value)
        voice_id = data.get('voice_id')
        options = data.get('options', {})
        
        # Extract post-processing options
        apply_post_processing = data.get('apply_post_processing', True)
        normalize_volume = data.get('normalize_volume', True)
        remove_silence = data.get('remove_silence', True)
        target_format = data.get('target_format', 'wav')
        sample_rate = data.get('sample_rate', 24000)
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        # Get TTS factory and service
        factory = TTSFactory()
        engine = TTSEngine(engine_id)
        
        # Set up options dictionary
        synthesis_options = options.copy()
        if voice_id:
            synthesis_options['voice_id'] = voice_id
        
        # Generate speech with post-processing
        output_path = factory.synthesize_speech(
            text, 
            engine=engine, 
            apply_post_processing=apply_post_processing,
            normalize_volume=normalize_volume,
            remove_silence=remove_silence,
            target_format=target_format,
            sample_rate=sample_rate,
            **synthesis_options
        )
        
        # Return file information
        if output_path:
            # Get quality metrics for the generated speech
            quality_metrics = factory.assess_speech_quality(output_path)
            
            # Extract key metrics for the response
            quality_info = {
                'duration': quality_metrics.get('duration', 0),
                'sample_rate': quality_metrics.get('sample_rate', 0),
                'bit_depth': quality_metrics.get('bit_depth', 0),
                'average_loudness_db': quality_metrics.get('average_loudness_db', 0),
                'is_suitable_for_playback': quality_metrics.get('is_suitable_for_tts', True)
            }
            
            return JsonResponse({
                'success': True,
                'file_url': f"/media/tts_output/{os.path.basename(output_path)}",
                'engine': engine_id,
                'voice_id': voice_id,
                'format': target_format,
                'quality': quality_info
            })
        else:
            return JsonResponse({'error': 'Failed to synthesize speech'}, status=500)
        
    except Exception as e:
        logger.error(f"Error in speech synthesis: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def voice_cloning(request):
    """Clone a voice from reference audio and synthesize speech"""
    try:
        text = request.data.get('text')
        engine_id = request.data.get('engine', TTSEngine.SPARK.value)
        
        # Extract post-processing options
        apply_post_processing = request.data.get('apply_post_processing', True)
        normalize_volume = request.data.get('normalize_volume', True)
        remove_silence = request.data.get('remove_silence', True)
        target_format = request.data.get('target_format', 'wav')
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        if 'reference_audio' not in request.FILES:
            return JsonResponse({'error': 'Reference audio is required'}, status=400)
        
        # Process the reference audio file
        reference_audio = request.FILES['reference_audio']
        audio_path = handle_uploaded_audio(reference_audio)
        
        # Preprocess reference audio for better cloning results
        from models_app.ai_models.utils.audio.processing import normalize_audio, convert_to_mono
        
        # Convert to mono if needed
        reference_audio_processed = convert_to_mono(audio_path)
        
        # Normalize audio for consistent voice cloning
        reference_audio_processed = normalize_audio(reference_audio_processed)
        
        # Validate audio quality before using for cloning
        quality = assess_audio_quality(reference_audio_processed)
        if not quality.get('is_suitable_for_tts', True):
            logger.warning(f"Reference audio might not be suitable for voice cloning: {quality}")
        
        # Get TTS factory
        factory = TTSFactory()
        engine = TTSEngine(engine_id)
        
        # Clone voice and synthesize with post-processing
        output_path = factory.synthesize_speech(
            text, 
            engine=engine,
            apply_post_processing=apply_post_processing,
            normalize_volume=normalize_volume,
            remove_silence=remove_silence,
            target_format=target_format,
            reference_audio=reference_audio_processed
        )
        
        # Clean up temporary files
        try:
            os.unlink(audio_path)
            if reference_audio_processed != audio_path:
                os.unlink(reference_audio_processed)
        except:
            pass
        
        # Return results
        if output_path:
            # Get quality metrics for the generated speech
            quality_metrics = factory.assess_speech_quality(output_path)
            
            return JsonResponse({
                'success': True,
                'file_url': f"/media/tts_output/{os.path.basename(output_path)}",
                'engine': engine_id,
                'format': target_format,
                'quality': {
                    'duration': quality_metrics.get('duration', 0),
                    'sample_rate': quality_metrics.get('sample_rate', 0),
                    'average_loudness_db': quality_metrics.get('average_loudness_db', 0)
                }
            })
        else:
            return JsonResponse({'error': 'Failed to clone voice'}, status=500)
        
    except Exception as e:
        logger.error(f"Error in voice cloning: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def optimize_audio(request):
    """Optimize an audio file for web playback"""
    try:
        if 'audio_file' not in request.FILES:
            return JsonResponse({'error': 'Audio file is required'}, status=400)
        
        audio_file = request.FILES['audio_file']
        target_format = request.data.get('target_format', 'mp3')
        normalize = request.data.get('normalize', True)
        
        # Save uploaded file
        audio_path = handle_uploaded_audio(audio_file)
        
        # Optimize audio using factory
        factory = TTSFactory()
        optimized_path = factory.optimize_audio_for_playback(
            audio_path,
            target_format=target_format,
            normalize=normalize
        )
        
        # Clean up original file
        try:
            os.unlink(audio_path)
        except:
            pass
            
        if optimized_path:
            return JsonResponse({
                'success': True,
                'file_url': f"/media/tts_output/{os.path.basename(optimized_path)}",
                'format': target_format
            })
        else:
            return JsonResponse({'error': 'Failed to optimize audio'}, status=500)
            
    except Exception as e:
        logger.error(f"Error in audio optimization: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

# Helper functions
def get_engine_features(engine):
    """Get features for a TTS engine"""
    features = []
    
    if engine == TTSEngine.SPARK:
        features = ['voice_cloning', 'custom_voices', 'bilingual']
    elif engine == TTSEngine.COQUI:
        features = ['multiple_languages', 'custom_voices']
    elif engine == TTSEngine.MOZILLA:
        features = ['multiple_languages', 'multispeaker']
    
    return features

def handle_uploaded_audio(audio_file):
    """Save uploaded audio file to temporary location"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        return tmp.name
