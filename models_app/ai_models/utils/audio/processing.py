import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import wave
import contextlib
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Generator, BinaryIO
import io

def convert_audio_format(audio_path, target_format="wav", sample_rate=16000):
    """Convert audio to target format with specific sample rate."""
    # Handle different input formats
    audio = AudioSegment.from_file(audio_path)
    
    # Create temp file path with new extension
    base_path = os.path.splitext(audio_path)[0]
    output_path = f"{base_path}.{target_format}"
    
    # Resample and export
    audio = audio.set_frame_rate(sample_rate)
    audio.export(output_path, format=target_format)
    
    return output_path

def split_audio_on_silence(audio_path, min_silence_len=500, silence_thresh=-40):
    """Split audio on silence for better processing."""
    audio = AudioSegment.from_file(audio_path)
    chunks = librosa.effects.split(
        np.array(audio.get_array_of_samples()),
        top_db=abs(silence_thresh),
        frame_length=min_silence_len
    )
    
    # Create output files for each chunk
    base_path = os.path.splitext(audio_path)[0]
    chunk_paths = []
    
    for i, (start, end) in enumerate(chunks):
        chunk = audio[start:end]
        chunk_path = f"{base_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    
    return chunk_paths

def normalize_audio(audio_path, target_db=-20.0):
    """Normalize audio volume to target dB."""
    audio = AudioSegment.from_file(audio_path)
    
    # Calculate difference and adjust
    change = target_db - audio.dBFS
    normalized = audio.apply_gain(change)
    
    # Save normalized audio
    output_path = f"{os.path.splitext(audio_path)[0]}_normalized.wav"
    normalized.export(output_path, format="wav")
    
    return output_path

def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds."""
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def convert_to_mono(audio_path):
    """Convert stereo audio to mono by averaging channels."""
    audio = AudioSegment.from_file(audio_path)
    mono_audio = audio.set_channels(1)
    
    output_path = f"{os.path.splitext(audio_path)[0]}_mono.wav"
    mono_audio.export(output_path, format="wav")
    
    return output_path

def detect_speech_segments(audio_path, frame_duration_ms=30, min_speech_duration_ms=300):
    """
    Detect segments containing speech using voice activity detection.
    
    Args:
        audio_path: Path to audio file
        frame_duration_ms: Duration of each frame for analysis
        min_speech_duration_ms: Minimum duration to consider as speech
        
    Returns:
        List of tuples containing (start_time, end_time) in seconds
    """
    try:
        import webrtcvad
        vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        
        # Load audio
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Convert to PCM s16le format required by webrtcvad
        pcm_audio = (audio * 32767).astype(np.int16).tobytes()
        
        # Calculate frame size
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        frames = [pcm_audio[i:i + frame_size] for i in range(0, len(pcm_audio), frame_size) if i + frame_size <= len(pcm_audio)]
        
        # Detect speech
        is_speech = []
        for frame in frames:
            is_speech.append(vad.is_speech(frame, sample_rate))
        
        # Group speech frames
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Speech started
                in_speech = True
                speech_start = i
            elif not speech and in_speech:
                # Speech ended
                in_speech = False
                duration = i - speech_start
                if duration * frame_duration_ms >= min_speech_duration_ms:
                    speech_segments.append((
                        speech_start * frame_duration_ms / 1000, 
                        i * frame_duration_ms / 1000
                    ))
        
        # Handle if audio ends during speech
        if in_speech:
            duration = len(is_speech) - speech_start
            if duration * frame_duration_ms >= min_speech_duration_ms:
                speech_segments.append((
                    speech_start * frame_duration_ms / 1000, 
                    len(is_speech) * frame_duration_ms / 1000
                ))
        
        return speech_segments
        
    except ImportError:
        # Fallback to simpler method if webrtcvad not available
        audio, sr = librosa.load(audio_path, sr=None)
        intervals = librosa.effects.split(
            audio, top_db=20, frame_length=1024, hop_length=256
        )
        
        segments = []
        for start, end in intervals:
            segments.append((start/sr, end/sr))
        
        return segments

def extract_audio_features(audio_path, n_mfcc=13):
    """
    Extract audio features useful for machine learning models.
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients to extract
        
    Returns:
        Dictionary with audio features
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Calculate statistics for each feature
    features = {
        'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
        'mfcc_var': np.var(mfccs, axis=1).tolist(),
        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
        'spectral_centroid_var': float(np.var(spectral_centroid)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_rolloff_var': float(np.var(spectral_rolloff)),
        'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
        'zero_crossing_rate_var': float(np.var(zero_crossing_rate)),
    }
    
    return features

def trim_silence(audio_path, silence_threshold_db=-50):
    """
    Trim silence from beginning and end of audio file.
    
    Args:
        audio_path: Path to audio file
        silence_threshold_db: Threshold for silence detection in dB
        
    Returns:
        Path to trimmed audio file
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    
    # Trim silence
    start_trim = 0
    end_trim = 0
    
    # Find start trim
    for i in range(len(audio)):
        if audio[i:i+1].dBFS > silence_threshold_db:
            start_trim = i
            break
    
    # Find end trim (going backwards)
    for i in range(len(audio)-1, 0, -1):
        if audio[i:i+1].dBFS > silence_threshold_db:
            end_trim = i + 1
            break
    
    # Trim the audio
    trimmed_audio = audio[start_trim:end_trim]
    
    # Save trimmed audio
    output_path = f"{os.path.splitext(audio_path)[0]}_trimmed.wav"
    trimmed_audio.export(output_path, format="wav")
    
    return output_path

def assess_audio_quality(audio_path):
    """
    Assess basic quality metrics of an audio file.
    Useful for determining if audio is suitable for STT or needs preprocessing.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with quality metrics
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    y, sr = librosa.load(audio_path)
    
    # Calculate metrics
    duration = len(audio) / 1000.0  # in seconds
    sample_rate = audio.frame_rate
    bit_depth = audio.sample_width * 8
    channels = audio.channels
    
    # Calculate signal-to-noise ratio (simple estimation)
    signal_power = np.mean(y**2)
    
    # Get sections with low energy (potential noise)
    noise_threshold = np.percentile(librosa.feature.rms(y=y)[0], 10)
    noise_mask = librosa.feature.rms(y=y)[0] <= noise_threshold
    noise_indices = np.where(noise_mask)[0]
    
    if len(noise_indices) > 0:
        noise = librosa.feature.rms(y=y)[0][noise_indices]
        noise_power = np.mean(noise**2) if len(noise) > 0 else 1e-10
        estimated_snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
    else:
        estimated_snr = 100  # No noise detected
    
    # Calculate dynamic range
    dynamic_range = audio.max_dBFS - audio.dBFS
    
    quality_metrics = {
        'duration': duration,
        'sample_rate': sample_rate,
        'bit_depth': bit_depth,
        'channels': channels,
        'average_loudness_db': audio.dBFS,
        'max_loudness_db': audio.max_dBFS,
        'dynamic_range_db': dynamic_range,
        'estimated_snr_db': estimated_snr,
        'is_suitable_for_stt': sample_rate >= 16000 and bit_depth >= 16 and estimated_snr > 15,
        'is_suitable_for_tts': sample_rate >= 22050 and bit_depth >= 16 and estimated_snr > 20
    }
    
    return quality_metrics

def change_speech_rate(audio_path: str, rate_factor: float = 1.0) -> str:
    """
    Change the speech rate of an audio file without changing the pitch.
    
    Args:
        audio_path: Path to the audio file
        rate_factor: Rate factor (1.0 = original, 0.5 = half speed, 2.0 = double speed)
        
    Returns:
        Path to the modified audio file
    """
    if rate_factor == 1.0:
        return audio_path

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Time stretch without changing pitch
    modified = librosa.effects.time_stretch(y, rate=rate_factor)
    
    # Save modified audio
    output_path = f"{os.path.splitext(audio_path)[0]}_rate{rate_factor}.wav"
    sf.write(output_path, modified, sr)
    
    return output_path

def change_pitch(audio_path: str, semitones: float = 0.0) -> str:
    """
    Change the pitch of an audio file without changing the speech rate.
    
    Args:
        audio_path: Path to the audio file
        semitones: Number of semitones to shift pitch (positive = higher, negative = lower)
        
    Returns:
        Path to the modified audio file
    """
    if semitones == 0.0:
        return audio_path
        
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Pitch shift
    modified = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    
    # Save modified audio
    output_path = f"{os.path.splitext(audio_path)[0]}_pitch{semitones}.wav"
    sf.write(output_path, modified, sr)
    
    return output_path

def compare_voices(reference_path: str, synthesized_path: str) -> Dict[str, float]:
    """
    Compare similarity between reference and synthesized voices.
    
    Args:
        reference_path: Path to the reference audio file
        synthesized_path: Path to the synthesized audio file
        
    Returns:
        Dictionary with similarity metrics
    """
    # Load audio files
    y_ref, sr_ref = librosa.load(reference_path, sr=None)
    y_syn, sr_syn = librosa.load(synthesized_path, sr=None)
    
    # Resample if needed
    if sr_ref != sr_syn:
        y_syn = librosa.resample(y_syn, orig_sr=sr_syn, target_sr=sr_ref)
        sr_syn = sr_ref
    
    # Extract features
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13)
    mfcc_syn = librosa.feature.mfcc(y=y_syn, sr=sr_syn, n_mfcc=13)
    
    # Adjust dimensions if different lengths
    min_len = min(mfcc_ref.shape[1], mfcc_syn.shape[1])
    mfcc_ref = mfcc_ref[:, :min_len]
    mfcc_syn = mfcc_syn[:, :min_len]
    
    # Calculate similarity metrics
    mfcc_distance = np.mean(np.sqrt(np.sum((mfcc_ref - mfcc_syn) ** 2, axis=0)))
    
    # Calculate pitch contours
    pitches_ref, _ = librosa.piptrack(y=y_ref, sr=sr_ref)
    pitches_syn, _ = librosa.piptrack(y=y_syn, sr=sr_syn)
    
    # Adjust dimensions
    min_len = min(pitches_ref.shape[1], pitches_syn.shape[1])
    pitches_ref = pitches_ref[:, :min_len]
    pitches_syn = pitches_syn[:, :min_len]
    
    # Calculate median pitch for each frame
    pitch_ref = np.median(pitches_ref, axis=0)
    pitch_syn = np.median(pitches_syn, axis=0)
    
    # Calculate pitch similarity (ignore zeros)
    nonzero_indices = (pitch_ref > 0) & (pitch_syn > 0)
    if np.any(nonzero_indices):
        pitch_distance = np.mean(np.abs(pitch_ref[nonzero_indices] - pitch_syn[nonzero_indices]))
    else:
        pitch_distance = float('inf')
    
    # Calculate overall similarity score (0-100)
    if mfcc_distance > 0 and pitch_distance < float('inf'):
        mfcc_score = 100 * np.exp(-0.1 * mfcc_distance)
        pitch_score = 100 * np.exp(-0.01 * pitch_distance)
        overall_score = 0.7 * mfcc_score + 0.3 * pitch_score
    else:
        overall_score = 0.0
    
    return {
        'mfcc_distance': float(mfcc_distance),
        'pitch_distance': float(pitch_distance) if pitch_distance < float('inf') else None,
        'overall_similarity': float(overall_score),
        'is_same_speaker': overall_score > 75.0
    }

def detect_speech_segments_with_transcription_timestamps(audio_path: str) -> List[Dict[str, Union[float, str]]]:
    """
    Advanced speech segment detection that returns segments suitable for transcription.
    Useful for both STT preprocessing and TTS quality verification.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        List of dictionaries with start_time, end_time, and duration for each speech segment
    """
    # Basic speech detection
    segments = detect_speech_segments(audio_path)
    
    # Load audio for feature extraction
    y, sr = librosa.load(audio_path, sr=None)
    
    result = []
    for i, (start, end) in enumerate(segments):
        # Calculate segment properties
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        # Safety check
        if start_sample >= end_sample or start_sample >= len(y) or end_sample > len(y):
            continue
            
        segment_audio = y[start_sample:end_sample]
        
        # Calculate audio features
        if len(segment_audio) > 0:
            rms = np.sqrt(np.mean(segment_audio**2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(segment_audio))
            
            result.append({
                'index': i,
                'start_time': start,
                'end_time': end,
                'duration': end - start,
                'avg_amplitude': float(rms),
                'zero_crossing_rate': float(zcr),
                'likely_speech': rms > 0.01 and 0.01 < zcr < 0.1
            })
    
    return result

def generate_audio_chunks(audio_path: str, chunk_size_ms: int = 1000) -> Generator[bytes, None, None]:
    """
    Stream an audio file in chunks. Useful for streaming TTS output or processing STT input.
    
    Args:
        audio_path: Path to the audio file
        chunk_size_ms: Size of each chunk in milliseconds
        
    Yields:
        Audio chunks as bytes
    """
    audio = AudioSegment.from_file(audio_path)
    
    # Convert ms to bytes
    chunk_length = int(chunk_size_ms * audio.frame_rate * audio.sample_width * audio.channels / 1000)
    
    # Create in-memory representation
    buffer = io.BytesIO()
    audio.export(buffer, format='wav')
    buffer.seek(0)
    
    # Read and yield chunks
    while True:
        chunk = buffer.read(chunk_length)
        if not chunk:
            break
        yield chunk

def mix_audio_files(audio_files: List[str], weights: Optional[List[float]] = None) -> str:
    """
    Mix multiple audio files, optionally with different weights.
    Useful for adding background effects to TTS output or preprocessing STT input.
    
    Args:
        audio_files: List of paths to audio files
        weights: List of weights for each audio file (default: equal weights)
        
    Returns:
        Path to the mixed audio file
    """
    if not audio_files:
        raise ValueError("No audio files provided")
    
    # Load first audio to get parameters
    primary = AudioSegment.from_file(audio_files[0])
    
    # Use equal weights if not provided
    if weights is None:
        weights = [1.0 / len(audio_files)] * len(audio_files)
    elif len(weights) != len(audio_files):
        raise ValueError("Number of weights must match number of audio files")
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Create empty result with same parameters as primary
    result = AudioSegment.silent(duration=len(primary), frame_rate=primary.frame_rate)
    
    # Mix audio files
    for audio_path, weight in zip(audio_files, weights):
        audio = AudioSegment.from_file(audio_path)
        
        # Ensure same length
        if len(audio) < len(primary):
            audio = audio + AudioSegment.silent(duration=len(primary) - len(audio))
        elif len(audio) > len(primary):
            audio = audio[:len(primary)]
        
        # Apply weight
        audio = audio - (20 * np.log10(1.0 / weight))  # Convert weight to dB
        
        # Mix
        result = result.overlay(audio)
    
    # Export result
    output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    result.export(output_path, format='wav')
    
    return output_path

def remove_background_noise(audio_path: str, noise_reduction_level: float = 0.5) -> str:
    """
    Remove background noise from audio file. Useful for both TTS reference audio
    and STT input preprocessing.
    
    Args:
        audio_path: Path to the audio file
        noise_reduction_level: Level of noise reduction (0.0-1.0)
        
    Returns:
        Path to the cleaned audio file
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Separate harmonics and percussives
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Keep primarily the harmonic part (speech) and some percussive
    y_clean = y_harmonic + y_percussive * (1.0 - noise_reduction_level)
    
    # Save cleaned audio
    output_path = f"{os.path.splitext(audio_path)[0]}_clean.wav"
    sf.write(output_path, y_clean, sr)
    
    return output_path

def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio track from a video file.
    Useful for processing video inputs in STT workflows.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Path to the extracted audio file
    """
    try:
        from moviepy.editor import VideoFileClip
        
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Save to temporary file
        output_path = f"{os.path.splitext(video_path)[0]}_audio.wav"
        audio.write_audiofile(output_path)
        
        return output_path
        
    except ImportError:
        # Fallback to ffmpeg if available
        import subprocess
        
        output_path = f"{os.path.splitext(video_path)[0]}_audio.wav"
        
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '2', output_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return output_path
            
        except subprocess.CalledProcessError:
            raise RuntimeError("Failed to extract audio from video. Ensure ffmpeg is installed.")
