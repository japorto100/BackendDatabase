"""
Text processing utilities for Speech-to-Text (STT) and Text-to-Speech (TTS) applications.

This module provides functions for handling text normalization, SSML processing,
language detection, and other text-related utilities useful for both STT and TTS.
"""

import re
import json
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple, Union, Set
import os

# Optional imports that enhance functionality
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory
    # Make language detection deterministic
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import phonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Common abbreviations and their expansions
COMMON_ABBREVIATIONS = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "Ms.": "Miss",
    "Prof.": "Professor",
    "etc.": "etcetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "vs.": "versus",
    "approx.": "approximately",
    # Add more as needed
}

# Number-to-word mapping
NUMBER_WORDS = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
    16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',
    30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy',
    80: 'eighty', 90: 'ninety'
}

def normalize_text(text: str) -> str:
    """
    Apply basic text normalization for text-to-speech or speech-to-text processing.
    
    This includes:
    - Converting to lowercase
    - Trimming whitespace
    - Normalizing unicode characters
    - Removing multiple spaces
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text

def expand_abbreviations(text: str, custom_abbreviations: Optional[Dict[str, str]] = None) -> str:
    """
    Expand common abbreviations in text.
    
    Args:
        text: Input text
        custom_abbreviations: Additional custom abbreviations to expand
        
    Returns:
        Text with expanded abbreviations
    """
    # Combine standard and custom abbreviations
    abbreviations = COMMON_ABBREVIATIONS.copy()
    if custom_abbreviations:
        abbreviations.update(custom_abbreviations)
    
    # Replace abbreviations
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)
    
    return text

def normalize_numbers(text: str, language: str = 'en') -> str:
    """
    Convert numbers to their word equivalent.
    
    Args:
        text: Input text
        language: Language code (currently only supports 'en')
        
    Returns:
        Text with numbers converted to words
    """
    # Simple number replacement for English
    if language != 'en':
        logger.warning(f"Number normalization for language '{language}' is not supported")
        return text
    
    def _replace_number(match):
        num = int(match.group(0))
        
        if num <= 20:
            return NUMBER_WORDS.get(num, str(num))
        
        if num < 100:
            tens = (num // 10) * 10
            units = num % 10
            if units == 0:
                return NUMBER_WORDS[tens]
            return f"{NUMBER_WORDS[tens]} {NUMBER_WORDS[units]}"
            
        # For larger numbers, keep them as is
        return str(num)
    
    # Replace standalone numbers
    pattern = r'\b\d+\b'
    return re.sub(pattern, _replace_number, text)

def normalize_dates(text: str, language: str = 'en') -> str:
    """
    Convert date formats to TTS-friendly text.
    
    Args:
        text: Input text
        language: Language code (currently only supports 'en')
        
    Returns:
        Text with normalized dates
    """
    if language != 'en':
        logger.warning(f"Date normalization for language '{language}' is not supported")
        return text
    
    # Convert MM/DD/YYYY to "Month DD, YYYY"
    def _replace_date(match):
        month, day, year = match.groups()
        
        month_names = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]
        
        try:
            month_int = int(month)
            if 1 <= month_int <= 12:
                month_name = month_names[month_int - 1]
                return f"{month_name} {int(day)}, {year}"
        except:
            pass
        
        return match.group(0)
    
    # Replace dates in MM/DD/YYYY format
    pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
    return re.sub(pattern, _replace_date, text)

def is_ssml(text: str) -> bool:
    """
    Detect if text contains SSML markup.
    
    Args:
        text: Input text
        
    Returns:
        True if text appears to contain SSML markup
    """
    # Simple check for common SSML tags
    ssml_patterns = [
        r'<speak\b[^>]*>.*?</speak>',
        r'<voice\b[^>]*>.*?</voice>',
        r'<prosody\b[^>]*>.*?</prosody>',
        r'<break\b[^>]*/>',
        r'<say-as\b[^>]*>.*?</say-as>',
        r'<phoneme\b[^>]*>.*?</phoneme>'
    ]
    
    for pattern in ssml_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    
    return False

def validate_ssml(ssml_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate SSML markup and fix common issues.
    
    Args:
        ssml_text: Input text with SSML markup
        
    Returns:
        Tuple containing:
        - Boolean indicating if the SSML is valid
        - Fixed SSML if fixable, otherwise None
        - Error message if invalid, otherwise None
    """
    # Check if wrapped in <speak> tags
    if not ssml_text.strip().startswith('<speak') or not ssml_text.strip().endswith('</speak>'):
        fixed_ssml = f"<speak>{ssml_text}</speak>"
        return False, fixed_ssml, "Added missing <speak> tags"
    
    # Check for unclosed tags
    opening_tags = re.findall(r'<(\w+)[^>]*>', ssml_text)
    closing_tags = re.findall(r'</(\w+)>', ssml_text)
    
    # Self-closing tags don't need to be matched
    self_closing_tags = re.findall(r'<(\w+)[^>]*?/>', ssml_text)
    opening_tags = [tag for tag in opening_tags if tag not in self_closing_tags]
    
    # Check if all opening tags have matching closing tags
    if len(opening_tags) != len(closing_tags):
        return False, None, "Mismatched opening and closing tags"
    
    # Check for invalid nesting
    stack = []
    tag_pattern = re.compile(r'<(/?)(\w+)[^>]*?(/?)>')
    
    for match in tag_pattern.finditer(ssml_text):
        is_closing, tag_name, is_self_closing = match.groups()
        
        if is_self_closing:
            continue
        
        if not is_closing:
            stack.append(tag_name)
        else:
            if not stack or stack.pop() != tag_name:
                return False, None, f"Invalid nesting for tag: {tag_name}"
    
    if stack:
        return False, None, f"Unclosed tags: {', '.join(stack)}"
    
    return True, ssml_text, None

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text
        
    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr', 'de')
    """
    if not LANGDETECT_AVAILABLE:
        logger.warning("Language detection requires the 'langdetect' package")
        return 'en'  # Default to English
    
    try:
        return detect(text)
    except:
        logger.warning("Failed to detect language, defaulting to English")
        return 'en'

def detect_language_parts(text: str) -> List[Dict[str, str]]:
    """
    Detect different language segments in multilingual text.
    
    Args:
        text: Input text, potentially containing multiple languages
        
    Returns:
        List of dictionaries with 'text' and 'language' keys
    """
    if not NLTK_AVAILABLE or not LANGDETECT_AVAILABLE:
        logger.warning("Multilingual detection requires 'nltk' and 'langdetect' packages")
        return [{'text': text, 'language': detect_language(text)}]
    
    try:
        # Ensure we have the punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Detect language for each sentence
        result = []
        current_lang = None
        current_text = []
        
        for sentence in sentences:
            try:
                lang = detect(sentence)
                
                if current_lang is None:
                    current_lang = lang
                    current_text.append(sentence)
                elif lang == current_lang:
                    current_text.append(sentence)
                else:
                    # Language changed, add the current segment and start a new one
                    result.append({
                        'text': ' '.join(current_text),
                        'language': current_lang
                    })
                    current_lang = lang
                    current_text = [sentence]
            except:
                # If detection fails, continue with current language
                current_text.append(sentence)
        
        # Add the last segment
        if current_text:
            result.append({
                'text': ' '.join(current_text),
                'language': current_lang or 'en'
            })
        
        return result
    except:
        # Fallback to treating everything as a single language
        return [{'text': text, 'language': detect_language(text)}]

def convert_to_phonemes(text: str, language: str = 'en-us') -> str:
    """
    Convert text to phonemes for better pronunciation control.
    
    Args:
        text: Input text
        language: Language code in format 'en-us', 'de-de', etc.
        
    Returns:
        Text with phonetic transcription
    """
    if not PHONEMIZER_AVAILABLE:
        logger.warning("Phoneme conversion requires the 'phonemizer' package")
        return text
    
    try:
        return phonemizer.phonemize(
            text,
            language=language,
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            with_stress=True
        )
    except:
        logger.warning(f"Failed to phonemize text for language '{language}'")
        return text

def preprocess_text_for_tts(text: str, language: str = 'en', 
                          expand_abbrev: bool = True, 
                          norm_numbers: bool = True,
                          norm_dates: bool = True) -> str:
    """
    Comprehensive text preprocessing for TTS.
    
    Args:
        text: Input text
        language: Language code
        expand_abbrev: Whether to expand abbreviations
        norm_numbers: Whether to normalize numbers
        norm_dates: Whether to normalize dates
        
    Returns:
        Preprocessed text ready for TTS
    """
    # Check if SSML and skip preprocessing if so
    if is_ssml(text):
        valid, fixed_ssml, error = validate_ssml(text)
        if valid:
            return text
        elif fixed_ssml:
            return fixed_ssml
    
    # Apply preprocessing steps
    if expand_abbrev:
        text = expand_abbreviations(text)
    
    if norm_numbers:
        text = normalize_numbers(text, language)
    
    if norm_dates:
        text = normalize_dates(text, language)
    
    # Final normalization
    text = normalize_text(text)
    
    return text

def verify_text_matches_audio(original_text: str, transcribed_text: str) -> Dict[str, Union[float, bool]]:
    """
    Compare original text with transcribed text to evaluate TTS quality.
    
    Args:
        original_text: Original text used for TTS
        transcribed_text: Text transcribed from the TTS audio output
        
    Returns:
        Dictionary with similarity metrics
    """
    # Normalize both texts for comparison
    orig_norm = normalize_text(original_text.lower())
    trans_norm = normalize_text(transcribed_text.lower())
    
    # Calculate word error rate
    if NLTK_AVAILABLE:
        try:
            # Tokenize into words
            orig_words = word_tokenize(orig_norm)
            trans_words = word_tokenize(trans_norm)
            
            # Simple Levenshtein distance for words
            from nltk.metrics.distance import edit_distance
            distance = edit_distance(orig_words, trans_words)
            
            # Word Error Rate
            wer = distance / max(len(orig_words), 1)
            
            # Word match rate
            word_match_set = set(orig_words).intersection(set(trans_words))
            word_match_rate = len(word_match_set) / max(len(set(orig_words)), 1)
            
            return {
                'word_error_rate': wer,
                'word_match_rate': word_match_rate,
                'is_good_match': wer < 0.3 or word_match_rate > 0.7
            }
            
        except:
            pass
    
    # Fallback to character-level comparison
    import difflib
    similarity = difflib.SequenceMatcher(None, orig_norm, trans_norm).ratio()
    
    return {
        'text_similarity': similarity,
        'is_good_match': similarity > 0.7
    }

def extract_spoken_phrases(transcription: str) -> List[Dict[str, str]]:
    """
    Extract spoken phrases from transcription, useful for STT post-processing.
    
    Args:
        transcription: Transcribed text from STT
        
    Returns:
        List of phrase dictionaries with text and speaker field
    """
    if not NLTK_AVAILABLE:
        # Simple splitting by common delimiters
        phrases = re.split(r'[.!?:]', transcription)
        return [{'text': phrase.strip(), 'speaker': 'unknown'} for phrase in phrases if phrase.strip()]
    
    try:
        # Use NLTK for better sentence splitting
        sentences = sent_tokenize(transcription)
        
        # Check for speaker patterns
        speaker_pattern = re.compile(r'^(?:([A-Za-z\s]+)[:]\s*)(.*)')
        
        results = []
        for sentence in sentences:
            match = speaker_pattern.match(sentence)
            if match:
                speaker, text = match.groups()
                results.append({
                    'text': text.strip(),
                    'speaker': speaker.strip()
                })
            else:
                results.append({
                    'text': sentence.strip(),
                    'speaker': 'unknown'
                })
        
        return results
    except:
        # Fallback to simple splitting
        phrases = re.split(r'[.!?:]', transcription)
        return [{'text': phrase.strip(), 'speaker': 'unknown'} for phrase in phrases if phrase.strip()]

def generate_ssml(text: str, voice: Optional[str] = None, rate: Optional[float] = None,
                pitch: Optional[float] = None, volume: Optional[float] = None,
                language: Optional[str] = None) -> str:
    """
    Generate SSML markup from plain text with configurable parameters.
    
    Args:
        text: Plain text input
        voice: Voice name
        rate: Speech rate (0.5-2.0, where 1.0 is normal)
        pitch: Voice pitch (-10 to 10, where 0 is normal)
        volume: Volume (0-100, where 100 is normal)
        language: Language code
        
    Returns:
        Text with SSML markup
    """
    # Start with speak tag
    ssml = "<speak>"
    
    # Add voice if specified
    if voice:
        ssml += f'<voice name="{voice}">'
    
    # Add prosody if any parameters specified
    prosody_attrs = []
    if rate is not None:
        if rate < 0.5:
            rate_str = "x-slow"
        elif rate < 0.8:
            rate_str = "slow"
        elif rate < 1.2:
            rate_str = "medium"
        elif rate < 1.5:
            rate_str = "fast"
        else:
            rate_str = "x-fast"
        prosody_attrs.append(f'rate="{rate_str}"')
    
    if pitch is not None:
        if pitch < -5:
            pitch_str = "x-low"
        elif pitch < -2:
            pitch_str = "low"
        elif pitch < 2:
            pitch_str = "medium"
        elif pitch < 5:
            pitch_str = "high"
        else:
            pitch_str = "x-high"
        prosody_attrs.append(f'pitch="{pitch_str}"')
    
    if volume is not None:
        if volume < 30:
            volume_str = "x-soft"
        elif volume < 70:
            volume_str = "soft"
        elif volume < 90:
            volume_str = "medium"
        elif volume < 110:
            volume_str = "loud"
        else:
            volume_str = "x-loud"
        prosody_attrs.append(f'volume="{volume_str}"')
    
    if prosody_attrs:
        ssml += f'<prosody {" ".join(prosody_attrs)}>'
    
    # Add language if specified
    if language:
        ssml += f'<lang xml:lang="{language}">'
    
    # Add the text content
    ssml += text
    
    # Close tags in reverse order
    if language:
        ssml += '</lang>'
    
    if prosody_attrs:
        ssml += '</prosody>'
    
    if voice:
        ssml += '</voice>'
    
    ssml += '</speak>'
    
    return ssml

def create_caching_key(text: str, voice_id: Optional[str] = None, 
                      options: Optional[Dict[str, any]] = None) -> str:
    """
    Create a unique key for caching TTS output.
    
    Args:
        text: Text to synthesize
        voice_id: Voice ID
        options: Additional options that affect synthesis
        
    Returns:
        String key suitable for caching
    """
    import hashlib
    
    # Normalize text to ensure consistent caching
    normalized_text = normalize_text(text)
    
    # Create key components
    key_parts = [normalized_text]
    
    if voice_id:
        key_parts.append(str(voice_id))
    
    if options:
        # Sort options to ensure consistent ordering
        sorted_options = sorted((str(k), str(v)) for k, v in options.items())
        key_parts.extend(f"{k}={v}" for k, v in sorted_options)
    
    # Join and hash
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def segment_long_text(text: str, max_length: int = 1000) -> List[str]:
    """
    Segment long text into smaller chunks for better TTS processing.
    
    Args:
        text: Input text
        max_length: Maximum length for each segment
        
    Returns:
        List of text segments
    """
    if len(text) <= max_length:
        return [text]
    
    if not NLTK_AVAILABLE:
        # Simple splitting by sentence boundaries
        segments = []
        current_segment = ""
        
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            if len(current_segment) + len(sentence) + 1 <= max_length:
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence
            else:
                if current_segment:
                    segments.append(current_segment)
                
                # If a single sentence is too long, split by commas
                if len(sentence) > max_length:
                    comma_parts = sentence.split(", ")
                    current_segment = comma_parts[0]
                    
                    for part in comma_parts[1:]:
                        if len(current_segment) + len(part) + 2 <= max_length:  # +2 for ", "
                            current_segment += ", " + part
                        else:
                            segments.append(current_segment)
                            current_segment = part
                else:
                    current_segment = sentence
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    # Use NLTK for better sentence splitting
    try:
        sentences = sent_tokenize(text)
        
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) + 1 <= max_length:
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence
            else:
                if current_segment:
                    segments.append(current_segment)
                
                # If a single sentence is too long, split it
                if len(sentence) > max_length:
                    words = word_tokenize(sentence)
                    current_segment = words[0]
                    
                    for word in words[1:]:
                        if len(current_segment) + len(word) + 1 <= max_length:
                            current_segment += " " + word
                        else:
                            segments.append(current_segment)
                            current_segment = word
                else:
                    current_segment = sentence
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    except:
        # Fallback to simple splitting
        return [text[i:i+max_length] for i in range(0, len(text), max_length)] 