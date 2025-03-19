from PIL import Image, ImageOps, ImageFile, UnidentifiedImageError
import numpy as np
import base64
import io
import re
import requests
import logging
import os
import time
from functools import wraps
from typing import List, Union, Optional, Tuple, Dict, Any, Callable

# Import error types for proper error handling
from models_app.ai_models.utils.common.errors import ImageProcessingError, MultiImageProcessingError

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# Cache to store already processed images (URL/path -> processed image)
_image_cache = {}
_cache_expiry = {}
_MAX_CACHE_SIZE = 100
_CACHE_TTL = 300  # 5 minutes

def with_timing(func: Callable) -> Callable:
    """Decorator to time image processing functions and log slow operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Log slow operations (more than 500ms)
        if elapsed_time > 0.5:
            logger.warning(f"Slow image processing operation: {func.__name__} took {elapsed_time:.2f}s")
        
        return result
    return wrapper

def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Convert a base64 string to a PIL Image object.
    
    Args:
        base64_string: The base64 encoded image data
        
    Returns:
        PIL.Image.Image: The decoded image
        
    Raises:
        ImageProcessingError: If decoding fails
    """
    try:
        # Remove header if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Remove whitespace or newlines that might break decoding
        base64_string = re.sub(r'\s+', '', base64_string)
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')  # Ensure consistent format
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ImageProcessingError(f"Failed to decode base64 image: {str(e)}", cause=e)

@with_timing
def encode_image_to_base64(image: Union[Image.Image, str], format: str = "JPEG") -> str:
    """
    Convert a PIL Image or image path to a base64 string.
    
    Args:
        image: The PIL Image object or path to image file
        format: The image format for encoding
        
    Returns:
        str: Base64 encoded image data
        
    Raises:
        ImageProcessingError: If encoding fails
    """
    try:
        if isinstance(image, str) and os.path.exists(image):
            image = Image.open(image)
        
        # Ensure image is in RGB format (handles RGBA, CMYK, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        buffered = io.BytesIO()
        image.save(buffered, format=format, quality=85)  # Reduced quality for smaller payload
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise ImageProcessingError(f"Failed to encode image to base64: {str(e)}", cause=e)

@with_timing
def resize_image_for_model(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resize image to target dimensions while preserving aspect ratio.
    
    Args:
        image: The PIL Image to resize
        target_size: The target dimensions (width, height)
        
    Returns:
        PIL.Image.Image: The resized image
        
    Raises:
        ImageProcessingError: If resizing fails
    """
    try:
        orig_width, orig_height = image.size
        ratio = min(target_size[0]/orig_width, target_size[1]/orig_height)
        
        # Check for very small images that might lose detail when resized
        if orig_width < 32 or orig_height < 32:
            logger.warning(f"Very small image ({orig_width}x{orig_height}) being resized")
        
        new_size = (int(orig_width*ratio), int(orig_height*ratio))
        resized = image.resize(new_size, Image.LANCZOS)
        
        # Create a black canvas of target size
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        # Paste resized image at center
        new_img.paste(resized, ((target_size[0]-new_size[0])//2, (target_size[1]-new_size[1])//2))
        
        return new_img
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise ImageProcessingError(f"Failed to resize image: {str(e)}", cause=e)

def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract image URLs from text.
    
    Args:
        text: The text to search for image URLs
        
    Returns:
        List[str]: A list of extracted image URLs
        
    Raises:
        ImageProcessingError: If URL extraction fails
    """
    try:
        url_pattern = r'https?://\S+(?:png|jpg|jpeg|gif|webp)'
        return re.findall(url_pattern, text)
    except Exception as e:
        logger.error(f"Error extracting URLs from text: {e}")
        raise ImageProcessingError(f"Failed to extract URLs from text: {str(e)}", cause=e)

@with_timing
def support_multiple_images(images_list: List[Union[str, Image.Image, bytes]], max_images: int = 5) -> List[Image.Image]:
    """
    Process multiple images for models that support multi-image input.
    
    Args:
        images_list: List of images (paths, URLs, PIL Images, bytes)
        max_images: Maximum number of images to process
        
    Returns:
        List[PIL.Image.Image]: List of processed PIL Images
        
    Raises:
        MultiImageProcessingError: If processing multiple images fails
    """
    try:
        if not images_list:
            raise MultiImageProcessingError("Empty image list provided")
            
        if len(images_list) > max_images:
            logger.warning(f"Too many images provided ({len(images_list)}). Using first {max_images} images.")
            images_list = images_list[:max_images]
        
        processed_images = []
        failed_images = 0
        errors = []
        
        for idx, img in enumerate(images_list):
            try:
                if isinstance(img, str):
                    # Check if image is already in cache
                    if img in _image_cache and time.time() < _cache_expiry.get(img, 0):
                        processed_images.append(_image_cache[img])
                        continue
                        
                    # Handle URL or base64 string
                    if img.startswith(('http://', 'https://')):
                        # Download and process image from URL
                        processed_img = download_and_process_image(img)
                    elif img.startswith('data:image'):
                        # Process base64 string
                        processed_img = decode_base64_image(img)
                    elif os.path.exists(img):
                        # Load from file path
                        processed_img = Image.open(img).convert('RGB')
                    else:
                        # Try as base64 without proper header
                        processed_img = decode_base64_image(img)
                    
                    # Add to cache with expiry
                    _add_to_cache(img, processed_img)
                    processed_images.append(processed_img)
                    
                elif isinstance(img, bytes):
                    # Process bytes
                    processed_img = Image.open(io.BytesIO(img)).convert('RGB')
                    processed_images.append(processed_img)
                    
                elif isinstance(img, Image.Image):
                    # Already a PIL Image
                    processed_images.append(img.convert('RGB'))
                    
                else:
                    err_msg = f"Unsupported image type at index {idx}: {type(img)}"
                    logger.warning(err_msg)
                    errors.append(err_msg)
                    failed_images += 1
                    
            except Exception as e:
                err_msg = f"Error processing image at index {idx}: {str(e)}"
                logger.error(err_msg)
                errors.append(err_msg)
                failed_images += 1
        
        if not processed_images:
            raise MultiImageProcessingError(
                f"Failed to process any images from the provided list. Errors: {'; '.join(errors)}"
            )
            
        if failed_images > 0:
            logger.warning(f"Failed to process {failed_images} out of {len(images_list)} images")
            
        return processed_images
    except MultiImageProcessingError:
        # Re-raise MultiImageProcessingError directly
        raise
    except Exception as e:
        logger.error(f"Error processing multiple images: {e}")
        raise MultiImageProcessingError(f"Failed to process multiple images: {str(e)}", cause=e)

@with_timing
def handle_high_resolution_image(image: Image.Image, method: str = "tile", 
                                 tile_size: int = 512, max_tiles: int = 6) -> Union[Image.Image, List[Image.Image]]:
    """
    Handle high-resolution images using tiling or downscaling.
    
    Args:
        image: The PIL Image to process
        method: The method to use ('tile' or 'resize')
        tile_size: The size of each tile when using tiling
        max_tiles: Maximum number of tiles to return
        
    Returns:
        Union[PIL.Image.Image, List[PIL.Image.Image]]: Processed image or list of tiles
        
    Raises:
        ImageProcessingError: If processing fails
    """
    try:
        width, height = image.size
        
        # Very large images - warn and handle specially
        if width * height > 25000000:  # 25 megapixels
            logger.warning(f"Very large image: {width}x{height} pixels")
            
            # Force resize method for extremely large images
            if width * height > 100000000:  # 100 megapixels
                logger.warning(f"Image too large for tiling, forcing resize")
                method = "resize"
        
        if method == "tile" and (width > tile_size or height > tile_size):
            # Create image tiles for models that support tiling
            tiles = []
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                    tile = image.crop(box)
                    tiles.append(tile)
            
            # Limit number of tiles if necessary
            if len(tiles) > max_tiles:
                logger.info(f"Image produced {len(tiles)} tiles; limiting to {max_tiles}")
                # Select tiles strategically (e.g., center and corners)
                center_idx = len(tiles) // 2
                corner_idxs = [0, len(tiles) - 1]
                
                # Prioritize center and corners, then evenly space other tiles
                selected_idxs = corner_idxs + [center_idx]
                remaining_count = max_tiles - len(selected_idxs)
                
                if remaining_count > 0:
                    step = len(tiles) // (remaining_count + 1)
                    for i in range(step, len(tiles), step):
                        if len(selected_idxs) < max_tiles and i not in selected_idxs:
                            selected_idxs.append(i)
                
                tiles = [tiles[i] for i in sorted(selected_idxs[:max_tiles])]
            
            return tiles
        else:
            # Simple downscaling for models that don't support tiling
            return resize_image_for_model(image)
    except Exception as e:
        logger.error(f"Error handling high-resolution image: {e}")
        raise ImageProcessingError(f"Failed to handle high-resolution image: {str(e)}", cause=e)

@with_timing
def download_and_process_image(url: str) -> Image.Image:
    """
    Download an image from a URL and convert to PIL Image.
    
    Args:
        url: The URL of the image to download
        
    Returns:
        PIL.Image.Image: The downloaded image
        
    Raises:
        ImageProcessingError: If download or processing fails
    """
    # Check if image is in cache
    if url in _image_cache and time.time() < _cache_expiry.get(url, 0):
        return _image_cache[url]
    
    try:
        # Set reasonable timeouts and headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
        }
        response = requests.get(url, stream=True, timeout=10, headers=headers)
        response.raise_for_status()  # Will raise an exception for 4XX/5XX responses
        
        # Check content type to ensure it's an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            raise ImageProcessingError(f"URL does not point to an image: {content_type}")
        
        # Process the image
        try:
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Add to cache
            _add_to_cache(url, img)
            
            return img
        except UnidentifiedImageError:
            raise ImageProcessingError(f"Could not identify image format from URL: {url}")
            
    except requests.RequestException as e:
        logger.error(f"Error downloading image from URL {url}: {e}")
        raise ImageProcessingError(f"Failed to download image from URL: {str(e)}", cause=e)
    except Exception as e:
        logger.error(f"Error processing downloaded image from URL {url}: {e}")
        raise ImageProcessingError(f"Failed to process downloaded image: {str(e)}", cause=e)

def validate_image(image: Union[str, Image.Image, bytes]) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validate an image to check if it's valid and process-able.
    
    Args:
        image: The image to validate (path, URL, PIL Image, bytes)
        
    Returns:
        Tuple[bool, str, Optional[PIL.Image.Image]]: (is_valid, error_message, processed_image)
    """
    try:
        # Process based on input type
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                # URL validation
                processed = download_and_process_image(image)
            elif image.startswith('data:image'):
                # Base64 image
                processed = decode_base64_image(image)
            elif os.path.exists(image):
                # File path
                processed = Image.open(image).convert('RGB')
            else:
                # Try as base64 without proper header
                processed = decode_base64_image(image)
        elif isinstance(image, bytes):
            processed = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            processed = image.convert('RGB')
        else:
            return False, f"Unsupported image type: {type(image)}", None
        
        # Additional validation checks
        width, height = processed.size
        if width < 16 or height < 16:
            return False, f"Image too small: {width}x{height}", processed
            
        if width > 10000 or height > 10000:
            logger.warning(f"Very large image: {width}x{height}")
            # We allow large images but warn about them
            
        # Check if image isn't just a solid color
        try:
            colors = processed.getcolors(256)
            if colors is not None and len(colors) < 3:
                return False, "Image has too few colors, might be blank or corrupt", processed
        except:
            # If getcolors fails, the image has more than 256 colors which is fine
            pass
            
        return True, "", processed
        
    except Exception as e:
        return False, str(e), None

def fix_corrupt_image(image: Union[Image.Image, bytes]) -> Image.Image:
    """
    Attempt to fix a corrupt image.
    
    Args:
        image: The corrupt image (PIL Image or bytes)
        
    Returns:
        PIL.Image.Image: The fixed image if possible
        
    Raises:
        ImageProcessingError: If fixing fails
    """
    try:
        if isinstance(image, bytes):
            # Try to open with PIL's recover mode
            try:
                return Image.open(io.BytesIO(image))
            except:
                pass
            
            # Try different formats if the standard one fails
            for format_name in ['JPEG', 'PNG', 'GIF', 'BMP']:
                try:
                    img = Image.open(io.BytesIO(image), formats=[format_name])
                    return img.convert('RGB')
                except:
                    continue
                    
            raise ImageProcessingError("Could not recover corrupt image data")
            
        elif isinstance(image, Image.Image):
            # If it's already a PIL Image, it's not corrupt at the data level
            # But we can try to fix visual corruption
            
            # Convert to RGB if needed
            img = image.convert('RGB')
            
            # Try to fix by re-saving to a new buffer
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            fixed = Image.open(buffer)
            
            return fixed
            
        else:
            raise ImageProcessingError(f"Unsupported image type for fixing: {type(image)}")
            
    except Exception as e:
        logger.error(f"Failed to fix corrupt image: {e}")
        raise ImageProcessingError(f"Failed to fix corrupt image: {str(e)}", cause=e)

def _add_to_cache(key: str, image: Image.Image) -> None:
    """Add an image to the cache with expiry time."""
    # Remove oldest items if cache is full
    if len(_image_cache) >= _MAX_CACHE_SIZE:
        oldest_key = min(_cache_expiry, key=_cache_expiry.get)
        _image_cache.pop(oldest_key, None)
        _cache_expiry.pop(oldest_key, None)
    
    # Add to cache with expiry
    _image_cache[key] = image
    _cache_expiry[key] = time.time() + _CACHE_TTL

def clear_image_cache() -> None:
    """Clear the image processing cache."""
    global _image_cache, _cache_expiry
    _image_cache = {}
    _cache_expiry = {}
