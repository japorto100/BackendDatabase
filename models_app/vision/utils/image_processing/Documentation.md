# Image Processing Utilities for Vision Processing

This module provides comprehensive image processing utilities for document analysis, OCR preprocessing, and visualization.

## Installation and Setup

The utilities are part of the localgpt_vision_django project and are located in `models_app/vision/utils/image_processing/`.

## Core Modules

### core.py
Basic image loading and conversion functions:
- `load_image(image_path_or_array)`: Load images from path or convert from array
- `save_image(image, output_path)`: Save image to disk
- `convert_to_array(image)`: Convert to NumPy array
- `convert_to_pil(image)`: Convert to PIL image
- `get_image_grayscale(image)`: Convert to grayscale

### enhancement.py
Image enhancement functions:
- `denoise_image(image, method='fastNL', strength=10)`: Remove noise using various methods
- `binarize_image(image, method='adaptive')`: Convert to binary image
- `deskew_image(image, max_angle=20.0)`: Fix skewed text
- `enhance_contrast(image, method='clahe')`: Improve contrast
- `sharpen_image(image, method='unsharp_mask')`: Sharpen details
- `enhance_for_ocr(image)`: Apply combined preprocessing for OCR

### detection.py
Feature detection functions:
- `detect_text_regions(image, min_area=100)`: Find text blocks
- `detect_tables(image, min_confidence=0.5)`: Detect tables
- `detect_formulas(image, min_confidence=0.5)`: Detect mathematical formulas
- `detect_lines(image, min_length=50)`: Find horizontal and vertical lines
- `detect_paragraphs(image, text_regions=None)`: Group text into paragraphs
- `detect_images(document_image)`: Find embedded images

### transformation.py
Geometric transformations:
- `resize_image(image, size, keep_aspect_ratio=True)`: Resize an image
- `rotate_image(image, angle)`: Rotate an image
- `crop_image(image, bbox)`: Crop to region of interest
- `apply_perspective_transform(image, points)`: Fix perspective distortion
- `normalize_image(image, alpha=0, beta=255)`: Normalize intensity values

### analysis.py
Image analysis functions:
- `analyze_image_complexity(image)`: Assess image complexity
- `analyze_image_quality(image)`: Check image quality metrics
- `estimate_text_density(image)`: Estimate amount of text
- `detect_image_content_type(image)`: Classify image content

### visualization.py
Visualization utilities:
- `draw_bounding_boxes(image, boxes, color=(0,255,0))`: Draw bounding boxes
- `draw_text_regions(image, regions)`: Highlight text regions
- `visualize_processing_steps(image, steps)`: Show multiple processing steps
- `create_debug_image(image, detection_results)`: Create debugging visualization
- `overlay_segmentation_mask(image, mask)`: Show segmentation results
- `draw_keypoints(image, keypoints)`: Visualize key points

## Example Usage

```python
from models_app.vision.utils.image_processing import enhance_for_ocr, detect_text_regions, draw_bounding_boxes

# Load and preprocess an image for OCR
processed_image = enhance_for_ocr('document.jpg')

# Detect text regions
text_regions = detect_text_regions(processed_image)

# Create visualization with bounding boxes
result_image = draw_bounding_boxes(processed_image, text_regions)
```

## OCR-Specific Functions

The module includes optimized preprocessing for different OCR engines:
- `enhance_for_tesseract(image)`: Optimal preprocessing for Tesseract
- `enhance_for_easyocr(image)`: Optimal preprocessing for EasyOCR
- `enhance_for_paddleocr(image)`: Optimal preprocessing for PaddleOCR

## Integration with Document Processing

These utilities are designed to work with both the OCR and document processing modules
to provide consistent image handling throughout the application.
