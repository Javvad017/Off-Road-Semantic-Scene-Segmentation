"""
Advanced Image Preprocessing Module for Off-Road Segmentation
Handles denoising, contrast enhancement, sharpening, and normalization
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Applies preprocessing pipeline to improve segmentation quality
    especially for challenging off-road conditions (dust, shadows, glare)
    """
    
    def __init__(self, 
                 enable_denoise: bool = True,
                 enable_clahe: bool = True,
                 enable_sharpen: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: int = 8):
        """
        Args:
            enable_denoise: Apply Non-Local Means Denoising
            enable_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            enable_sharpen: Apply Unsharp Masking
            clahe_clip_limit: CLAHE contrast limit (higher = more contrast)
            clahe_tile_size: CLAHE grid size (smaller = more local adaptation)
        """
        self.enable_denoise = enable_denoise
        self.enable_clahe = enable_clahe
        self.enable_sharpen = enable_sharpen
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        
        # Initialize CLAHE object
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(clahe_tile_size, clahe_tile_size)
            )
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Non-Local Means Denoising
        Effective for removing camera noise while preserving edges
        """
        # Fast denoising for color images
        return cv2.fastNlMeansDenoisingColored(
            image, 
            None, 
            h=10,  # Filter strength (higher = more smoothing)
            hColor=10,  # Color component filter strength
            templateWindowSize=7,  # Template patch size
            searchWindowSize=21  # Search area size
        )
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to improve contrast in shadows and highlights
        Works in LAB color space to preserve color fidelity
        """
        # Convert BGR to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        l_clahe = self.clahe.apply(l)
        
        # Merge and convert back to BGR
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def sharpen(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Apply Unsharp Masking to enhance edges and details
        
        Args:
            image: Input image
            strength: Sharpening strength (1.0 = no change, >1.0 = sharper)
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Unsharp mask formula: sharpened = original + strength * (original - blurred)
        sharpened = cv2.addWeighted(image, strength, blurred, 1 - strength, 0)
        
        return sharpened
    
    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize brightness to handle varying lighting conditions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Normalize V channel to [0, 255]
        v_normalized = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
        
        # Merge and convert back
        hsv_normalized = cv2.merge([h, s, v_normalized])
        normalized = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)
        
        return normalized
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline
        
        Args:
            image: Input BGR image from OpenCV
            
        Returns:
            Preprocessed BGR image
        """
        processed = image.copy()
        
        try:
            # Step 1: Denoise (removes camera noise)
            if self.enable_denoise:
                processed = self.denoise(processed)
            
            # Step 2: Enhance contrast (improves shadow/highlight detail)
            if self.enable_clahe:
                processed = self.enhance_contrast(processed)
            
            # Step 3: Sharpen (enhances edges for better segmentation)
            if self.enable_sharpen:
                processed = self.sharpen(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return image  # Return original on error


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) for improved stability and accuracy
    Averages predictions across multiple augmented versions of the input
    """
    
    def __init__(self, use_flip: bool = True, use_scales: bool = False):
        """
        Args:
            use_flip: Apply horizontal flip augmentation
            use_scales: Apply multi-scale augmentation (slower but more accurate)
        """
        self.use_flip = use_flip
        self.use_scales = use_scales
        self.scales = [0.9, 1.0, 1.1] if use_scales else [1.0]
    
    def augment_image(self, image: np.ndarray) -> list:
        """
        Generate augmented versions of input image
        
        Returns:
            List of (augmented_image, transform_info) tuples
        """
        augmented = []
        
        for scale in self.scales:
            # Scale augmentation
            if scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled = image
            
            # Original
            augmented.append((scaled, {'flip': False, 'scale': scale}))
            
            # Horizontal flip
            if self.use_flip:
                flipped = cv2.flip(scaled, 1)
                augmented.append((flipped, {'flip': True, 'scale': scale}))
        
        return augmented
    
    def merge_predictions(self, predictions: list, transforms: list, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Merge multiple predictions by averaging
        
        Args:
            predictions: List of prediction masks
            transforms: List of transform info dicts
            original_shape: (height, width) of original image
            
        Returns:
            Merged prediction mask
        """
        h, w = original_shape
        merged = np.zeros((h, w), dtype=np.float32)
        
        for pred, transform in zip(predictions, transforms):
            # Undo scale
            if transform['scale'] != 1.0:
                pred = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Undo flip
            if transform['flip']:
                pred = cv2.flip(pred, 1)
            
            merged += pred.astype(np.float32)
        
        # Average and convert back to class indices
        merged = (merged / len(predictions)).astype(np.uint8)
        
        return merged
