"""
Advanced Post-Processing Module for Segmentation Refinement
Handles morphological operations, noise removal, and temporal smoothing
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class SegmentationPostprocessor:
    """
    Refines raw segmentation masks using morphological operations
    and temporal smoothing for video stability
    """
    
    def __init__(self,
                 enable_morphology: bool = True,
                 enable_temporal_smooth: bool = True,
                 temporal_window: int = 5,
                 min_region_size: int = 500):
        """
        Args:
            enable_morphology: Apply morphological operations (opening/closing)
            enable_temporal_smooth: Apply temporal smoothing across frames
            temporal_window: Number of frames to average for temporal smoothing
            min_region_size: Minimum region size in pixels (smaller regions removed)
        """
        self.enable_morphology = enable_morphology
        self.enable_temporal_smooth = enable_temporal_smooth
        self.temporal_window = temporal_window
        self.min_region_size = min_region_size
        
        # Temporal buffer for smoothing
        self.frame_buffer = deque(maxlen=temporal_window)
        
        # Morphological kernels
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    def morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up segmentation mask
        - Opening: Removes small noise/speckles
        - Closing: Fills small holes
        
        Args:
            mask: Segmentation mask (H, W) with class indices
            
        Returns:
            Cleaned mask
        """
        # Opening: erosion followed by dilation (removes small objects)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
        
        # Closing: dilation followed by erosion (fills small holes)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_medium)
        
        return closed
    
    def remove_small_regions(self, mask: np.ndarray, safe_classes: set) -> np.ndarray:
        """
        Remove small disconnected regions that are likely noise
        
        Args:
            mask: Segmentation mask
            safe_classes: Set of class IDs considered safe
            
        Returns:
            Mask with small regions removed
        """
        cleaned = mask.copy()
        
        # Process safe and unsafe regions separately
        for is_safe in [True, False]:
            if is_safe:
                binary = np.isin(mask, list(safe_classes)).astype(np.uint8)
            else:
                binary = (~np.isin(mask, list(safe_classes))).astype(np.uint8)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            # Remove small components
            for i in range(1, num_labels):  # Skip background (0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.min_region_size:
                    # Set small regions to background or majority neighbor class
                    cleaned[labels == i] = 0
        
        return cleaned
    
    def temporal_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing by averaging predictions across frames
        Reduces flickering in video segmentation
        
        Args:
            mask: Current frame segmentation mask
            
        Returns:
            Temporally smoothed mask
        """
        # Add current frame to buffer
        self.frame_buffer.append(mask.astype(np.float32))
        
        if len(self.frame_buffer) < 2:
            return mask  # Not enough frames yet
        
        # Average across temporal window
        averaged = np.mean(self.frame_buffer, axis=0)
        
        # Convert back to class indices (majority voting)
        smoothed = np.round(averaged).astype(np.uint8)
        
        return smoothed
    
    def bilateral_filter_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to smooth mask while preserving edges
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Smoothed mask
        """
        # Convert to float for filtering
        mask_float = mask.astype(np.float32)
        
        # Apply bilateral filter (preserves edges)
        filtered = cv2.bilateralFilter(mask_float, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Convert back to uint8
        return np.round(filtered).astype(np.uint8)
    
    def process(self, mask: np.ndarray, safe_classes: set) -> np.ndarray:
        """
        Apply full post-processing pipeline
        
        Args:
            mask: Raw segmentation mask from model
            safe_classes: Set of class IDs considered safe terrain
            
        Returns:
            Refined segmentation mask
        """
        processed = mask.copy()
        
        try:
            # Step 1: Morphological cleanup
            if self.enable_morphology:
                processed = self.morphological_cleanup(processed)
            
            # Step 2: Remove small regions
            processed = self.remove_small_regions(processed, safe_classes)
            
            # Step 3: Temporal smoothing (for video)
            if self.enable_temporal_smooth:
                processed = self.temporal_smoothing(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            return mask  # Return original on error
    
    def reset_temporal_buffer(self):
        """Reset temporal buffer (call when switching videos)"""
        self.frame_buffer.clear()


class EdgeRefinement:
    """
    Refines segmentation boundaries using edge-aware techniques
    """
    
    def __init__(self, edge_threshold: int = 50):
        """
        Args:
            edge_threshold: Threshold for edge detection (lower = more sensitive)
        """
        self.edge_threshold = edge_threshold
    
    def refine_boundaries(self, mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Refine segmentation boundaries using image edges
        
        Args:
            mask: Segmentation mask
            original_image: Original BGR image
            
        Returns:
            Mask with refined boundaries
        """
        # Detect edges in original image
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        
        # Dilate edges slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Use edges to guide boundary refinement
        # Where there's a strong edge, trust the original mask more
        refined = mask.copy()
        
        # Apply guided filter or edge-aware smoothing
        # (Simplified version - full implementation would use guided filter)
        refined = cv2.bilateralFilter(
            refined.astype(np.float32), 
            d=5, 
            sigmaColor=75, 
            sigmaSpace=75
        ).astype(np.uint8)
        
        return refined
