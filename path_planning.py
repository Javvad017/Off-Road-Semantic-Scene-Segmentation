"""
Safe Path Detection and Visualization Module
Computes optimal traversable path through safe terrain
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class SafePathDetector:
    """
    Analyzes segmentation mask to detect safe traversable paths
    and compute optimal navigation trajectory
    """
    
    def __init__(self, 
                 safe_classes: set,
                 path_width: int = 80,
                 horizon_ratio: float = 0.6,
                 min_safe_area: float = 0.3):
        """
        Args:
            safe_classes: Set of class IDs considered safe terrain
            path_width: Width of path corridor in pixels
            horizon_ratio: Vertical position to start path search (0.0=top, 1.0=bottom)
            min_safe_area: Minimum ratio of safe pixels required for valid path
        """
        self.safe_classes = safe_classes
        self.path_width = path_width
        self.horizon_ratio = horizon_ratio
        self.min_safe_area = min_safe_area
    
    def create_safe_binary_mask(self, seg_mask: np.ndarray) -> np.ndarray:
        """
        Convert segmentation mask to binary safe/unsafe mask
        
        Args:
            seg_mask: Segmentation mask with class indices
            
        Returns:
            Binary mask (255=safe, 0=unsafe)
        """
        safe_mask = np.isin(seg_mask, list(self.safe_classes)).astype(np.uint8) * 255
        return safe_mask
    
    def compute_safe_score_map(self, safe_mask: np.ndarray) -> np.ndarray:
        """
        Compute safety score for each pixel using distance transform
        Higher values = farther from obstacles
        
        Args:
            safe_mask: Binary safe mask
            
        Returns:
            Safety score map (float32)
        """
        # Distance transform: distance to nearest unsafe pixel
        dist_transform = cv2.distanceTransform(safe_mask, cv2.DIST_L2, 5)
        
        # Normalize to [0, 1]
        if dist_transform.max() > 0:
            dist_transform = dist_transform / dist_transform.max()
        
        return dist_transform
    
    def find_optimal_path(self, safe_mask: np.ndarray, score_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find optimal path through safe terrain using dynamic programming
        
        Args:
            safe_mask: Binary safe mask
            score_map: Safety score map
            
        Returns:
            List of (x, y) waypoints defining the path
        """
        h, w = safe_mask.shape
        start_y = int(h * self.horizon_ratio)
        
        # Initialize path from bottom center
        path = []
        current_x = w // 2
        
        # Scan from horizon to bottom of image
        for y in range(start_y, h, 5):  # Step by 5 pixels for efficiency
            # Search window around current x position
            search_left = max(0, current_x - self.path_width // 2)
            search_right = min(w, current_x + self.path_width // 2)
            
            # Find x position with highest safety score in search window
            row_scores = score_map[y, search_left:search_right]
            
            if len(row_scores) > 0 and row_scores.max() > 0:
                best_x_offset = np.argmax(row_scores)
                current_x = search_left + best_x_offset
            
            path.append((current_x, y))
        
        return path
    
    def smooth_path(self, path: List[Tuple[int, int]], window: int = 5) -> List[Tuple[int, int]]:
        """
        Smooth path using moving average
        
        Args:
            path: List of waypoints
            window: Smoothing window size
            
        Returns:
            Smoothed path
        """
        if len(path) < window:
            return path
        
        smoothed = []
        path_array = np.array(path)
        
        for i in range(len(path)):
            start = max(0, i - window // 2)
            end = min(len(path), i + window // 2 + 1)
            avg_x = int(np.mean(path_array[start:end, 0]))
            avg_y = int(np.mean(path_array[start:end, 1]))
            smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def compute_path_statistics(self, safe_mask: np.ndarray, path: List[Tuple[int, int]]) -> dict:
        """
        Compute statistics about the detected path
        
        Args:
            safe_mask: Binary safe mask
            path: Detected path waypoints
            
        Returns:
            Dictionary with path statistics
        """
        h, w = safe_mask.shape
        total_pixels = h * w
        safe_pixels = np.sum(safe_mask > 0)
        
        # Compute path safety (percentage of path on safe terrain)
        path_safe_count = 0
        for x, y in path:
            if 0 <= y < h and 0 <= x < w:
                if safe_mask[y, x] > 0:
                    path_safe_count += 1
        
        path_safety = path_safe_count / len(path) if len(path) > 0 else 0.0
        
        stats = {
            'safe_area_ratio': safe_pixels / total_pixels,
            'path_safety': path_safety,
            'path_length': len(path),
            'is_valid': path_safety > self.min_safe_area
        }
        
        return stats
    
    def visualize_path(self, 
                       image: np.ndarray, 
                       path: List[Tuple[int, int]], 
                       stats: dict,
                       color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw path and statistics on image
        
        Args:
            image: BGR image to draw on
            path: Path waypoints
            stats: Path statistics
            color: Path color (BGR)
            
        Returns:
            Image with path visualization
        """
        vis = image.copy()
        
        # Draw path as thick polyline
        if len(path) > 1:
            path_array = np.array(path, dtype=np.int32)
            cv2.polylines(vis, [path_array], isClosed=False, 
                         color=color, thickness=4, lineType=cv2.LINE_AA)
            
            # Draw path corridor
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                
                # Draw semi-transparent corridor
                overlay = vis.copy()
                cv2.line(overlay, (x1, y1), (x2, y2), color, self.path_width)
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
        
        # Draw start point
        if len(path) > 0:
            start_x, start_y = path[0]
            cv2.circle(vis, (start_x, start_y), 10, (255, 255, 0), -1)
            cv2.circle(vis, (start_x, start_y), 12, (0, 0, 0), 2)
        
        # Draw statistics
        y_offset = 130
        cv2.putText(vis, f"Safe Area: {stats['safe_area_ratio']*100:.1f}%", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 35
        path_status = "VALID" if stats['is_valid'] else "BLOCKED"
        status_color = (0, 255, 0) if stats['is_valid'] else (0, 0, 255)
        cv2.putText(vis, f"Path: {path_status} ({stats['path_safety']*100:.1f}%)", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return vis
    
    def process(self, seg_mask: np.ndarray, original_image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Complete path detection pipeline
        
        Args:
            seg_mask: Segmentation mask
            original_image: Original BGR image
            
        Returns:
            Tuple of (visualized image, path statistics)
        """
        try:
            # Create binary safe mask
            safe_mask = self.create_safe_binary_mask(seg_mask)
            
            # Compute safety scores
            score_map = self.compute_safe_score_map(safe_mask)
            
            # Find optimal path
            path = self.find_optimal_path(safe_mask, score_map)
            
            # Smooth path
            path = self.smooth_path(path, window=7)
            
            # Compute statistics
            stats = self.compute_path_statistics(safe_mask, path)
            
            # Visualize
            vis = self.visualize_path(original_image, path, stats)
            
            return vis, stats
            
        except Exception as e:
            logger.error(f"Path detection error: {e}")
            return original_image, {'is_valid': False, 'safe_area_ratio': 0.0}


class ObstacleDetector:
    """
    Detects and highlights critical obstacles in the scene
    """
    
    def __init__(self, unsafe_classes: set, min_obstacle_size: int = 1000):
        """
        Args:
            unsafe_classes: Set of class IDs considered obstacles
            min_obstacle_size: Minimum obstacle size in pixels to highlight
        """
        self.unsafe_classes = unsafe_classes
        self.min_obstacle_size = min_obstacle_size
    
    def detect_obstacles(self, seg_mask: np.ndarray) -> List[dict]:
        """
        Detect and localize obstacles
        
        Args:
            seg_mask: Segmentation mask
            
        Returns:
            List of obstacle dictionaries with bbox and class info
        """
        obstacles = []
        
        # Create binary obstacle mask
        obstacle_mask = np.isin(seg_mask, list(self.unsafe_classes)).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obstacle_mask, connectivity=8
        )
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= self.min_obstacle_size:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                obstacles.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': (int(centroids[i][0]), int(centroids[i][1]))
                })
        
        return obstacles
    
    def visualize_obstacles(self, image: np.ndarray, obstacles: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes around detected obstacles
        
        Args:
            image: BGR image
            obstacles: List of obstacle dictionaries
            
        Returns:
            Image with obstacle visualization
        """
        vis = image.copy()
        
        for obs in obstacles:
            x, y, w, h = obs['bbox']
            cx, cy = obs['centroid']
            
            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw warning icon
            cv2.circle(vis, (cx, cy), 15, (0, 0, 255), -1)
            cv2.putText(vis, "!", (cx - 5, cy + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return vis
