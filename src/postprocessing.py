import cv2
import numpy as np

class Postprocessor:
    def __init__(self, safe_classes):
        self.safe_classes = safe_classes

    def apply_morphology(self, mask):
        """Applies morphological operations to smooth the mask."""
        return cv2.medianBlur(mask, 5)

    def extract_safe_mask(self, pred_seg):
        """Creates a binary mask where 255 is safe, 0 is unsafe."""
        safe_mask = np.zeros_like(pred_seg, dtype=np.uint8)
        for cls_id in self.safe_classes:
            safe_mask[pred_seg == cls_id] = 255
        return safe_mask

    def find_safe_path(self, safe_mask):
        """
        Finds the largest safe area and computes a direction vector (path).
        Returns: (largest_contour, (cX, cY)) or None if no path found.
        """
        # Morphological Cleanup on binary mask
        kernel = np.ones((5, 5), np.uint8)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, kernel)

        # Find Contours
        contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        # Get largest contour (main path)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter small noise
        if cv2.contourArea(largest_contour) < 500:
            return None

        # Compute Moment to find center of the bottom part of the contour is better
        # For simplicity, just use standard moment
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        return largest_contour, (cX, cY)

    def overlay_mask(self, frame, seg_color, alpha=0.4):
        """Blends the original frame with the segmentation color mask in a stylish way."""
        # Resize if needed
        if frame.shape[:2] != seg_color.shape[:2]:
            seg_color = cv2.resize(seg_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Add a glow effect to the segmentation mask
        # 1. Blur the seg_color slightly to blend edges
        seg_blurred = cv2.GaussianBlur(seg_color, (0, 0), sigmaX=2)
        
        # 2. Weighted add
        overlay = cv2.addWeighted(frame, 1.0, seg_blurred, alpha, 0)
        return overlay

    def draw_safe_path(self, frame, path_data):
        """Draws the safe path and boundaries on the frame with cyber-style overlays."""
        if path_data is None:
            return frame

        largest_contour, centroid = path_data
        
        if largest_contour is None or centroid is None:
            return frame

        # 1. Draw Boundary of safe zone (Neon Green)
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        
        # 2. Draw Semi-Transparent Fill for Safe Zone (Optional - maybe too cluttered with the full mask)
        # Skipping fill to keep it clean

        # 3. Draw HUD Elements
        height, width = frame.shape[:2]
        
        # Draw Center Point (Target)
        cv2.circle(frame, centroid, 8, (0, 255, 255), -1) # Yellow Dot
        cv2.circle(frame, centroid, 12, (0, 255, 255), 1) # Ring
        
        # Draw Direction Arrow (from bottom center to centroid)
        start_point = (width // 2, height - 20)
        
        # Dashed Line Effect (Simulated with simple line here, dashed is complex in cv2)
        cv2.arrowedLine(frame, start_point, centroid, (50, 255, 50), 4, tipLength=0.2)
        
        # Add Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "SAFE PATH", (centroid[0] - 40, centroid[1] - 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        return frame
