import cv2
import numpy as np

class Preprocessor:
    def __init__(self, target_size=(640, 480)):
        self.target_size = target_size

    def resize_frame(self, frame):
        """Resizes frame maintaining aspect ratio or to exact size."""
        # Simple resize to target size for consistency
        return cv2.resize(frame, self.target_size)

    def enhance_contrast(self, frame):
        """Applies CLAHE for contrast enhancement."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def sharpen_image(self, image):
        """Applies sharpening kernel."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def denoise_image(self, image):
        """Applies Gaussian Blur for noise reduction."""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def preprocess_pipeline(self, frame):
        """
        Full pipeline: Resize -> Contrast -> Sharpen -> Denoise (minimal)
        """
        frame = self.resize_frame(frame)
        frame = self.enhance_contrast(frame)
        # frame = self.denoise_image(frame) # Keep minimal for speed
        # frame = self.sharpen_image(frame) # Can introduce artifacts
        return frame
