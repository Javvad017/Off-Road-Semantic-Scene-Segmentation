import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import cv2
import logging
from PIL import Image

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentationEngine:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            logger.info(f"Loading model: {model_name}...")
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
            # Use FP16 if on CUDA
            if self.device == "cuda":
                self.model.half() 
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # ADE20K Labels (Subset relevant to Off-Road)
        # Safe Terrain: Road (6), Floor (3), Grass (9), Sidewalk (11), Earth (13), Path (52), Dirt (91)
        self.safe_classes = {3, 6, 9, 11, 13, 46, 52, 91} 
        
        # --- Cyberpunk Palette (BGR) ---
        self.colors = np.zeros((151, 3), dtype=np.uint8)
        self.colors[:] = [20, 20, 20]  # Default Unsafe (Dark Grey)

        # Safe Zones -> Neon Green
        for cls_id in self.safe_classes:
            self.colors[cls_id] = [0, 255, 128] # Spring Green
        
        # Specific overrides for visuals
        self.colors[2] = [255, 191, 0]   # Sky (DeepSkyBlue) -> Cyan/Blue
        self.colors[4] = [0, 100, 0]     # Tree (DarkGreen) -> Subdued Green
        self.colors[12] = [0, 0, 255]    # Person (Red) -> Alert!
        self.colors[20] = [0, 165, 255]  # Car (Orange)
        
    def process_frame(self, frame_rgb_pil):
        """
        Takes a PIL Image (RGB), runs segmentation, and returns the raw prediction mask (numpy).
        """
        try:
            width, height = frame_rgb_pil.size
            
            # Inference using Autocast
            with torch.cuda.amp.autocast(enabled=(self.device=="cuda")):
                with torch.no_grad():
                    inputs = self.processor(images=frame_rgb_pil, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
            
            # Upsample logits
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(height, width), # (h, w)
                mode="bilinear",
                align_corners=False,
            )
            
            # Get class predictions
            pred_seg = upsampled_logits.argmax(dim=1)[0] # (height, width)
            
            return pred_seg.cpu().numpy().astype(np.uint8)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None

    def get_color_mask(self, pred_seg_np):
        """Converts prediction mask to color image."""
        if pred_seg_np is None:
            return None
        # Use numpy indexing for speed
        return self.colors[pred_seg_np]
