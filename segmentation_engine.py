import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2
import logging
from preprocessing import ImagePreprocessor, TestTimeAugmentation
from postprocessing import SegmentationPostprocessor, EdgeRefinement
from path_planning import SafePathDetector, ObstacleDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentationEngine:
    def __init__(self, 
                 model_name="nvidia/segformer-b0-finetuned-ade-512-512", 
                 device=None,
                 use_preprocessing=True,
                 use_postprocessing=True,
                 use_path_detection=True,
                 use_fp16=False,
                 use_tta=False):
        """
        Initializes the SegFormer model and processing pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
            use_preprocessing: Enable image preprocessing
            use_postprocessing: Enable segmentation post-processing
            use_path_detection: Enable safe path detection
            use_fp16: Use FP16 mixed precision (faster on modern GPUs)
            use_tta: Use Test-Time Augmentation (slower but more accurate)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.use_tta = use_tta
        
        logger.info(f"Using device: {self.device}")
        if self.use_fp16:
            logger.info("FP16 mixed precision enabled")

        try:
            logger.info(f"Loading model: {model_name}...")
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
            
            # Enable FP16 if requested
            if self.use_fp16:
                self.model = self.model.half()
            
            self.model.eval()
            logger.info("Model loaded successfully.")
            
            # Log GPU info if available
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Define Safe/Unsafe Class IDs based on ADE20K dataset (150 classes)
        # 0=wall, 1=building, 2=sky, 3=floor, 4=tree, 5=ceiling, 6=road, 7=bed, 8=window, 9=grass, 10=cabinet, 11=sidewalk, 12=person, 13=earth, 14=door, 15=table, 16=mountain, 17=plant, 18=curtain, 19=chair, 20=car, 21=water ...
        # Safe Terrain: Road (6), Floor (3), Grass (9), Sidewalk (11), Earth (13), Path (52), Dirt (91), Sand (46)
        self.safe_classes = {3, 6, 9, 11, 13, 46, 52, 91} 
        
        # Unsafe obstacles (for highlighting)
        self.unsafe_classes = {12, 20, 21, 1, 0}  # person, car, water, building, wall
        
        # Color Map (BGR for OpenCV)
        self.colors = np.zeros((150, 3), dtype=np.uint8)
        self.colors[:] = [0, 0, 255]  # Default Unsafe (Red)
        for cls_id in self.safe_classes:
            self.colors[cls_id] = [0, 255, 0] # Safe (Green)
        self.colors[2] = [255, 0, 0] # Sky (Blue) - Neutral
        self.colors[4] = [255, 0, 0] # Tree (Blue) - Neutral (or treat as unsafe obstacle)
        self.colors[17] = [255, 0, 0] # Plant (Blue) - Neutral
        
        # Initialize processing modules
        self.preprocessor = ImagePreprocessor() if use_preprocessing else None
        self.postprocessor = SegmentationPostprocessor() if use_postprocessing else None
        self.path_detector = SafePathDetector(self.safe_classes) if use_path_detection else None
        self.obstacle_detector = ObstacleDetector(self.unsafe_classes)
        self.edge_refiner = EdgeRefinement()
        
        # TTA module
        if self.use_tta:
            self.tta = TestTimeAugmentation(use_flip=True, use_scales=False)
            logger.info("Test-Time Augmentation enabled")

    def _run_inference(self, image_pil):
        """
        Run model inference on a PIL image
        
        Args:
            image_pil: PIL Image in RGB format
            
        Returns:
            Segmentation mask as numpy array
        """
        # Preprocess
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Convert to FP16 if enabled
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_pil.size[::-1], # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # Get class predictions
        pred_seg = upsampled_logits.argmax(dim=1)[0] # (height, width)
        
        # Move to CPU for visualization
        pred_seg_np = pred_seg.cpu().numpy().astype(np.uint8)
        
        return pred_seg_np
    
    def process_frame(self, frame_bgr, show_path=True, show_obstacles=True):
        """
        Takes a BGR frame (OpenCV), runs full processing pipeline, and returns the overlay.
        
        Args:
            frame_bgr: Input frame in BGR format (OpenCV)
            show_path: Whether to visualize safe path
            show_obstacles: Whether to highlight obstacles
            
        Returns:
            Tuple of (overlay_image, segmentation_mask, statistics_dict)
        """
        try:
            # Step 1: Preprocessing
            if self.preprocessor:
                frame_processed = self.preprocessor.process(frame_bgr)
            else:
                frame_processed = frame_bgr
            
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Step 2: Inference (with optional TTA)
            if self.use_tta:
                # Generate augmented versions
                augmented = self.tta.augment_image(np.array(image))
                predictions = []
                
                for aug_img, transform in augmented:
                    aug_pil = Image.fromarray(aug_img)
                    pred = self._run_inference(aug_pil)
                    predictions.append(pred)
                
                # Merge predictions
                pred_seg_np = self.tta.merge_predictions(
                    predictions, 
                    [t for _, t in augmented],
                    image.size[::-1]
                )
            else:
                # Standard inference
                pred_seg_np = self._run_inference(image)
            
            # Step 3: Post-processing
            if self.postprocessor:
                pred_seg_np = self.postprocessor.process(pred_seg_np, self.safe_classes)

            # Step 4: Create Color Mask
            seg_color = self.colors[pred_seg_np]

            # Step 5: Blend Original Frame with Segmentation Mask
            alpha = 0.5
            overlay = cv2.addWeighted(frame_bgr, 1 - alpha, seg_color, alpha, 0)

            # Step 6: Path Detection
            stats = {}
            if self.path_detector and show_path:
                overlay, stats = self.path_detector.process(pred_seg_np, overlay)
            
            # Step 7: Obstacle Detection
            if show_obstacles:
                obstacles = self.obstacle_detector.detect_obstacles(pred_seg_np)
                overlay = self.obstacle_detector.visualize_obstacles(overlay, obstacles)
                stats['num_obstacles'] = len(obstacles)

            return overlay, pred_seg_np, stats

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame_bgr, None, {}
    
    def reset_temporal_state(self):
        """Reset temporal buffers (call when switching videos)"""
        if self.postprocessor:
            self.postprocessor.reset_temporal_buffer()
    
    def get_performance_stats(self):
        """Get GPU memory usage and performance statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9  # GB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / 1e9  # GB
            stats['gpu_utilization'] = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
        
        return stats
