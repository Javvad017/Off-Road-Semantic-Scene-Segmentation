"""
Model Evaluation Script
Computes accuracy metrics and generates performance reports
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from segmentation_engine import SegmentationEngine
import logging
from sklearn.metrics import confusion_matrix, classification_report
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationEvaluator:
    """
    Evaluates segmentation model performance
    """
    
    def __init__(self, model_path, num_classes=150):
        """
        Args:
            model_path: Path to model or HuggingFace model name
            num_classes: Number of segmentation classes
        """
        self.engine = SegmentationEngine(
            model_name=model_path,
            use_preprocessing=True,
            use_postprocessing=True,
            use_path_detection=False
        )
        self.num_classes = num_classes
        
    def compute_iou(self, pred, target, class_id):
        """
        Compute Intersection over Union for a specific class
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            class_id: Class to compute IoU for
            
        Returns:
            IoU score
        """
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            return float('nan')
        
        return intersection / union
    
    def compute_metrics(self, pred, target, ignore_index=255):
        """
        Compute comprehensive metrics
        
        Args:
            pred: Predicted mask (H, W)
            target: Ground truth mask (H, W)
            ignore_index: Index to ignore in computation
            
        Returns:
            Dictionary of metrics
        """
        # Create valid mask (ignore certain pixels)
        valid_mask = target != ignore_index
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # Pixel accuracy
        correct = (pred_valid == target_valid).sum()
        total = valid_mask.sum()
        pixel_accuracy = correct / total if total > 0 else 0
        
        # Per-class IoU
        iou_per_class = []
        for class_id in range(self.num_classes):
            iou = self.compute_iou(pred, target, class_id)
            if not np.isnan(iou):
                iou_per_class.append(iou)
        
        mean_iou = np.mean(iou_per_class) if iou_per_class else 0
        
        # Class accuracy
        class_accuracies = {}
        for class_id in range(self.num_classes):
            class_mask = target_valid == class_id
            if class_mask.sum() > 0:
                class_correct = (pred_valid[class_mask] == class_id).sum()
                class_accuracies[class_id] = class_correct / class_mask.sum()
        
        return {
            'pixel_accuracy': float(pixel_accuracy),
            'mean_iou': float(mean_iou),
            'class_accuracies': class_accuracies,
            'iou_per_class': iou_per_class
        }
    
    def evaluate_dataset(self, images_dir, masks_dir, output_dir="./evaluation_results"):
        """
        Evaluate model on a dataset
        
        Args:
            images_dir: Directory containing test images
            masks_dir: Directory containing ground truth masks
            output_dir: Where to save results
            
        Returns:
            Dictionary of aggregate metrics
        """
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all images
        image_files = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
        
        logger.info(f"Evaluating on {len(image_files)} images...")
        
        all_metrics = []
        inference_times = []
        
        for img_file in tqdm(image_files, desc="Evaluating"):
            # Load image
            image = cv2.imread(str(img_file))
            
            # Load ground truth mask
            mask_file = masks_path / img_file.name.replace('.jpg', '.png')
            if not mask_file.exists():
                mask_file = masks_path / img_file.name
            
            if not mask_file.exists():
                logger.warning(f"Mask not found for {img_file.name}, skipping")
                continue
            
            gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Run inference
            start_time = time.time()
            _, pred_mask, _ = self.engine.process_frame(image, show_path=False, show_obstacles=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Resize prediction to match ground truth
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Compute metrics
            metrics = self.compute_metrics(pred_mask, gt_mask)
            metrics['image'] = img_file.name
            metrics['inference_time'] = inference_time
            all_metrics.append(metrics)
        
        # Aggregate results
        aggregate = {
            'num_images': len(all_metrics),
            'mean_pixel_accuracy': np.mean([m['pixel_accuracy'] for m in all_metrics]),
            'std_pixel_accuracy': np.std([m['pixel_accuracy'] for m in all_metrics]),
            'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
            'std_iou': np.std([m['mean_iou'] for m in all_metrics]),
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'fps': 1.0 / np.mean(inference_times)
        }
        
        # Save results
        with open(output_path / "metrics.json", "w") as f:
            json.dump({
                'aggregate': aggregate,
                'per_image': all_metrics
            }, f, indent=2)
        
        # Generate visualizations
        self.plot_results(all_metrics, output_path)
        
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Images evaluated: {aggregate['num_images']}")
        logger.info(f"Mean Pixel Accuracy: {aggregate['mean_pixel_accuracy']:.4f} ± {aggregate['std_pixel_accuracy']:.4f}")
        logger.info(f"Mean IoU: {aggregate['mean_iou']:.4f} ± {aggregate['std_iou']:.4f}")
        logger.info(f"Mean Inference Time: {aggregate['mean_inference_time']*1000:.2f}ms ± {aggregate['std_inference_time']*1000:.2f}ms")
        logger.info(f"FPS: {aggregate['fps']:.2f}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 60)
        
        return aggregate
    
    def plot_results(self, metrics, output_dir):
        """
        Generate visualization plots
        
        Args:
            metrics: List of per-image metrics
            output_dir: Where to save plots
        """
        output_path = Path(output_dir)
        
        # Pixel accuracy distribution
        plt.figure(figsize=(10, 6))
        accuracies = [m['pixel_accuracy'] for m in metrics]
        plt.hist(accuracies, bins=20, edgecolor='black')
        plt.xlabel('Pixel Accuracy')
        plt.ylabel('Frequency')
        plt.title('Pixel Accuracy Distribution')
        plt.axvline(np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "accuracy_distribution.png", dpi=150)
        plt.close()
        
        # IoU distribution
        plt.figure(figsize=(10, 6))
        ious = [m['mean_iou'] for m in metrics]
        plt.hist(ious, bins=20, edgecolor='black')
        plt.xlabel('Mean IoU')
        plt.ylabel('Frequency')
        plt.title('Mean IoU Distribution')
        plt.axvline(np.mean(ious), color='r', linestyle='--', label=f'Mean: {np.mean(ious):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "iou_distribution.png", dpi=150)
        plt.close()
        
        # Inference time distribution
        plt.figure(figsize=(10, 6))
        times = [m['inference_time'] * 1000 for m in metrics]
        plt.hist(times, bins=20, edgecolor='black')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.axvline(np.mean(times), color='r', linestyle='--', label=f'Mean: {np.mean(times):.1f}ms')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "inference_time_distribution.png", dpi=150)
        plt.close()
        
        logger.info(f"Plots saved to {output_path}")


def benchmark_performance(model_path, input_size=(1080, 1920), num_iterations=100):
    """
    Benchmark model performance
    
    Args:
        model_path: Path to model
        input_size: Input image size (height, width)
        num_iterations: Number of iterations to run
        
    Returns:
        Performance statistics
    """
    logger.info("=" * 60)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Iterations: {num_iterations}")
    logger.info("=" * 60)
    
    # Initialize engine
    engine = SegmentationEngine(
        model_name=model_path,
        use_preprocessing=True,
        use_postprocessing=True,
        use_path_detection=True,
        use_fp16=True
    )
    
    # Create dummy input
    dummy_frame = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(10):
        engine.process_frame(dummy_frame)
    
    # Benchmark
    logger.info("Benchmarking...")
    times = []
    for _ in tqdm(range(num_iterations)):
        start = time.time()
        engine.process_frame(dummy_frame)
        times.append(time.time() - start)
    
    # Statistics
    times = np.array(times)
    stats = {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times)),
        'p95_time': float(np.percentile(times, 95)),
        'p99_time': float(np.percentile(times, 99)),
        'mean_fps': float(1.0 / np.mean(times)),
        'input_size': input_size
    }
    
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean time: {stats['mean_time']*1000:.2f}ms ± {stats['std_time']*1000:.2f}ms")
    logger.info(f"Median time: {stats['median_time']*1000:.2f}ms")
    logger.info(f"Min time: {stats['min_time']*1000:.2f}ms")
    logger.info(f"Max time: {stats['max_time']*1000:.2f}ms")
    logger.info(f"P95 time: {stats['p95_time']*1000:.2f}ms")
    logger.info(f"P99 time: {stats['p99_time']*1000:.2f}ms")
    logger.info(f"Mean FPS: {stats['mean_fps']:.2f}")
    logger.info("=" * 60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Model")
    parser.add_argument("--mode", type=str, choices=['evaluate', 'benchmark'], required=True,
                       help="Evaluation mode")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512",
                       help="Model path or HuggingFace name")
    parser.add_argument("--images", type=str, help="Test images directory (for evaluate mode)")
    parser.add_argument("--masks", type=str, help="Test masks directory (for evaluate mode)")
    parser.add_argument("--output", type=str, default="./evaluation_results",
                       help="Output directory")
    parser.add_argument("--input-size", type=int, nargs=2, default=[1080, 1920],
                       help="Input size for benchmark (height width)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations for benchmark")
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        if not args.images or not args.masks:
            parser.error("--images and --masks required for evaluate mode")
        
        evaluator = SegmentationEvaluator(args.model)
        results = evaluator.evaluate_dataset(args.images, args.masks, args.output)
        
    elif args.mode == 'benchmark':
        results = benchmark_performance(
            args.model,
            input_size=tuple(args.input_size),
            num_iterations=args.iterations
        )
        
        # Save results
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        with open(output_path / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
