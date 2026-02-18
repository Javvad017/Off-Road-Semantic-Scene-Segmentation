"""
Enhanced Off-Road Semantic Segmentation System
Production-quality implementation with full preprocessing, post-processing, and path detection
"""

import cv2
import time
import argparse
import yt_dlp
import os
import torch
import numpy as np
from segmentation_engine import SegmentationEngine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_youtube(url, output_filename="input_video.mp4"):
    """Downloads a YouTube video using yt-dlp."""
    logger.info(f"Downloading YouTube video: {url}...")
    if os.path.exists(output_filename):
        os.remove(output_filename)
        
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_filename,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    logger.info("Download complete.")
    return output_filename


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Off-Road Semantic Segmentation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument("--source", type=str, required=True, 
                       help="Video path, YouTube URL, or 0 for webcam")
    parser.add_argument("--output", type=str, default="output.mp4", 
                       help="Output video filename")
    
    # Model Configuration
    parser.add_argument("--model", type=str, 
                       default="nvidia/segformer-b0-finetuned-ade-512-512",
                       help="HuggingFace model name")
    parser.add_argument("--fp16", action="store_true", 
                       help="Use FP16 mixed precision (2x faster on RTX GPUs)")
    parser.add_argument("--tta", action="store_true", 
                       help="Use Test-Time Augmentation (slower but more accurate)")
    
    # Processing Options
    parser.add_argument("--no-preprocessing", action="store_true", 
                       help="Disable image preprocessing (denoise, CLAHE, sharpen)")
    parser.add_argument("--no-postprocessing", action="store_true", 
                       help="Disable segmentation post-processing (morphology, temporal smoothing)")
    parser.add_argument("--no-path", action="store_true", 
                       help="Disable safe path detection and visualization")
    parser.add_argument("--no-obstacles", action="store_true", 
                       help="Disable obstacle detection and highlighting")
    
    # Performance Options
    parser.add_argument("--resize", type=int, default=None, 
                       help="Resize input width (maintains aspect ratio, improves FPS)")
    parser.add_argument("--skip-frames", type=int, default=1, 
                       help="Process every Nth frame (1=all frames, 2=every other frame)")
    
    # Display Options
    parser.add_argument("--no-display", action="store_true", 
                       help="Hide display window (headless mode)")
    parser.add_argument("--show-stats", action="store_true", 
                       help="Show detailed performance statistics")
    
    args = parser.parse_args()

    # 1. Setup Source
    source = args.source
    if "youtube.com" in source or "youtu.be" in source:
        source = download_youtube(source)
    elif source == "0":
        source = 0
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        return

    # 2. Setup Model
    logger.info(f"Initializing Segmentation Engine...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  FP16: {args.fp16}")
    logger.info(f"  TTA: {args.tta}")
    logger.info(f"  Preprocessing: {not args.no_preprocessing}")
    logger.info(f"  Post-processing: {not args.no_postprocessing}")
    logger.info(f"  Path Detection: {not args.no_path}")
    
    try:
        engine = SegmentationEngine(
            model_name=args.model,
            use_preprocessing=not args.no_preprocessing,
            use_postprocessing=not args.no_postprocessing,
            use_path_detection=not args.no_path,
            use_fp16=args.fp16,
            use_tta=args.tta
        )
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        return

    # 3. Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 120:
        fps = 30
    
    # Apply resize if requested
    if args.resize:
        aspect_ratio = height / width
        width = args.resize
        height = int(width * aspect_ratio)
        logger.info(f"Resizing to: {width}x{height}")

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        logger.info(f"Output will be saved to: {args.output}")

    logger.info("Starting processing... Press 'q' to quit.")
    
    # Performance tracking
    prev_time = time.time()
    frame_count = 0
    total_inference_time = 0
    fps_history = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if frame_count % args.skip_frames != 0:
                frame_count += 1
                continue
            
            # Resize frame if requested
            if args.resize:
                frame = cv2.resize(frame, (width, height))

            # Processing
            start_time = time.time()
            overlay, seg_mask, stats = engine.process_frame(
                frame, 
                show_path=not args.no_path,
                show_obstacles=not args.no_obstacles
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # FPS Calculation
            curr_time = time.time()
            fps_curr = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_history.append(fps_curr)
            
            # Keep only last 30 FPS measurements
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Overlay FPS and Stats
            cv2.putText(overlay, f"FPS: {fps_curr:.1f} (avg: {avg_fps:.1f})", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(overlay, "SAFE: Green | UNSAFE: Red", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show additional stats if requested
            if args.show_stats and stats:
                y_offset = 200
                if 'safe_area_ratio' in stats:
                    cv2.putText(overlay, f"Inference: {inference_time*1000:.1f}ms", 
                               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30
                
                # GPU stats
                gpu_stats = engine.get_performance_stats()
                if 'gpu_memory_allocated' in gpu_stats:
                    cv2.putText(overlay, f"GPU Mem: {gpu_stats['gpu_memory_allocated']:.2f}GB", 
                               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write to file
            if out:
                out.write(overlay)

            # Display
            if not args.no_display:
                cv2.imshow('Off-Road Segmentation', overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested quit")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_name = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_name, overlay)
                    logger.info(f"Screenshot saved: {screenshot_name}")
            
            frame_count += 1
            
            # Periodic logging
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames | FPS: {avg_fps:.1f} | "
                          f"Inference: {inference_time*1000:.1f}ms")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Average FPS: {np.mean(fps_history):.2f}")
        logger.info(f"Average inference time: {(total_inference_time/frame_count)*1000:.1f}ms")
        
        if args.output and os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            logger.info(f"Output saved: {args.output} ({file_size:.1f} MB)")
        
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
