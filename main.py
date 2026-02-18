import cv2
import time
import argparse
import os
import torch
import numpy as np
from PIL import Image

from src.segmentation import SegmentationEngine
from src.preprocessing import Preprocessor
from src.postprocessing import Postprocessor
from src.utils import download_youtube_video, FPSMeter, get_gpu_memory_usage, create_video_writer

def main():
    parser = argparse.ArgumentParser(description="Advanced Off-Road Semantic Segmentation CLI (Video Only)")
    parser.add_argument("--source", type=str, required=True, help="Path to video file OR YouTube URL")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="HuggingFace model name")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Output video filename")
    parser.add_argument("--no-display", action="store_true", help="Hide display window (headless mode)")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold (if applicable)")
    args = parser.parse_args()

    print("=== Off-Road Semantic Segmentation System (Offline/Video Mode) ===")
    
    # 1. Setup Source
    source = args.source
    
    # Block webcam usage intentionally
    if source == "0" or source.lower() == "webcam":
        print("Error: Webcam usage is disabled for this version. Please provide a video file or YouTube URL.")
        return

    if "youtube.com" in source or "youtu.be" in source:
        print(f"Detected YouTube URL: {source}")
        source = download_youtube_video(source)
        if not source:
            print("Failed to download video. Exiting.")
            return
    
    if not os.path.exists(source):
        print(f"Error: Video file not found: {source}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # 2. Setup Modules
    print("Initializing segmentation engine...")
    engine = SegmentationEngine(model_name=args.model)
    postprocessor = Postprocessor(safe_classes=engine.safe_classes)
    
    # Preprocessor Strategy
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize large 4K videos to 640p for speed, keep small videos as is or upsample
    target_width = 640 
    if original_width > target_width:
        target_height = int(original_height * (target_width / original_width))
        print(f"Resizing input from {original_width}x{original_height} to {target_width}x{target_height} for performance.")
        preprocessor = Preprocessor(target_size=(target_width, target_height))
        output_size = (target_width, target_height)
    else:
        preprocessor = Preprocessor(target_size=(original_width, original_height))
        output_size = (original_width, original_height)
    
    # Video Writer
    out = None
    if args.output:
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in <= 0: fps_in = 30
        out = create_video_writer(args.output, fps_in, output_size[0], output_size[1])

    # FPS Tracker
    fps_meter = FPSMeter()
    
    print("Processing started. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # A. Preprocessing
        processed_frame = preprocessor.preprocess_pipeline(frame)
        
        # B. Inference
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        pred_seg_np = engine.process_frame(pil_image)
        
        if pred_seg_np is not None:
            # C. Post-processing
            color_mask = engine.get_color_mask(pred_seg_np)
            safe_mask = postprocessor.extract_safe_mask(pred_seg_np)
            path_data = postprocessor.find_safe_path(safe_mask) # returns (contour, centroid) or None
            
            # D. Visualization
            output_frame = postprocessor.overlay_mask(processed_frame, color_mask, alpha=0.4)
            output_frame = postprocessor.draw_safe_path(output_frame, path_data)
        else:
            output_frame = processed_frame

        # E. Stats Display
        fps = fps_meter.update()
        frame_count += 1
        
        # Overlay UI
        cv2.rectangle(output_frame, (10, 10), (280, 130), (0, 0, 0), -1) 
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(output_frame, f"GPU RAM: {get_gpu_memory_usage()}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if out:
            out.write(output_frame)

        if not args.no_display:
            cv2.imshow('Offline Segmentation Demo', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\nProcessing Complete.")
    print(f"Total Frames: {frame_count}")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
