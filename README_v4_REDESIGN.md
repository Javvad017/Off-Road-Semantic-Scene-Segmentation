# Off-Road Segmentation & Path Planning (Redesign)

## Overview
This is a complete redesign of the segmentation system to include:
1.  **Risk-Aware Cost Mapping**: Distinguishes between Safe (Road), Moderate (Grass/Dirt), and Dangerous (Water/Obstacles).
2.  **A* Path Planning**: Finds the global optimal path instead of greedy local search.
3.  **Visualization**: Clear overlays for Safe/Danger zones and the planned path.

## Setup
1.  Ensure you have the required libraries:
    ```bash
    pip install torch torchvision transformers opencv-python numpy pillow
    ```
    *Note: If you have `opencv-python-headless` installed, `cv2.imshow` might not work. Install `opencv-python` for the display window.*

## Usage
Run the script with a video file:
```bash
python main_redesign.py "path/to/your/video.mp4" --output "results.mp4"
```

## Key Features
*   **Model**: Uses `nvidia/segformer-b1-finetuned-ade-512-512` for better accuracy than B0 while keeping high FPS.
*   **Path Planning**: Using A* algorithm on a downsampled cost map (1/8 resolution) ensures real-time performance (>30 FPS).
*   **Risk Map**: 
    *   Green: Safe
    *   Red: Dangerous/Obstacles
    *   Yellow Line: Optimal Path

## Troubleshooting
*   **No Path Found**: If the path is blocked by obstacles (red zones), the system will return no path.
*   **Low FPS**: If running on CPU, performance will be slower. Ensure CUDA is available.
