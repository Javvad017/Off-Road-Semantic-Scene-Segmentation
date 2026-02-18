# 🎬 Video-Only Processing Guide

## Complete Guide for Pre-Recorded Video Segmentation

This system is specifically designed for **offline video processing** - no webcam or live camera support.

---

## 🎯 Supported Input Methods

### ✅ Supported
1. **Local Video Files**
   - MP4, AVI, MOV, MKV, WebM
   - Any resolution (auto-resized for performance)
   - Any frame rate

2. **YouTube Videos**
   - Direct URL download
   - Automatic format selection
   - Saved locally for processing

### ❌ Not Supported
- Webcam/live camera input
- Real-time streaming
- RTSP/RTMP streams
- IP cameras

---

## 🚀 Quick Start - Video Processing

### Method 1: Command Line (Recommended for Hackathon)

```powershell
# Process local video file
python main_enhanced.py --source path/to/video.mp4 --output result.mp4 --fp16

# Process YouTube video
python main_enhanced.py --source "https://youtube.com/watch?v=VIDEO_ID" --output result.mp4 --fp16

# Batch process multiple videos
python main_enhanced.py --source video1.mp4 --output result1.mp4 --fp16
python main_enhanced.py --source video2.mp4 --output result2.mp4 --fp16
python main_enhanced.py --source video3.mp4 --output result3.mp4 --fp16
```

### Method 2: Web Interface (Best for Demos)

```powershell
# Start web server
python app.py

# Open browser
http://localhost:5000

# Upload video file OR paste YouTube URL
# Watch processing in real-time
```

---

## 📁 Organizing Your Videos

### Recommended Folder Structure

```
project/
├── demo_videos/              # Input videos for demo
│   ├── 01_easy_terrain.mp4
│   ├── 02_challenging.mp4
│   └── 03_mixed_terrain.mp4
├── test_videos/              # Videos for testing
│   ├── test1.mp4
│   └── test2.mp4
├── output_videos/            # Processed results
│   ├── 01_easy_terrain_result.mp4
│   ├── 02_challenging_result.mp4
│   └── 03_mixed_terrain_result.mp4
└── uploads/                  # Web interface uploads
    └── (auto-managed)
```

### Create Folders

```powershell
# Windows
mkdir demo_videos test_videos output_videos

# Linux
mkdir -p demo_videos test_videos output_videos
```

---

## 🎥 Downloading Videos from YouTube

### Using yt-dlp (Recommended)

```powershell
# Install yt-dlp
pip install yt-dlp

# Download single video
yt-dlp "https://youtube.com/watch?v=VIDEO_ID" -o demo_videos/terrain1.mp4

# Download with specific quality
yt-dlp "https://youtube.com/watch?v=VIDEO_ID" -f "best[height<=1080]" -o demo_videos/terrain1.mp4

# Download playlist
yt-dlp "https://youtube.com/playlist?list=PLAYLIST_ID" -o "demo_videos/%(title)s.%(ext)s"

# Download only first 60 seconds (for demos)
yt-dlp "https://youtube.com/watch?v=VIDEO_ID" --download-sections "*0-60" -o demo_videos/short_clip.mp4
```

### Finding Good Off-Road Videos

**Search Terms:**
- "off road driving POV"
- "4x4 trail camera"
- "off road terrain"
- "dirt road driving"
- "forest trail driving"
- "desert off road"

**Recommended Channels:**
- Off-road adventure channels
- 4x4 enthusiast channels
- Trail riding channels
- Overlanding channels

**Quality Criteria:**
- ✅ Clear terrain visibility
- ✅ Good lighting (daytime)
- ✅ Stable camera mount
- ✅ 720p or 1080p resolution
- ✅ 30-60 seconds duration
- ❌ Avoid night footage
- ❌ Avoid very shaky footage
- ❌ Avoid low resolution

---

## ⚙️ Processing Options

### Basic Processing

```powershell
# Simplest command
python main_enhanced.py --source video.mp4 --output result.mp4
```

### Optimized for Speed

```powershell
# Enable FP16 + resize for maximum FPS
python main_enhanced.py --source video.mp4 --output result.mp4 --fp16 --resize 640
```

### Optimized for Quality

```powershell
# Enable Test-Time Augmentation for better accuracy
python main_enhanced.py --source video.mp4 --output result.mp4 --tta
```

### Headless Mode (No Display)

```powershell
# Process without showing window (faster)
python main_enhanced.py --source video.mp4 --output result.mp4 --fp16 --no-display
```

### With Statistics

```powershell
# Show detailed performance stats
python main_enhanced.py --source video.mp4 --output result.mp4 --fp16 --show-stats
```

### Disable Features (Troubleshooting)

```powershell
# Minimal processing (fastest)
python main_enhanced.py --source video.mp4 --output result.mp4 --no-preprocessing --no-postprocessing --no-path
```

---

## 📊 Batch Processing Script

### Process Multiple Videos Automatically

Create `batch_process.py`:

```python
"""
Batch Video Processing Script
Processes all videos in a folder automatically
"""

import os
import subprocess
from pathlib import Path

# Configuration
INPUT_DIR = "demo_videos"
OUTPUT_DIR = "output_videos"
ENABLE_FP16 = True
RESIZE_WIDTH = 640

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all video files
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
input_path = Path(INPUT_DIR)
video_files = []

for ext in video_extensions:
    video_files.extend(input_path.glob(f'*{ext}'))

print(f"Found {len(video_files)} videos to process")

# Process each video
for i, video_file in enumerate(video_files, 1):
    print(f"\n{'='*60}")
    print(f"Processing {i}/{len(video_files)}: {video_file.name}")
    print(f"{'='*60}")
    
    # Build output filename
    output_name = video_file.stem + "_result" + video_file.suffix
    output_path = Path(OUTPUT_DIR) / output_name
    
    # Build command
    cmd = [
        "python", "main_enhanced.py",
        "--source", str(video_file),
        "--output", str(output_path),
        "--no-display"  # Don't show window for batch processing
    ]
    
    if ENABLE_FP16:
        cmd.append("--fp16")
    
    if RESIZE_WIDTH:
        cmd.extend(["--resize", str(RESIZE_WIDTH)])
    
    # Run processing
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Successfully processed: {output_name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process: {video_file.name}")
        print(f"  Error: {e}")
        continue

print(f"\n{'='*60}")
print(f"Batch processing complete!")
print(f"Processed {len(video_files)} videos")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
```

Run:
```powershell
python batch_process.py
```

---

## 🎬 Creating Demo Clips

### Extract Specific Segments

```powershell
# Extract 30 seconds starting at 1:00
ffmpeg -i input.mp4 -ss 00:01:00 -t 00:00:30 -c copy demo_clip.mp4

# Extract multiple segments
ffmpeg -i input.mp4 -ss 00:00:30 -t 00:00:20 -c copy clip1.mp4
ffmpeg -i input.mp4 -ss 00:02:00 -t 00:00:20 -c copy clip2.mp4
```

### Compress Videos (Reduce File Size)

```powershell
# Compress to smaller size (good quality)
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 compressed.mp4

# Compress more aggressively
ffmpeg -i input.mp4 -vcodec libx264 -crf 32 -preset fast compressed.mp4

# Resize and compress
ffmpeg -i input.mp4 -vf scale=1280:720 -vcodec libx264 -crf 28 compressed_720p.mp4
```

### Concatenate Multiple Videos

Create `filelist.txt`:
```
file 'video1.mp4'
file 'video2.mp4'
file 'video3.mp4'
```

Run:
```powershell
ffmpeg -f concat -safe 0 -i filelist.txt -c copy combined.mp4
```

---

## 📈 Performance Optimization

### Resolution vs Speed Trade-off

| Resolution | RTX 3060 FPS | RTX 4070 FPS | Quality | Use Case |
|------------|--------------|--------------|---------|----------|
| 1920x1080 | 25 | 35 | Best | Final output |
| 1280x720 | 40 | 60 | Good | Balanced |
| 640x480 | 80 | 120 | Fair | Live demo |

### Recommended Settings by Use Case

**For Final Output (Quality Priority):**
```powershell
python main_enhanced.py --source video.mp4 --output result.mp4 --tta
```

**For Live Demo (Speed Priority):**
```powershell
python main_enhanced.py --source video.mp4 --output result.mp4 --fp16 --resize 640
```

**For Batch Processing (Balanced):**
```powershell
python main_enhanced.py --source video.mp4 --output result.mp4 --fp16 --no-display
```

---

## 🎯 Demo Preparation Workflow

### 1. Select Videos (1 hour)

```powershell
# Download 5-10 candidate videos
yt-dlp "URL1" -o candidates/video1.mp4
yt-dlp "URL2" -o candidates/video2.mp4
# ... etc
```

### 2. Test Process All (30 minutes)

```powershell
# Quick test with low quality
for video in candidates/*.mp4; do
    python main_enhanced.py --source "$video" --output "test_$(basename $video)" --fp16 --resize 640 --no-display
done
```

### 3. Select Best 3 (15 minutes)

Watch all test outputs and select:
- 1 easy terrain (shows baseline)
- 1 challenging terrain (shows robustness)
- 1 mixed terrain (shows path planning)

### 4. Generate Final Outputs (30 minutes)

```powershell
# Process selected videos with best quality
python main_enhanced.py --source demo_videos/easy.mp4 --output output_videos/easy_result.mp4 --fp16
python main_enhanced.py --source demo_videos/challenging.mp4 --output output_videos/challenging_result.mp4 --fp16
python main_enhanced.py --source demo_videos/mixed.mp4 --output output_videos/mixed_result.mp4 --fp16
```

### 5. Create Comparison Videos (15 minutes)

```powershell
# Side-by-side comparison
ffmpeg -i demo_videos/challenging.mp4 -i output_videos/challenging_result.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" comparison.mp4
```

---

## 🔍 Quality Assessment

### Check Output Quality

```python
# Create quality_check.py
import cv2
import numpy as np

def check_video_quality(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Video: {video_path}")
    print(f"{'='*60}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f}s")
    
    # Sample frames for quality check
    sample_frames = []
    for i in range(0, frame_count, max(1, frame_count // 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
    
    # Check brightness
    brightnesses = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in sample_frames]
    avg_brightness = np.mean(brightnesses)
    
    print(f"Average Brightness: {avg_brightness:.1f}/255")
    
    if avg_brightness < 50:
        print("⚠️  Warning: Video is very dark")
    elif avg_brightness > 200:
        print("⚠️  Warning: Video is overexposed")
    else:
        print("✓ Brightness is good")
    
    # Check sharpness (Laplacian variance)
    sharpnesses = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() 
                   for f in sample_frames]
    avg_sharpness = np.mean(sharpnesses)
    
    print(f"Average Sharpness: {avg_sharpness:.1f}")
    
    if avg_sharpness < 100:
        print("⚠️  Warning: Video is blurry")
    else:
        print("✓ Sharpness is good")
    
    cap.release()
    print(f"{'='*60}\n")

# Check all demo videos
import glob
for video in glob.glob("demo_videos/*.mp4"):
    check_video_quality(video)
```

Run:
```powershell
python quality_check.py
```

---

## 🎬 Web Interface Usage

### Starting the Server

```powershell
python app.py
```

### Uploading Videos

1. Open browser: `http://localhost:5000`
2. Click "Choose File" or drag-and-drop
3. Or paste YouTube URL
4. Click "Upload" or "Download"
5. Watch processing in real-time

### Features

- ✅ Upload local video files
- ✅ Download from YouTube
- ✅ Real-time processing visualization
- ✅ Automatic video management
- ✅ Multiple video support

### Tips for Demo

1. **Pre-upload videos** before demo starts
2. **Test upload speed** with your video sizes
3. **Have backup** if upload fails
4. **Use short videos** (30-60s) for faster processing

---

## 🐛 Troubleshooting Video Issues

### Issue: "Could not open video source"

**Solutions:**
```powershell
# Check if file exists
dir path\to\video.mp4  # Windows
ls path/to/video.mp4   # Linux

# Try different path format
python main_enhanced.py --source "C:\full\path\to\video.mp4"

# Check video codec
ffmpeg -i video.mp4
```

### Issue: "Video processing is slow"

**Solutions:**
```powershell
# Enable FP16
python main_enhanced.py --source video.mp4 --fp16

# Reduce resolution
python main_enhanced.py --source video.mp4 --resize 640 --fp16

# Skip frames
python main_enhanced.py --source video.mp4 --skip-frames 2 --fp16
```

### Issue: "Output video is corrupted"

**Solutions:**
```powershell
# Try different codec
# Edit main_enhanced.py, change:
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # to
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Or use ffmpeg to re-encode
ffmpeg -i output.mp4 -c:v libx264 output_fixed.mp4
```

### Issue: "YouTube download fails"

**Solutions:**
```powershell
# Update yt-dlp
pip install --upgrade yt-dlp

# Try different format
yt-dlp "URL" -f "best[ext=mp4]" -o video.mp4

# Download manually and process
yt-dlp "URL" -o video.mp4
python main_enhanced.py --source video.mp4 --output result.mp4
```

---

## 📦 Creating Portable Demo Package

### Package Structure

```
demo_package/
├── main_enhanced.py
├── segmentation_engine.py
├── preprocessing.py
├── postprocessing.py
├── path_planning.py
├── requirements.txt
├── models/
│   └── segformer-b0/  (pre-downloaded)
├── demo_videos/
│   ├── easy.mp4
│   ├── challenging.mp4
│   └── mixed.mp4
├── output_videos/
│   ├── easy_result.mp4
│   ├── challenging_result.mp4
│   └── mixed_result.mp4
├── run_demo.bat
└── README.txt
```

### Create Package

```powershell
# 1. Pre-download model
python -c "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation; SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512').save_pretrained('./models/segformer-b0'); SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512').save_pretrained('./models/segformer-b0')"

# 2. Copy demo videos
copy demo_videos\*.mp4 demo_package\demo_videos\

# 3. Pre-generate results
python main_enhanced.py --source demo_videos/easy.mp4 --output demo_package/output_videos/easy_result.mp4 --fp16 --no-display

# 4. Create ZIP
Compress-Archive -Path demo_package -DestinationPath offroad_demo.zip
```

---

## 🎯 Final Checklist

### Before Hackathon

- [ ] Download 5-10 candidate videos
- [ ] Test process all videos
- [ ] Select best 3 videos
- [ ] Generate final outputs
- [ ] Create comparison videos
- [ ] Pre-download model
- [ ] Test offline (disconnect internet)
- [ ] Create portable package
- [ ] Backup everything to USB

### At Hackathon

- [ ] Test GPU: `nvidia-smi`
- [ ] Test processing: `python main_enhanced.py --source demo_videos/easy.mp4 --fp16`
- [ ] Have pre-generated results ready
- [ ] Have backup plan ready
- [ ] Stay calm and confident!

---

## 🚀 You're Ready!

You now have everything you need to process pre-recorded videos for your hackathon demo. No webcam or live camera required!

**Good luck! 🏆**
