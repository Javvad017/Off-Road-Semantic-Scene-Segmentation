# ⚡ Quick Reference Card

## Essential Commands

### Setup (One-Time)
```powershell
# Create environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify setup
python quick_start.py
```

### Basic Usage
```powershell
# Process video (basic)
python main_enhanced.py --source video.mp4 --output result.mp4

# Process video (optimized)
python main_enhanced.py --source video.mp4 --fp16 --output result.mp4

# Web interface
python app.py
```

### Advanced Options
```powershell
# Maximum performance
python main_enhanced.py --source video.mp4 --fp16 --resize 640

# Maximum accuracy
python main_enhanced.py --source video.mp4 --tta

# Show statistics
python main_enhanced.py --source video.mp4 --show-stats --fp16

# Headless mode
python main_enhanced.py --source video.mp4 --no-display --fp16
```

---

## Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--source` | Input video/URL/webcam | `--source video.mp4` |
| `--output` | Output filename | `--output result.mp4` |
| `--fp16` | Enable FP16 (2x faster) | `--fp16` |
| `--tta` | Test-time augmentation | `--tta` |
| `--resize` | Resize width | `--resize 640` |
| `--skip-frames` | Process every Nth frame | `--skip-frames 2` |
| `--show-stats` | Show detailed stats | `--show-stats` |
| `--no-display` | Headless mode | `--no-display` |
| `--no-preprocessing` | Disable preprocessing | `--no-preprocessing` |
| `--no-postprocessing` | Disable post-processing | `--no-postprocessing` |
| `--no-path` | Disable path detection | `--no-path` |

---

## Performance Tuning

### If FPS is Low
```powershell
# Try in order:
python main_enhanced.py --source video.mp4 --fp16
python main_enhanced.py --source video.mp4 --fp16 --resize 640
python main_enhanced.py --source video.mp4 --fp16 --no-preprocessing
python main_enhanced.py --source video.mp4 --fp16 --skip-frames 2
```

### If CUDA Out of Memory
```powershell
# Try in order:
python main_enhanced.py --source video.mp4 --resize 640
python main_enhanced.py --source video.mp4 --resize 640 --no-postprocessing
python main_enhanced.py --source video.mp4 --resize 480
```

---

## File Locations

| File | Purpose |
|------|---------|
| `main_enhanced.py` | Advanced CLI |
| `app.py` | Web interface |
| `segmentation_engine.py` | Core model |
| `preprocessing.py` | Image enhancement |
| `postprocessing.py` | Mask refinement |
| `path_planning.py` | Path detection |
| `fine_tuning.py` | Model training |
| `evaluate_model.py` | Metrics |

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| CUDA not available | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| Out of memory | Add `--resize 640` |
| Low FPS | Add `--fp16` |
| Module not found | `pip install -r requirements.txt` |
| FFmpeg not found | Download from ffmpeg.org, add to PATH |
| Video won't open | Check file path, try different format |

---

## Keyboard Shortcuts (During Processing)

| Key | Action |
|-----|--------|
| `q` | Quit processing |
| `s` | Save screenshot |

---

## Expected Performance

| GPU | FP32 FPS | FP16 FPS |
|-----|----------|----------|
| RTX 4090 | 50 | 90 |
| RTX 4070 | 35 | 60 |
| RTX 3060 | 25 | 45 |
| RTX 2060 | 18 | 32 |

---

## Fine-Tuning Quick Start

```powershell
# Create dataset structure
python fine_tuning.py --create-structure

# Train model
python fine_tuning.py \
  --train-images ./dataset/train/images \
  --train-masks ./dataset/train/masks \
  --val-images ./dataset/val/images \
  --val-masks ./dataset/val/masks \
  --epochs 20 \
  --batch-size 4
```

---

## Evaluation Quick Start

```powershell
# Evaluate accuracy
python evaluate_model.py --mode evaluate \
  --images ./test/images \
  --masks ./test/masks

# Benchmark speed
python evaluate_model.py --mode benchmark \
  --iterations 100
```

---

## Web Interface

```powershell
# Start server
python app.py

# Open browser
http://localhost:5000

# Upload video and watch results
```

---

## Diagnostic Commands

```powershell
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check packages
pip list

# Verify setup
python quick_start.py
```

---

## Documentation

| File | When to Read |
|------|--------------|
| `README.md` | First time |
| `SETUP_GUIDE.md` | Installation issues |
| `HACKATHON_GUIDE.md` | Before presentation |
| `TROUBLESHOOTING.md` | When stuck |
| `PROJECT_SUMMARY.md` | Understanding architecture |
| `QUICK_REFERENCE.md` | This file (quick lookup) |

---

## Support

1. Check error message
2. Read TROUBLESHOOTING.md
3. Verify GPU: `nvidia-smi`
4. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
5. Reinstall if needed

---

## Hackathon Checklist

### Before Demo
- [ ] Test on demo videos
- [ ] Verify FPS acceptable
- [ ] Practice pitch (2:30)
- [ ] Prepare backup plan
- [ ] Charge laptop

### During Demo
- [ ] Start with impact
- [ ] Show live demo
- [ ] Explain innovation
- [ ] Quantify results
- [ ] End with vision

---

## Quick Tips

💡 **Always use `--fp16` on RTX GPUs** (2x speedup)  
💡 **Use `--resize 640` if FPS < 20**  
💡 **Test with `--show-stats` to see bottlenecks**  
💡 **Pre-download model before demo** (run once)  
💡 **Have backup video ready** (don't rely on live processing)

---

## One-Liners

```powershell
# Best quality
python main_enhanced.py --source video.mp4 --tta --output best.mp4

# Best speed
python main_enhanced.py --source video.mp4 --fp16 --resize 640 --no-preprocessing --output fast.mp4

# Balanced
python main_enhanced.py --source video.mp4 --fp16 --output balanced.mp4

# YouTube
python main_enhanced.py --source "https://youtube.com/watch?v=..." --fp16 --output youtube.mp4
```

---

**Print this page for quick reference during hackathon! 📄**
