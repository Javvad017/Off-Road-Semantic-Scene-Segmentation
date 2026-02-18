# 🚀 Complete Setup Guide - Off-Road Semantic Segmentation System

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [CUDA Installation](#cuda-installation)
3. [PyTorch Installation](#pytorch-installation)
4. [Project Setup](#project-setup)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Windows 10/11

#### Prerequisites
1. **Python 3.8-3.11** (3.10 recommended)
   - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   - Verify: `python --version`

2. **Git** (optional, for cloning)
   - Download from [git-scm.com](https://git-scm.com/)

3. **Visual Studio Build Tools** (for some packages)
   - Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"

#### Step 1: Check GPU Compatibility
```powershell
# Open PowerShell and run:
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    15W / 250W |    500MiB /  8192MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
```

Note your CUDA Version (e.g., 12.2, 11.8)

---

## CUDA Installation

### Option 1: CUDA 11.8 (Recommended for RTX 20/30 series)

1. Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Run installer (Express Installation)
3. Verify installation:
```powershell
nvcc --version
```

### Option 2: CUDA 12.1+ (For RTX 40 series)

1. Download [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
2. Run installer
3. Verify installation

### cuDNN Installation (Optional but Recommended)

1. Download cuDNN from [NVIDIA Developer](https://developer.nvidia.com/cudnn) (requires free account)
2. Extract and copy files to CUDA installation directory:
   - Copy `bin` files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - Copy `include` files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
   - Copy `lib` files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64`

---

## PyTorch Installation

### Step 1: Create Virtual Environment

```powershell
# Navigate to your project folder
cd C:\path\to\your\project

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate (Windows CMD)
.\venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Install PyTorch with CUDA

**For CUDA 11.8:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU Only (Not Recommended):**
```powershell
pip install torch torchvision torchaudio
```

### Step 3: Verify PyTorch Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA Available: True
CUDA Version: 11.8
GPU: NVIDIA GeForce RTX 3060
```

---

## Project Setup

### Step 1: Install Dependencies

```powershell
# Make sure virtual environment is activated
pip install -r requirements.txt
```

### Step 2: Download Model (First Run)

The model will auto-download on first run (~100MB). To pre-download:

```powershell
python -c "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation; SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512'); SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')"
```

### Step 3: Test Installation

```powershell
# Test with a sample video (replace with your video path)
python main_enhanced.py --source "uploads/your_video.mp4" --output test_output.mp4 --show-stats
```

---

## Verification

### GPU Utilization Check

While processing is running, open another terminal:

```powershell
# Monitor GPU usage in real-time
nvidia-smi -l 1
```

You should see:
- GPU Utilization: 70-95%
- Memory Usage: 2-6 GB (depending on model and resolution)
- Temperature: 60-80°C

### Performance Benchmarks

Expected FPS on different GPUs (1080p video):

| GPU | FP32 FPS | FP16 FPS | VRAM Usage |
|-----|----------|----------|------------|
| RTX 4090 | 45-60 | 80-100 | 3-4 GB |
| RTX 4070 | 30-40 | 50-70 | 3-4 GB |
| RTX 3060 | 20-30 | 35-50 | 4-5 GB |
| RTX 2060 | 15-25 | 25-40 | 4-6 GB |
| GTX 1660 Ti | 10-18 | N/A | 5-6 GB |

---

## Troubleshooting

### Issue 1: "CUDA out of memory"

**Solutions:**
```powershell
# Option 1: Resize input video
python main_enhanced.py --source video.mp4 --resize 640

# Option 2: Skip frames
python main_enhanced.py --source video.mp4 --skip-frames 2

# Option 3: Disable some features
python main_enhanced.py --source video.mp4 --no-preprocessing --no-postprocessing
```

### Issue 2: "DLL load failed" or "ImportError: torch"

**Cause:** Wrong PyTorch version for your CUDA

**Solution:**
```powershell
# Uninstall PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: "FFmpeg not found" (for YouTube downloads)

**Solution:**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH
4. Restart terminal

### Issue 4: Low FPS / Slow Processing

**Solutions:**
```powershell
# Enable FP16 (2x speedup on RTX GPUs)
python main_enhanced.py --source video.mp4 --fp16

# Reduce resolution
python main_enhanced.py --source video.mp4 --resize 640 --fp16

# Disable expensive features
python main_enhanced.py --source video.mp4 --fp16 --no-preprocessing
```

### Issue 5: "RuntimeError: CUDA error: no kernel image available"

**Cause:** PyTorch compiled for different GPU architecture

**Solution:**
```powershell
# Check your GPU compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"

# Reinstall PyTorch from source or use compatible wheel
```

### Issue 6: Model Download Fails

**Solution:**
```powershell
# Set proxy if behind firewall
set HTTP_PROXY=http://proxy:port
set HTTPS_PROXY=http://proxy:port

# Or download manually and load from local path
# Edit segmentation_engine.py to use local model path
```

---

## Linux Setup (Ubuntu 20.04/22.04)

### Quick Setup Script

```bash
#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-11-8 -y

# Create project directory
mkdir -p ~/offroad_segmentation
cd ~/offroad_segmentation

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Next Steps

1. ✅ Run test video: `python main_enhanced.py --source test.mp4 --show-stats`
2. ✅ Try YouTube video: `python main_enhanced.py --source "https://youtube.com/watch?v=..." --fp16`
3. ✅ Optimize for your GPU: Experiment with `--fp16`, `--resize`, `--skip-frames`
4. ✅ Read [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md) for custom dataset training
5. ✅ Read [HACKATHON_GUIDE.md](HACKATHON_GUIDE.md) for presentation tips

---

## Support

If you encounter issues:
1. Check GPU compatibility: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check logs in terminal for specific error messages
4. Try CPU mode first: Remove `--index-url` from PyTorch install command

Good luck! 🚀
