# 🔧 Troubleshooting Guide

## Common Issues and Solutions

---

## GPU and CUDA Issues

### Issue 1: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions (in order of preference):**

1. **Reduce Input Resolution**
   ```powershell
   python main_enhanced.py --source video.mp4 --resize 640 --fp16
   ```

2. **Skip Frames**
   ```powershell
   python main_enhanced.py --source video.mp4 --skip-frames 2
   ```

3. **Disable Features**
   ```powershell
   python main_enhanced.py --source video.mp4 --no-preprocessing --no-postprocessing
   ```

4. **Clear GPU Memory**
   ```powershell
   # Close other GPU applications
   # Restart Python script
   # Or run:
   python -c "import torch; torch.cuda.empty_cache()"
   ```

5. **Use Smaller Batch Size (for fine-tuning)**
   ```powershell
   python fine_tuning.py --batch-size 2  # Instead of 4
   ```

---

### Issue 2: "CUDA not available" or "torch.cuda.is_available() returns False"

**Cause:** PyTorch not installed with CUDA support

**Solution:**
```powershell
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

**If still not working:**
1. Check NVIDIA driver: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure CUDA version matches PyTorch version

---

### Issue 3: "RuntimeError: CUDA error: no kernel image available"

**Cause:** PyTorch compiled for different GPU architecture

**Solution:**
```powershell
# Check your GPU compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"

# Reinstall PyTorch (latest version supports most GPUs)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Issue 4: "DLL load failed" or "ImportError: DLL load failed while importing"

**Cause:** Missing or incompatible CUDA libraries

**Solutions:**

1. **Reinstall PyTorch with correct CUDA version**
   ```powershell
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install Visual C++ Redistributable**
   - Download from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   - Install both x64 and x86 versions

3. **Check CUDA installation**
   ```powershell
   # Verify CUDA is in PATH
   echo %PATH%
   # Should include: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   ```

---

## Performance Issues

### Issue 5: Low FPS (< 10 FPS)

**Diagnosis:**
```powershell
python main_enhanced.py --source video.mp4 --show-stats
# Check GPU utilization with: nvidia-smi -l 1
```

**Solutions:**

1. **Enable FP16 (2x speedup)**
   ```powershell
   python main_enhanced.py --source video.mp4 --fp16
   ```

2. **Reduce Resolution**
   ```powershell
   python main_enhanced.py --source video.mp4 --resize 640 --fp16
   ```

3. **Disable Expensive Features**
   ```powershell
   python main_enhanced.py --source video.mp4 --fp16 --no-preprocessing
   ```

4. **Check GPU Utilization**
   - If GPU utilization < 50%, bottleneck is CPU or I/O
   - If GPU utilization > 90%, GPU is maxed out (reduce resolution)

5. **Close Background Applications**
   - Close browser (especially with hardware acceleration)
   - Close games or other GPU applications
   - Disable Windows Game Bar

---

### Issue 6: High GPU Memory Usage but Low Utilization

**Cause:** Model loaded but not processing efficiently

**Solutions:**

1. **Reduce Batch Size (for fine-tuning)**
   ```powershell
   python fine_tuning.py --batch-size 2
   ```

2. **Enable Gradient Checkpointing (for fine-tuning)**
   - Edit `fine_tuning.py` and add `gradient_checkpointing=True` to TrainingArguments

3. **Clear Unused Memory**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

## Installation Issues

### Issue 7: "FFmpeg not found" or "yt-dlp error"

**Cause:** FFmpeg not installed or not in PATH

**Solution (Windows):**

1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit PATH variable
   - Add `C:\ffmpeg\bin`
4. Restart terminal
5. Verify: `ffmpeg -version`

**Solution (Linux):**
```bash
sudo apt update
sudo apt install ffmpeg
```

---

### Issue 8: "ModuleNotFoundError: No module named 'X'"

**Cause:** Missing dependencies

**Solution:**
```powershell
# Reinstall all dependencies
pip install -r requirements.txt

# Or install specific package
pip install transformers opencv-python albumentations
```

---

### Issue 9: "Permission denied" or "Access denied"

**Cause:** Insufficient permissions

**Solutions:**

1. **Run as Administrator (Windows)**
   - Right-click PowerShell → Run as Administrator

2. **Check File Permissions**
   ```powershell
   # Windows
   icacls your_file.py

   # Linux
   chmod +x your_file.py
   ```

3. **Use User Install**
   ```powershell
   pip install --user -r requirements.txt
   ```

---

## Model Issues

### Issue 10: "Model download fails" or "Connection timeout"

**Cause:** Network issues or firewall blocking HuggingFace

**Solutions:**

1. **Use Proxy (if behind firewall)**
   ```powershell
   set HTTP_PROXY=http://proxy:port
   set HTTPS_PROXY=http://proxy:port
   pip install -r requirements.txt
   ```

2. **Download Model Manually**
   ```powershell
   # Download from HuggingFace website
   # Place in: C:\Users\YourName\.cache\huggingface\hub\
   ```

3. **Use Mirror (China users)**
   ```python
   # Edit segmentation_engine.py
   # Add: mirror="https://hf-mirror.com"
   ```

---

### Issue 11: "RuntimeError: Expected all tensors to be on the same device"

**Cause:** Model and input on different devices (CPU vs GPU)

**Solution:**
- This should be handled automatically by the code
- If you modified the code, ensure:
  ```python
  model = model.to(device)
  inputs = {k: v.to(device) for k, v in inputs.items()}
  ```

---

## Video Processing Issues

### Issue 12: "Could not open video source"

**Cause:** Invalid video file or codec not supported

**Solutions:**

1. **Check Video File**
   ```powershell
   # Try opening with VLC or Windows Media Player
   ```

2. **Convert Video Format**
   ```powershell
   ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
   ```

3. **Check File Path**
   ```powershell
   # Use absolute path
   python main_enhanced.py --source "C:\full\path\to\video.mp4"
   ```

---

### Issue 13: "YouTube download fails"

**Cause:** yt-dlp outdated or video restricted

**Solutions:**

1. **Update yt-dlp**
   ```powershell
   pip install --upgrade yt-dlp
   ```

2. **Download Manually**
   ```powershell
   yt-dlp "https://youtube.com/watch?v=..." -o video.mp4
   python main_enhanced.py --source video.mp4
   ```

3. **Check Video Availability**
   - Some videos are region-restricted
   - Some videos require authentication

---

## Fine-Tuning Issues

### Issue 14: "Training loss not decreasing"

**Causes and Solutions:**

1. **Learning Rate Too High**
   ```powershell
   python fine_tuning.py --lr 1e-5  # Instead of 5e-5
   ```

2. **Insufficient Data**
   - Need at least 100 training images
   - Ensure diverse examples

3. **Data Quality Issues**
   - Check mask annotations are correct
   - Verify class IDs match model expectations

4. **Enable Data Augmentation**
   - Already enabled by default in `fine_tuning.py`

---

### Issue 15: "Validation accuracy worse than training"

**Cause:** Overfitting

**Solutions:**

1. **Reduce Epochs**
   ```powershell
   python fine_tuning.py --epochs 10  # Instead of 20
   ```

2. **Add More Training Data**
   - Collect more diverse examples

3. **Increase Augmentation**
   - Edit `fine_tuning.py` to add more augmentation transforms

---

## Web Interface Issues

### Issue 16: "Flask app not starting" or "Port already in use"

**Solutions:**

1. **Change Port**
   ```python
   # Edit app.py, change:
   app.run(host='0.0.0.0', port=5001)  # Instead of 5000
   ```

2. **Kill Existing Process**
   ```powershell
   # Windows
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F

   # Linux
   lsof -ti:5000 | xargs kill -9
   ```

---

### Issue 17: "Video not streaming in browser"

**Cause:** Browser compatibility or codec issues

**Solutions:**

1. **Try Different Browser**
   - Chrome/Edge usually work best
   - Firefox may have issues with MJPEG streams

2. **Check Console for Errors**
   - Open browser DevTools (F12)
   - Check Console and Network tabs

3. **Reduce Frame Rate**
   - Edit `app.py` to skip frames if needed

---

## System-Specific Issues

### Windows-Specific

**Issue 18: "PowerShell execution policy error"**

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue 19: "Long path names not supported"**

**Solution:**
```powershell
# Enable long paths in Windows
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

---

### Linux-Specific

**Issue 20: "libGL.so.1: cannot open shared object file"**

**Solution:**
```bash
sudo apt install libgl1-mesa-glx
```

**Issue 21: "Permission denied: /dev/nvidia0"**

**Solution:**
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Logout and login again
```

---

## Diagnostic Commands

### Check System Status

```powershell
# GPU status
nvidia-smi

# Python packages
pip list

# PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Version: {torch.__version__}')"

# Disk space
dir  # Windows
df -h  # Linux

# Memory usage
wmic OS get FreePhysicalMemory  # Windows
free -h  # Linux
```

### Performance Profiling

```powershell
# Monitor GPU in real-time
nvidia-smi -l 1

# Profile Python script
python -m cProfile -o profile.stats main_enhanced.py --source video.mp4
python -m pstats profile.stats
```

---

## Getting Help

If none of these solutions work:

1. **Check Logs**
   - Look for error messages in terminal output
   - Check Python traceback for specific errors

2. **Verify Installation**
   ```powershell
   python quick_start.py
   ```

3. **Minimal Reproducible Example**
   ```powershell
   # Test with minimal features
   python main_enhanced.py --source video.mp4 --no-preprocessing --no-postprocessing --no-path
   ```

4. **System Information**
   ```powershell
   # Collect system info
   nvidia-smi > system_info.txt
   python -c "import torch; print(torch.__version__)" >> system_info.txt
   pip list >> system_info.txt
   ```

---

## Prevention Tips

1. **Keep Software Updated**
   ```powershell
   pip install --upgrade torch torchvision transformers
   ```

2. **Use Virtual Environments**
   - Prevents dependency conflicts
   - Isolates project dependencies

3. **Monitor Resources**
   - Check GPU memory before running
   - Close unnecessary applications

4. **Test Incrementally**
   - Start with basic features
   - Add advanced features one at a time

5. **Backup Working Configuration**
   ```powershell
   pip freeze > working_requirements.txt
   ```

---

## Still Having Issues?

1. Read [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup
2. Check [HACKATHON_GUIDE.md](HACKATHON_GUIDE.md) for demo tips
3. Review error messages carefully
4. Search for specific error messages online
5. Check PyTorch and CUDA compatibility matrix

Good luck! 🚀
