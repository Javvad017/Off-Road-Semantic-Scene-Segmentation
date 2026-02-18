# 🚙 Off-Road Semantic Segmentation (Hackathon Edition)

> **Premium AI-Powered Terrain Analysis System**
>
> *Built for high-performance offline processing with modern UI/UX.*

![Status](https://img.shields.io/badge/Status-Hackathon--Ready-brightgreen)
![Tech](https://img.shields.io/badge/Tech-Flask%20%7C%20PyTorch%20%7C%20CUDA%20%7C%20SegFormer-blue)

## 🌟 Key Features

*   **Advanced AI**: Powered by NVIDIA SegFormer (Transformer-based) for state-of-the-art accuracy.
*   **Local GPU Processing**: Fully optimized for CUDA (NVIDIA GPUs) to run privately and securely.
*   **Modern Dashboard**: Smooth, dark-themed responsive Web UI with drag-and-drop.
*   **Real-time Feedback**: Live progress bars and status updates during video processing.
*   **Safe Path Visualization**: Dynamic overlay highlighting safe terrain and optimal path vectors.

---

## 🚀 Quick Start (Windows)

1.  **Install Prerequisites**
    *   [Python 3.8+](https://www.python.org/downloads/)
    *   [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (if you have an NVIDIA GPU)
    *   [Git](https://git-scm.com/)

2.  **Setup Environment**
    Open PowerShell in this folder:
    ```powershell
    # Create Virtual Env
    python -m venv venv
    .\venv\Scripts\activate

    # Install Dependencies
    pip install -r requirements.txt
    
    # Install PyTorch with CUDA (Command varies by CUDA version)
    # Example for CUDA 11.8:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

3.  **Run the App**
    Double-click `run_demo.bat` OR run:
    ```powershell
    python app.py
    ```

4.  **Access Dashboard**
    Open your browser to: `http://localhost:5000`

---

## 🎮 How to Use

1.  **Upload**: Drag & drop a video file (`.mp4`, `.avi`) or paste a YouTube link.
2.  **Process**: Click "Process Video". The system will initialize the AI model (might take 10s first time).
3.  **Monitor**: Watch the real-time progress bar.
4.  **Analyze**: View the Side-by-Side comparison of Original vs. AI Output.
5.  **Download**: Click the "Result" button to save the processed video.

---

## 🛠 Project Structure

```
hackathon_v3/
├── app.py              # Main Flask Application (Backend)
├── src/
│   ├── segmentation.py # AI Engine (SegFormer logic)
│   ├── postprocessing.py # Visualization & Path Finding
│   └── utils.py        # Helpers (YouTube DL, Logging)
├── static/
│   ├── css/style.css   # Premium Dark Theme
│   ├── js/main.js      # Frontend Logic
│   └── outputs/        # Processed Videos go here
├── templates/
│   └── index.html      # Main Dashboard
└── requirements.txt    # Python Dependencies
```

## ⚠️ Troubleshooting

*   **"CUDA out of memory"**: The model is efficient (SegFormer-B0), but if you have <4GB VRAM, close other apps.
*   **Video Codec Error**: If the output video doesn't play in Chrome, try VLC Player. We use `avc1` or `mp4v` codecs.
*   **Slow Processing?**: Ensure you installed the *CUDA* version of PyTorch. Check by running: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 👨‍💻 For Judges

*   This system runs **completely offline** (except for YouTube downloads).
*   It demonstrates **End-to-End Deep Learning deployment**.
*   The "Safe Path" logic uses **computer vision heuristics** on top of semantic masks.
