"""
Quick Start Script - Automated Setup and Testing
Helps verify installation and run first demo
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"→ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Success")
            return True, result.stdout
        else:
            print(f"  ✗ Failed: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, str(e)


def check_gpu():
    """Check NVIDIA GPU availability"""
    print_header("GPU Check")
    
    success, output = run_command("nvidia-smi", "Checking NVIDIA GPU")
    if success:
        print("\nGPU Information:")
        print(output[:500])  # Print first 500 chars
        return True
    else:
        print("\n⚠️  Warning: No NVIDIA GPU detected or nvidia-smi not found")
        print("   The system will run on CPU (much slower)")
        return False


def check_python():
    """Check Python version"""
    print_header("Python Check")
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False


def check_pytorch():
    """Check PyTorch and CUDA"""
    print_header("PyTorch Check")
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print("✓ PyTorch with CUDA is ready")
            return True
        else:
            print("⚠️  PyTorch installed but CUDA not available")
            print("   Install PyTorch with CUDA support:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        print("   Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_dependencies():
    """Check if all dependencies are installed"""
    print_header("Dependencies Check")
    
    required = [
        'transformers',
        'cv2',
        'numpy',
        'PIL',
        'flask',
        'albumentations',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True


def download_test_model():
    """Pre-download the model"""
    print_header("Model Download")
    
    print("Downloading SegFormer model (this may take a few minutes)...")
    
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        print(f"  → Loading {model_name}...")
        
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        print("  ✓ Model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download model: {e}")
        return False


def run_test():
    """Run a quick test"""
    print_header("Quick Test")
    
    print("Creating a test frame and running inference...")
    
    try:
        import numpy as np
        import cv2
        from segmentation_engine import SegmentationEngine
        
        # Create dummy frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Initialize engine
        print("  → Initializing segmentation engine...")
        engine = SegmentationEngine(use_path_detection=False)
        
        # Process frame
        print("  → Running inference...")
        overlay, mask, stats = engine.process_frame(test_frame, show_path=False, show_obstacles=False)
        
        print("  ✓ Test successful!")
        print(f"     Output shape: {overlay.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_sample_videos():
    """Check if sample videos exist"""
    print_header("Sample Videos")
    
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        videos = list(uploads_dir.glob("*.mp4"))
        if videos:
            print(f"Found {len(videos)} sample video(s):")
            for video in videos[:3]:  # Show first 3
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"  • {video.name} ({size_mb:.1f} MB)")
            return True
    
    print("No sample videos found in uploads/ directory")
    print("You can add your own videos or download from YouTube")
    return False


def print_next_steps():
    """Print next steps for the user"""
    print_header("Next Steps")
    
    print("🎉 Setup complete! Here's what you can do next:\n")
    
    print("1. Process a video:")
    print("   python main_enhanced.py --source your_video.mp4 --fp16\n")
    
    print("2. Start web interface:")
    print("   python app.py\n")
    
    print("3. Process YouTube video:")
    print('   python main_enhanced.py --source "https://youtube.com/watch?v=..." --fp16\n')
    
    print("4. Read the guides:")
    print("   • SETUP_GUIDE.md - Detailed setup instructions")
    print("   • HACKATHON_GUIDE.md - Presentation tips\n")
    
    print("For help: python main_enhanced.py --help")
    print("\nGood luck! 🚀")


def main():
    """Main setup verification"""
    print("\n" + "=" * 60)
    print("  🚙 Off-Road Segmentation - Quick Start")
    print("=" * 60)
    
    checks = []
    
    # Run all checks
    checks.append(("Python", check_python()))
    checks.append(("GPU", check_gpu()))
    checks.append(("PyTorch", check_pytorch()))
    checks.append(("Dependencies", check_dependencies()))
    
    # Summary
    print_header("Setup Summary")
    
    for name, status in checks:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")
    
    all_passed = all(status for _, status in checks)
    
    if all_passed:
        print("\n✓ All checks passed!")
        
        # Optional: Download model
        print("\nWould you like to pre-download the model? (y/n)")
        response = input("> ").strip().lower()
        if response == 'y':
            download_test_model()
        
        # Optional: Run test
        print("\nWould you like to run a quick test? (y/n)")
        response = input("> ").strip().lower()
        if response == 'y':
            run_test()
        
        # Check for sample videos
        check_sample_videos()
        
        # Print next steps
        print_next_steps()
        
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("   Refer to SETUP_GUIDE.md for detailed instructions.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
