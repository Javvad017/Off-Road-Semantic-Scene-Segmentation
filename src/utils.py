import cv2
import time
import os
import yt_dlp
import logging
import torch

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("App")

logger = setup_logger()

def download_youtube_video(url, output_path="input_video.mp4"):
    """Downloads a YouTube video using yt-dlp."""
    logger.info(f"Downloading YouTube video: {url}...")
    
    if os.path.exists(output_path):
        os.remove(output_path) # Clean up old file
        
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'writesubtitles': False,
        'writethumbnail': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info("Download complete.")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return None

class FPSMeter:
    """Tracks Frame Rate over a window of time."""
    def __init__(self, buffer_len=30):
        self.buffer_len = buffer_len
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        curr_time = time.time()
        delta = curr_time - self.prev_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.prev_time = curr_time
        return self.fps

def get_gpu_memory_usage():
    """Returns GPU memory usage in MB as a string."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**2
        return f"{used:.2f} MB / {total / 1024**2:.2f} MB"
    return "N/A"

def create_video_writer(output_path, fps, width, height):
    """Creates a VideoWriter object."""
    # Try different codecs if one fails
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Classic
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
             # Fallback
             fourcc = cv2.VideoWriter_fourcc(*'XVID')
             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return out
    except Exception as e:
        logger.error(f"Error creating video writer: {e}")
        return None
