from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import cv2
import time
import threading
import uuid
import logging
import torch
from werkzeug.utils import secure_filename
from src.segmentation import SegmentationEngine
from src.postprocessing import Postprocessor
from src.utils import download_youtube_video, get_gpu_memory_usage
from PIL import Image

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'hackathon_secret_key'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global Variables
TASKS = {}
MODEL_LOCK = threading.Lock()
engine = None
postprocessor = None

def load_model():
    """Lazy load the model to allow app to start fast."""
    global engine, postprocessor
    if engine is None:
        print("Loading SegFormer Model... (This may take a moment)")
        engine = SegmentationEngine() # Default model
        postprocessor = Postprocessor(safe_classes=engine.safe_classes)
        print("Model Loaded!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_task(task_id, input_path, output_filename):
    """Background worker to process the video."""
    global TASKS
    
    try:
        # Load Model if needed (Thread Safe)
        with MODEL_LOCK:
             load_model()

        # Update Status
        TASKS[task_id]['status'] = 'Initializing Video...'
        TASKS[task_id]['progress'] = 5

        # Open Video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0: total_frames = 1 # Prevent div/0

        # Resize Strategy: Cap at 720p for display/processing speed
        target_width = 1280
        if width > target_width:
            aspect_ratio = height / width
            width = target_width
            height = int(target_width * aspect_ratio)
            print(f"Resizing video to {width}x{height} for processing.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0: total_frames = 1 # Prevent div/0

        # Setup Output
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # We need to re-encode to likely H.264 for browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
             print("Falling back to mp4v")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        TASKS[task_id]['status'] = 'Processing Frames...'
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize Frame
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            # --- AI Pipeline ---
            # 1. Preprocess (Resize handled by model internally/efficiently)
            # Convert to PIL for SegFormer
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # 2. Inference
            pred_seg = engine.process_frame(pil_image)
            
            # 3. Postprocess
            if pred_seg is not None:
                color_mask = engine.get_color_mask(pred_seg)
                safe_mask = postprocessor.extract_safe_mask(pred_seg)
                path_data = postprocessor.find_safe_path(safe_mask)
                
                final_frame = postprocessor.overlay_mask(frame, color_mask)
                final_frame = postprocessor.draw_safe_path(final_frame, path_data)
            else:
                final_frame = frame

            # 4. Write
            out.write(final_frame)
            
            # Update Progress
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            
            # Throttle updates to shared dict
            if frame_count % 5 == 0:
                TASKS[task_id]['progress'] = progress
                TASKS[task_id]['status'] = f"Processing: {progress}%"
                
                # Update Runtime Stats
                current_duration = time.time() - start_time
                current_fps = frame_count / current_duration if current_duration > 0 else 0
                TASKS[task_id]['fps'] = f"{current_fps:.1f}"
                TASKS[task_id]['gpu_usage'] = get_gpu_memory_usage()

        # Cleanup
        cap.release()
        out.release()
        
        # Calculate Stats
        end_time = time.time()
        duration = end_time - start_time
        avg_fps = frame_count / duration if duration > 0 else 0
        
        TASKS[task_id]['progress'] = 100
        TASKS[task_id]['status'] = 'Finalizing...'
        TASKS[task_id]['state'] = 'COMPLETED'
        TASKS[task_id]['output_url'] = f'/static/outputs/{output_filename}'
        TASKS[task_id]['download_url'] = f'/download/{output_filename}'
        TASKS[task_id]['input_url'] = f'/static/{os.path.basename(UPLOAD_FOLDER)}/{os.path.basename(input_path)}' # Need to expose uploads via static
        TASKS[task_id]['fps'] = f"{avg_fps:.1f}"

    except Exception as e:
        print(f"Task Failed: {e}")
        TASKS[task_id]['state'] = 'FAILED'
        TASKS[task_id]['error'] = str(e)
        TASKS[task_id]['progress'] = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {'state': 'QUEUED', 'progress': 0, 'status': 'Queued...'}
    
    input_path = None
    filename = None
    
    # Handle YouTube
    if 'youtube_url' in request.form:
        url = request.form['youtube_url']
        filename = f"{task_id}.mp4"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Run download in the same thread (blocking) for simplicity validation, 
        # or thread it. Blocking is better to catch download errors early.
        downloaded_path = download_youtube_video(url, input_path)
        if not downloaded_path:
            return jsonify({'error': 'Failed to download YouTube video'}), 400
            
    # Handle File Upload
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{task_id}.{ext}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
        else:
             return jsonify({'error': 'Invalid file type'}), 400
    else:
        return jsonify({'error': 'No input provided'}), 400

    # Start Processing Thread
    output_filename = f"processed_{filename.rsplit('.', 1)[0]}.mp4"
    thread = threading.Thread(target=process_video_task, args=(task_id, input_path, output_filename))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'started'})

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = TASKS.get(task_id)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/download/<filename>')
def download_result(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# Expose Uploads folder for previewing input (Hack for demo)
@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
if __name__ == '__main__':
    # Load model on startup to be ready
    # load_model() # Optional: Uncomment to load on start, but might slow down dev restart
    app.run(host='0.0.0.0', port=4000, debug=True, threaded=True)
