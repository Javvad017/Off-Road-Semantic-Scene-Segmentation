
import cv2
import torch
import numpy as np
import argparse
import collections
import heapq
import time
import sys
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# ADE20K Class Indices (Common classes relevant to off-road)
# 0-indexed in HuggingFace transformers
CLASS_MAP = {
    'wall': 0, 'building': 1, 'sky': 2, 'floor': 3, 'tree': 4, 
    'ceiling': 5, 'road': 6, 'bed': 7, 'window': 8, 'grass': 9, 
    'cabinet': 10, 'sidewalk': 11, 'person': 12, 'earth': 13, 
    'door': 14, 'table': 15, 'mountain': 16, 'plant': 17, 
    'curtain': 18, 'chair': 19, 'car': 20, 'water': 21, 
    'painting': 22, 'sofa': 23, 'shelf': 24, 'house': 25, 
    'sea': 26, 'mirror': 27, 'rug': 28, 'field': 29, 
    'armchair': 30, 'seat': 31, 'fence': 32, 'desk': 33, 
    'rock': 34, 'wardrobe': 35, 'lamp': 36, 'bathtub': 37, 
    'rail': 38, 'cushion': 39, 'base': 40, 'box': 41, 
    'column': 42, 'signboard': 43, 'chest': 44, 'counter': 45, 
    'sand': 46, 'sink': 47, 'skyscraper': 48, 'fireplace': 49, 
    'refrigerator': 50, 'grandstand': 51, 'path': 52, 'stairs': 53, 
    'runway': 54, 'case': 55, 'pool_table': 56, 'pillow': 57, 
    'screen_door': 58, 'stairway': 59, 'river': 60, 'bridge': 61, 
    'bookcase': 62, 'blind': 63, 'coffee_table': 64, 'toilet': 65, 
    'flower': 66, 'book': 67, 'hill': 68, 'bench': 69, 
    'countertop': 70, 'stove': 71, 'palm': 72, 'kitchen_island': 73, 
    'computer': 74, 'swivel_chair': 75, 'boat': 76, 'bar': 77, 
    'arcade_machine': 78, 'hovel': 79, 'bus': 80, 'towel': 81, 
    'light': 82, 'truck': 83, 'tower': 84, 'chandelier': 85, 
    'awning': 86, 'streetlight': 87, 'booth': 88, 'tv': 89, 
    'airplane': 90, 'dirt': 91, 'apparel': 92, 'pole': 93, 
    'land': 94, 'bannister': 95, 'escalator': 96, 'ottoman': 97, 
    'bottle': 98, 'buffet': 99, 'poster': 100, 'stage': 101, 
    'van': 102, 'ship': 103, 'fountain': 104
}

# Risk Levels: Lower is safer. 
# 1 = Safe, 5-10 = Moderate, 50+ = Dangerous, 255 = Obstacle
RISK_MAPPING = collections.defaultdict(lambda: 20) # Default to caution
RISK_MAPPING.update({
    # SAFE (Cost 1-3)
    CLASS_MAP['road']: 1,
    CLASS_MAP['sidewalk']: 1,
    CLASS_MAP['floor']: 1,
    CLASS_MAP['path']: 2,
    CLASS_MAP['earth']: 3, # Dry dirt
    CLASS_MAP['field']: 3,
    CLASS_MAP['runway']: 1,
    
    # MODERATE / UNSTABLE (Cost 5-15)
    CLASS_MAP['grass']: 8,    # slippery? hidden obstacles?
    CLASS_MAP['sand']: 12,    # sink risk
    CLASS_MAP['dirt']: 10,     
    CLASS_MAP['land']: 10,
    CLASS_MAP['hill']: 15,    # slope risk
    
    # DANGEROUS (Cost 20-50)
    CLASS_MAP['water']: 50,
    CLASS_MAP['sea']: 50,
    CLASS_MAP['river']: 50,
    CLASS_MAP['rock']: 40,    # unless very small
    CLASS_MAP['mountain']: 50, # too steep
    
    # OBSTACLES (Cost 255 - Impassable)
    CLASS_MAP['wall']: 255,
    CLASS_MAP['building']: 255,
    CLASS_MAP['sky']: 255,
    CLASS_MAP['tree']: 255,
    CLASS_MAP['person']: 255,
    CLASS_MAP['car']: 255,
    CLASS_MAP['fence']: 255,
    CLASS_MAP['pole']: 255,
    CLASS_MAP['signboard']: 255,
    CLASS_MAP['truck']: 255,
    CLASS_MAP['bus']: 255,
    CLASS_MAP['van']: 255,
})

# Visualization Colors (BGR)
COLORS = {
    'safe': (0, 255, 0),        # Green
    'warning': (0, 255, 255),   # Yellow
    'danger': (0, 0, 255),      # Red
    'path': (0, 215, 255),      # Gold/Yellow
    'obstacle': (128, 128, 128) # Gray
}

# ==========================================
# SEGMENTATION ENGINE
# ==========================================
class SegmentationEngine:
    def __init__(self, model_name="nvidia/segformer-b1-finetuned-ade-512-512"):
        """Initialize the SegFormer model"""
        print(f"[INFO] Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)
            
    def run_inference(self, frame):
        """Run standard inference and return label mask"""
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process logits
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        
        # Upsample to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        # Get class indices
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        return pred_seg.cpu().numpy().astype(np.uint8)

# ==========================================
# RISK ASSESSMENT & MAPPING
# ==========================================
class RiskEstimator:
    def __init__(self):
        # Create a lookup table for fast risk mapping
        self.risk_lut = np.zeros(256, dtype=np.uint8)
        for cls_id, risk_val in RISK_MAPPING.items():
            if 0 <= cls_id < 256:
                self.risk_lut[cls_id] = risk_val
                
    def create_cost_map(self, seg_mask):
        """
        Convert segmentation mask to cost map.
        Low value = Safe, High value = Risky/Obstacle
        """
        # 1. Direct Mapping
        cost_map = self.risk_lut[seg_mask]
        
        # 2. Add safety margin around obstacles (Morphological Dilation)
        # Identify obstacles (Cost >= 200)
        obstacles = (cost_map >= 200).astype(np.uint8)
        
        # Dilate obstacles to create a buffer zone
        kernel = np.ones((9, 9), np.uint8) # 9x9 kernel
        dilated_obstacles = cv2.dilate(obstacles, kernel, iterations=2)
        
        # Increase cost in buffer zones
        buffer_mask = (dilated_obstacles > 0) & (obstacles == 0)
        cost_map[buffer_mask] = np.maximum(cost_map[buffer_mask], 50) # High cost for being near obstacle
        
        return cost_map.astype(np.float32)

# ==========================================
# PATH PLANNING (A*)
# ==========================================
class PathPlanner:
    def __init__(self, plan_scale=0.125):
        """
        plan_scale: Downgrades resolution for A* to run fast (e.g. 1/8th scale)
        """
        self.plan_scale = plan_scale
        
    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan_path(self, cost_map, start_ratio=(0.5, 0.95), goal_ratio=(0.5, 0.1)):
        """
        Run A* on the cost map.
        start_ratio: (x, y) relative position (0.5, 0.95 is bottom center)
        goal_ratio: (x, y) relative position (0.5, 0.1 is top center)
        """
        h, w = cost_map.shape
        
        # Resize cost map for planning
        small_h, small_w = int(h * self.plan_scale), int(w * self.plan_scale)
        small_cost_map = cv2.resize(cost_map, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply Gaussian Blur to smooth costs (gradual risk)
        small_cost_map = cv2.GaussianBlur(small_cost_map, (5, 5), 0)
        
        # Define Start and Goal indices
        start_node = (int(start_ratio[1] * small_h), int(start_ratio[0] * small_w))
        goal_node = (int(goal_ratio[1] * small_h), int(goal_ratio[0] * small_w))
        
        # Clamp coordinates
        start_node = (np.clip(start_node[0], 0, small_h-1), np.clip(start_node[1], 0, small_w-1))
        goal_node = (np.clip(goal_node[0], 0, small_h-1), np.clip(goal_node[1], 0, small_w-1))
        
        # A* Algorithm
        queue = []
        heapq.heappush(queue, (0, start_node))
        came_from = {}
        cost_so_far = {}
        came_from[start_node] = None
        cost_so_far[start_node] = 0
        
        found = False
        
        while queue:
            _, current = heapq.heappop(queue)
            
            if current == goal_node:
                found = True
                break
                
            # If we are close enough to goal (e.g., top row region), consider it done
            if current[0] <= goal_node[0] + 2 and abs(current[1] - goal_node[1]) < 5:
                goal_node = current
                found = True
                break
                
            # Neighbors (4-connectivity)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_node = (current[0] + dy, current[1] + dx)
                
                # Check bounds
                if 0 <= next_node[0] < small_h and 0 <= next_node[1] < small_w:
                    # Get traversing cost
                    # Add base movement cost (1) + terrain cost
                    # If terrain is obstacle (>200), cost is VERY high
                    cell_cost = small_cost_map[next_node]
                    
                    if cell_cost >= 200:
                        move_cost = 9999 # Effective Infinity
                    else:
                        move_cost = 1 + cell_cost
                    
                    new_cost = cost_so_far[current] + move_cost
                    
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + self.heuristic(goal_node, next_node)
                        heapq.heappush(queue, (priority, next_node))
                        came_from[next_node] = current
                        
        if not found:
            return None, 0
            
        # Reconstruct Path
        path = []
        curr = goal_node
        while curr != start_node:
            path.append(curr)
            curr = came_from.get(curr)
            if curr is None: break # Should not happen if found
        path.append(start_node)
        path.reverse()
        
        # Calculate Path Confidence (Inverse of average cost)
        if len(path) == 0: return None, 0
        
        total_risk = sum([small_cost_map[p] for p in path])
        avg_risk = total_risk / len(path)
        confidence = max(0, 100 - avg_risk) # Simple confidence score
        
        # Scale path back to original resolution
        full_path = []
        scale_factor = 1.0 / self.plan_scale
        for p in path:
            full_path.append((int(p[1] * scale_factor), int(p[0] * scale_factor)))
            
        return full_path, confidence

# ==========================================
# VISUALIZATION
# ==========================================
class Visualizer:
    def overlay_risk(self, frame, cost_map):
        """Draw Safe/Danger zones"""
        # Create colored masks
        h, w = frame.shape[:2]
        
        # Mask definitions
        safe_mask = (cost_map < 5).astype(np.uint8)
        danger_mask = ((cost_map >= 20) & (cost_map < 200)).astype(np.uint8)
        
        # Create overlays
        green_layer = np.zeros_like(frame, dtype=np.uint8)
        green_layer[:] = COLORS['safe']
        
        red_layer = np.zeros_like(frame, dtype=np.uint8)
        red_layer[:] = COLORS['danger']
        
        # Alpha blending using numpy for stability
        # Safe zones (0.3 opacity)
        safe_blend = cv2.addWeighted(frame, 0.7, green_layer, 0.3, 0)
        
        # Danger zones (0.4 opacity)
        danger_blend = cv2.addWeighted(frame, 0.6, red_layer, 0.4, 0)
        
        # Combine
        frame_aug = frame.copy()
        
        # Apply Safe mask
        mask_3ch_safe = np.stack([safe_mask]*3, axis=2)
        np.copyto(frame_aug, safe_blend, where=mask_3ch_safe > 0)
        
        # Apply Danger mask (overwrite safe if overlap, though shouldn't overlap based on cost logic)
        mask_3ch_danger = np.stack([danger_mask]*3, axis=2)
        np.copyto(frame_aug, danger_blend, where=mask_3ch_danger > 0)
        
        return frame_aug

    def draw_path(self, frame, path, confidence):
        """Draw the computed path"""
        if not path or len(path) < 2:
            return frame
            
        # Draw path line
        pts = np.array(path, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, COLORS['path'], thickness=4, lineType=cv2.LINE_AA)
        
        # Draw Start/End
        cv2.circle(frame, path[0], 8, (0, 255, 0), -1) # Start Green
        cv2.circle(frame, path[-1], 8, (0, 0, 255), -1) # End Red
        
        # Stats Box
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        
        # Confidence color
        conf_color = (0, 255, 0) if confidence > 80 else (0, 165, 255) if confidence > 50 else (0, 0, 255)
        
        cv2.putText(frame, "Terrain Analytics", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Path Conf: {confidence:.1f}%", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
        
        return frame

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', default='output_analysis.mp4', help='Path to output video')
    args = parser.parse_args()

    # Initialize Modules
    seg_engine = SegmentationEngine()
    risk_estimator = RiskEstimator()
    path_planner = PathPlanner(plan_scale=0.125) # 1/8 scale for speed
    visualizer = Visualizer()
    
    # Open Video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error opening video: {args.video_path}")
        return

    # Video Writer Setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process Loop
    frame_count = 0
    start_time = time.time()
    
    print("\n[INFO] Starting Video Processing...")
    print("Press 'q' to stop.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        
        # 1. Segmentation
        # Resize for inference speed if needed (SegFormer is fast, but 4K is slow)
        # keeping original for now, usually 720p or 1080p
        seg_mask = seg_engine.run_inference(frame)
        
        # 2. Risk Mapping
        cost_map = risk_estimator.create_cost_map(seg_mask)
        
        # 3. Path Planning
        # Start: Bottom Center, Goal: Top Center
        path, confidence = path_planner.plan_path(
            cost_map, 
            start_ratio=(0.5, 0.95), 
            goal_ratio=(0.5, 0.15)
        )
        
        # 4. Visualization
        # Overlay terrain risks
        vis_frame = visualizer.overlay_risk(frame, cost_map)
        # Draw path
        final_frame = visualizer.draw_path(vis_frame, path, confidence)
        
        # FPS Calc
        t1 = time.time()
        proc_fps = 1.0 / (t1 - t0)
        cv2.putText(final_frame, f"FPS: {proc_fps:.1f}", (width-150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Display & Save
        cv2.imshow('Off-Road Path Planning', final_frame)
        out.write(final_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames. Current FPS: {proc_fps:.1f}")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\n[DONE] Processing complete.")
    print(f"Total Frames: {frame_count}")
    print(f"Average FPS: {frame_count / total_time:.1f}")
    print(f"Output saved to: {args.output}")

if __name__ == '__main__':
    main()
