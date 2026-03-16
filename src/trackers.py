import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

# Add the sibling directory to sys.path to allow importing model_definitions and inference_scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
wasb_root = os.path.join(os.path.dirname(project_root), "wasb-sbdt-inference")
if wasb_root not in sys.path:
    sys.path.append(wasb_root)

from model_definitions.wasb import HRNet
from inference_scripts.wasb_inference import predict_ball_position

class BallTracker:
    def __init__(self, model_path):
        # EXACT configuration from wasb_inference.py
        self.config = {
            "name": "hrnet",
            "frames_in": 3,
            "frames_out": 3,
            "inp_height": 288,
            "inp_width": 512,
            "out_height": 288,
            "out_width": 512,
            "rgb_diff": False,
            "out_scales": [0],
            "MODEL": {
                "EXTRA": {
                    "FINAL_CONV_KERNEL": 1,
                    "PRETRAINED_LAYERS": ['*'],
                    "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
                    "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": 'BOTTLENECK', "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": 'SUM'},
                    "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": 'BASIC', "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": 'SUM'},
                    "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": 'BASIC', "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": 'SUM'},
                    "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": 'BASIC', "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": 'SUM'},
                    "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2}
                },
                "INIT_WEIGHTS": True
            },
            "model_path": model_path
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HRNet(cfg=self.config).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"[INIT] BallTracker: Loaded original WASB weights from {model_path}")
        else:
            raise FileNotFoundError(f"WASB weights not found at {model_path}")
            
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['inp_height'], self.config['inp_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.prev_positions = []

    def infer_batch(self, frames):
        """
        Processes a batch of 3 frames EXACTLY as the original run_inference.
        Returns a list of 3 coordinates [(x,y), ...].
        """
        h, w = frames[0].shape[:2]
        frames_processed = [self.transform(f) for f in frames]
        input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)[0]
            
        batch_results = []
        for i in range(self.config['frames_out']):
            output = torch.sigmoid(outputs[0][i])
            heatmap = output.squeeze().cpu().numpy()
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Original post-processing from wasb_inference.py
            heatmap_thresh = (heatmap > 0.5).astype(np.float32) * heatmap
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((heatmap_thresh > 0).astype(np.uint8), connectivity=8)
            
            blob_centers = []
            for j in range(1, num_labels):
                mask = labels_im == j
                blob_sum = heatmap_thresh[mask].sum()
                if blob_sum > 0:
                    cx = np.sum(np.where(mask)[1] * heatmap_thresh[mask]) / blob_sum
                    cy = np.sum(np.where(mask)[0] * heatmap_thresh[mask]) / blob_sum
                    blob_centers.append((cx, cy, blob_sum))
            
            detected_pos = None
            if blob_centers:
                # Use imported predict_ball_position
                pred = predict_ball_position(self.prev_positions, w, h)
                if pred is not None:
                    dists = [np.sqrt((x - pred[0])**2 + (y - pred[1])**2) for x, y, _ in blob_centers]
                    idx = np.argmin(dists)
                    detected_pos = (blob_centers[idx][0], blob_centers[idx][1])
                else:
                    blob_centers.sort(key=lambda x: x[2], reverse=True)
                    detected_pos = (blob_centers[0][0], blob_centers[0][1])
                
                self.prev_positions.append(np.array(detected_pos))
                if len(self.prev_positions) > 3:
                    self.prev_positions.pop(0)
            
            batch_results.append(detected_pos)
            
        return batch_results

    def smooth_trajectory(self, raw_trajectory):
        """
        No smoothing. Just format conversion.
        """
        return [(float(pt[0]), float(pt[1])) if pt is not None else (0.0, 0.0) for pt in raw_trajectory]


class PlayerTracker:
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load YOLO Pose model (e.g., yolov8n-pose.pt or yolov11n-pose.pt)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"[INIT] PlayerTracker: Loaded YOLO Pose model from {model_path}")

    def infer_frame(self, frame):
        """
        Detects and tracks players, extracting bounding boxes and keypoints.
        Uses Ultralytics tracking with BoT-SORT or ByteTrack.
        """
        # Run tracking (classes=[0] for person)
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)[0]
        
        players_info = {}
        
        # Check if we have detections and tracks
        if results.boxes is None or results.boxes.id is None or results.keypoints is None:
            return players_info
            
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        keypoints = results.keypoints.xy.cpu().numpy() # Shape: [N, 17, 2]
        
        for i, player_id in enumerate(ids):
            # COCO Pose Keypoints indices:
            # 0: nose, 5: l_shoulder, 6: r_shoulder, 9: l_wrist, 10: r_wrist, 15: l_ankle, 16: r_ankle
            kps = keypoints[i]
            
            players_info[int(player_id)] = {
                "bbox": boxes[i].tolist(), # [x1, y1, x2, y2]
                "keypoints": {
                    "nose": (float(kps[0][0]), float(kps[0][1])),
                    "l_shoulder": (float(kps[5][0]), float(kps[5][1])),
                    "r_shoulder": (float(kps[6][0]), float(kps[6][1])),
                    "l_wrist": (float(kps[9][0]), float(kps[9][1])),
                    "r_wrist": (float(kps[10][0]), float(kps[10][1])),
                    "l_ankle": (float(kps[15][0]), float(kps[15][1])),
                    "r_ankle": (float(kps[16][0]), float(kps[16][1]))
                }
            }
            
        return players_info
