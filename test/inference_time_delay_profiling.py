import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.shared_data import SharedData # Assuming this is a valid import
from models.tennis_tracker.wasb import HRNet # Assuming this is a valid import
from utils.time_utils import createTimeStamp # Assuming this is a valid import

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

import time

class TennisTracker(SharedData):
    def __init__(self):
        super().__init__()
        
        # --- Time Model Initialization ---
        init_start_time = time.perf_counter()

        model_load_start = time.perf_counter()
        self.model = HRNet(cfg=self.config).to(self.device)
        model_load_end = time.perf_counter()
        print(f"Time for HRNet model instantiation: {model_load_end - model_load_start:.4f} seconds")

        checkpoint_load_start = time.perf_counter()
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checkpoint_load_end = time.perf_counter()
        print(f"Time for loading checkpoint and state dict: {checkpoint_load_end - checkpoint_load_start:.4f} seconds")

        self.model.eval()
        
        init_end_time = time.perf_counter()
        print(f"Total __init__ time: {init_end_time - init_start_time:.4f} seconds")
        # --- End Time Model Initialization ---

        self.frames_buffer = []
        self.prev_positions = []
        
        # To store timing results for process method (optional, for aggregation)
        self.timing_stats = {
            "process_total": [],
            "input_prep": [],
            "model_inference": [],
            "output_loop_total": [],
            "heatmap_processing": [],
            "connected_components": [],
            "blob_processing": [],
            "predict_ball_position": [],
            "standardize_coordinates": [],
            "create_time_stamp": [],
        }
            
    def process(self, frames, event_draw=False):
        process_start_time = time.perf_counter()

        self.frames_buffer = frames # Assuming frames is already a list of preprocessed tensors
        # self.frames_buffer.append(frames) # If frames is a single frame, you'd append

        if len(self.frames_buffer) == self.config['frames_in']:
            # --- Time Input Preparation ---
            input_prep_start = time.perf_counter()
            # frames_processed = [self.preprocess_frame(f) for f in self.frames_buffer] # If you have a preprocess_frame method
            input_tensor = torch.cat(self.frames_buffer, dim=0).unsqueeze(0).to(self.device)
            input_prep_end = time.perf_counter()
            self.timing_stats["input_prep"].append((input_prep_end - input_prep_start)*1000)
            # --- End Time Input Preparation ---
            
            # --- Time Model Inference ---
            inference_start_time = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(input_tensor)[0]
            inference_end_time = time.perf_counter()
            self.timing_stats["model_inference"].append((inference_end_time - inference_start_time)*1000)
            # --- End Time Model Inference ---
                
            detected = False
            center_x, center_y, confidence = 0, 0, 0
            
            # --- Time Output Processing Loop ---
            output_loop_start_time = time.perf_counter()
            for i in range(self.config['frames_out']):
                output_frame_processing_start_time = time.perf_counter() # For timing each iteration if needed

                output = outputs[0][i] # Assuming outputs is structured like this

                # --- Time Heatmap Processing ---
                heatmap_proc_start = time.perf_counter()
                output = torch.sigmoid(output)
                heatmap = output.squeeze().cpu().numpy()
                heatmap = cv2.resize(heatmap, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                heatmap = (heatmap > 0.5).astype(np.float32) * heatmap
                heatmap_proc_end = time.perf_counter()
                self.timing_stats["heatmap_processing"].append((heatmap_proc_end - heatmap_proc_start)*1000)
                # --- End Time Heatmap Processing ---
                
                # --- Time Connected Components ---
                conn_comp_start = time.perf_counter()
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((heatmap > 0).astype(np.uint8), connectivity=8)
                conn_comp_end = time.perf_counter()
                self.timing_stats["connected_components"].append((conn_comp_end - conn_comp_start)*1000)
                # --- End Time Connected Components ---
                
                # --- Time Blob Processing & Selection ---
                blob_proc_start = time.perf_counter()
                blob_centers = []
                for j in range(1, num_labels): # Start from 1 to exclude background
                    mask = labels_im == j
                    blob_sum = heatmap[mask].sum()
                    if blob_sum > 0: # Ensure blob_sum is positive before division
                        center_x_blob = np.sum(np.where(mask)[1] * heatmap[mask]) / blob_sum
                        center_y_blob = np.sum(np.where(mask)[0] * heatmap[mask]) / blob_sum
                        blob_centers.append((center_x_blob, center_y_blob, blob_sum))
            
                current_center_x, current_center_y, current_confidence = 0, 0, 0 # Use temporary vars for current frame
                current_detected = False

                if blob_centers:
                    # --- Time Ball Position Prediction (conditional) ---
                    predict_ball_pos_start = time.perf_counter()
                    predicted_position = self.predict_ball_position(self.prev_positions)
                    predict_ball_pos_end = time.perf_counter()
                    if predicted_position is not None: # Only record time if it was actually called significantly
                        self.timing_stats["predict_ball_position"].append((predict_ball_pos_end - predict_ball_pos_start)*1000)
                    # --- End Time Ball Position Prediction ---

                    if predicted_position is not None:
                        distances = [np.sqrt((x - predicted_position[0]) ** 2 + (y - predicted_position[1]) ** 2) for x, y, _ in blob_centers]
                        if distances: # Ensure distances is not empty
                            closest_blob_idx = np.argmin(distances)
                            current_center_x, current_center_y, current_confidence = blob_centers[closest_blob_idx]
                    else:
                        blob_centers.sort(key=lambda x: x[2], reverse=True)
                        current_center_x, current_center_y, current_confidence = blob_centers[0]
                    
                    current_detected = True
                    self.prev_positions.append(np.array([current_center_x, current_center_y]))
                    if len(self.prev_positions) > self.config.get('max_prev_positions', 3): # Use a configurable max
                        self.prev_positions.pop(0)
                
                # Update main center_x, center_y, confidence and detected status for this output frame
                center_x, center_y, confidence, detected = current_center_x, current_center_y, current_confidence, current_detected
                blob_proc_end = time.perf_counter()
                self.timing_stats["blob_processing"].append((blob_proc_end - blob_proc_start)*1000) # Includes prediction time if it happens
                # --- End Time Blob Processing & Selection ---
                        
                # --- Time Coordinate Standardization ---
                std_coords_start = time.perf_counter()
                x_standardized, y_standardized = self.standardize_coordinates((center_x, center_y))
                std_coords_end = time.perf_counter()
                self.timing_stats["standardize_coordinates"].append((std_coords_end - std_coords_start)*1000)
                # --- End Time Coordinate Standardization ---
                create_time_stamp_start = time.perf_counter()
                timestamp = createTimeStamp() # Assuming this is a fast operation, or time it if needed
                create_time_stamp_end = time.perf_counter()
                self.timing_stats["create_time_stamp"].append((create_time_stamp_end - create_time_stamp_start)*1000)
                
                coordinates = [self.frame_number, timestamp, 1 if detected else 0, center_x, center_y, x_standardized if detected else 0, y_standardized if detected else 0, confidence]
                
                self.frame_number += 1
                yield coordinates
            
            output_loop_end_time = time.perf_counter()
            self.timing_stats["output_loop_total"].append((output_loop_end_time - output_loop_start_time)*1000)
            # --- End Time Output Processing Loop ---
                
            self.frames_buffer = [] # Clear buffer after processing
        
        process_end_time = time.perf_counter()
        self.timing_stats["process_total"].append((process_end_time - process_start_time)*1000)
        
    def predict_ball_position(self, prev_positions):
        if len(prev_positions) < 3: # Ensure enough data points
            return None
        # No internal timing here as it's usually very fast, but you could add it if needed
        p_t = prev_positions[-1]
        a_t = p_t - 2 * prev_positions[-2] + prev_positions[-3]
        v_t = p_t - prev_positions[-2] + a_t
        predicted_position = p_t + v_t + 0.5 * a_t
        predicted_position = np.clip(predicted_position, [0, 0], [self.frame_width -1, self.frame_height -1]) # Ensure within bounds
        return predicted_position
    
    def standardize_coordinates(self, coordinates):
        # No internal timing here as it's usually very fast, but you could add it if needed
        projected = self.homography_matrix @ np.array([coordinates[0], coordinates[1], 1])
        if projected[2] == 0: # Avoid division by zero
            return 0, 0 
        x_proj = projected[0] / projected[2]
        y_proj = projected[1] / projected[2]
        
        x_standardized = (x_proj - 72) / 216 # Make these configurable or constants
        y_standardized = (y_proj - 126) / 468 # Make these configurable or constants
        
        return x_standardized, y_standardized
        
    def draw(self, frame, center_x, center_y):
        cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), 2)

    def print_timing_summary(self):
        print("\n--- Timing Summary (Average per call) ---")
        for key, times in self.timing_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"{key}: {avg_time:.6f} ms")
            else:
                print(f"{key}: No calls recorded")
        print("----------------------------------------")

# Example Usage (modify your main loop)
if __name__ == "__main__":
    tennis_tracker = TennisTracker()
    tennis_tracker.print_timing_summary()