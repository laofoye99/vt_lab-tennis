import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
from algorithms.sort import Sort # Assuming this path and class are correct
from utils.shared_data import SharedData # Assuming this path and class are correct
import torch

class PlayerTracker(SharedData):
    def __init__(self):
        super().__init__()
        self.detector = YOLO(self.yolo_path) # Ensure self.yolo_path is set by SharedData
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3) # Adjusted max_age for potentially gappy 3-frame input
        
        self.roi_expansion = 0.3
        self.history_size = 15 # Increased history for better prediction with batched input
        self.roi_border_buffer = 25 # Renamed for clarity
        
        self.current_roi = None
        self.track_history = {}
        self.frame_number_global = 0
        
        # Ensure frame_width and frame_height are available from SharedData
        # If not, they will be initialized from the first processed frame.

    def _initialize_roi(self):
        if hasattr(self, 'frame_width') and hasattr(self, 'frame_height'):
            self.current_roi = np.array([[0, 0, self.frame_width, self.frame_height]])
        else:
            # This should not be reached if dimensions are set from first frame
            pass

    def _update_history(self, track_id: int, bbox: np.ndarray):
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'positions': [], 'velocities': [], 'accelerations': []
            }
        history = self.track_history[track_id]
        history['positions'].append(bbox.copy())
        if len(history['positions']) > self.history_size:
            history['positions'].pop(0)
        
        if len(history['positions']) >= 2:
            velocity = history['positions'][-1] - history['positions'][-2]
            history['velocities'].append(velocity)
            if len(history['velocities']) > self.history_size:
                history['velocities'].pop(0)
        
        if len(history['positions']) >= 3:
            accel = history['positions'][-1] - 2 * history['positions'][-2] + history['positions'][-3]
            history['accelerations'].append(accel)
            if len(history['accelerations']) > self.history_size:
                history['accelerations'].pop(0)

    def _predict_position(self, track_id: int) -> np.ndarray:
        history = self.track_history.get(track_id)
        if not history or not history['positions']:
            return None
        
        current = history['positions'][-1]
        velocity = history['velocities'][-1] if history['velocities'] else np.zeros(4)
        acceleration = history['accelerations'][-1] if history['accelerations'] else np.zeros(4)
        
        return current + velocity + 0.5 * acceleration

    def _calculate_target_roi(self, track_id: int) -> list:
        predicted = self._predict_position(track_id)
        if predicted is None:
            return None
        
        x1, y1, x2, y2 = predicted
        w, h = x2 - x1, y2 - y1
        
        exp_x1 = max(0, x1 - w * self.roi_expansion)
        exp_y1 = max(0, y1 - h * self.roi_expansion)
        exp_x2 = min(self.frame_width, x2 + w * self.roi_expansion)
        exp_y2 = min(self.frame_height, y2 + h * self.roi_expansion)
        
        return [
            max(0, exp_x1 - self.roi_border_buffer),
            max(0, exp_y1 - self.roi_border_buffer),
            min(self.frame_width, exp_x2 + self.roi_border_buffer),
            min(self.frame_height, exp_y2 + self.roi_border_buffer)
        ]

    def _merge_rois(self, rois: list) -> np.ndarray:
        if not rois:
            return np.array([[0, 0, self.frame_width, self.frame_height]])
        
        return np.array([[
            min(r[0] for r in rois), min(r[1] for r in rois),
            max(r[2] for r in rois), max(r[3] for r in rois)
        ]])

    def process_input_frames(self, three_frames_list: list):
        if not three_frames_list or len(three_frames_list) != 3:
            # Or raise an error, depending on desired behavior
            return 

        if not hasattr(self, 'frame_width') or not hasattr(self, 'frame_height'):
            first_frame = three_frames_list[0]
            self.frame_height, self.frame_width = first_frame.shape[:2]
            # Optionally update SharedData if it's the source of truth
            # super().frame_height = self.frame_height 
            # super().frame_width = self.frame_width

        for frame in three_frames_list:
            self.frame_number_global += 1

            if self.current_roi is None:
                self._initialize_roi()
                if self.current_roi is None: # Still None if dimensions were missing
                    print(f"Error: Frame dimensions not available for frame {self.frame_number_global}. Skipping.")
                    continue


            x1_roi, y1_roi, x2_roi, y2_roi = self.current_roi[0].astype(int)
            y1_roi, y2_roi = max(0, y1_roi), min(frame.shape[0], y2_roi)
            x1_roi, x2_roi = max(0, x1_roi), min(frame.shape[1], x2_roi)

            roi_frame = frame
            detection_offset_x, detection_offset_y = 0, 0

            if y1_roi < y2_roi and x1_roi < x2_roi:
                roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]
                detection_offset_x, detection_offset_y = x1_roi, y1_roi
            
            tracked_this_frame = np.empty((0,5))
            if roi_frame.size > 0:
                results_list = self.detector(roi_frame, classes=0, conf=0.3, verbose=False)
                
                detections = []
                if results_list:
                    result_for_frame = results_list[0]
                    boxes_xyxy = result_for_frame.boxes.xyxy.cpu().numpy()
                    confs = result_for_frame.boxes.conf.cpu().numpy()
                    for i in range(len(boxes_xyxy)):
                        box = boxes_xyxy[i]
                        confidence = float(confs[i])
                        global_box = box.copy()
                        global_box[[0, 2]] += detection_offset_x
                        global_box[[1, 3]] += detection_offset_y
                        detections.append(np.array([*global_box, confidence]))
                
                if len(detections) > 0:
                    # Pass the original full frame to Sort if it uses image data
                    tracked_this_frame = self.tracker.update(np.array(detections), np.array([])) 
                else:
                    tracked_this_frame = self.tracker.update(np.empty((0, 5)), np.array([]))
            else: # roi_frame was empty
                tracked_this_frame = self.tracker.update(np.empty((0, 5)), np.array([]))
            
            current_frame_active_track_ids = set()
            for t in tracked_this_frame:
                track_id = int(t[4])
                current_frame_active_track_ids.add(track_id)
                bbox = t[:4].astype(int)
                self._update_history(track_id, bbox)
                
                x_min, y_min, x_max, y_max = bbox
                output_data = [
                    self.frame_number_global, track_id,
                    x_min, y_min, x_max, y_min,
                    x_max, y_max, x_min, y_max
                ]
                yield output_data
            
            # Manage track history for tracks that might have been lost by Sort
            # Sort manages max_age internally, self.tracker.trackers reflects active/recent tracks
            rois_for_merge = []
            # Use track_ids from Sort's internal state for ROI prediction
            # These are tracks that Sort considers active (not older than max_age)
            sort_active_ids = {int(trk.id) for trk in self.tracker.trackers}

            for tid in sort_active_ids:
                 if tid in self.track_history and self.track_history[tid]['positions']:
                    roi = self._calculate_target_roi(tid)
                    if roi is not None:
                        rois_for_merge.append(roi)
            
            if not rois_for_merge:
                self._initialize_roi() # Fallback to full frame ROI
            else:
                self.current_roi = self._merge_rois(rois_for_merge)

    def visualize(self, frame: np.ndarray, tracked_data_for_frame: list):
        vis_frame = frame.copy()
        if self.current_roi is not None:
            x1_r, y1_r, x2_r, y2_r = self.current_roi[0].astype(int)
            cv2.rectangle(vis_frame, (x1_r, y1_r), (x2_r, y2_r), (0, 255, 0), 2) # Green for ROI
        
        for data in tracked_data_for_frame:
            # data = [frame_num, track_id, x1tl, y1tl, x2tr, y2tr, x3br, y3br, x4bl, y4bl]
            track_id = data[1]
            x1_tl, y1_tl = data[2], data[3]
            # x2_tr, y2_tr = data[4], data[5] # Not needed for simple rectangle
            x3_br, y3_br = data[6], data[7]
            # x4_bl, y4_bl = data[8], data[9] # Not needed for simple rectangle
            
            cv2.rectangle(vis_frame, (x1_tl, y1_tl), (x3_br, y3_br), (255, 0, 0), 2) # Blue for tracks
            cv2.putText(vis_frame, f"ID: {track_id}", (x1_tl, y1_tl - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return vis_frame

# Example Usage (adjust as needed for your setup)
if __name__ == "__main__":
    # This example assumes SharedData provides yolo_path, frame_width, frame_height
    # For standalone testing, you might need to mock or set these:
    
    # To run this example, you would replace 'SharedData' in PlayerTracker's inheritance
    # with 'MockSharedDataForPlayerTracker' or ensure your actual SharedData is configured.
    # For simplicity, assuming PlayerTracker can get yolo_path and infer dimensions:

    # tracker = PlayerTracker() # If using actual SharedData correctly set up
    # If you need to mock SharedData for PlayerTracker:

    tracker = PlayerTracker()

    cap = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4') # Or your video file path
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    frames_batch_list = []
    BATCH_SIZE = 3

    # Initialize frame_width and frame_height for tracker from the first frame if not in SharedData
    # This is now handled inside process_input_frames
    # ret, first_frame_for_dim = cap.read()
    # if ret and not hasattr(tracker, 'frame_width'):
    #     tracker.frame_height, tracker.frame_width = first_frame_for_dim.shape[:2]
    #     tracker._initialize_roi() # Initialize ROI after getting dimensions
    #     frames_batch_list.append(first_frame_for_dim)
    # elif not ret:
    #     print("Error: Could not read first frame.")
    #     exit()
    # else: # Dimensions already set
    #     frames_batch_list.append(first_frame_for_dim)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Process any remaining frames in the buffer if it's not empty and less than BATCH_SIZE
            # For this example, we'll just break.
            break
        
        frames_batch_list.append(frame.copy()) # Use copy if frame is reused

        if len(frames_batch_list) == BATCH_SIZE:
            all_tracked_data_in_batch = []
            current_frame_for_vis = frames_batch_list[-1] # Visualize the last frame of the batch

            for data_item in tracker.process_input_frames(frames_batch_list):
                # data_item is [frame_number, tracked_id, x1tl,y1tl, x2tr,y2tr, x3br,y3br, x4bl,y4bl]
                print(data_item) # Print the output
                # Collect data if you need to visualize based on a specific frame's tracks
                if data_item[0] == tracker.frame_number_global: # Check if data is for the current (last) frame in batch
                    all_tracked_data_in_batch.append(data_item)
            
            if frames_batch_list: # Ensure there are frames to visualize
                vis_frame = tracker.visualize(frames_batch_list[-1], all_tracked_data_in_batch)
                cv2.imshow('Player Tracking', vis_frame)

            frames_batch_list = [] # Clear the batch

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()