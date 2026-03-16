import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.shared_data import SharedData
from models.tennis_tracker.wasb import HRNet
from utils.time_utils import createTimeStamp
from utils.traj2 import RealTimeSegmenter, shot_and_bounce
from models.speed_calculator.speed_calculator import SpeedCalculator
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

import time

class TennisTracker(SharedData):
    def __init__(self):
        super().__init__()
        
        self.model = HRNet(cfg=self.config).to(self.device)
        self.speed_calculator = SpeedCalculator()
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        
        self.frames_buffer = []
        self.original_frames_buffer = []
        self.prev_positions = []
        self.last_coord_index = 0
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config["inp_height"], self.config["inp_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.inner_frame_number = 0
        self.segmenter = RealTimeSegmenter(shot_and_bounce)
        self.top_bounces = []
        self.bottom_bounces = []
        self.top_shots = []
        self.bottom_shots = []
        self.stop_event_bounces = []
        self.output_dir = '/home/max/Desktop/tennis/tennis-v5/data/OutputVideos'
        self.out = cv2.VideoWriter(os.path.join(self.output_dir, 'traj_output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (self.frame_width, self.frame_height))
        
        self.original_map = cv2.imread('/home/max/Desktop/tennis/tennis-v5/data/images/standard_court_reference.png')
        self.map_height, self.map_width, _ = self.original_map.shape
        self.out_map = cv2.VideoWriter(os.path.join(self.output_dir, 'map_output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (self.map_width, self.map_height))
        
        self.time_elapsed = 0
        
    def preprocess_frame(self, frame):
        return self.transform(frame)
            
    def process(self, frame, original_frame, event_draw=False):
        # start_time = time.time()
        # self.frames_buffer = frames
        self.frames_buffer.append(frame)
        self.original_frames_buffer.append(original_frame)
        if len(self.frames_buffer) == self.config['frames_in']:
            frames_processed = [self.preprocess_frame(f) for f in self.frames_buffer]
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)[0]
                
            detected = False
            center_x, center_y, confidence = 0, 0, 0
            
            for i in range(self.config['frames_out']):
                output = outputs[0][i]
                output = torch.sigmoid(output)
                heatmap = output.squeeze().cpu().numpy()
                heatmap = cv2.resize(heatmap, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                heatmap = (heatmap > 0.5).astype(np.float32) * heatmap
                
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((heatmap > 0).astype(np.uint8), connectivity=8)
                
                blob_centers = []
                for j in range(1, num_labels):
                    mask = labels_im == j
                    blob_sum = heatmap[mask].sum()
                    if blob_sum > 0:
                        center_x = np.sum(np.where(mask)[1] * heatmap[mask]) / blob_sum
                        center_y = np.sum(np.where(mask)[0] * heatmap[mask]) / blob_sum
                        blob_centers.append((center_x, center_y, blob_sum))
            
                if blob_centers:
                    predicted_position = self.predict_ball_position(self.prev_positions)
                    if predicted_position is not None:
                        distances = [np.sqrt((x - predicted_position[0]) ** 2 + (y - predicted_position[1]) ** 2) for x, y, _ in blob_centers]
                        closest_blob_idx = np.argmin(distances)
                        center_x, center_y, confidence = blob_centers[closest_blob_idx]
                    else:
                        blob_centers.sort(key=lambda x: x[2], reverse=True)
                        center_x, center_y, confidence = blob_centers[0]
                    detected = True
                    self.prev_positions.append(np.array([center_x, center_y]))
                    if len(self.prev_positions) > 3:
                        self.prev_positions.pop(0)
                        
                x_proj, y_proj, x_standardized, y_standardized = self.standardize_coordinates((center_x, center_y))
                
                timestamp = createTimeStamp()
                
                if detected:
                    cv2.circle(self.original_frames_buffer[i], (int(center_x), int(center_y)), 5, (0, 0, 255), 2)
                    
                current_map = self.original_map.copy()
                if detected:
                    cv2.circle(current_map, (int(x_proj), int(y_proj)), 3, (255, 255, 255), -1)
                    
                self.out_map.write(current_map)
                    
                if detected:
                    self.coordinates.append([self.inner_frame_number, timestamp, 1, center_x, center_y, x_proj, y_proj, x_standardized, y_standardized, confidence])
                    results = self.segmenter.process_row(self.inner_frame_number, center_x, center_y)
                else:
                    self.coordinates.append([self.inner_frame_number, timestamp, 0, 0, 0, 0, 0, 0, 0, 0])
                    results = self.segmenter.process_row(self.inner_frame_number, np.nan, np.nan)
                    
                self.speed_calculator.add_coordinate(x_proj, y_proj)
                speed_results = self.speed_calculator.get_results()
                if speed_results.get('speed') > 0:
                    cv2.putText(self.original_frames_buffer[i], f"Speed: {speed_results.get('speed'):.2f} m/s, Direction: {speed_results.get('direction')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                if results is not None:
                    detected_bounces, detected_shot, stop_event_occurred = results
                    if detected_bounces:
                        for bounce_info in detected_bounces:
                            print('detect_bounces: ', bounce_info.get('bounce'))
                            bounce_ptx = bounce_info.get('bounce_x')
                            bounce_pty = bounce_info.get('bounce_y')
                            bounce_type = bounce_info.get('bounce_type', '')
                            
                            if bounce_ptx is not None and bounce_pty is not None:
                                if stop_event_occurred:
                                    print(f"Detected Stop Event Bounce: Frame {bounce_info.get('bounce')}, Original Type: {bounce_type}")
                                    self.stop_event_bounces.append(bounce_info)
                                else:
                                    if 'top' in bounce_type:
                                        print(f"Detected Top Bounce: Frame {bounce_info.get('bounce')}, Type: {bounce_type}")
                                        self.top_bounces.append(bounce_info)
                                    elif 'bottom' in bounce_type:
                                        print(f"Detected Bottom Bounce: Frame {bounce_info.get('bounce')}, Type: {bounce_type}")
                                        self.bottom_bounces.append(bounce_info)
                                    else:
                                        print(f"Warning: Detected bounce with unknown type: {bounce_type}")
                                        # pass
                            else:
                                print(f"Warning: Bounce detected with invalid coordinates: {bounce_info}")
                                # pass
                    
                    if detected_shot:
                        shot_type = detected_shot.get('shot_type')
                        print('detect_shots: ', detected_shot.get('shot'))
                        if shot_type == 'top':
                            self.top_shots.append(detected_shot)
                        elif shot_type == 'bottom':
                            self.bottom_shots.append(detected_shot)
                        else:
                            print(f"Warning: Detected shot with unknown type: {shot_type}")
                            # pass
                            
                for bounce_info in self.top_bounces[-5:]:
                    bounce_ptx = bounce_info.get('bounce_x')
                    bounce_pty = bounce_info.get('bounce_y')
                    if bounce_ptx is not None and bounce_pty is not None:
                        bounce_coords = (int(bounce_ptx), int(bounce_pty))
                        color = (0, 128, 0)
                        thickness = -1
                        cv2.circle(self.original_frames_buffer[i], bounce_coords, 5, color, thickness, cv2.LINE_AA)
                        
                for bounce_info in self.bottom_bounces[-5:]:
                    bounce_ptx = bounce_info.get('bounce_x')
                    bounce_pty = bounce_info.get('bounce_y')
                    if bounce_ptx is not None and bounce_pty is not None:
                        bounce_coords = (int(bounce_ptx), int(bounce_pty))
                        color = (255, 144, 0)
                        thickness = -1
                        cv2.circle(self.original_frames_buffer[i], bounce_coords, 5, color, thickness, cv2.LINE_AA)
                        
                # for bounce_info in self.stop_event_bounces:
                #     bounce_ptx = bounce_info.get('bounce_x')
                #     bounce_pty = bounce_info.get('bounce_y')
                #     if bounce_ptx is not None and bounce_pty is not None:
                #         bounce_coords = (int(bounce_ptx), int(bounce_pty))
                #         color = (0, 0, 255)
                #         thickness = 2
                #         cv2.circle(self.frames_buffer[i], bounce_coords, 5, color, thickness, cv2.LINE_AA)
                        
                for shot_info in self.top_shots[-5:]:
                    shot_ptx = shot_info.get('shot_x')
                    shot_pty = shot_info.get('shot_y')
                    if shot_ptx is not None and shot_pty is not None:
                        shot_coords = (int(shot_ptx), int(shot_pty))
                        color = (255, 144, 0)
                        thickness = 2
                        cv2.circle(self.original_frames_buffer[i], shot_coords, 4, color, thickness, cv2.LINE_AA)
                        
                for shot_info in self.bottom_shots[-5:]:
                    shot_ptx = shot_info.get('shot_x')
                    shot_pty = shot_info.get('shot_y')
                    if shot_ptx is not None and shot_pty is not None:
                        shot_coords = (int(shot_ptx), int(shot_pty))
                        color = (0, 128, 0)
                        thickness = 2
                        cv2.circle(self.original_frames_buffer[i], shot_coords, 4, color, thickness, cv2.LINE_AA)
                
                if event_draw:
                    cv2.imshow('frame', self.original_frames_buffer[i])
                    cv2.imshow('map', current_map)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.cleanup()
                        return
                
                self.out.write(self.original_frames_buffer[i])
                
                self.inner_frame_number += 1
            
            self.frames_buffer = []
            self.original_frames_buffer = []
            
        # end_time = time.time()
        # self.time_elapsed += (end_time - start_time)*1000
        # print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    
    def get_latest_event(self):
        if len(self.top_bounces) > 0:
            top_bounces = self.top_bounces[-1]
        else:
            top_bounces = None
        if len(self.bottom_bounces) > 0:
            bottom_bounces = self.bottom_bounces[-1]
        else:
            bottom_bounces = None
        if len(self.top_shots) > 0:
            top_shots = self.top_shots[-1]
        else:
            top_shots = None
        if len(self.bottom_shots) > 0:
            bottom_shots = self.bottom_shots[-1]
        else:
            bottom_shots = None
        # stop_event_bounces = self.stop_event_bounces[-1]
        
        return [top_bounces, bottom_bounces, top_shots, bottom_shots]
        
    def cleanup(self):
        self.out.release()
        self.out_map.release()
        cv2.destroyWindow('frame')
        cv2.destroyWindow('map')
        
    def get_latest_coordinates(self):
        new_coords = self.coordinates[self.last_coord_index:]
        self.last_coord_index = len(self.coordinates)
        return new_coords
    
    def predict_ball_position(self, prev_positions):
        if len(prev_positions) < 3:
            return None
        p_t = prev_positions[-1]  # 当前位置
        a_t = p_t - 2 * prev_positions[-2] + prev_positions[-3]  # 加速度
        v_t = p_t - prev_positions[-2] + a_t  # 速度
        predicted_position = p_t + v_t + 0.5 * a_t  # 预测位置: s = s0 + vt + 0.5at²
        predicted_position = np.clip(predicted_position, [0, 0], [self.frame_width, self.frame_height])  # 限制在图像范围内
        return predicted_position
    
    def standardize_coordinates(self, coordinates):
        projected = self.homography_matrix @ np.array([coordinates[0], coordinates[1], 1])
        x_proj = projected[0] / projected[2]
        y_proj = projected[1] / projected[2]
        
        x_standardized = (x_proj - 72) / 216
        y_standardized = (y_proj - 126) / 468
        
        return x_proj, y_proj, x_standardized, y_standardized


if __name__ == "__main__":
    tracker = TennisTracker()
    cap = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        tracker.process(frame, event_draw=True)
        
        new_coords = tracker.get_latest_coordinates()
        for coord in new_coords:
            # pass
            print(f"New coordinate: {coord}")
            
    # print(f"Time taken average: {tracker.time_elapsed/total_frames:.2f} ms")
    cap.release()
    cv2.destroyAllWindows()
        
#         for coordinate in tracker.process(frame):
#             with open('/home/max/Desktop/tennis/tennis-v5/data/OutputVideos/output.csv', 'a') as f:
#                 f.write(f"{coordinate}\n")
#             # print(coordinate)
            
#     cap.release()
#     cv2.destroyAllWindows()
    