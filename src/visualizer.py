import cv2
import numpy as np

class Visualizer:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width = width
        self.height = height
        # Colors (BGR)
        self.color_ball = (0, 255, 255)      # Yellow
        self.color_player = (255, 0, 0)      # Blue
        self.color_hit = (0, 0, 255)        # Red
        self.color_bounce = (0, 255, 0)     # Green
        self.color_text = (255, 255, 255)    # White

    def draw_frame(self, frame, frame_idx, smoothed_ball, players_data, events, movement_distances):
        """
        Overlays analysis results on a single frame.
        """
        # 1. Current Ball Position (No trajectory lines)
        curr_ball = smoothed_ball[frame_idx]
        if curr_ball != (0, 0):
            cv2.circle(frame, (int(curr_ball[0]), int(curr_ball[1])), 8, self.color_ball, -1)
            cv2.circle(frame, (int(curr_ball[0]), int(curr_ball[1])), 12, self.color_ball, 2) # Outer ring

        # 2. Players Bboxes and Keypoints
        if frame_idx in players_data:
            for pid, pinfo in players_data[frame_idx].items():
                bbox = pinfo['bbox']
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.color_player, 2)
                cv2.putText(frame, f"P{pid}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_player, 2)
                
                # Draw wrist keypoints for visual check
                kps = pinfo.get('keypoints', {})
                if 'r_wrist' in kps:
                    cv2.circle(frame, (int(kps['r_wrist'][0]), int(kps['r_wrist'][1])), 5, (0, 165, 255), -1)

        # 4. Events (Display for 20 frames after they occur)
        display_window = 20
        for event in events:
            if event['frame'] <= frame_idx < event['frame'] + display_window:
                pos = event['pixel_pos']
                etype = event['type'].upper()
                label = f"EVENT: {etype}"
                color = self.color_hit if event['type'] == 'hit' else self.color_bounce
                
                if event['type'] == 'hit':
                    label += f" | {event.get('stroke', '')} | {event.get('speed_kmh', 0):.1f} km/h"
                
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 15, color, 3)
                cv2.putText(frame, label, (int(pos[0]) + 20, int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 5. Dashboard (Summary Overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, "Match Analysis Dashboard", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)
        y_offset = 75
        for pid, dist in movement_distances.items():
            cv2.putText(frame, f"Player {pid} Dist: {dist:.2f} m", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_text, 1)
            y_offset += 25
        
        cv2.putText(frame, f"Frame: {frame_idx}", (self.width - 150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)

        return frame
