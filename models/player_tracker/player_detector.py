import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import cv2
import numpy as np
import time
from ultralytics import YOLO
from algorithms.sort import Sort
from utils.shared_data import SharedData

class PersonDetector(SharedData):
    def __init__(self):
        super().__init__()
        self.model = YOLO(self.yolo_path)
        self.min_score = 0.3
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # 初始化 SORT 跟踪器
        self.tracks = {}       # 记录 player1 和 player2 的 ID
        self.player1_boxes = []
        self.player2_boxes = []
        self.roi_mask = None

        self.person_output_path = os.path.join(self.output_video_dir, 'person_output.mp4')
        self.person_output = cv2.VideoWriter(
            self.person_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25,
            (self.frame_width, self.frame_height)
        )

    def apply_roi(self, frame):
        """应用 ROI 掩码（如果有），否则不处理"""
        if self.roi_mask is not None and self.roi_mask.shape[:2] == frame.shape[:2]:
            return cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        else:
            return frame  # 没有掩码就返回原图

    def generate_roi_mask(self, box1, box2, margin=300):
        """根据两个框生成掩码"""
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        for box in [box1, box2]:
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(self.frame_width, x2 + margin)
                y2 = min(self.frame_height, y2 + margin)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def detect(self, frame, event_draw=False):
        processed = self.apply_roi(frame)
        results = self.model(processed, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        valid_idx = (classes == 0) & (scores > self.min_score)
        person_boxes = boxes[valid_idx]
        person_scores = scores[valid_idx]

        if len(person_boxes) == 0:
            self.person_output.write(frame)
            return [self.frame_number, 0, 0, 0, 0, 0, 0, 0, 0]

        detections = np.hstack((person_boxes, person_scores.reshape(-1, 1)))
        tracked_objects = self.tracker.update(detections, np.array([]))

        player_boxes = []
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            area = (x2 - x1) * (y2 - y1)
            player_boxes.append((area, int(track_id), [int(x1), int(y1), int(x2), int(y2)]))

        player_boxes.sort(reverse=True)
        players = player_boxes[:2]

        # 初始化 player1 和 player2
        if len(self.tracks) < 2:
            for i, (_, tid, box) in enumerate(players):
                if tid not in self.tracks.values():
                    key = 'player1' if i == 0 else 'player2'
                    self.tracks[key] = tid
            if len(players) == 1:
                self.tracks['player2'] = self.tracks['player1']

        player1_box = [0, 0, 0, 0]
        player2_box = [0, 0, 0, 0]
        for _, tid, box in players:
            if tid == self.tracks.get('player1'):
                player1_box = box
            elif tid == self.tracks.get('player2'):
                player2_box = box

        self.player1_boxes.append(np.array(player1_box))
        self.player2_boxes.append(np.array(player2_box))

        # 更新 ROI 掩码
        self.roi_mask = self.generate_roi_mask(player1_box, player2_box)

        # 绘图
        if event_draw:
            cv2.rectangle(frame, tuple(player1_box[:2]), tuple(player1_box[2:]), (0,255,0), 2)
            cv2.rectangle(frame, tuple(player2_box[:2]), tuple(player2_box[2:]), (0,255,0), 2)
            cv2.imshow('person detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                return

        self.person_output.write(frame)
        return [self.frame_number, *player1_box, *player2_box]

    def cleanup(self):
        cv2.destroyWindow('person detection')
        self.person_output.release()

if __name__ == "__main__":
    detector = PersonDetector()
    cap = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time_elapsed = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        detector.detect(frame, event_draw=True)
        end_time = time.time()
        total_time_elapsed += (end_time - start_time)*1000
    print(f"Time taken average: {total_time_elapsed/total_frames:.2f}ms")
    detector.cleanup()