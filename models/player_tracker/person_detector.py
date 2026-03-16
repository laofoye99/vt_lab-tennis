import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from utils.shared_data import SharedData

class PersonDetector(SharedData):
    def __init__(self):
        super().__init__()
        self.model = YOLO(self.yolo_path)
        self.min_score = 0.3
        self.player1_boxes = []
        self.player2_boxes = []
        self.roi_mask = None  # 添加ROI掩码属性
        self.person_output_path = os.path.join(self.output_video_dir, 'person_output.mp4')
        self.person_output = cv2.VideoWriter(self.person_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (self.frame_width, self.frame_height))
        
    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / float(areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0

    def set_roi_mask(self, mask):
        """设置感兴趣区域(ROI)掩码"""
        self.roi_mask = mask
        # print(f"ROI掩码已设置: 掩码大小 {mask.shape}")
        return self

    def apply_roi(self, frame):
        # 如果存在外部设置的ROI掩码，优先使用它
        if self.roi_mask is not None and self.roi_mask.shape[:2] == frame.shape[:2]:
            # 确保掩码维度匹配
            mask = self.roi_mask.copy()
        else:
            # 否则使用默认的ROI生成逻辑
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            if not self.player1_boxes and not self.player2_boxes:
                cv2.rectangle(mask, (int(self.frame_width*0.1), int(self.frame_height*0.1)), 
                             (int(self.frame_width*0.85), int(self.frame_height*0.4)), 255, -1)
                cv2.rectangle(mask, (int(self.frame_width*0.1), int(self.frame_height*0.5)), 
                             (int(self.frame_width*0.85), int(self.frame_height*0.9)), 255, -1)
            else:
                if self.player1_boxes:
                    last_p1 = self.player1_boxes[-1]
                    if np.any(last_p1):
                        x1,y1,x2,y2 = last_p1.astype(int)
                        margin = 650
                        cv2.rectangle(mask, 
                                    (max(0,x1-margin), max(0,y1-margin)),
                                    (min(self.frame_width,x2+margin), min(self.frame_height,y2+margin)),
                                    255, -1)
        
        # 应用掩码到原始帧
        return cv2.bitwise_and(frame, frame, mask=mask)

    def detect(self, frame, original_frame, event_draw=False):
        processed = self.apply_roi(frame)
        results = self.model(processed, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        persons = boxes[(results.boxes.cls.cpu().numpy() == 0) & (scores > self.min_score)]
        
        if not self.player1_boxes and not self.player2_boxes:
            lower_boxes = [b for b in persons if b[3] > self.frame_height/2]
            upper_boxes = [b for b in persons if (b[3] <= self.frame_height*0.4) & (b[1] >= self.frame_height*0.1)]
            p1 = max(lower_boxes, key=lambda x: x[3], default=np.zeros(4))
            p2 = max(upper_boxes, key=lambda x: x[3], default=np.zeros(4))
        else:
            p1 = self.player1_boxes[-1].copy() if self.player1_boxes else np.zeros(4)
            p2 = self.player2_boxes[-1].copy() if self.player2_boxes else np.zeros(4)
            max_iou1, max_iou2 = 0.3, 0.3
            
            for box in persons:
                if self.player1_boxes:
                    iou1 = self.calculate_iou(self.player1_boxes[-1], box)
                    if iou1 > max_iou1:
                        max_iou1 = iou1
                        p1 = box
                if self.player2_boxes:
                    iou2 = self.calculate_iou(self.player2_boxes[-1], box)
                    if iou2 > max_iou2:
                        max_iou2 = iou2
                        p2 = box
        
        self.player1_boxes.append(p1)
        self.player2_boxes.append(p2)

        if event_draw:
            cv2.rectangle(original_frame, (int(p1[0]), int(p1[1])), (int(p1[2]), int(p1[3])), (0,255,0), 2)
            cv2.rectangle(original_frame, (int(p2[0]), int(p2[1])), (int(p2[2]), int(p2[3])), (0,255,0), 2)
            cv2.imshow('person detection', original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                return
            
        self.person_output.write(original_frame)
        
        return [self.frame_number, *map(int, p1), *map(int, p2)]

    def cleanup(self):
        cv2.destroyWindow('person detection')
        self.person_output.release()

# if __name__ == "__main__":
#     total_time_elapsed = 0
#     detector = PersonDetector()
#     cap = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     while cap.isOpened():
#         start_time = time.time()
#         ret, frame = cap.read()
#         if not ret: break
#         detector.detect(frame, event_draw=True)
#         end_time = time.time()
#         total_time_elapsed += (end_time - start_time)*1000
#     print(f"Time taken average: {total_time_elapsed/total_frames:.2f}ms")
#     detector.cleanup()
# detector = PersonDetector()
# cap = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# out = cv2.VideoWriter('/home/max/Desktop/tennis/tennis-v5/data/OutputVideos/output_person.mp4', fourcc, fps, (width, height))
# results = []
# frame_number = 0

# # start_time = time.time()
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
    
#     result = detector.detect(frame, frame_number)
#     results.append(result)
    
#     cv2.rectangle(frame, (result[1], result[2]), (result[3], result[4]), (0,255,0), 2)
#     cv2.rectangle(frame, (result[5], result[6]), (result[7], result[8]), (0,0,255), 2)
#     out.write(frame)
#     frame_number += 1

# cap.release()
# out.release()
# pd.DataFrame(results, columns=['frame_number','x1','y1','x2','y2','x3','y3','x4','y4']).to_csv('/home/max/Desktop/tennis/tennis-v3/models/1_combined_output.csv', index=False)
# # print(f"Processed in {time.time()-start_time:.2f}s, {frame_number/(time.time()-start_time):.2f}fps")
