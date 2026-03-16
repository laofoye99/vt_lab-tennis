import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
from algorithms.sort import Sort
from utils.shared_data import SharedData

class PlayerTracker(SharedData):
    def __init__(self):
        super().__init__()
        # 初始化模型组件
        self.detector = YOLO(self.yolo_path)
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
        
        # 配置参数
        self.roi_expansion = 0.3
        self.history_size = 10
        self.frame_buffer = 25
        
        # 状态存储
        self.current_roi = None
        
        self.track_history = {}  # {track_id: {'positions': [], 'velocities': [], 'accelerations': []}}

    def _initialize_roi(self):
        """初始化ROI为全帧"""
        self.current_roi = np.array([[0, 0, self.frame_width, self.frame_height]])

    def _update_history(self, track_id: int, bbox: np.ndarray):
        """
        更新指定track_id的运动历史数据
        :param track_id: 目标ID
        :param bbox: 当前边界框 [x1,y1,x2,y2]
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'positions': [],
                'velocities': [],
                'accelerations': []
            }
        
        history = self.track_history[track_id]
        
        # 更新位置数据
        history['positions'].append(bbox.copy())
        if len(history['positions']) > self.history_size:
            history['positions'].pop(0)
        
        # 计算速度
        if len(history['positions']) >= 2:
            velocity = bbox - history['positions'][-2]
            history['velocities'].append(velocity)
            if len(history['velocities']) > self.history_size:
                history['velocities'].pop(0)
        
        # 计算加速度
        if len(history['positions']) >= 3:
            acc = history['positions'][-1] - 2*history['positions'][-2] + history['positions'][-3]
            history['accelerations'].append(acc)
            if len(history['accelerations']) > self.history_size:
                history['accelerations'].pop(0)

    def _predict_position(self, track_id: int) -> np.ndarray:
        """
        预测目标下一个位置
        :return: 预测的边界框 [x1,y1,x2,y2]
        """
        history = self.track_history.get(track_id, None)
        if not history or len(history['positions']) == 0:
            return None
        
        current = history['positions'][-1]
        velocity = history['velocities'][-1] if history['velocities'] else np.zeros(4)
        acceleration = history['accelerations'][-1] if history['accelerations'] else np.zeros(4)
        
        return current + velocity + 0.5 * acceleration

    def _calculate_target_roi(self, track_id: int) -> list:
        """计算单个目标的ROI区域"""
        predicted = self._predict_position(track_id)
        if predicted is None:
            return None
        
        # 计算扩展区域
        x1, y1, x2, y2 = predicted
        w = x2 - x1
        h = y2 - y1
        expanded = [
            max(0, x1 - w * self.roi_expansion),
            max(0, y1 - h * self.roi_expansion),
            min(self.frame_width, x2 + w * self.roi_expansion),
            min(self.frame_height, y2 + h * self.roi_expansion)
        ]
        
        # 添加边界缓冲
        return [
            max(0, expanded[0] - self.frame_buffer),
            max(0, expanded[1] - self.frame_buffer),
            min(self.frame_width, expanded[2] + self.frame_buffer),
            min(self.frame_height, expanded[3] + self.frame_buffer)
        ]

    def _merge_rois(self, rois: list) -> np.ndarray:
        """合并所有ROI区域"""
        if not rois:
            return np.array([[0, 0, self.frame_width, self.frame_height]])
        
        return np.array([[
            min(r[0] for r in rois),
            min(r[1] for r in rois),
            max(r[2] for r in rois),
            max(r[3] for r in rois)
        ]])

    def _preprocess_frame(self, frame):
        pass
    
    def process_frame(self, frames: np.ndarray) -> dict:
        """
        处理视频帧的核心方法
        :return: 包含跟踪结果和ROI的字典
        """
        for frame in frames:
            # 更新帧尺寸
            if self.current_roi is None:
                self._initialize_roi()
            
            # ROI裁剪
            x1, y1, x2, y2 = self.current_roi[0].astype(int)
            roi_frame = frame[y1:y2, x1:x2]
            
            # YOLO检测
            results = self.detector(roi_frame, classes=0, conf=0.3, verbose=False)  # 只检测人物
            
            # 转换检测坐标到原图
            detections = []
            for result in results:
                for box in result.boxes.xyxy.cpu().numpy():
                    global_box = box.copy()
                    global_box[[0, 2]] += x1
                    global_box[[1, 3]] += y1
                    confidence = float(result.boxes.conf[0].item())
                    detections.append(np.array([*global_box, confidence]))  # 添加置信度
            
            # SORT跟踪
            if len(detections) > 0:
                tracked = self.tracker.update(np.array(detections), np.array([]))
            else:
                tracked = self.tracker.update(np.empty((0, 5)), np.array([]))
            
            # 更新历史数据
            for t in tracked:
                track_id = int(t[4])
                bbox = t[:4].astype(int)
                self._update_history(track_id, bbox)
            
            # 计算各目标ROI
            active_rois = [self._calculate_target_roi(tid) for tid in self.track_history]
            valid_rois = [r for r in active_rois if r is not None]
            
            # 合并ROI并更新
            self.current_roi = self._merge_rois(valid_rois)
            
            yield {
                'frame': frame,
                'tracked_players': tracked,
                'current_roi': self.current_roi.copy(),
                'track_history': self.track_history.copy()
            }
        
        # # 返回结构化数据
        # return {
        #     'frame': frame,
        #     'tracked_players': tracked,
        #     'current_roi': self.current_roi.copy(),
        #     'track_history': self.track_history.copy()
        # }

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """可视化跟踪结果"""
        # 绘制ROI
        x1, y1, x2, y2 = self.current_roi[0].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制跟踪框
        for t in self.track_history.values():
            if len(t['positions']) > 0:
                x1, y1, x2, y2 = t['positions'][-1].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return frame

# # # 使用示例
# if __name__ == "__main__":
#     tracker = PlayerTracker()
#     cap = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # 处理帧并获取结果
#         result = tracker.process_frame(frame)
        
#         # 可视化
#         vis_frame = tracker.visualize(result['frame'])
#         cv2.imshow('Tracking', vis_frame)
        
#         if cv2.waitKey(1) == 27:
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()