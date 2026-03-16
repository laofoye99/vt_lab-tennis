import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithms.sort import Sort
from ultralytics import YOLO

class PeopleTracker:
    def __init__(self, model_path="/home/max/Desktop/tennis/tennis-v5/weights/yolo11x.pt", target_people=2, roi_expansion=0.3):
        # 初始化YOLO模型
        self.model = YOLO(model_path)
        self.model_params = {
            'classes': [0],     # 只检测人物
            'conf': 0.3,        # 置信度阈值
            'verbose': False    
        }
        
        # 初始化SORT跟踪器
        self.sort_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
        
        # 配置运行参数
        self.target_people = target_people
        self.roi_expansion = roi_expansion
        self.current_roi = None    # 当前有效ROI区域（合并后的单一区域）
        self.track_history = {}    # 跟踪状态记录

    def process_frame(self, frame):
        # 决策检测模式
        if not self.current_roi:
            detections = self._full_detection(frame)
        else:
            detections = self._roi_detection(frame)
            # 当ROI检测失败时回退全图检测
            if detections.size == 0:
                self.current_roi = None
                detections = self._full_detection(frame)

        # 更新跟踪器
        tracked_objs = self.sort_tracker.update(np.array(detections) if detections.size else np.empty((0,5)), np.array([]))
        
        # 更新系统状态
        self._update_tracking_context(tracked_objs, frame.shape[1], frame.shape[0])
        
        return self._format_output(tracked_objs)

    def _full_detection(self, frame):
        """全图检测模式"""
        results = self.model.predict(frame, **self.model_params)
        return self._parse_detections(results)

    def _roi_detection(self, frame):
        """ROI区域检测"""
        x1, y1, x2, y2 = self.current_roi
        roi_img = frame[y1:y2, x1:x2]
        
        if roi_img.size == 0:
            return np.empty((0, 5))
            
        results = self.model.predict(roi_img, **self.model_params)
        detections = []
        for det in self._parse_detections(results):
            # 坐标转换到原图
            det[0] += x1
            det[1] += y1
            det[2] += x1
            det[3] += y1
            detections.append(det)
        
        return np.array(detections) if detections else np.empty((0, 5))

    def _parse_detections(self, results):
        """解析检测结果"""
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                conf = box.conf.item()
                detections.append([*xyxy, conf])
        return np.array(detections) if detections else np.empty((0, 5))

    def _update_tracking_context(self, tracked_objs, img_w, img_h):
        """更新跟踪上下文"""
        # 维护跟踪历史
        current_ids = set(int(obj[4]) for obj in tracked_objs)
        
        # 清理丢失的目标
        self.track_history = {
            k: v for k, v in self.track_history.items() 
            if k in current_ids
        }
        
        # 更新现有目标状态
        individual_rois = []
        for obj in tracked_objs:
            obj_id = int(obj[4])
            # 记录个体ROI
            individual_roi = self._expand_roi(obj[:4], img_w, img_h)
            individual_rois.append(individual_roi)
            
            if obj_id not in self.track_history:
                self.track_history[obj_id] = {
                    'age': 0,
                    'trajectory': []
                }
            self.track_history[obj_id]['age'] += 1
            self.track_history[obj_id]['trajectory'].append(obj[:4])
        
        # 生成合并后的全局ROI
        if individual_rois:
            self.current_roi = self._merge_rois(individual_rois, img_w, img_h)
        else:
            self.current_roi = None

    def _expand_roi(self, bbox, img_w, img_h):
        """生成单个目标的扩展ROI"""
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
        
        return [
            max(0, x1 - int(w * self.roi_expansion)),
            max(0, y1 - int(h * self.roi_expansion)),
            min(img_w, x2 + int(w * self.roi_expansion)),
            min(img_h, y2 + int(h * self.roi_expansion))
        ]

    def _merge_rois(self, rois, img_w, img_h):
        """合并多个ROI为单一区域"""
        # 计算合并后的边界
        min_x = min(r[0] for r in rois)
        min_y = min(r[1] for r in rois)
        max_x = max(r[2] for r in rois)
        max_y = max(r[3] for r in rois)
        
        # 应用安全边界
        return [
            max(0, min_x),
            max(0, min_y),
            min(img_w, max_x),
            min(img_h, max_y)
        ]

    def _format_output(self, tracked_objs):
        """生成最终输出"""
        # 按跟踪稳定性排序
        sorted_objs = sorted(
            tracked_objs,
            key=lambda x: self.track_history.get(int(x[4]), {'age': 0})['age'],
            reverse=True
        )
        
        # 提取目标框
        selected = [list(map(int, obj[:4])) for obj in sorted_objs[:self.target_people]]
        
        # 使用历史数据补足数量
        if len(selected) < self.target_people:
            missing = self.target_people - len(selected)
            candidates = sorted(
                self.track_history.values(),
                key=lambda x: x['age'],
                reverse=True
            )[:missing]
            selected.extend(
                list(map(int, c['trajectory'][-1]))
                for c in candidates if c['trajectory']
            )
        
        return selected[:self.target_people]

# 使用示例
if __name__ == "__main__":
    tracker = PeopleTracker(
        model_path="/home/max/Desktop/tennis/tennis-v5/weights/yolo11x.pt",
        target_people=4,
        roi_expansion=0.4
    )

    cap = cv2.VideoCapture("/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        bboxes = tracker.process_frame(frame)
        
        # 可视化合并后的ROI
        if tracker.current_roi:
            x1, y1, x2, y2 = tracker.current_roi
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        
        # 绘制追踪框
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()