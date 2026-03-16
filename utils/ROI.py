"""
姿态检测模块
提供运动检测、姿态估计和可视化功能
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from ultralytics import YOLO
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Optional


class PoseDetector:
    """
    姿态检测器类
    负责从两帧图像中检测人物并估计姿态
    """
    
    def __init__(
        self,
        model_path: str = "weights/yolo11x-pose.pt",
        motion_threshold: int = 25,
        dbscan_eps: int = 100,
        dbscan_min_samples: int = 20,
        roi_margin: int = 150,
        bbox_padding: Tuple[int, int] = (50, 50),
        iou_threshold: float = 0.01,
        keypoint_conf_threshold: float = 0.5
    ):
        """
        初始化姿态检测器
        
        参数:
            model_path: YOLO模型路径
            motion_threshold: 运动检测阈值
            dbscan_eps: DBSCAN聚类的邻域半径
            dbscan_min_samples: DBSCAN聚类的最小样本数
            roi_margin: ROI边界扩展距离
            bbox_padding: 边界框扩展填充 (padding_x, padding_y)
            iou_threshold: IOU合并阈值
            keypoint_conf_threshold: 关键点置信度阈值
        """
        try:
            self.model = YOLO('weights/yolo11x-pose.pt')
        except:
            self.model = YOLO(model_path)
        self.motion_threshold = motion_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.roi_margin = roi_margin
        self.bbox_padding = bbox_padding
        self.iou_threshold = iou_threshold
        self.keypoint_conf_threshold = keypoint_conf_threshold
        
        # 关键点标签
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    
    def detect(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        roi: Optional[np.ndarray] = None,
        max_players: int = 2
    ) -> Tuple[List[List[Dict]], List[Dict], List[Dict]]:
        """
        从两帧图像中检测人物姿态
        
        参数:
            img1: 第一帧图像 (BGR格式)
            img2: 第二帧图像 (BGR格式)
            roi: ROI多边形顶点坐标 (N x 2 数组)，None表示使用整个图像
            max_players: 最大检测人数
        
        返回:
            (all_keypoints, all_detection_boxes, region_proposals)
            - all_keypoints: 所有人物的关键点列表
            - all_detection_boxes: 所有检测框
            - region_proposals: 区域提案
        """
        # 步骤1: 运动检测生成Region Proposals
        region_proposals = self._generate_region_proposals(
            img1, img2, roi, max_players
        )
        
        if not region_proposals:
            print("can not generate Region Proposals")
            return [], [], []
        
        # 步骤2: YOLO姿态检测
        all_keypoints, all_detection_boxes = self._detect_pose_in_proposals(
            img2, region_proposals
        )
        
        return all_keypoints, all_detection_boxes, region_proposals
    
    def _generate_region_proposals(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        roi: Optional[np.ndarray],
        max_players: int
    ) -> List[Dict]:
        """生成运动区域提案"""
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 计算帧差
        frame_diff = cv2.absdiff(gray1, gray2)
        
        # 二值化
        _, motion_mask = cv2.threshold(
            frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY
        )
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # 获取运动像素坐标
        motion_pixels = np.where(motion_mask > 0)
        motion_coords = np.column_stack((motion_pixels[1], motion_pixels[0]))
        
        # print(f"检测到 {len(motion_coords)} 个运动像素")
        
        # ROI过滤
        if roi is not None:
            filtered_coords = self._filter_by_roi(motion_coords, roi)
            # print(f"ROI过滤后剩余 {len(filtered_coords)} 个运动像素")
        else:
            filtered_coords = motion_coords
        
        if len(filtered_coords) == 0:
            return []
        
        # 聚类
        labels = self._cluster_motion_points(filtered_coords, max_players)
        
        # 生成proposals
        region_proposals = self._create_proposals(
            filtered_coords, labels, img2.shape, max_players
        )
        
        # IOU合并
        region_proposals = self._merge_proposals_by_iou(
            region_proposals, self.iou_threshold
        )
        
        # 重新编号
        for i, proposal in enumerate(region_proposals):
            proposal['player_id'] = i + 1
        
        return region_proposals
    
    def _filter_by_roi(
        self,
        coords: np.ndarray,
        roi: np.ndarray
    ) -> np.ndarray:
        """ROI过滤"""
        filtered_coords = []
        for coord in coords:
            point_tuple = (float(coord[0]), float(coord[1]))
            dist = cv2.pointPolygonTest(roi, point_tuple, True)
            if dist >= -self.roi_margin:
                filtered_coords.append(coord)
        return np.array(filtered_coords)
    
    def _cluster_motion_points(
        self,
        coords: np.ndarray,
        max_players: int
    ) -> np.ndarray:
        """聚类运动点"""
        # print(f"\n=== 使用DBSCAN进行聚类 ===")
        
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = dbscan.fit_predict(coords)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        # print(f"DBSCAN检测到 {n_clusters} 个区域")
        # print(f"噪声点: {n_noise}")
        
        # 处理聚类数量
        if n_clusters > max_players:
            # print(f"检测到 {n_clusters} 个区域，选择最大的{max_players}个")
            labels = self._select_top_clusters(labels, max_players)
        elif n_clusters < max_players:
            # print(f"只检测到 {n_clusters} 个区域，使用KMeans强制分为{max_players}类")
            kmeans = KMeans(n_clusters=max_players, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)
        
        return labels
    
    def _select_top_clusters(
        self,
        labels: np.ndarray,
        n_top: int
    ) -> np.ndarray:
        """选择点数最多的n_top个簇"""
        unique_labels = set(labels)
        cluster_sizes = []
        
        for label in unique_labels:
            if label != -1:
                cluster_size = np.sum(labels == label)
                cluster_sizes.append((label, cluster_size))
        
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        top_labels = [cluster_sizes[i][0] for i in range(min(n_top, len(cluster_sizes)))]
        
        # 重新映射标签
        label_mapping = {top_labels[i]: i for i in range(len(top_labels))}
        new_labels = np.array([
            label_mapping[l] if l in label_mapping else -1 
            for l in labels
        ])
        
        return new_labels
    
    def _create_proposals(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        img_shape: Tuple,
        n_clusters: int
    ) -> List[Dict]:
        """创建区域提案"""
        # print(f"\n=== 生成Region Proposals ===")
        
        proposals = []
        padding_x, padding_y = self.bbox_padding
        
        for cluster_id in range(n_clusters):
            cluster_points = coords[labels == cluster_id]
            n_points = len(cluster_points)
            
            if n_points == 0:
                continue
            
            # 计算紧凑边界框
            x_min, y_min = cluster_points.min(axis=0)
            x_max, y_max = cluster_points.max(axis=0)
            
            # 计算中心
            center_x, center_y = cluster_points.mean(axis=0)
            
            # 扩展边界框
            x_min_expanded = max(0, int(x_min - padding_x))
            y_min_expanded = max(0, int(y_min - padding_y))
            x_max_expanded = min(img_shape[1], int(x_max + padding_x))
            y_max_expanded = min(img_shape[0], int(y_max + padding_y))
            
            width = x_max_expanded - x_min_expanded
            height = y_max_expanded - y_min_expanded
            
            proposal = {
                'player_id': cluster_id + 1,
                'bbox': (x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded),
                'bbox_tight': (int(x_min), int(y_min), int(x_max), int(y_max)),
                'center': (int(center_x), int(center_y)),
                'n_points': n_points,
                'width': width,
                'height': height
            }
            
            proposals.append(proposal)
            
            # print(f"\nPlayer {cluster_id + 1} Region Proposal:")
            # print(f"  Tight BBox: {proposal['bbox_tight']}")
            # print(f"  Expanded BBox: {proposal['bbox']}")
            # print(f"  Size: {width} x {height}")
            # print(f"  Motion points: {n_points}")
        
        return proposals
    
    def _merge_proposals_by_iou(
        self,
        proposals: List[Dict],
        iou_threshold: float
    ) -> List[Dict]:
        """根据IOU合并重叠的proposals"""
        if len(proposals) <= 1:
            return proposals
        
        # print(f"\n=== IOU合并 (阈值={iou_threshold}) ===")
        
        merged_proposals = []
        merged_flags = [False] * len(proposals)
        
        for i in range(len(proposals)):
            if merged_flags[i]:
                continue
            
            current = proposals[i]
            to_merge = [current]
            merged_flags[i] = True
            
            for j in range(i + 1, len(proposals)):
                if merged_flags[j]:
                    continue
                
                iou = self._calculate_iou(current['bbox'], proposals[j]['bbox'])
                print(f"Proposal {current['player_id']} 与 Proposal {proposals[j]['player_id']} 的IOU: {iou:.3f}")
                
                if iou >= iou_threshold:
                    print(f"  → 合并 Proposal {proposals[j]['player_id']} 到 Proposal {current['player_id']}")
                    to_merge.append(proposals[j])
                    merged_flags[j] = True
            
            if len(to_merge) == 1:
                merged_proposals.append(current)
            else:
                merged_proposal = self._merge_multiple_proposals(to_merge)
                merged_proposals.append(merged_proposal)
        
        # print(f"\n合并后剩余 {len(merged_proposals)} 个proposals")
        return merged_proposals
    
    @staticmethod
    def _calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """计算两个边界框的IOU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        else:
            inter_area = 0
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    @staticmethod
    def _merge_multiple_proposals(proposals: List[Dict]) -> Dict:
        """合并多个proposals"""
        all_bboxes = [p['bbox'] for p in proposals]
        x_mins = [bbox[0] for bbox in all_bboxes]
        y_mins = [bbox[1] for bbox in all_bboxes]
        x_maxs = [bbox[2] for bbox in all_bboxes]
        y_maxs = [bbox[3] for bbox in all_bboxes]
        
        merged_bbox = (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))
        
        all_tight_bboxes = [p['bbox_tight'] for p in proposals]
        tight_x_mins = [bbox[0] for bbox in all_tight_bboxes]
        tight_y_mins = [bbox[1] for bbox in all_tight_bboxes]
        tight_x_maxs = [bbox[2] for bbox in all_tight_bboxes]
        tight_y_maxs = [bbox[3] for bbox in all_tight_bboxes]
        
        merged_tight_bbox = (
            min(tight_x_mins), min(tight_y_mins),
            max(tight_x_maxs), max(tight_y_maxs)
        )
        
        total_points = sum(p['n_points'] for p in proposals)
        merged_width = merged_bbox[2] - merged_bbox[0]
        merged_height = merged_bbox[3] - merged_bbox[1]
        merged_center = (
            (merged_bbox[0] + merged_bbox[2]) // 2,
            (merged_bbox[1] + merged_bbox[3]) // 2
        )
        
        return {
            'player_id': proposals[0]['player_id'],
            'bbox': merged_bbox,
            'bbox_tight': merged_tight_bbox,
            'center': merged_center,
            'n_points': total_points,
            'width': merged_width,
            'height': merged_height,
            'merged_from': [p['player_id'] for p in proposals]
        }
    
    def _detect_pose_in_proposals(
        self,
        image: np.ndarray,
        region_proposals: List[Dict]
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """在region proposals中进行姿态检测"""
        all_keypoints = []
        all_detection_boxes = []
        
        for proposal in region_proposals:
            player_id = proposal['player_id']
            x1, y1, x2, y2 = proposal['bbox']
            
            # print(f"\n处理 Player {player_id} 的区域...")
            
            # 裁剪图像
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                print(f"  warning: Player {player_id} crop region is None")
                continue
            
            # YOLO预测
            results = self.model.predict(cropped_image, verbose=False)
            
            # 处理检测框
            if len(results) > 0 and results[0].boxes is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                boxes_conf = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else None
                
                # print(f"  detect {len(boxes_xyxy)} people")
                
                for i, box in enumerate(boxes_xyxy):
                    box_x1, box_y1, box_x2, box_y2 = box
                    
                    original_box = {
                        'player_id': player_id,
                        'coords': [box_x1 + x1, box_y1 + y1, box_x2 + x1, box_y2 + y1],
                        'cropped_coords': [float(box_x1), float(box_y1), float(box_x2), float(box_y2)],
                        'confidence': float(boxes_conf[i]) if boxes_conf is not None else None
                    }
                    all_detection_boxes.append(original_box)
            
            # 处理关键点
            if len(results) > 0 and results[0].keypoints is not None:
                for person_idx in range(len(results[0].keypoints)):
                    keypoints_xy = results[0].keypoints.xy[person_idx].cpu().numpy()
                    keypoints_conf = None
                    
                    if hasattr(results[0].keypoints, 'conf') and results[0].keypoints.conf is not None:
                        keypoints_conf = results[0].keypoints.conf[person_idx].cpu().numpy()
                    
                    person_keypoints = []
                    
                    for i, (x, y) in enumerate(keypoints_xy):
                        if x > 0 or y > 0:
                            conf_value = None
                            if keypoints_conf is not None and i < len(keypoints_conf):
                                conf_value = float(keypoints_conf[i])
                            
                            if conf_value is None or conf_value >= self.keypoint_conf_threshold:
                                original_x = x + x1
                                original_y = y + y1
                                
                                person_keypoints.append({
                                    'player_id': player_id,
                                    'person_idx': person_idx,
                                    'label': self.keypoint_labels[i] if i < len(self.keypoint_labels) else f'keypoint_{i}',
                                    'original_coords': [float(original_x), float(original_y)],
                                    'cropped_coords': [float(x), float(y)],
                                    'confidence': conf_value,
                                    'index': i
                                })
                    
                    if person_keypoints:
                        all_keypoints.append(person_keypoints)
                        # print(f"  Person {person_idx}: detect {len(person_keypoints)} keypoints")
        
        return all_keypoints, all_detection_boxes


class PoseVisualizer:
    """
    姿态可视化器类
    负责绘制检测结果
    """
    
    # COCO格式的人体骨架连接关系
    SKELETON = [
        (0, 1), (0, 2),      # 鼻子到眼睛
        (1, 3), (2, 4),      # 眼睛到耳朵
        (0, 5), (0, 6),      # 鼻子到肩膀
        (5, 6),              # 左右肩膀
        (5, 7), (7, 9),      # 左臂
        (6, 8), (8, 10),     # 右臂
        (5, 11), (6, 12),    # 肩膀到臀部
        (11, 12),            # 左右臀部
        (11, 13), (13, 15),  # 左腿
        (12, 14), (14, 16)   # 右腿
    ]
    
    def __init__(
        self,
        show_roi: bool = True,
        show_proposals: bool = True,
        show_detection_boxes: bool = True,
        show_keypoints: bool = True,
        show_skeleton: bool = True
    ):
        """
        初始化可视化器
        
        参数:
            show_roi: 是否显示ROI
            show_proposals: 是否显示region proposals
            show_detection_boxes: 是否显示检测框
            show_keypoints: 是否显示关键点
            show_skeleton: 是否显示骨架连线
        """
        self.show_roi = show_roi
        self.show_proposals = show_proposals
        self.show_detection_boxes = show_detection_boxes
        self.show_keypoints = show_keypoints
        self.show_skeleton = show_skeleton
    
    def visualize(
        self,
        image: np.ndarray,
        all_keypoints: List[List[Dict]],
        all_detection_boxes: List[Dict],
        region_proposals: List[Dict],
        roi: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        可视化检测结果
        
        参数:
            image: 输入图像 (BGR格式)
            all_keypoints: 所有人物的关键点
            all_detection_boxes: 所有检测框
            region_proposals: 区域提案
            roi: ROI多边形顶点坐标
        
        返回:
            绘制后的图像 (BGR格式)
        """
        # 转换为PIL图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制ROI
        if self.show_roi and roi is not None:
            self._draw_roi(draw, roi)
        
        # 绘制region proposals
        if self.show_proposals:
            self._draw_proposals(draw, region_proposals)
        
        # 绘制检测框
        if self.show_detection_boxes:
            self._draw_detection_boxes(draw, all_detection_boxes)
        
        # 绘制姿态
        if self.show_skeleton or self.show_keypoints:
            self._draw_poses(draw, all_keypoints)
        
        # 转换回OpenCV格式
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image
    
    def _draw_roi(self, draw: ImageDraw.Draw, roi: np.ndarray):
        """绘制ROI"""
        roi_points = [(int(p[0]), int(p[1])) for p in roi]
        draw.polygon(roi_points, outline='yellow', width=2)
    
    def _draw_proposals(self, draw: ImageDraw.Draw, proposals: List[Dict]):
        """绘制region proposals"""
        for proposal in proposals:
            x1, y1, x2, y2 = proposal['bbox']
            self._draw_dashed_rectangle(
                draw, [x1, y1, x2, y2], 
                outline='cyan', width=2, dash_length=10
            )
            
            if 'merged_from' in proposal:
                label = f"Proposal {proposal['player_id']} (merged)"
            else:
                label = f"Proposal {proposal['player_id']}"
            draw.text((x1, y1 - 25), label, fill='cyan')
    
    def _draw_detection_boxes(self, draw: ImageDraw.Draw, boxes: List[Dict]):
        """绘制检测框"""
        for box in boxes:
            box_coords = box['coords']
            self._draw_dashed_rectangle(
                draw, box_coords, 
                outline='green', width=2, dash_length=5
            )
            
            if box['confidence'] is not None:
                conf_text = f"Person: {box['confidence']:.2f}"
                draw.text((box_coords[0], box_coords[1] - 15), conf_text, fill='green')
    
    def _draw_poses(self, draw: ImageDraw.Draw, all_keypoints: List[List[Dict]]):
        """绘制姿态（骨架和关键点）"""
        for person_keypoints in all_keypoints:
            # 创建关键点字典
            keypoints_dict = {kp['index']: kp for kp in person_keypoints}
            
            # 绘制骨架连线
            if self.show_skeleton:
                self._draw_skeleton(draw, keypoints_dict)
            
            # 绘制关键点
            if self.show_keypoints:
                self._draw_keypoints(draw, person_keypoints)
    
    def _draw_skeleton(self, draw: ImageDraw.Draw, keypoints_dict: Dict):
        """绘制骨架连线"""
        for start_idx, end_idx in self.SKELETON:
            if start_idx in keypoints_dict and end_idx in keypoints_dict:
                start_kp = keypoints_dict[start_idx]
                end_kp = keypoints_dict[end_idx]
                
                x1, y1 = start_kp['original_coords']
                x2, y2 = end_kp['original_coords']
                
                # 根据置信度选择颜色
                start_conf = start_kp['confidence'] if start_kp['confidence'] is not None else 0
                end_conf = end_kp['confidence'] if end_kp['confidence'] is not None else 0
                avg_conf = (start_conf + end_conf) / 2
                
                if avg_conf >= 0.7:
                    line_color = 'lime'
                    line_width = 3
                elif avg_conf >= 0.5:
                    line_color = 'yellow'
                    line_width = 2
                else:
                    line_color = 'orange'
                    line_width = 2
                
                draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)
    
    def _draw_keypoints(self, draw: ImageDraw.Draw, keypoints: List[Dict]):
        """绘制关键点"""
        for kp in keypoints:
            x, y = kp['original_coords']
            conf = kp['confidence']
            
            # 根据置信度选择颜色
            if conf is not None and conf >= 0.7:
                point_color = 'red'
                outline_color = 'white'
            elif conf is not None and conf >= 0.5:
                point_color = 'orange'
                outline_color = 'white'
            else:
                point_color = 'yellow'
                outline_color = 'white'
            
            radius = 5
            draw.ellipse(
                [x-radius, y-radius, x+radius, y+radius],
                fill=point_color, outline=outline_color, width=2
            )
    
    @staticmethod
    def _draw_dashed_rectangle(
        draw: ImageDraw.Draw,
        coords: List,
        outline: str = 'green',
        width: int = 2,
        dash_length: int = 5
    ):
        """绘制虚线矩形"""
        x1, y1, x2, y2 = coords
        
        # 上边
        for x in range(int(x1), int(x2), dash_length * 2):
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=outline, width=width)
        # 下边
        for x in range(int(x1), int(x2), dash_length * 2):
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=outline, width=width)
        # 左边
        for y in range(int(y1), int(y2), dash_length * 2):
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=outline, width=width)
        # 右边
        for y in range(int(y1), int(y2), dash_length * 2):
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=outline, width=width)

if __name__ == "__main__":
    from ball_point_info import BallPointInfo
    print("=" * 70)
    print("姿态检测模块测试")
    print("=" * 70)
    
    # ==================== 步骤1: 加载测试数据 ====================
    print("\n[步骤1] 加载测试图像...")
    
    try:
        bp_info = BallPointInfo(r"D:\tennis-dataset\game13\2560_1440\Clip2")
        imgs = list(bp_info.get_frame_paths())
        img1 = cv2.imread(str(imgs[28]))
        img2 = cv2.imread(str(imgs[29]))
        
        if img1 is None or img2 is None:
            raise ValueError("图像加载失败")
        
        print(f"  ✓ 图像1尺寸: {img1.shape}")
        print(f"  ✓ 图像2尺寸: {img2.shape}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        exit(1)
    
    # ==================== 步骤2: 定义ROI ====================
    print("\n[步骤2] 定义ROI区域...")
    
    roi = np.array([
        [1050, 544],
        [1593, 541],
        [2247, 1191],
        [411, 1178]
    ], dtype=np.int32)
    
    print(f"  ✓ ROI顶点数: {len(roi)}")
    
    # ==================== 步骤3: 初始化检测器 ====================
    print("\n[步骤3] 初始化姿态检测器...")
    
    detector = PoseDetector(
        model_path="yolo11x-pose.pt",
        motion_threshold=25,
        dbscan_eps=100,
        dbscan_min_samples=20,
        roi_margin=150,
        bbox_padding=(50, 50),
        iou_threshold=0.01,
        keypoint_conf_threshold=0.5
    )
    
    print(f"  ✓ 检测器初始化完成")
    print(f"    - 模型: yolo11x-pose.pt")
    print(f"    - 运动阈值: 25")
    print(f"    - 关键点置信度阈值: 0.5")
    
    # ==================== 步骤4: 执行姿态检测 ====================
    print("\n[步骤4] 执行姿态检测...")
    print("-" * 70)
    
    all_keypoints, all_detection_boxes, region_proposals = detector.detect(
        img1=img1,
        img2=img2,
        roi=roi,
        max_players=2
    )
    
    print("-" * 70)
    print(f"\n  检测结果:")
    print(f"    - Region Proposals: {len(region_proposals)}")
    print(f"    - 检测到的人数: {len(all_detection_boxes)}")
    print(f"    - 检测到的姿态: {len(all_keypoints)}")
    
    # 输出详细信息
    if region_proposals:
        print(f"\n  Region Proposals 详情:")
        for prop in region_proposals:
            print(f"    Player {prop['player_id']}: bbox={prop['bbox']}, points={prop['n_points']}")
    
    if all_keypoints:
        print(f"\n  关键点详情:")
        for i, person_kps in enumerate(all_keypoints):
            print(f"    人物 {i+1}: {len(person_kps)} 个关键点")
            # 显示前3个关键点作为示例
            for kp in person_kps[:3]:
                print(f"      - {kp['label']}: coords={kp['original_coords']}, conf={kp['confidence']:.2f}")
    
    # ==================== 步骤5: 可视化结果 ====================
    print("\n[步骤5] 可视化检测结果...")
    
    # 创建完整可视化
    print("  生成完整可视化 (包含ROI、proposals、检测框、骨架)...")
    visualizer_full = PoseVisualizer(
        show_roi=True,
        show_proposals=True,
        show_detection_boxes=True,
        show_keypoints=True,
        show_skeleton=True
    )
    
    result_full = visualizer_full.visualize(
        image=img2,
        all_keypoints=all_keypoints,
        all_detection_boxes=all_detection_boxes,
        region_proposals=region_proposals,
        roi=roi
    )
    
    output_full = 'test_result_full.jpg'
    cv2.imwrite(output_full, result_full)
    print(f"  ✓ 完整结果已保存: {output_full}")
    
    # 创建简化可视化（只显示骨架）
    print("  生成简化可视化 (只显示骨架和关键点)...")
    visualizer_simple = PoseVisualizer(
        show_roi=False,
        show_proposals=False,
        show_detection_boxes=False,
        show_keypoints=True,
        show_skeleton=True
    )
    
    result_simple = visualizer_simple.visualize(
        image=img2,
        all_keypoints=all_keypoints,
        all_detection_boxes=all_detection_boxes,
        region_proposals=region_proposals,
        roi=None
    )
    
    output_simple = 'test_result_simple.jpg'
    cv2.imwrite(output_simple, result_simple)
    print(f"  ✓ 简化结果已保存: {output_simple}")
    
    # ==================== 步骤6: 显示结果 ====================
    print("\n[步骤6] 显示结果窗口...")
    print("  提示: 按任意键关闭窗口")
    
    # 缩放图像以适应屏幕
    scale = 0.5
    result_display = cv2.resize(result_full, None, fx=scale, fy=scale)
    
    cv2.imshow('Pose Detection Test - Full', result_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # ==================== 测试总结 ====================
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    print("\n总结:")
    print(f"  ✓ 成功检测到 {len(all_detection_boxes)} 个人物")
    print(f"  ✓ 成功估计 {len(all_keypoints)} 个姿态")
    print(f"  ✓ 生成了 2 个可视化结果")
    print(f"\n输出文件:")
    print(f"  - {output_full}")
    print(f"  - {output_simple}")
    print("\n" + "=" * 70)
