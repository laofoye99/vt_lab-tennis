import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import threading
import queue
import time
from collections import deque
from models.tennis_tracker.wasb import HRNet
from utils.ROI import PoseDetector, PoseVisualizer

class TennisProcessor:
    def __init__(self, config=None, output_dir=None):
        # 线程控制
        self.processing_thread = None
        self.pose_thread = None
        self.running = False
        self.paused = False
        
        # 队列管理
        self.pending_queue = queue.Queue()  # 待处理帧队列
        self.processing_buffer = deque(maxlen=3)  # 正在处理的帧缓冲区
        self.processed_frames = deque(maxlen=75)  # 已处理帧队列（75帧追溯）
        
        # 输出目录
        self.output_dir = output_dir or "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 事件检测
        self.shot_event_detected = False
        self.shot_frame_info = None
        self.detection_results = []  # 存储检测结果
        
        # 模型配置
        self.config = config or self._get_default_config()
        self.device = self._setup_device()
        
        # 初始化模型
        self.ball_detector = None
        self.pose_detector = None
        self.pose_visualizer = None
        
        self._initialize_models()
        
    def _get_default_config(self):
        """获取默认配置 - 修复版本"""
        return {
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
                "STEM": {
                    "INPLANES": 64,
                    "STRIDES": [1, 1]
                },
                "STAGE1": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 1,
                    "BLOCK": 'BOTTLENECK',
                    "NUM_BLOCKS": [1],
                    "NUM_CHANNELS": [32],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2],
                    "NUM_CHANNELS": [16, 32],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE3": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 3,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE4": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 4,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64, 128],
                    "FUSE_METHOD": 'SUM'
                },
                "DECONV": {
                    "NUM_DECONVS": 0,
                    "KERNEL_SIZE": [],
                    "NUM_BASIC_BLOCKS": 2
                }
            },
            "INIT_WEIGHTS": True
        },
        "model_path": "weights/wasb_tennis_best.pth.tar",  # Update with your model path
    }
    
    def _setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _initialize_models(self):
        """初始化所有模型"""
        print("Initializing models...")
        
        # 初始化网球检测模型
        try:
            self.ball_detector = HRNet(cfg=self.config).to(self.device)
            checkpoint = torch.load(self.config['model_path'], map_location=self.device)
            self.ball_detector.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.ball_detector.eval()
            print("Ball detection model loaded successfully")
        except Exception as e:
            print(f"Error loading ball detection model: {e}")
            # 如果模型加载失败，创建一个虚拟的检测器用于测试
            self.ball_detector = None
        
        # 初始化姿势检测模型
        try:
            self.pose_detector = PoseDetector()
            self.pose_visualizer = PoseVisualizer(
                show_roi=True,
                show_proposals=True,
                show_detection_boxes=True,
                show_keypoints=True,
                show_skeleton=True
            )
            print("Pose detection model initialized")
        except Exception as e:
            print(f"Error initializing pose detection model: {e}")
            self.pose_detector = None
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['inp_height'], self.config['inp_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def add_frame(self, frame_id, frame):
        """添加帧到待处理队列"""
        if self.running and not self.paused:
            self.pending_queue.put((frame_id, frame))
    
    def start_processing(self):
        """启动处理线程"""
        if self.running:
            print("Processing is already running")
            return
        
        self.running = True
        self.paused = False
        
        # 启动主处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Tennis processing started")
    
    def pause_processing(self):
        """暂停处理线程"""
        self.paused = True
        print("Processing paused")
    
    def resume_processing(self):
        """恢复处理线程"""
        if self.running and self.paused:
            self.paused = False
            print("Processing resumed")
    
    def stop_processing(self):
        """停止所有处理"""
        self.running = False
        self.paused = False
        
        # 清空队列
        while not self.pending_queue.empty():
            try:
                self.pending_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # 保存检测结果
        self._save_detection_results()
        
        print("Tennis processing stopped")
    
    def _processing_loop(self):
        """主处理循环"""
        while self.running:
            try:
                # 如果暂停，等待一段时间后继续检查
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 收集3帧进行处理
                if self.pending_queue.qsize() >= 3:
                    frames_batch = []
                    for _ in range(3):
                        if not self.pending_queue.empty():
                            frame_id, frame = self.pending_queue.get(timeout=1.0)
                            frames_batch.append((frame_id, frame))
                    
                    if len(frames_batch) == 3:
                        self._process_frame_batch(frames_batch)
                
                # 控制处理速度
                time.sleep(0.01)  # 10ms间隔
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_frame_batch(self, frames_batch):
        """处理帧批次"""
        try:
            frame_ids = [frame_id for frame_id, _ in frames_batch]
            frames = [frame for _, frame in frames_batch]
            
            # 记录检测结果
            batch_results = {
                'frame_ids': frame_ids,
                'ball_detections': [],
                'shot_detected': False
            }
            
            # 如果网球检测器不可用，跳过检测逻辑
            if self.ball_detector is None:
                # 模拟处理，直接将帧添加到已处理队列
                for frame_id, frame in frames_batch:
                    self.processed_frames.append({
                        'frame_id': frame_id,
                        'frame': frame,
                        'processed': True,
                        'ball_detected': False,
                        'ball_position': None
                    })
                    batch_results['ball_detections'].append({
                        'detected': False,
                        'position': None
                    })
                return
            
            # 预处理帧
            frames_processed = [self._preprocess_frame(frame) for frame in frames]
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(self.device)
            
            # 进行网球检测推理
            with torch.no_grad():
                outputs = self.ball_detector(input_tensor)[0]
            
            # 处理每一帧的检测结果
            for i in range(len(frames)):
                output = outputs[0][i]
                output = torch.sigmoid(output)
                heatmap = output.squeeze().cpu().numpy()
                
                # 检测网球位置
                ball_detected, ball_position = self._detect_ball_position(heatmap, frames[i].shape)
                
                # 记录检测结果
                batch_results['ball_detections'].append({
                    'detected': ball_detected,
                    'position': ball_position
                })
                
                # 在帧上绘制检测结果（可选）
                if ball_detected:
                    annotated_frame = self._draw_detection_result(frames[i], ball_position)
                else:
                    annotated_frame = frames[i]
                
                # 将处理后的帧添加到已处理队列
                self.processed_frames.append({
                    'frame_id': frame_ids[i],
                    'frame': annotated_frame,
                    'processed': True,
                    'ball_detected': ball_detected,
                    'ball_position': ball_position
                })
            
            # 检测击球事件
            shot_detected = self._detect_shot_event(batch_results, frame_ids)
            
            if shot_detected:
                print(f"Shot event detected at frame {self.shot_frame_info['frame_id']}")
                batch_results['shot_detected'] = True
                self._handle_shot_event()
            
            # 保存批次结果
            self.detection_results.append(batch_results)
                
        except Exception as e:
            print(f"Error processing frame batch: {e}")
    
    def _preprocess_frame(self, frame):
        """预处理单帧"""
        return self.transform(frame)
    
    def _detect_ball_position(self, heatmap, frame_shape):
        """检测网球位置"""
        try:
            heatmap = cv2.resize(heatmap, (frame_shape[1], frame_shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
            heatmap = (heatmap > 0.5).astype(np.float32) * heatmap
            
            # 寻找连通组件
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                (heatmap > 0).astype(np.uint8), connectivity=8)
            
            if num_labels > 1:
                # 找到最大的连通区域
                largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                center_x, center_y = centroids[largest_component]
                
                return True, (center_x, center_y)
            
        except Exception as e:
            print(f"Error detecting ball position: {e}")
        
        return False, None
    
    def _draw_detection_result(self, frame, ball_position):
        """在帧上绘制检测结果"""
        annotated_frame = frame.copy()
        center_x, center_y = ball_position
        
        # 绘制网球位置
        cv2.circle(annotated_frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Tennis Ball", 
                   (int(center_x) + 15, int(center_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _detect_shot_event(self, batch_results, frame_ids):
        """检测击球事件"""
        # 这里需要实现复杂的击球检测逻辑
        # 包括速度分析、轨迹分析、位置分析等
        
        # 简化版：基于连续帧中网球位置的剧烈变化来检测击球
        for i, detection in enumerate(batch_results['ball_detections']):
            if detection['detected']:
                frame_id = frame_ids[i]
                
                # 检查是否满足击球条件（简化逻辑）
                if self._analyze_shot_characteristics(detection['position'], frame_id):
                    self.shot_frame_info = {
                        'frame_id': frame_id,
                        'ball_position': detection['position']
                    }
                    return True
        
        return False
    
    def _analyze_shot_characteristics(self, ball_position, frame_id):
        """分析击球特征"""
        # 这里需要实现复杂的击球检测逻辑
        # 包括速度分析、轨迹分析、位置分析等
        
        # 简化版：随机检测击球事件用于演示
        # 在实际应用中应该替换为真实的击球检测算法
        import random
        return random.random() < 0.005  # 0.5%的概率检测到击球
    
    def _handle_shot_event(self):
        """处理击球事件"""
        if not self.shot_frame_info:
            return
        
        # 暂停主处理线程
        self.pause_processing()
        
        # 向前追溯75帧，找到击球帧及其前后帧
        shot_frames = self._get_shot_context_frames()
        
        if shot_frames and self.pose_detector:
            # 启动姿势检测线程
            self.pose_thread = threading.Thread(
                target=self._pose_detection_worker,
                args=(shot_frames,)
            )
            self.pose_thread.daemon = True
            self.pose_thread.start()
        else:
            # 如果没有姿势检测器，直接恢复处理
            self.resume_processing()
    
    def _get_shot_context_frames(self):
        """获取击球上下文帧（击球帧及其前后帧）"""
        shot_frame_id = self.shot_frame_info['frame_id']
        context_frames = []
        
        # 在已处理队列中寻找击球帧及其前后帧
        processed_list = list(self.processed_frames)
        
        for i, frame_info in enumerate(processed_list):
            if frame_info['frame_id'] == shot_frame_id:
                # 获取前一帧、当前帧和后一帧（共三帧）
                start_idx = max(0, i - 1)
                end_idx = min(len(processed_list), i + 2)  # i+1 +1 为了包含后一帧
                
                context_frames = processed_list[start_idx:end_idx]
                break
        
        # 如果没找到足够的帧，使用当前击球帧
        if not context_frames:
            # 从待处理队列中获取当前帧
            context_frames = [{
                'frame_id': shot_frame_id,
                'frame': None,  # 这里需要实际获取帧数据
                'processed': True
            }]
        
        return context_frames
    
    def _pose_detection_worker(self, shot_frames):
        """姿势检测工作线程"""
        try:
            print("Starting pose detection for shot event...")
            
            if not self.pose_detector:
                print("Pose detector not initialized")
                return
            
            # 处理击球相关的帧进行姿势检测
            pose_results = []
            
            for i in range(len(shot_frames) - 1):
                if i + 1 >= len(shot_frames):
                    break
                
                img1 = shot_frames[i]['frame']
                img2 = shot_frames[i + 1]['frame']
                
                # 确保帧数据有效
                if img1 is None or img2 is None:
                    continue
                
                # 进行姿势检测
                all_keypoints, all_detection_boxes, region_proposals = self.pose_detector.detect(
                    img1=img1,
                    img2=img2,
                    roi=None,  # 可以设置ROI区域
                    max_players=2
                )
                
                # 可视化结果
                if self.pose_visualizer and len(all_keypoints) > 0:
                    result_frame = self.pose_visualizer.visualize(
                        image=img2,
                        all_keypoints=all_keypoints,
                        all_detection_boxes=all_detection_boxes,
                        region_proposals=region_proposals,
                        roi=None
                    )
                    
                    # 保存姿势检测结果
                    self._save_pose_result(result_frame, shot_frames[i + 1]['frame_id'])
                
                pose_results.append({
                    'frame_id': shot_frames[i + 1]['frame_id'],
                    'keypoints': all_keypoints,
                    'detection_boxes': all_detection_boxes,
                    'region_proposals': region_proposals
                })
            
            print(f"Pose detection completed for {len(pose_results)} frames")
            
        except Exception as e:
            print(f"Error in pose detection worker: {e}")
        finally:
            # 姿势检测完成后，恢复主处理线程
            self.resume_processing()
    
    def _save_pose_result(self, result_frame, frame_id):
        """保存姿势检测结果"""
        try:
            pose_output_dir = os.path.join(self.output_dir, "pose_detection")
            os.makedirs(pose_output_dir, exist_ok=True)
            
            output_path = os.path.join(pose_output_dir, f"pose_frame_{frame_id}.jpg")
            cv2.imwrite(output_path, result_frame)
            print(f"Pose result saved: {output_path}")
            
        except Exception as e:
            print(f"Error saving pose result: {e}")
    
    def _save_detection_results(self):
        """保存检测结果到CSV文件"""
        try:
            results_csv_path = os.path.join(self.output_dir, "detection_results.csv")
            
            # 整理结果数据
            rows = []
            for batch in self.detection_results:
                for i, detection in enumerate(batch['ball_detections']):
                    row = {
                        'frame_id': batch['frame_ids'][i],
                        'ball_detected': detection['detected'],
                        'ball_x': detection['position'][0] if detection['position'] else None,
                        'ball_y': detection['position'][1] if detection['position'] else None,
                        'shot_detected': batch['shot_detected']
                    }
                    rows.append(row)
            
            # 创建DataFrame并保存
            df = pd.DataFrame(rows)
            df.to_csv(results_csv_path, index=False)
            print(f"Detection results saved to: {results_csv_path}")
            
        except Exception as e:
            print(f"Error saving detection results: {e}")
    
    def get_status(self):
        """获取处理状态"""
        return {
            'running': self.running,
            'paused': self.paused,
            'pending_frames': self.pending_queue.qsize(),
            'processed_frames': len(self.processed_frames),
            'shot_event_detected': self.shot_event_detected,
            'ball_detector_available': self.ball_detector is not None,
            'pose_detector_available': self.pose_detector is not None
        }