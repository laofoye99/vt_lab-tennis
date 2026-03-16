import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from model_definitions.wasb import HRNet

class BallDetector:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = None
        self.transform = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化网球检测模型"""
        self.model = HRNet(cfg=self.config).to(self.device)
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['inp_height'], self.config['inp_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def detect_balls(self, frames):
        """检测多帧中的网球"""
        if len(frames) != self.config['frames_in']:
            raise ValueError(f"Expected {self.config['frames_in']} frames, got {len(frames)}")
        
        # 预处理帧
        frames_processed = [self._preprocess_frame(frame) for frame in frames]
        input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)[0]
        
        # 后处理
        detections = []
        for i in range(self.config['frames_out']):
            detection = self._postprocess_output(outputs[0][i], frames[i].shape)
            detections.append(detection)
        
        return detections
    
    def _preprocess_frame(self, frame):
        """预处理单帧"""
        return self.transform(frame)
    
    def _postprocess_output(self, output, frame_shape):
        """后处理输出"""
        output = torch.sigmoid(output)
        heatmap = output.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (frame_shape[1], frame_shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
        
        # 检测网球位置
        ball_detected, position, confidence = self._find_ball_position(heatmap)
        
        return {
            'ball_detected': ball_detected,
            'position': position,
            'confidence': confidence,
            'heatmap': heatmap
        }
    
    def _find_ball_position(self, heatmap):
        """在热图中寻找网球位置"""
        # 二值化
        binary_heatmap = (heatmap > 0.5).astype(np.uint8)
        
        # 寻找连通组件
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            binary_heatmap, connectivity=8)
        
        if num_labels > 1:
            # 找到最大的连通区域
            largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            center_x, center_y = centroids[largest_component]
            confidence = stats[largest_component, cv2.CC_STAT_AREA]
            
            return True, (center_x, center_y), confidence
        
        return False, None, 0.0