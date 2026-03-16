import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from streams.video_streamer import VideoStreamer

class FrameProcessor(VideoStreamer):
    def __init__(self):
        super().__init__()
        self.black_layer = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self.load_mask()
        
    def process(self, frame):
        original_frame = frame.copy()
        frame_for_processing = self.apply_court_mask(frame)
        
        return original_frame, frame_for_processing
    
    def calibrate_court(self, frame):
        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.real_court_points.append([int(x), int(y)])
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                print(f"Selected point: {x}, {y}")
        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
        cv2.imshow("calibration", frame)
        
        cv2.setMouseCallback("calibration", _mouse_callback)
        
        while True:
            cv2.imshow("calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        print(self.real_court_points)
        return self.real_court_points
    
    def compute_homography(self):
        reference_points = np.array(self.reference_court_points, dtype=np.float32).reshape(-1, 1, 2)
        real_points = np.array(self.real_court_points, dtype=np.float32).reshape(-1, 1, 2)

        self.homography_matrix = cv2.findHomography(real_points, reference_points, cv2.RANSAC)[0]  # 球场点投影到参考点
        self.inverse_homography_matrix = cv2.findHomography(reference_points, real_points, cv2.RANSAC)[0]  # 参考点投影到球场点

        return self.homography_matrix, self.inverse_homography_matrix
    
    def apply_homography(self, points):
        transformed_points = []
        for point in points:
            homogeneous_point = np.array([point[0], point[1], 1]).reshape(3, 1)
            transformed_homogeneous_point = np.dot(self.homography_matrix, homogeneous_point)
            transformed_point = transformed_homogeneous_point / transformed_homogeneous_point[2]
            transformed_points.append(transformed_point[:2])
        return transformed_points
    
    def apply_inverse_homography(self, points):
        transformed_points = []
        for point in points:
            homogeneous_point = np.array([point[0], point[1], 1]).reshape(3, 1)
            transformed_homogeneous_point = np.dot(self.inverse_homography_matrix, homogeneous_point)
            transformed_point = transformed_homogeneous_point / transformed_homogeneous_point[2]
            transformed_point = transformed_point[:2].squeeze()
            transformed_points.append(transformed_point)
        return transformed_points
    
    def apply_court_mask(self, frame, opacity=1):
        """应用球场掩码"""
        copied_frame = frame.copy()
        masked_frame = copied_frame * self.court_mask
        # darkened_frame = cv2.addWeighted(copied_frame, 1-opacity, self.black_layer, opacity, 0)
        # masked_frame = np.where(self.court_mask[:, :, np.newaxis] == 255, copied_frame, darkened_frame)
        return masked_frame
    
    def compute_court_mask(self, opacity=1):
        """计算球场掩码形状"""
        left_top = (0, 0)
        right_top = (360, 0)
        left_bottom = (0, 720)
        right_bottom = (360, 720)
        
        src_points = np.array([[left_top], [right_top], [left_bottom], [right_bottom]], dtype=np.float32).squeeze()
        transformed_points = self.apply_inverse_homography(src_points)
        
        left_top_proj = (int(transformed_points[0][0]), int(transformed_points[0][1]))
        right_top_proj = (int(transformed_points[1][0]), int(transformed_points[1][1]))
        left_bottom_proj = (int(transformed_points[2][0]), int(transformed_points[2][1]))
        right_bottom_proj = (int(transformed_points[3][0]), int(transformed_points[3][1]))
        
        left_top_boundary = (left_top_proj[0], 0)
        right_top_boundary = (right_top_proj[0], 0)
        
        polygon = np.array([left_top_boundary, left_top_proj, left_bottom_proj, right_bottom_proj, right_top_proj, right_top_boundary])
        
        mask = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], (1, 1, 1))
        
        return mask

    def save_mask(self):
        """保存球场掩码"""
        mask = self.compute_court_mask()
        cv2.imwrite(self.base_dir + "/data/images/test_court_mask.png", mask)
        
    def load_mask(self):
        """加载球场掩码"""
        self.court_mask = cv2.imread(self.base_dir + "/data/images/test_court_mask.png")
                
    def create_standard_court_reference(self):
        """创建标准网球场参考图"""
        # 定义图片尺寸
        width = 360
        height = 720
        
        # 创建黑色背景
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制网球场参考图
        cv2.line(image, self.reference_court_points[0], self.reference_court_points[3], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[0], self.reference_court_points[15], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[1], self.reference_court_points[16], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[2], self.reference_court_points[17], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[3], self.reference_court_points[18], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[15], self.reference_court_points[18], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[7], self.reference_court_points[11], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[4], self.reference_court_points[6], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[12], self.reference_court_points[14], (255,255,255), 2)
        cv2.line(image, self.reference_court_points[5], self.reference_court_points[13], (255,255,255), 2)
        
        cv2.imshow('standard_court_reference', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存图片
        cv2.imwrite(self.base_dir + '/data/images/standard_court_reference.png', image)


if __name__ == "__main__":
    frame_processor = FrameProcessor()
    frame_processor.load_mask()