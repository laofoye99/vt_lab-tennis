import numpy as np
import cv2

class CourtGeometry:
    def __init__(self, pts_2d, pts_3d):
        self.pts_2d = np.array(pts_2d, dtype=np.float32)
        self.pts_3d = np.array(pts_3d, dtype=np.float32)
        # 计算单应性矩阵 (2D -> 2D 平面)
        self.H, _ = cv2.findHomography(self.pts_2d, self.pts_3d)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
        else:
            self.H_inv = None

    def pixel_to_real_2d(self, u, v):
        """将像素坐标转换为地面(Z=0)的物理坐标"""
        if self.H is None:
            return 0.0, 0.0
        pt = np.array([u, v, 1.0])
        mapped_pt = np.dot(self.H, pt)
        return mapped_pt[0] / mapped_pt[2], mapped_pt[1] / mapped_pt[2]

    def compensate_perspective_single_view(self, u, v, z_height, cam_height):
        """
        单视角透视补偿: 
        如果物体在高度 Z_height，直接使用单应性矩阵会产生误差（因为 H 假设 Z=0）。
        此方法通过简单的几何相似三角形原理，估算其在 Z=0 平面的真实投影点。
        
        参数:
        u, v: 像素坐标
        z_height: 物体实际高度 (例如网球击球高度 1.0m)
        cam_height: 相机高度
        """
        # 1. 先映射到假定 Z=0 的地面得到 (X0, Y0)
        x0, y0 = self.pixel_to_real_2d(u, v)
        
        # 2. 几何补偿: 
        # 假设相机投影中心在 (0, Y_cam, cam_height) 附近，
        # 物体在空间中高度为 z_height。根据截距定理，我们需要把点往相机方向拉回。
        # 这是一个简化的径向补偿模型。
        
        # 假设相机在 Y 轴的远处 (例如底线后侧 10 米)
        cam_x, cam_y = 0.0, 21.885 
        
        # 相似三角形比例
        scale = (cam_height - z_height) / cam_height
        
        # 补偿后的真实XY坐标
        x_real = cam_x + (x0 - cam_x) * scale
        y_real = cam_y + (y0 - cam_y) * scale
        
        return x_real, y_real
