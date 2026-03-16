import cv2
import numpy as np
from utils.shared_data import SharedData

shared_data = SharedData()

reference_court_points = [
    [72, 126],
    [99, 126],
    [261, 126],
    [288, 126],
    [99, 234],
    [180, 234],
    [261, 234],
    [72, 360],
    [99, 360],
    [180, 360],
    [261, 360],
    [288, 360],
    [99, 486],
    [180, 486],
    [261, 486],
    [72, 594],
    [99, 594],
    [261, 594],
    [288, 594]
]

real_court_points = [
    [526, 273],
    [560, 273],
    [762, 272],
    [796, 272],
    [541, 298],
    [661, 298],
    [781, 297],
    [454, 343],
    [505, 344],
    [661, 343],
    [818, 344],
    [871, 344],
    [440, 428],
    [661, 427],
    [884, 429],
    [208, 586],
    [317, 588],
    [1007, 592],
    [1119, 593]
]

def create_standard_court_reference():
    """创建标准网球场参考图"""
    # 定义图片尺寸
    width = 360
    height = 720
    
    # 创建黑色背景
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 绘制网球场参考图
    cv2.line(image, reference_court_points[0], reference_court_points[3], (255,255,255), 2)
    cv2.line(image, reference_court_points[0], reference_court_points[15], (255,255,255), 2)
    cv2.line(image, reference_court_points[1], reference_court_points[16], (255,255,255), 2)
    cv2.line(image, reference_court_points[2], reference_court_points[17], (255,255,255), 2)
    cv2.line(image, reference_court_points[3], reference_court_points[18], (255,255,255), 2)
    cv2.line(image, reference_court_points[15], reference_court_points[18], (255,255,255), 2)
    cv2.line(image, reference_court_points[7], reference_court_points[11], (255,255,255), 2)
    cv2.line(image, reference_court_points[4], reference_court_points[6], (255,255,255), 2)
    cv2.line(image, reference_court_points[12], reference_court_points[14], (255,255,255), 2)
    cv2.line(image, reference_court_points[5], reference_court_points[13], (255,255,255), 2)
    
    cv2.imshow('standard_court_reference', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存图片
    cv2.imwrite(shared_data.base_dir + '/data/images/standard_court_reference.png', image)
    
def compute_homography():
    """计算实际场地到标准场地的单应性矩阵"""
    reference_points = np.array(reference_court_points, dtype=np.float32)
    real_points = np.array(real_court_points, dtype=np.float32)

    homography_matrix = cv2.findHomography(real_points, reference_points)[0]
    inverse_homography_matrix = cv2.findHomography(reference_points, real_points)[0]

    return homography_matrix, inverse_homography_matrix

def apply_homography(point_input):
    """应用单应性矩阵"""
    return_single = False

    if isinstance(point_input, (list, tuple)):
        if len(point_input) == 2 and all(isinstance(coord, (int, float)) for coord in point_input):
            return_single = True
            points_for_cv = np.array([[point_input]], dtype=np.float32)
            
        elif len(point_input) > 0 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in point_input):
            return_single = False
            try:
                temp_array = np.array(point_input, dtype=np.float32)
                if temp_array.ndim == 2 and temp_array.shape[1] == 2:
                    points_for_cv = temp_array.reshape(-1, 1, 2)
                else:
                    raise ValueError("Each item in the list/tuple must be a 2-element point.")
            except Exception as e:
                raise ValueError(f"Could not convert input list/tuple to points array: {e}")

        else:
            raise ValueError("Input list/tuple must be a single point (2 elements) or a list/tuple of points (each with 2 elements).")

    elif isinstance(point_input, np.ndarray):
        if point_input.ndim == 1 and point_input.shape[0] == 2:
            return_single = True
            points_for_cv = point_input.astype(np.float32).reshape(1, 1, 2)
        elif point_input.ndim == 2 and point_input.shape[1] == 2:
            return_single = False
            points_for_cv = point_input.astype(np.float32).reshape(-1, 1, 2)
        elif point_input.ndim == 3 and point_input.shape[2] == 2 and point_input.shape[1] == 1:
            return_single = False
            points_for_cv = point_input.astype(np.float32)
        else:
            raise ValueError(f"Unsupported NumPy array shape: {point_input.shape}. Expected (2,), (N, 2), or (N, 1, 2).")
    else:
        raise TypeError("Input must be a tuple, list, or NumPy array.")
    
    transformed_point = cv2.perspectiveTransform(points_for_cv, shared_data.homography_matrix)
    if return_single:
        return transformed_point[0][0]
    else:
        return transformed_point.squeeze()
    
def apply_inverse_homography(point_input):
    """应用逆单应性矩阵"""
    return_single = False

    if isinstance(point_input, (list, tuple)):
        if len(point_input) == 2 and all(isinstance(coord, (int, float)) for coord in point_input):
            return_single = True
            points_for_cv = np.array([[point_input]], dtype=np.float32)
            
        elif len(point_input) > 0 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in point_input):
            return_single = False
            try:
                temp_array = np.array(point_input, dtype=np.float32)
                if temp_array.ndim == 2 and temp_array.shape[1] == 2:
                    points_for_cv = temp_array.reshape(-1, 1, 2)
                else:
                    raise ValueError("Each item in the list/tuple must be a 2-element point.")
            except Exception as e:
                raise ValueError(f"Could not convert input list/tuple to points array: {e}")

        else:
            raise ValueError("Input list/tuple must be a single point (2 elements) or a list/tuple of points (each with 2 elements).")

    elif isinstance(point_input, np.ndarray):
        if point_input.ndim == 1 and point_input.shape[0] == 2:
            return_single = True
            points_for_cv = point_input.astype(np.float32).reshape(1, 1, 2)
        elif point_input.ndim == 2 and point_input.shape[1] == 2:
            return_single = False
            points_for_cv = point_input.astype(np.float32).reshape(-1, 1, 2)
        elif point_input.ndim == 3 and point_input.shape[2] == 2 and point_input.shape[1] == 1:
            return_single = False
            points_for_cv = point_input.astype(np.float32)
        else:
            raise ValueError(f"Unsupported NumPy array shape: {point_input.shape}. Expected (2,), (N, 2), or (N, 1, 2).")
    else:
        raise TypeError("Input must be a tuple, list, or NumPy array.")
    
    transformed_point = cv2.perspectiveTransform(points_for_cv, shared_data.inverse_homography_matrix)
    if return_single:
        return transformed_point[0][0]
    else:
        return transformed_point.squeeze()
    
def create_court_mask(frame, opacity=1):
    """创建球场掩码"""
    copied_frame = frame.copy()
    h, w = shared_data.frame_height, shared_data.frame_width
    
    left_top = (0, 0)
    right_top = (360, 0)
    left_bottom = (0, 720)
    right_bottom = (360, 720)
    
    src_points = np.array([[left_top], [right_top], [left_bottom], [right_bottom]], dtype=np.float32)
    transformed_points = apply_inverse_homography(src_points)
    
    left_top_proj = (int(transformed_points[0][0]), int(transformed_points[0][1]))
    right_top_proj = (int(transformed_points[1][0]), int(transformed_points[1][1]))
    left_bottom_proj = (int(transformed_points[2][0]), int(transformed_points[2][1]))
    right_bottom_proj = (int(transformed_points[3][0]), int(transformed_points[3][1]))
    
    left_top_boundary = (left_top_proj[0], 0)
    right_top_boundary = (right_top_proj[0], 0)
    
    polygon = np.array([left_top_boundary, left_top_proj, left_bottom_proj, right_bottom_proj, right_top_proj, right_top_boundary])
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    
    black_layer = np.zeros_like(copied_frame)
    
    darkened_frame = cv2.addWeighted(copied_frame, 1-opacity, black_layer, opacity, 0)
    masked_frame = np.where(mask[:, :, np.newaxis] == 255, copied_frame, darkened_frame)
    
    return masked_frame