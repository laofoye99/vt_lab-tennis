import cv2
import numpy as np
import time
from utils.ROI import PoseDetector

def benchmark_pose_detection():
    """
    基准测试：只统计处理时间，不保存输出
    """
    video_path = "/home/max/Desktop/tennis/tennis-v5/data/new_test/game13/output.mp4"
    
    detector = PoseDetector(
        model_path="weights/yolo11n-pose.pt",
        motion_threshold=25,
        dbscan_eps=100,
        dbscan_min_samples=20,
        roi_margin=150,
        bbox_padding=(50, 50),
        iou_threshold=0.01,
        keypoint_conf_threshold=0.5
    )
    
    roi = np.array([
        [1050, 544],
        [1593, 541],
        [2247, 1191],
        [411, 1178]
    ], dtype=np.int32)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = None
    processed_frames = 0
    
    print("开始基准测试...")
    time1 = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if prev_frame is None:
            prev_frame = frame.copy()
            continue
        
        # 执行检测（不进行可视化）
        all_keypoints, all_detection_boxes, region_proposals = detector.detect(
            img1=prev_frame,
            img2=frame,
            roi=roi,
            max_players=2
        )
        
        processed_frames += 1
        prev_frame = frame.copy()
        
        # 每处理100帧显示一次进度
        if processed_frames % 100 == 0:
            print(f"已处理 {processed_frames} 帧...")
    
    time2 = time.time()
    cap.release()
    
    total_time = time2 - time1
    
    print(f"\n基准测试结果:")
    print(f"总帧数: {total_frames}")
    print(f"处理帧数: {processed_frames}")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"FPS: {processed_frames / total_time:.2f}")
    print(f"平均每帧时间: {total_time * 1000 / processed_frames:.4f} ms")

if __name__ == "__main__":
    benchmark_pose_detection()