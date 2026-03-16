import cv2
import time
import os
from models.tennis_tracker.tennisProcessor import TennisProcessor

def main():
    # 输入和输出路径
    input_video_path = "/home/max/Desktop/tennis/tennis-v5/data/new_test/game13/2560_1440/Clip3/output.mp4"
    output_dir = "/home/max/Desktop/tennis/tennis-v5/data/new_test/output/"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化处理器
    processor = TennisProcessor()
    
    # 启动处理

    processor.start_processing()
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # 创建视频写入器
    output_video_path = os.path.join(output_dir, "processed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 创建姿势检测结果目录
    pose_output_dir = os.path.join(output_dir, "pose_detection")
    os.makedirs(pose_output_dir, exist_ok=True)
    
    frame_id = 0
    processed_frames_count = 0
    
    strat_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 添加帧到处理器
            processor.add_frame(frame_id, frame.copy())
            frame_id += 1
            
            # 获取处理状态
            status = processor.get_status()
            
            # 显示状态信息
            status_text = f"Frame {frame_id}/{total_frames} | " \
                         f"Pending: {status['pending_frames']} | " \
                         f"Processed: {status['processed_frames']} | " \
                         f"Paused: {status['paused']}"
            
            # 在帧上绘制状态信息
            display_frame = frame.copy()
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示处理中的帧
            cv2.namedWindow('Tennis Processing', cv2.WINDOW_NORMAL)
            cv2.imshow('Tennis Processing', display_frame)
            
            # 将原始帧写入输出视频（您可以根据需要修改为写入处理后的帧）
            out.write(frame)
            processed_frames_count += 1
            
            # 控制显示帧率
            wait_time = max(1, int(1000 / fps) - 10)  # 稍微快于原速
            key = cv2.waitKey(wait_time) & 0xFF
            
            # 按q退出
            if key == ord('q'):
                break
            # 按p暂停/继续处理
            elif key == ord('p'):
                if status['paused']:
                    processor.resume_processing()
                else:
                    processor.pause_processing()
            
            # 每100帧打印一次状态
            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{total_frames} frames "
                      f"({frame_id/total_frames*100:.1f}%)")
                

        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # 清理资源
        print("Cleaning up resources...")
        processor.stop_processing()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Processing completed: {processed_frames_count} frames processed")
        print(f"Output video saved to: {output_video_path}")
        print(f"Pose detection results saved to: {pose_output_dir}")

        end_time = time.time()
        elapsed_time = end_time - strat_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()