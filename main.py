import argparse
import os
import cv2
from src.config import Config
from src.trackers import BallTracker, PlayerTracker
from src.geometry import CourtGeometry
from src.analytics import MatchAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Tennis Video Analysis System")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--wasb_weights", type=str, default="../wasb-sbdt-inference/model_weights/wasb_tennis_best.pth.tar", help="Path to WASB weights")
    parser.add_argument("--yolo_weights", type=str, default="yolo11n-pose.pt", help="Path to YOLO pose weights")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[INFO] Initializing analysis for: {args.video_path}")
    
    # 1. Initialize Components
    geometry = CourtGeometry(Config.REFERENCE_POINTS_2D, Config.REFERENCE_POINTS_3D)
    
    # In a real run, these trackers will load deep learning models
    ball_tracker = BallTracker(model_path=args.wasb_weights)
    player_tracker = PlayerTracker(model_path=args.yolo_weights)
    analyzer = MatchAnalyzer(geometry, fps=Config.FPS)
    
    # 2. Open Video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video_path}")
        return
        
    Config.FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer.fps = Config.FPS
    
    print("[INFO] Processing frames (Extracting Trajectories)...")
    frame_idx = 0
    
    # Dictionary to store tracking data
    tracking_data = {
        "ball": [],      # List of (x, y) or None
        "players": {}    # Dict mapping player_id to list of poses/bboxes
    }
    
    frame_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames in buffer if video ends and buffer not empty
            if frame_buffer:
                # Padding buffer to 3 frames if necessary (though WASB original drops them, 
                # we'll pad with the last frame to keep length consistent)
                while len(frame_buffer) < 3:
                    frame_buffer.append(frame_buffer[-1])
                ball_results = ball_tracker.infer_batch(frame_buffer)
                tracking_data["ball"].extend(ball_results)
            break
            
        # a. Track Players & Pose (Frame-by-frame)
        players_info = player_tracker.infer_frame(frame)
        tracking_data["players"][frame_idx] = players_info
        
        # b. Buffer frames for WASB Ball Tracking
        frame_buffer.append(frame)
        
        if len(frame_buffer) == 3:
            # Run WASB Batch Inference (Exactly as original)
            ball_results = ball_tracker.infer_batch(frame_buffer)
            tracking_data["ball"].extend(ball_results)
            frame_buffer = []
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")
            
    cap.release()
    
    # Ensure tracking_data["ball"] has the same length as processed frames
    # (In case of slight mismatch during loop)
    tracking_data["ball"] = tracking_data["ball"][:frame_idx]
    print("[INFO] Feature extraction complete. Starting analytics...")
    
    # 3. Analytics Pipeline
    # a. Smooth Ball Trajectory
    smoothed_ball = ball_tracker.smooth_trajectory(tracking_data["ball"])
    
    # b. Detect Hit & Bounce Events
    events = analyzer.detect_events(smoothed_ball, tracking_data["players"])
    print(f"[INFO] Detected {len(events)} events (Hits/Bounces).")
    
    # c. Classify Forehand/Backhand & Calculate Speed for Hits
    results = []
    for event in events:
        if event['type'] == 'hit':
            # Identify stroke type
            stroke_type = analyzer.classify_stroke(event, tracking_data["players"])
            event['stroke'] = stroke_type
            
            speed_kmh = event.get('speed_kmh', 0)
            print(f"  -> {stroke_type} at frame {event['frame']}, Speed: {speed_kmh:.2f} km/h")
        results.append(event)
        
    # d. Calculate Near-Court Player Movement
    movement_distances = analyzer.calculate_player_movement(tracking_data["players"])
    for pid, dist in movement_distances.items():
        print(f"[INFO] Player {pid} total movement: {dist:.2f} meters")

    # 4. Visualization Pass
    print("[INFO] Starting visualization pass...")
    from src.visualizer import Visualizer
    
    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_filename = os.path.join(args.output_dir, f"analyzed_{os.path.basename(args.video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    visualizer = Visualizer(fps, width, height)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw all results on the frame
        frame = visualizer.draw_frame(
            frame, 
            frame_idx, 
            smoothed_ball, 
            tracking_data["players"], 
            results, 
            movement_distances
        )
        
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Visualized {frame_idx} frames...")
            
    cap.release()
    out.release()

    # 5. Save Results
    print(f"[INFO] Analysis complete. Results saved to {args.output_dir}")
    print(f"[INFO] Final video: {output_filename}")

if __name__ == "__main__":
    main()
