import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import websocket
import requests

from model_definitions.wasb import HRNet
from others.person_detector import PersonDetector
from others.traj2 import RealTimeSegmenter, shot_and_bounce
from others.ROI import *
# from others.tennis_court_minimap import TennisCourtCompact


def get_current_path():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    destination_dir = os.path.dirname(current_dir)
    return destination_dir

current_path = get_current_path()

def preprocess_frame(frame, transform):
    return transform(frame)

def predict_ball_position(prev_positions, width, height):
    if len(prev_positions) < 3:
        return None
    p_t = prev_positions[-1]
    a_t = p_t - 2 * prev_positions[-2] + prev_positions[-3]
    v_t = p_t - prev_positions[-2] + a_t
    predicted_position = p_t + v_t + 0.5 * a_t
    predicted_position = np.clip(predicted_position, [0, 0], [width, height])
    return predicted_position

def run_inference(input_path, overlay=False):
    # others
    ws = websocket.WebSocket()
    ws.connect('wss://jason-shawjaine.top:8086/general')
    person_dector = PersonDetector()

    config = {
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
        "model_path": f"{current_path}/model_weights/wasb_tennis_best.pth.tar",  # Update with your model path
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = HRNet(cfg=config).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_wasb.mp4")
    output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_wasb.csv")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    coordinates = []
    frame_number = 0
    frames_buffer = []
    prev_positions = []

    # envent proecessor
    real_time_processor = RealTimeSegmenter(shot_and_bounce)

    top_bounces = []
    top_shots = []

    bottom_bounces = []
    bottom_shots = []
    stop_event_bounces = []
    
    # player detector
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

    visualizer = PoseVisualizer(
        show_roi=False,
        show_proposals=False,
        show_detection_boxes=False,
        show_keypoints=True,
        show_skeleton=True
    )

    roi = np.array([
        [1050, 544],
        [1593, 541],
        [2247, 1191],
        [411, 1178]
    ], dtype=np.int32)

    # tennis court minimap
    H = np.array([
        [-1.81958953e-02, -1.15356310e-04,  2.40952617e+01],
        [ 1.28312178e-04,  7.39963838e-02, -5.10228573e+01],
        [-2.93199985e-05, -3.42417421e-03,  1.00000000e+00]
    ])

    # 初始化网球场mini map
    # mini_court = TennisCourtCompact(
    #     homography_matrix=H,
    #     position='top-right',  # 右上角
    #     map_width=250,         # 宽度250像素
    #     padding=15,            # 距离边缘15像素
    #     alpha=0.75             # 透明度75%
    # )
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) == config['frames_in']:

            # player detection
            all_keypoints, all_detection_boxes, region_proposals = detector.detect(
                img1=frames_buffer[0],
                img2=frames_buffer[1],
                roi=roi if 'roi' in locals() else None,
                max_players=2
            )
    
            # Preprocess the frames
            frames_processed = [preprocess_frame(f, transform) for f in frames_buffer]
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(device)

    
            # Perform inference
            with torch.no_grad():
                outputs = model(input_tensor)[0]  # Get the raw logits
            

            
            detected = False
            center_x, center_y, confidence = 0, 0, 0

            for i in range(config['frames_out']):
                output = outputs[0][i]
                # Post-process the output
                output = torch.sigmoid(output)  # Apply sigmoid to the output to get probabilities
                heatmap = output.squeeze().cpu().numpy()
                heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                heatmap = (heatmap > 0.5).astype(np.float32) * heatmap

                if overlay:
                    heatmap_normalized_visualization = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap_normalized_visualization = heatmap_normalized_visualization.astype(np.uint8)
                    # Apply color map to the heatmap
                    heatmap_colored = cv2.applyColorMap(heatmap_normalized_visualization, cv2.COLORMAP_JET)
                    # Overlay the heatmap on the original frame
                    overlayed_frame = cv2.addWeighted(frames_buffer[i], 0.6, heatmap_colored, 0.4, 0)

                # Find connected components
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((heatmap > 0).astype(np.uint8), connectivity=8)

                # Calculate centers of blobs
                blob_centers = []
                for j in range(1, num_labels):  # Skip the background label 0
                    mask = labels_im == j
                    blob_sum = heatmap[mask].sum()
                    if blob_sum > 0:
                        center_x = np.sum(np.where(mask)[1] * heatmap[mask]) / blob_sum
                        center_y = np.sum(np.where(mask)[0] * heatmap[mask]) / blob_sum
                        blob_centers.append((center_x, center_y, blob_sum))

                if blob_centers:
                    predicted_position = predict_ball_position(prev_positions, width, height)
                    if predicted_position is not None:
                        # Select the blob closest to the predicted position
                        distances = [np.sqrt((x - predicted_position[0]) ** 2 + (y - predicted_position[1]) ** 2) for x, y, _ in blob_centers]
                        closest_blob_idx = np.argmin(distances)
                        center_x, center_y, confidence = blob_centers[closest_blob_idx]
                    else:
                        # Select the blob with the highest confidence if no prediction is available
                        blob_centers.sort(key=lambda x: x[2], reverse=True)
                        center_x, center_y, confidence = blob_centers[0]
                    detected = True
                    prev_positions.append(np.array([center_x, center_y]))
                    if len(prev_positions) > 3:
                        prev_positions.pop(0)

                # Draw a circle on the detected ball
                if detected:
                    if overlay:
                        cv2.circle(overlayed_frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
                    else:
                        cv2.circle(frames_buffer[i], (int(center_x), int(center_y)), 10, (0, 255, 0), 2)

                # player_result = person_dector.track_players(frame, frames_buffer[i])
                # player_bottom_x1, player_bottom_y1, player_bottom_x2, player_bottom_y2, player_top_x1, player_top_y1, player_top_x2, player_top_y2 = player_result[1:]
                # cv2.rectangle(frames_buffer[i], (player_bottom_x1, player_bottom_y1), (player_bottom_x2, player_bottom_y2), (0, 0, 255), 2)
                # cv2.rectangle(frames_buffer[i], (player_top_x1, player_top_y1), (player_top_x2, player_top_y2), (0, 0, 255), 2)
                
                # Write the frame to the output video and save the coordinates
                
                if detected:
                    coordinates.append([frame_number, 1, center_x, center_y, confidence])
                    results = real_time_processor.process_row(frame_number, center_x, center_y)
                else:
                    coordinates.append([frame_number, 0, np.nan, np.nan, np.nan])
                    results = real_time_processor.process_row(frame_number, np.nan, np.nan)

                # If real_time_processor returned results for a completed segment
                if results is not None:
                    # Unpack the results correctly: bounces is a list, shot is a dict, stop is boolean
                    detected_bounces, detected_shot, stop_event_occurred = results

                    # Process detected bounces based on the stop_event_occurred flag
                    if detected_bounces:
                        for bounce_info in detected_bounces:
                            print('detect_bounces: ', bounce_info.get('bounce')) # Use .get() for safety
                            bounce_ptx = bounce_info.get('bounce_x')
                            bounce_pty = bounce_info.get('bounce_y')
                            bounce_type = bounce_info.get('bounce_type', '') # Get bounce type safely

                            if bounce_ptx is not None and bounce_pty is not None: # Ensure coordinates are valid
                                if stop_event_occurred:
                                    # If stop event occurred for this segment, add all bounces to stop_event_bounces
                                    print(f"Detected Stop Event Bounce: Frame {bounce_info.get('bounce')}, Original Type: {bounce_type}")
                                    stop_event_bounces.append(bounce_info) # Store the whole dictionary

                                    # mini_court.add_bounce(bounce_info, 'stop')
                                else:
                                    # If no stop event, categorize as normal top or bottom bounce
                                    if 'top' in bounce_type:
                                        print(f"Detected Top Bounce: Frame {bounce_info.get('bounce')}, Type: {bounce_type}")
                                        top_bounces.append(bounce_info) # Store the whole dictionary

                                        # mini_court.add_bounce(bounce_info, 'top')
                                    elif 'bottom' in bounce_type:
                                        print(f"Detected Bottom Bounce: Frame {bounce_info.get('bounce')}, Type: {bounce_type}")
                                        bottom_bounces.append(bounce_info) # Store the whole dictionary

                                        # mini_court.add_bounce(bounce_info, 'bottom')
                                    else:
                                        print(f"Warning: Detected bounce with unknown type: {bounce_type}")
                            else:
                                print(f"Warning: Bounce detected with invalid coordinates: {bounce_info}")


                    # Append the newly detected shot to the appropriate historical list (Logic remains the same)
                    if detected_shot: # Check if the shot dictionary is not empty
                        shot_type = detected_shot.get('shot_type')
                        print('detect_shots: ', detected_shot.get('shot')) # Access shot frame number
                        if shot_type == 'top':
                            top_shots.append(detected_shot) # Store the whole dictionary
                        elif shot_type == 'bottom':
                            bottom_shots.append(detected_shot) # Store the whole dictionary
                        else:
                            print(f"Warning: Detected shot with unknown type: {shot_type}")

                    # The 'stop_event_occurred' flag indicates if secondary segmentation happened for this segment.
                    # You can use this flag for other logic if needed.
                    # print(f"Stop event detected for this segment: {stop_event_occurred}") # Example usage


                # --- Drawing Section: Draw ALL historical points on the CURRENT frame ---
                # Use the current frame (overlayed_frame or frames_buffer[i]) for drawing

                # Draw all historical top bounces (Orange, Solid)
                for bounce_info in top_bounces:
                    bounce_ptx = bounce_info.get('bounce_x')
                    bounce_pty = bounce_info.get('bounce_y')
                    if bounce_ptx is not None and bounce_pty is not None:
                        bounce_coords = (int(bounce_ptx), int(bounce_pty))
                        color = (0, 128, 0) # Orange
                        thickness = -1 # Solid circle
                        cv2.circle(overlayed_frame if overlay else frames_buffer[i], bounce_coords, radius=5, color=color, thickness=thickness, lineType=cv2.LINE_AA)

                # Draw all historical bottom bounces (Green, Solid)
                for bounce_info in bottom_bounces:
                    bounce_ptx = bounce_info.get('bounce_x')
                    bounce_pty = bounce_info.get('bounce_y')
                    if bounce_ptx is not None and bounce_pty is not None:
                        bounce_coords = (int(bounce_ptx), int(bounce_pty))
                        color = (255, 144, 0) # Green
                        thickness = -1 # Solid circle
                        cv2.circle(overlayed_frame if overlay else frames_buffer[i], bounce_coords, radius=5, color=color, thickness=thickness, lineType=cv2.LINE_AA)

                # Draw all historical stop event bounces (Red, Hollow)
                for bounce_info in stop_event_bounces:
                    bounce_ptx = bounce_info.get('bounce_x')
                    bounce_pty = bounce_info.get('bounce_y')
                    if bounce_ptx is not None and bounce_pty is not None:
                        bounce_coords = (int(bounce_ptx), int(bounce_pty))
                        color = (0, 0, 255) # Red
                        thickness = 2 # Hollow circle
                        cv2.circle(overlayed_frame if overlay else frames_buffer[i], bounce_coords, radius=5, color=color, thickness=thickness, lineType=cv2.LINE_AA)


                # Draw all historical top shots (Orange, Solid)
                for shot_info in top_shots:
                    shot_ptx = shot_info.get('shot_x')
                    shot_pty = shot_info.get('shot_y')
                    if shot_ptx is not None and shot_pty is not None:
                        shot_coords = (int(shot_ptx), int(shot_pty))
                        color = (255, 144, 0) # Orange
                        thickness = 2 # Solid circle
                        # cv2.circle(overlayed_frame if overlay else frames_buffer[i], shot_coords, radius=4, color=color, thickness=thickness, lineType=cv2.LINE_AA)

                # Draw all historical bottom shots (Green, Solid)
                for shot_info in bottom_shots:
                    shot_ptx = shot_info.get('shot_x')
                    shot_pty = shot_info.get('shot_y')
                    if shot_ptx is not None and shot_pty is not None:
                        shot_coords = (int(shot_ptx), int(shot_pty))
                        color = (0, 128, 0) # Green
                        thickness = 2 # Solid circle
                        # cv2.circle(overlayed_frame if overlay else frames_buffer[i], shot_coords, radius=4, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            
                if overlay:
                    cv2.imshow("Frame", overlayed_frame)
                # else:
                    # cv2.imshow("Frame", frames_buffer[0])

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # draw players keypoints and boxes
                if 'all_keypoints' in locals() and all_keypoints:
                    current_frame = overlayed_frame if overlay else frames_buffer[i]
                    current_frame = visualizer.visualize(
                        image=current_frame,
                        all_keypoints=all_keypoints,
                        all_detection_boxes=all_detection_boxes,
                        region_proposals=region_proposals,
                        roi=roi if 'roi' in locals() else None
                    )
                    if overlay:
                        overlayed_frame = current_frame
                    else:
                        frames_buffer[i] = current_frame

                # current_frame = mini_court.draw_on_frame(current_frame)
                out.write(overlayed_frame if overlay else frames_buffer[i])

                frame_number += 1  
            frames_buffer = []  # Clear the buffer for the next set of frames

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save coordinates to CSV file
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y", "confidence (blob size)"]) #"x1_man_up", "y1_man_up", "x2_man_up", "y2_man_up", "x1_man_bottom", "y1_man_bottom", "x2_man_bottom", "y2_man_bottom"])
    coordinates_df.to_csv(output_csv_path, index=False)

# Example usage:
# run_inference(weights='example_weights', input_path='example_video.mp4', overlay=True)