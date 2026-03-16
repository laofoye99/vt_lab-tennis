from models.tennis_tracker.inference import TennisTracker
# from test.inference_time_delay_profiling import TennisTracker as TennisTracker_time_delay_profiling
from utils.shared_data import SharedData
from streams.video_streamer import VideoStreamer
from streams.frame_process import FrameProcessor
from models.player_tracker.player_tracker import PlayerTracker
from models.player_tracker.player_tracker1 import PlayerTracker as PlayerTracker1
import os
import cv2
import matplotlib.pyplot as plt
import time

def main():
    video_stream = VideoStreamer()
    # video_stream = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')
    frame_processor = FrameProcessor()
    tennis_tracker = TennisTracker()
    # tennis_tracker = TennisTracker_time_delay_profiling()
    player_tracker = PlayerTracker()
    frames_delay_buffer = []
    process_delay_buffer = []
    i = 0
    start_time = time.time()
    while i < 1000:
    # while True:
        frame = video_stream.get_frame()
        # video_stream.frame_number += 1
        # frame_for_processing = frame.copy()
        # ret, frame = video_stream.read()
        original_frame, frame_for_processing = frame_processor.process(frame)
        # cv2.imshow('frame', frame_for_processing)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
        frames_delay_buffer.append(original_frame)
        process_delay_buffer.append(frame_for_processing)
        
        if len(frames_delay_buffer) == 3:
            player_result = player_tracker.process_frame(process_delay_buffer)
            coordinates = tennis_tracker.process(process_delay_buffer)
            for result in player_result:
                print(result)

            for coordinate in coordinates:
                print(coordinate)
            # print(coordinates)
            # inncer_counter = 0
            # for coordinate in coordinates:
                # print(coordinate)
                # x = coordinate[3]
                # y = coordinate[4]
        #         # cv2.circle(frames_delay_buffer[inncer_counter], (int(x), int(y)), 3, (0, 0, 255), 2)
        # #         # cv2.imshow('frame', frames_delay_buffer[inncer_counter])
                # inncer_counter += 1
            frames_delay_buffer = []
            process_delay_buffer = []
        
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        i += 1
    # tennis_tracker.print_timing_summary()
    end_time = time.time()
    print(f"Time taken average: {(end_time - start_time)} ms")
    # cv2.destroyAllWindows()
    #     for t in player_tracker.track_history.values():
    #         if len(t['positions']) > 0:
    #             x1, y1, x2, y2 = t['positions'][-1].astype(int)
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     # cv2.imshow('masked_frame', masked_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
