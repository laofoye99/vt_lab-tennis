import requests
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shared_data import SharedData

# class VideoStream(SharedData):
#     def __init__(self):
#         super().__init__()
#         self.url = "http://jason-shawjaine.top:7001/tennisVideo"
#         self.rtsp_url = self.get_rtsp_url()
#         self.cap = cv2.VideoCapture(self.rtsp_url)
#         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
#         self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#         self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
#     def get_rtsp_url(self):
#         response = requests.get(self.url)
#         data = response.json()
#         if data.get('code') == 200:
#             inner_data = data.get('data', {})
#             return inner_data.get('videoUrl')
#         return None

#     def get_frame(self):
#         if not self.rtsp_url:
#             return []

#         ret, frame = self.cap.read()
#         if ret:
#             frame_copy = frame.copy()
#             width = frame_copy.shape[1]
#             height = frame_copy.shape[0]
#             print(width, height)
#             print(self.fps, self.width, self.height)
#             return frame_copy
#         else:
#             return None
        
# if __name__ == "__main__":
#     stream = VideoStream()
#     while True:
#         frame = stream.get_frame()
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()
