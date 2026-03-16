import requests
import cv2


class VideoStreamFetcher:
    def __init__(self):
        self.url = "http://jason-shawjaine.top:7001/tennisVideo"
        self.rtsp_url = self.get_rtsp_url()
        print(self.rtsp_url)
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
    def get_rtsp_url(self):
        response = requests.get(self.url)
        data = response.json()
        if data.get('code') == 200:
            inner_data = data.get('data', {})
            return inner_data.get('videoUrl')
        return None

    def capture_and_display_frames(self):
        if not self.rtsp_url:
            return []

        ret, frame = self.cap.read()
        if ret:
            frame_copy = frame.copy()
            return frame_copy
        else:
            return None

# if __name__ == "__main__":
#     streamer = VideoStreamFetcher()
#     while True:
#         frame = streamer.capture_and_display_frames()
#         print("successfully captured frame")
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()