import cv2
import numpy as np
import torch
from models.tennis_tracker.wasb import HRNet
from torchvision import transforms

class SharedData:
    def __init__(self):
        self.base_url = "http://jason-shawjaine.top:7001/tennisVideo"
        self.base_dir = '/home/max/Desktop/tennis/tennis-v5'
        self.frame_number = 0
        self.frame = None
        self.model_path = self.base_dir + "/weights/wasb_tennis_best.pth.tar"
        self.yolo_path = self.base_dir + "/weights/yolo11l.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_video_dir = self.base_dir + "/data/OutputVideos"
        self.input_video_dir = self.base_dir + "/data/InputVideos"
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.minimap_frame = cv2.imread(self.base_dir + "/data/images/standard_court_reference.png")
        # self.frame_width = 2560
        # self.frame_height = 1440
        self.frame_width = 1280
        self.frame_height = 720
        # self.y_net = 670
        self.y_net = 344
        self.frame_number = 0
        self.fps = 15
        # self.y_upper = 360
        self.y_upper = 273
        self.y_lower = 593
        # self.y_lower = 486
        # self.x_left = 86
        self.x_left = 208
        # self.x_right = 274
        self.x_right = 1119
        self.court_width_std = 10.97
        self.court_height_std = 23.77
        self.court_width_pixel = 216
        self.court_height_pixel = 468
        self.coordinates = []
        
        
        self.config = {
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
                        "BLOCK": "BOTTLENECK",
                        "NUM_BLOCKS": [1],
                        "NUM_CHANNELS": [32],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE2": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 2,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [2, 2],
                        "NUM_CHANNELS": [16, 32],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE3": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 3,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [2, 2, 2],
                        "NUM_CHANNELS": [16, 32, 64],
                        "FUSE_METHOD": "SUM",
                    },
                    "STAGE4": {
                        "NUM_MODULES": 1,
                        "NUM_BRANCHES": 4,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS": [2, 2, 2, 2],
                        "NUM_CHANNELS": [16, 32, 64, 128],
                        "FUSE_METHOD": "SUM",
                    },
                    "DECONV": {
                        "NUM_DECONVS": 0,
                        "KERNEL_SIZE": [],
                        "NUM_BASIC_BLOCKS":2
                    }
                },
                "INIT_WEIGHTS": True,
            },
            "model_path": self.model_path
        }
                
        self.real_court_points = [
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
        self.reference_court_points = [
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
        
        self.homography_matrix = np.array(
            [[-7.14286298e-01, -1.21821324e+00,  6.46583061e+02],
             [-2.32709946e-02, -5.29318537e+00,  1.34724446e+03],
             [-5.79720388e-05, -6.73708271e-03,  1.00000000e+00]]
        )

        self.inverse_homography_matrix = np.array(
            [[ 1.03099305e+00, -8.35256064e-01,  4.73242817e+02],
             [-5.41340407e-03, -1.75855404e-01,  2.50701552e+02],
             [-2.03330127e-05, -1.26229572e-03,  1.00000000e+00]]
        )
    def _update_config(self, frame=None):
        if frame is not None:
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]
            self.y_net = self.frame_height // 2

