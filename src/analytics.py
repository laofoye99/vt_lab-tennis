import numpy as np
from .config import Config

class MatchAnalyzer:
    def __init__(self, geometry, fps):
        self.geometry = geometry
        self.fps = fps
        self.bounce_history = [] # 记录最近的触地帧，用于判断截击

    def detect_events(self, ball_trajectory, players_data):
        """
        分层检测引擎：
        Layer 1: 物理运动学初筛 (Kinematic Screening)
        Layer 2: 姿态语义验证 (Pose Semantic Validation)
        """
        # 1. 物理初筛提取候选池
        candidates = self._kinematic_screening(ball_trajectory)
        
        # 2. 姿态精验并细化分类
        final_events = self._pose_validation(candidates, ball_trajectory, players_data)
        
        return final_events

    def _kinematic_screening(self, ball_trajectory):
        """
        Layer 1: 基于2D轨迹特征提取候选帧。
        """
        candidates = {'hit': [], 'bounce': [], 'toss': []}
        if len(ball_trajectory) < 10: return candidates

        coords = np.array(ball_trajectory)
        # 计算速度矢量
        v = coords[1:] - coords[:-1]
        
        for i in range(5, len(ball_trajectory) - 10):
            u, v_px = ball_trajectory[i]
            if u == 0 and v_px == 0: continue

            # --- 1. Toss (发球抛球起始) 候选 ---
            # 特征：低水平位移 + Y轴先下沉再急速上升
            if v_px > 400:
                vx_avg = np.mean(np.abs(v[i-3:i+3, 0]))
                vy_prep = ball_trajectory[i][1] - ball_trajectory[i-3][1] # 下沉
                vy_toss = ball_trajectory[i+3][1] - ball_trajectory[i][1] # 上升
                if vx_avg < 5 and vy_prep > 2 and vy_toss < -15:
                    candidates['toss'].append(i)

            # --- 2. Hit (击球) 候选 ---
            # 特征：水平速度 X 剧烈反转 或 矢量方向巨大改变
            vx_in = ball_trajectory[i][0] - ball_trajectory[i-3][0]
            vx_out = ball_trajectory[i+3][0] - ball_trajectory[i][0]
            if vx_in * vx_out < -15: # X方向反转 (被打回去了)
                candidates['hit'].append(i)
            else:
                # 检查矢量夹角突变 (用于发球或切球)
                v_in = coords[i] - coords[i-3]
                v_out = coords[i+3] - coords[i]
                cos_theta = np.dot(v_in, v_out) / (np.linalg.norm(v_in)*np.linalg.norm(v_out) + 1e-6)
                if cos_theta < 0: # 夹角 > 90度
                    candidates['hit'].append(i)

            # --- 3. Bounce (触地) 候选 ---
            # 特征：Y轴局部极大值 (落地点)
            # 增加灵敏度：检查更小的反转，且确保这是局部极值
            v_y_in_b = ball_trajectory[i][1] - ball_trajectory[i-1][1]
            v_y_out_b = ball_trajectory[i+1][1] - ball_trajectory[i][1]
            if v_y_in_b > 0.5 and v_y_out_b < -0.5: # 只要有反转趋势
                # 进一步验证周围 3 帧确保是极大值
                if ball_trajectory[i][1] >= ball_trajectory[i-1][1] and ball_trajectory[i][1] >= ball_trajectory[i+1][1]:
                    # 计算此点的 X 轴稳定性
                    vx_in_b = ball_trajectory[i][0] - ball_trajectory[i-2][0]
                    vx_out_b = ball_trajectory[i+2][0] - ball_trajectory[i][0]
                    # Bounce 时 X 轴方向不应反转 (vx_in * vx_out > 0)
                    if vx_in_b * vx_out_b >= 0:
                        candidates['bounce'].append(i)

        return candidates

    def _pose_validation(self, candidates, ball_trajectory, players_data):
        """
        Layer 2: 结合人体姿态和空间规则进行验证。
        """
        validated = []
        cooldown = {} # 帧冷却
        self.bounce_history = []

        # 1. 验证 Bounce (必须在地面)
        for idx in candidates['bounce']:
            if idx in cooldown: continue
            if idx not in players_data: continue
            
            ball_y = ball_trajectory[idx][1]
            # 找到全场最低的脚踝高度
            floor_y = 0
            for pid, p in players_data[idx].items():
                floor_y = max(floor_y, p['keypoints']['l_ankle'][1], p['keypoints']['r_ankle'][1])
            
            # 只有当球在脚踝附近或下方时才算触地 (图像中Y更大)
            if ball_y > floor_y - 30:
                validated.append({"frame": idx, "type": "bounce", "pixel_pos": ball_trajectory[idx]})
                self.bounce_history.append(idx)
                for f in range(idx-10, idx+10): cooldown[f] = True

        # 2. 验证 Serve (抛球手-头部空间链)
        for t0 in candidates['toss']:
            if t0 in cooldown: continue
            if t0 not in players_data: continue
            
            for pid, p in players_data[t0].items():
                kps = p['keypoints']
                # T0: 球在手，手在肩下
                for w_key in ['l_wrist', 'r_wrist']:
                    dist = np.linalg.norm(np.array(ball_trajectory[t0]) - np.array(kps[w_key]))
                    shoulder_y = kps['l_shoulder'][1] if 'l' in w_key else kps['r_shoulder'][1]
                    
                    if dist < 50 and kps[w_key][1] > shoulder_y:
                        # 检查 T0 + 10: 手举起，球在头顶
                        t10 = t0 + 10
                        if t10 in players_data and players_data[t10].get(pid):
                            kp10 = players_data[t10][pid]['keypoints']
                            if kp10[w_key][1] < kp10['l_shoulder'][1] and ball_trajectory[t10][1] < kp10['nose'][1]:
                                # 找到抛球后的爆发点 (Hit)
                                for f in range(t0 + 10, t0 + 40):
                                    if f >= len(ball_trajectory)-1: break
                                    v_accel = np.linalg.norm(np.array(ball_trajectory[f+1]) - np.array(ball_trajectory[f]))
                                    if v_accel > 15: # 击球爆发
                                        validated.append({
                                            "frame": f, "type": "hit", "is_serve": True, "stroke": "SERVE",
                                            "pixel_pos": ball_trajectory[f], "player_id": pid
                                        })
                                        for c in range(t0-5, f+30): cooldown[c] = True
                                        break
                    if t0 in cooldown: break
                if t0 in cooldown: break

        # 3. 验证 Hits (正反手/截击)
        for idx in candidates['hit']:
            if idx in cooldown: continue
            if idx not in players_data: continue
            
            for pid, p in players_data[idx].items():
                kps = p['keypoints']
                # 击球点必须在手腕附近
                dist_l = np.linalg.norm(np.array(ball_trajectory[idx]) - np.array(kps['l_wrist']))
                dist_r = np.linalg.norm(np.array(ball_trajectory[idx]) - np.array(kps['r_wrist']))
                
                if min(dist_l, dist_r) < 100:
                    # 区分 Volley (截击)
                    # 规则：脚在网前 或 最近1.5秒没弹地
                    is_volley = False
                    if kps['l_ankle'][1] < 700: # 站位靠前
                        is_volley = True
                    if not any(idx - b < 45 for b in self.bounce_history):
                        is_volley = True
                    
                    stroke = "Volley" if is_volley else self._classify_side(kps)
                    validated.append({
                        "frame": idx, "type": "hit", "is_serve": False, "stroke": stroke,
                        "pixel_pos": ball_trajectory[idx], "player_id": pid
                    })
                    for f in range(idx-15, idx+15): cooldown[f] = True
                    break

        return sorted(validated, key=lambda x: x['frame'])

    def classify_stroke(self, event, players_data):
        """
        Returns the pre-classified stroke type from the event.
        Compatibility method for main.py.
        """
        return event.get('stroke', 'Hit')

    def _classify_side(self, kps):
        """区分正反手"""
        body_x = (kps['l_shoulder'][0] + kps['r_shoulder'][0]) / 2
        # 简单逻辑：右手球员，球在身右为正手
        return "Forehand" if kps['r_wrist'][0] > body_x else "Backhand"

    def calculate_instant_speed(self, event, ball_trajectory):
        f = event['frame']
        window = 3
        if f + window >= len(ball_trajectory): return 0.0
        p1, p2 = ball_trajectory[f], ball_trajectory[f+window]
        if p1 == (0,0) or p2 == (0,0): return 0.0
        h = Config.ASSUMED_SERVE_HEIGHT if event.get("is_serve") else Config.ASSUMED_HIT_HEIGHT
        x1, y1 = self.geometry.compensate_perspective_single_view(p1[0], p1[1], h, Config.CAMERA_HEIGHT)
        x2, y2 = self.geometry.compensate_perspective_single_view(p2[0], p2[1], h, Config.CAMERA_HEIGHT)
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return min((dist / (window/self.fps)) * 3.6, Config.MAX_BALL_SPEED_KMH)

    def calculate_player_movement(self, players_data):
        movement = {}
        last_pos = {}
        for f in sorted(players_data.keys()):
            for pid, p in players_data[f].items():
                foot_u = (p['keypoints']['l_ankle'][0] + p['keypoints']['r_ankle'][0])/2
                foot_v = max(p['keypoints']['l_ankle'][1], p['keypoints']['r_ankle'][1])
                rx, ry = self.geometry.pixel_to_real_2d(foot_u, foot_v)
                if pid not in movement: movement[pid] = 0.0
                if pid in last_pos:
                    dist = np.sqrt((rx-last_pos[pid][0])**2 + (ry-last_pos[pid][1])**2)
                    if dist > 0.05: movement[pid] += dist
                last_pos[pid] = (rx, ry)
        return movement
