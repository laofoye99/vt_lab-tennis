import pandas as pd
import numpy as np
import os

# 导入分析组件
from src.config import Config
from src.geometry import CourtGeometry
from src.analytics import MatchAnalyzer

def run_comprehensive_verification(csv_path):
    geometry = CourtGeometry(Config.REFERENCE_POINTS_2D, Config.REFERENCE_POINTS_3D)
    analyzer = MatchAnalyzer(geometry, fps=30.0)
    
    # 1. 加载轨迹
    df = pd.read_csv(csv_path)
    ball_trajectory = []
    for _, row in df.iterrows():
        if row['detected'] == 1:
            ball_trajectory.append((row['x'], row['y']))
        else:
            ball_trajectory.append((0, 0))
            
    # 2. 运行初筛 (Layer 1)
    candidates = analyzer._kinematic_screening(ball_trajectory)
    
    # Ground Truth 定义 (基于您的提供和分析)
    gt = {
        'SERVE': [9, 144, 769, 1354, 1473, 2099],
        'BOUNCE': [1115, 1142, 1554, 2159, 2270, 2826],
        'VOLLEY': [213, 273, 816],
        'FOREHAND': [2484] 
    }

    print("\n=== COMPREHENSIVE VERIFICATION REPORT ===")
    
    def evaluate(name, gt_list, pred_list, window=8):
        print(f"\nCategory: {name}")
        hits = 0
        for g in gt_list:
            # 检查是否有预测帧在 GT 的窗口范围内
            found = [p for p in pred_list if abs(p - g) <= window]
            status = f"MATCH (found at {found[0]})" if found else "MISS"
            if found: hits += 1
            print(f"  GT Frame {g:4} : {status}")
        
        recall = (hits / len(gt_list)) * 100
        print(f"  -> Recall: {recall:.1f}% ({hits}/{len(gt_list)})")

    # 注意：由于验证脚本目前没有真实的 Pose 数据，这里主要验证初筛层能否捕捉到候选点
    # 对于 Hit 类别 (Serve, Volley, Forehand)，它们都应该在 candidates['hit'] 或 candidates['toss'] 中
    all_hit_candidates = candidates['hit'] + candidates['toss']
    
    evaluate("SERVE (in Hit/Toss Pool)", gt['SERVE'], all_hit_candidates)
    evaluate("BOUNCE (in Bounce Pool)", gt['BOUNCE'], candidates['bounce'])
    evaluate("VOLLEY (in Hit Pool)", gt['VOLLEY'], candidates['hit'])
    evaluate("FOREHAND (in Hit Pool)", gt['FOREHAND'], candidates['hit'])

    print("\n[NOTE] Actual classification (FH/BH/Volley) requires running main.py with full Pose data.")

if __name__ == "__main__":
    csv_file = '../recording/cam66_20260307_173403_2min_output_wasb.csv'
    if os.path.exists(csv_file):
        run_comprehensive_verification(csv_file)
    else:
        print(f"CSV not found: {csv_file}")
