#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def test_output_format():
    """测试输出格式是否符合要求"""
    
    # 检查输出文件是否存在
    output_file = 'tennis_results.json'
    if not os.path.exists(output_file):
        print(f"错误: 输出文件 {output_file} 不存在")
        return False
    
    # 读取JSON文件
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取JSON文件: {e}")
        return False
    
    # 检查数据结构
    if not isinstance(results, list):
        print("错误: 结果应该是列表格式")
        return False
    
    if len(results) == 0:
        print("警告: 结果列表为空")
        return True
    
    # 检查每个结果项的格式
    required_fields = ["x", "y", "type", "speed", "timestamp", 
                      "farCountPerson_x", "farCountPerson_y", 
                      "nearCountPerson_x", "nearCountPerson_y"]
    
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            print(f"错误: 第 {i} 项不是字典格式")
            return False
        
        # 检查必需字段
        for field in required_fields:
            if field not in result:
                print(f"错误: 第 {i} 项缺少字段 '{field}'")
                return False
        
        # 检查数据类型
        if not isinstance(result["x"], (int, float)):
            print(f"错误: 第 {i} 项的 'x' 字段类型错误")
            return False
        
        if not isinstance(result["y"], (int, float)):
            print(f"错误: 第 {i} 项的 'y' 字段类型错误")
            return False
        
        if not isinstance(result["type"], str):
            print(f"错误: 第 {i} 项的 'type' 字段类型错误")
            return False
        
        if not isinstance(result["speed"], (int, float)):
            print(f"错误: 第 {i} 项的 'speed' 字段类型错误")
            return False
        
        if not isinstance(result["timestamp"], int):
            print(f"错误: 第 {i} 项的 'timestamp' 字段类型错误")
            return False
    
    print("✓ 输出格式验证通过!")
    print(f"总共处理了 {len(results)} 帧数据")
    
    # 显示前几个结果作为示例
    if len(results) > 0:
        print("\n前3个结果示例:")
        for i, result in enumerate(results[:3]):
            print(f"  第{i+1}帧: {result}")
    
    return True

if __name__ == "__main__":
    test_output_format() 