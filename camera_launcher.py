#!/usr/bin/env python3
"""
camera_launcher.py
摄像头车牌识别启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import LicensePlateSystem, camera_mode
import argparse

def main():
    """摄像头启动器主函数"""
    parser = argparse.ArgumentParser(description="摄像头车牌识别启动器")
    
    # 摄像头参数
    parser.add_argument("--camera-index", type=int, default=0, 
                       help="摄像头索引（0=默认，1=第二个摄像头）")
    parser.add_argument("--resolution", type=str, default="1280x720",
                       help="分辨率（格式：宽x高）")
    parser.add_argument("--fps", type=int, default=30,
                       help="帧率")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                       help="YOLO模型路径")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="检测置信度阈值")
    parser.add_argument("--interval", type=int, default=10,
                       help="检测间隔帧数")
    parser.add_argument("--no-preprocess", action="store_true",
                       help="禁用预处理")
    parser.add_argument("--output-dir", type=str, default="camera_results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 解析分辨率
    try:
        width, height = map(int, args.resolution.split('x'))
    except:
        width, height = 1280, 720
        print(f"警告：分辨率格式错误，使用默认值 {width}x{height}")
    
    print("=" * 60)
    print("摄像头车牌识别系统")
    print("=" * 60)
    print(f"摄像头索引: {args.camera_index}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {args.fps}fps")
    print(f"检测间隔: {args.interval}帧")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 创建系统
    system = LicensePlateSystem(
        detection_model_path=args.model,
        detection_conf_threshold=args.conf,
        use_preprocessing=not args.no_preprocess
    )
    
    # 模拟args对象
    class Args:
        pass
    
    camera_args = Args()
    camera_args.camera_index = args.camera_index
    camera_args.frame_width = width
    camera_args.frame_height = height
    camera_args.fps = args.fps
    camera_args.detection_interval = args.interval
    camera_args.output_dir = args.output_dir
    
    # 启动摄像头模式
    import main
    main.camera_mode(system, camera_args)

if __name__ == "__main__":
    main()