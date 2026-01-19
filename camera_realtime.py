# camera_realtime.py
import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List
from license_plate_detection import LicensePlateDetector
from license_plate_preprocessor import LicensePlatePreprocessor
from license_plate_ocr_engine import get_license_plate_info


def get_all_cameras(max_test: int = 10):
    """
    检测所有可用的摄像头并返回详细信息
    
    Args:
        max_test: 最大测试摄像头数量
        
    Returns:
        摄像头信息列表 [(index, name, resolution, fps)]
    """
    cameras = []
    
    print("正在检测摄像头...")
    
    # 首先检查常见的后端
    backends = [
        cv2.CAP_ANY,  # 自动选择
        cv2.CAP_DSHOW,  # DirectShow (Windows)
        cv2.CAP_MSMF,   # Microsoft Media Foundation (Windows)
        cv2.CAP_V4L2,   # Video4Linux2 (Linux)
        cv2.CAP_AVFOUNDATION,  # AVFoundation (macOS)
    ]
    
    for backend in backends:
        for i in range(max_test):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend_name = "未知"
                    
                    if backend == cv2.CAP_ANY:
                        backend_name = "自动"
                    elif backend == cv2.CAP_DSHOW:
                        backend_name = "DirectShow"
                    elif backend == cv2.CAP_MSMF:
                        backend_name = "MSMF"
                    elif backend == cv2.CAP_V4L2:
                        backend_name = "Video4Linux2"
                    elif backend == cv2.CAP_AVFOUNDATION:
                        backend_name = "AVFoundation"
                    
                    camera_name = f"摄像头 {i} ({backend_name})"
                    print(f"  找到: {camera_name} - {width}x{height} @ {fps:.1f}fps")
                    
                    cameras.append({
                        'index': i,
                        'backend': backend,
                        'name': camera_name,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'available': True
                    })
                cap.release()
            else:
                cap.release()
    
    # 去重（同一个摄像头可能被多个后端检测到）
    unique_cameras = {}
    for cam in cameras:
        if cam['index'] not in unique_cameras:
            unique_cameras[cam['index']] = cam
        else:
            # 选择分辨率最大的后端
            if cam['width'] * cam['height'] > unique_cameras[cam['index']]['width'] * unique_cameras[cam['index']]['height']:
                unique_cameras[cam['index']] = cam
    
    return list(unique_cameras.values())


def select_camera_interactive():
    """
    交互式选择摄像头
    """
    print("=" * 60)
    print("摄像头检测")
    print("=" * 60)
    
    cameras = get_all_cameras()
    
    if not cameras:
        print("未找到任何摄像头！")
        print("请检查：")
        print("  1. 摄像头是否正确连接")
        print("  2. 摄像头驱动程序是否已安装")
        print("  3. 摄像头是否被其他程序占用")
        return None, None
    
    print(f"\n找到 {len(cameras)} 个可用摄像头:")
    for i, cam in enumerate(cameras):
        print(f"  [{i}] {cam['name']}")
        print(f"      分辨率: {cam['width']}x{cam['height']}")
        print(f"      帧率: {cam['fps']:.1f}fps")
    
    # 让用户选择
    while True:
        try:
            choice = input("\n请选择摄像头编号 (输入 'q' 退出): ").strip()
            
            if choice.lower() == 'q':
                return None, None
            
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                selected_camera = cameras[choice_idx]
                print(f"\n已选择: {selected_camera['name']}")
                print(f"  索引: {selected_camera['index']}")
                print(f"  后端: {selected_camera['backend']}")
                print(f"  分辨率: {selected_camera['width']}x{selected_camera['height']}")
                print(f"  帧率: {selected_camera['fps']:.1f}fps")
                
                # 询问是否修改参数
                modify = input("\n是否修改摄像头参数？(y/n): ").strip().lower()
                if modify == 'y':
                    width = input(f"帧宽度 (当前: {selected_camera['width']}): ").strip()
                    height = input(f"帧高度 (当前: {selected_camera['height']}): ").strip()
                    
                    if width:
                        selected_camera['width'] = int(width)
                    if height:
                        selected_camera['height'] = int(height)
                
                return selected_camera['index'], selected_camera['backend']
            else:
                print(f"错误: 请输入 0 到 {len(cameras)-1} 之间的数字")
        except ValueError:
            print("错误: 请输入有效的数字")


class RealTimeLicensePlateDetector:
    """实时车牌检测器"""
    
    def __init__(self, 
                 model_path: str = 'yolov8s.pt',
                 conf_threshold: float = 0.5,
                 use_preprocessing: bool = True):
        """
        初始化实时检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 检测置信度阈值
            use_preprocessing: 是否使用预处理
        """
        print("初始化实时车牌检测系统...")
        
        # 初始化检测器
        self.detector = LicensePlateDetector(
            model_path=model_path,
            conf_threshold=conf_threshold
        )
        
        # 初始化预处理器
        self.preprocessor = LicensePlatePreprocessor(
            target_size=(640, 480)
        )
        
        self.use_preprocessing = use_preprocessing
        
        # 检测统计
        self.frame_count = 0
        self.detection_count = 0
        self.plate_count = 0
        
        # 输出目录
        self.output_dir = Path("camera_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print("✓ 实时检测系统初始化完成")
    
    def start_camera(self, 
                    camera_index: int = 0,
                    camera_backend: int = cv2.CAP_ANY,
                    frame_width: int = 1280,
                    frame_height: int = 720,
                    detection_interval: int = 10):
        """
        启动摄像头检测
        
        Args:
            camera_index: 摄像头索引
            camera_backend: 摄像头后端
            frame_width: 帧宽度
            frame_height: 帧高度
            detection_interval: 检测间隔帧数
        """
        print(f"\n启动摄像头 (索引: {camera_index}, 后端: {camera_backend})...")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index, camera_backend)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {camera_index}")
            print("请尝试:")
            print("  1. 检查摄像头是否正确连接")
            print("  2. 尝试其他摄像头索引")
            print("  3. 确保没有其他程序占用摄像头")
            
            # 显示可用摄像头
            cameras = get_all_cameras()
            if cameras:
                print("\n可用摄像头:")
                for cam in cameras:
                    print(f"  索引 {cam['index']}: {cam['name']} ({cam['width']}x{cam['height']})")
            
            return False
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # 获取实际参数
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"摄像头参数: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        print(f"检测间隔: 每{detection_interval}帧检测一次")
        print("\n控制说明:")
        print("  Q: 退出程序")
        print("  S: 保存当前帧")
        print("  P: 暂停/继续检测")
        print("  I: 显示摄像头信息")
        print("=" * 60)
        
        # 状态变量
        is_paused = False
        start_time = time.time()
        
        try:
            while True:
                if not is_paused:
                    # 读取帧
                    ret, frame = cap.read()
                    if not ret:
                        print("摄像头读取失败")
                        break
                    
                    self.frame_count += 1
                    display_frame = frame.copy()
                    
                    # 检测车牌
                    if self.frame_count % detection_interval == 0:
                        self.detection_count += 1
                        
                        # 检测车牌
                        plates_info = self._detect_plates_in_frame(frame)
                        
                        if plates_info:
                            self.plate_count += len(plates_info)
                            
                            # 处理并显示每个车牌
                            for plate_info in plates_info:
                                display_frame = self._annotate_detection(display_frame, plate_info)
                    
                    # 计算并显示FPS
                    elapsed_time = time.time() - start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # 添加统计信息
                    self._add_statistics_overlay(display_frame, fps)
                    
                    if is_paused:
                        cv2.putText(display_frame, "已暂停", 
                                   (display_frame.shape[1]//2 - 50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    # 暂停时显示最后一帧
                    if 'display_frame' in locals():
                        cv2.putText(display_frame, "已暂停", 
                                   (display_frame.shape[1]//2 - 50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # 显示图像
                cv2.imshow('车牌识别 - 实时检测', display_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 退出
                    break
                elif key == ord('s'):  # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = self.output_dir / f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"已保存截图: {filename}")
                elif key == ord('p'):  # 暂停/继续
                    is_paused = not is_paused
                    status = "已暂停" if is_paused else "已继续"
                    print(status)
                elif key == ord('i'):  # 显示摄像头信息
                    self._print_camera_info(cap)
                elif key == 27:  # ESC键
                    break
        
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"运行时出错: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 保存统计信息
            self._save_statistics(elapsed_time)
        
        return True
    
    def _print_camera_info(self, cap):
        """显示摄像头信息"""
        print("\n" + "=" * 60)
        print("摄像头信息")
        print("=" * 60)
        
        properties = {
            cv2.CAP_PROP_FRAME_WIDTH: "帧宽度",
            cv2.CAP_PROP_FRAME_HEIGHT: "帧高度",
            cv2.CAP_PROP_FPS: "帧率",
            cv2.CAP_PROP_BRIGHTNESS: "亮度",
            cv2.CAP_PROP_CONTRAST: "对比度",
            cv2.CAP_PROP_SATURATION: "饱和度",
            cv2.CAP_PROP_HUE: "色调",
            cv2.CAP_PROP_GAIN: "增益",
            cv2.CAP_PROP_EXPOSURE: "曝光",
            cv2.CAP_PROP_AUTO_EXPOSURE: "自动曝光",
            cv2.CAP_PROP_AUTOFOCUS: "自动对焦",
            cv2.CAP_PROP_FOCUS: "焦距",
        }
        
        for prop_id, prop_name in properties.items():
            try:
                value = cap.get(prop_id)
                print(f"{prop_name}: {value:.2f}")
            except:
                pass
    
    def _detect_plates_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """在帧中检测车牌"""
        plates_info = []
        
        try:
            # 保存临时文件用于检测
            temp_path = self.output_dir / f"temp_frame_{self.frame_count}.jpg"
            cv2.imwrite(str(temp_path), frame)
            
            # 使用检测器检测车牌
            plates_info = self.detector.detect_all_and_rectify(str(temp_path))
            
            # 处理每个检测到的车牌
            for i, plate_info in enumerate(plates_info):
                # 尝试识别车牌文字
                plate_result = self._recognize_plate_text(plate_info, i)
                
                if plate_result and 'plate_text' in plate_result:
                    plate_info['recognition'] = plate_result
                    print(f"检测到车牌: {plate_result['plate_text']} "
                          f"({plate_result.get('plate_type', '未知')}) "
                          f"置信度: {plate_result.get('ocr_confidence', 0):.2f}")
            
            # 清理临时文件
            try:
                os.remove(str(temp_path))
            except:
                pass
                
        except Exception as e:
            print(f"检测过程中出错: {e}")
        
        return plates_info
    
    def _recognize_plate_text(self, plate_info: Dict, plate_index: int) -> Dict:
        """识别车牌文字"""
        result = {
            'plate_index': plate_index,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 获取矫正后的车牌图像
        rectified_image = plate_info['rectified']
        if rectified_image is None or rectified_image.size == 0:
            return result
        
        try:
            # 预处理
            preprocessed_image = rectified_image
            if self.use_preprocessing:
                preprocessed_image, _ = self.preprocessor.preprocess_with_color_recovery(
                    rectified_image,
                    detect_plate_region=True
                )
            
            # 保存临时文件用于OCR
            temp_path = self.output_dir / f"temp_plate_{self.frame_count}_{plate_index}.jpg"
            cv2.imwrite(str(temp_path), preprocessed_image)
            
            # OCR识别
            ocr_result = get_license_plate_info(str(temp_path))
            
            if ocr_result:
                plate_text, ocr_confidence, plate_type = ocr_result
                result.update({
                    'plate_text': plate_text,
                    'ocr_confidence': ocr_confidence,
                    'plate_type': plate_type,
                    'recognition_success': True
                })
            
            # 保存识别成功的车牌图像
            if 'plate_text' in result and result['plate_text'] != "未知":
                save_path = self.output_dir / f"plate_{self.frame_count}_{plate_index}_{result['plate_text']}.jpg"
                cv2.imwrite(str(save_path), preprocessed_image)
            
            # 清理临时文件
            try:
                os.remove(str(temp_path))
            except:
                pass
                
        except Exception as e:
            print(f"车牌识别出错: {e}")
            result['error'] = str(e)
        
        return result
    
    def _annotate_detection(self, frame: np.ndarray, plate_info: Dict) -> np.ndarray:
        """在帧上标注检测结果"""
        if 'bbox' not in plate_info:
            return frame
        
        x1, y1, x2, y2 = plate_info['bbox']
        
        # 根据识别结果选择颜色
        recognition = plate_info.get('recognition', {})
        if recognition.get('recognition_success', False):
            color = (0, 255, 0)  # 绿色 - 识别成功
        else:
            color = (0, 0, 255)  # 红色 - 仅检测到
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 准备文本信息
        text_lines = []
        
        if 'plate_text' in recognition and recognition['plate_text'] != "未知":
            text_lines.append(f"车牌: {recognition['plate_text']}")
            
            if 'plate_type' in recognition:
                text_lines.append(f"类型: {recognition['plate_type']}")
            
            if 'ocr_confidence' in recognition:
                text_lines.append(f"置信度: {recognition['ocr_confidence']:.2f}")
        else:
            text_lines.append("车牌检测")
        
        text_lines.append(f"检测置信度: {plate_info['confidence']:.2f}")
        
        # 添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # 计算文本位置
        total_height = len(text_lines) * 20 + 10
        
        # 文本背景位置
        bg_x1 = x1
        bg_y1 = max(0, y1 - total_height - 10)
        bg_x2 = x1 + 200
        bg_y2 = y1 - 5
        
        # 如果上方空间不足，放在下方
        if bg_y1 < 0:
            bg_y1 = y2 + 5
            bg_y2 = bg_y1 + total_height + 10
        
        # 绘制文本背景
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        
        # 绘制文本
        y_offset = bg_y1 + 15
        text_color = (255, 255, 255)
        
        for line in text_lines:
            cv2.putText(frame, line, (bg_x1 + 5, y_offset), 
                       font, font_scale, text_color, thickness)
            y_offset += 20
        
        return frame
    
    def _add_statistics_overlay(self, frame: np.ndarray, fps: float):
        """添加统计信息叠加层"""
        stats = [
            f"FPS: {fps:.1f}",
            f"帧数: {self.frame_count}",
            f"检测次数: {self.detection_count}",
            f"检测到车牌: {self.plate_count}",
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        y_offset = frame.shape[0] - 10
        
        for stat in reversed(stats):
            (text_width, text_height), _ = cv2.getTextSize(stat, font, font_scale, thickness)
            
            # 绘制背景
            bg_x1 = 5
            bg_y1 = y_offset - text_height - 5
            bg_x2 = text_width + 15
            bg_y2 = y_offset + 5
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # 绘制文本
            cv2.putText(frame, stat, (10, y_offset), 
                       font, font_scale, (0, 255, 255), thickness)
            
            y_offset -= 25
    
    def _save_statistics(self, elapsed_time: float):
        """保存统计信息"""
        stats_file = self.output_dir / "statistics.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("车牌识别实时检测统计\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"统计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总运行时间: {elapsed_time:.2f}秒\n")
            f.write(f"总帧数: {self.frame_count}\n")
            f.write(f"检测次数: {self.detection_count}\n")
            f.write(f"检测到车牌总数: {self.plate_count}\n")
            
            if elapsed_time > 0:
                avg_fps = self.frame_count / elapsed_time
                f.write(f"平均FPS: {avg_fps:.2f}\n")
            
            f.write(f"\n检测间隔: 每10帧检测一次\n")
            f.write(f"使用预处理: {'是' if self.use_preprocessing else '否'}\n")
        
        print(f"\n统计信息已保存: {stats_file}")
        print("=" * 60)
        print("检测完成！")
        print(f"检测到 {self.plate_count} 个车牌")
        print(f"平均FPS: {self.frame_count/elapsed_time:.2f}" if elapsed_time > 0 else "")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="车牌识别实时摄像头检测")
    
    # 摄像头参数
    parser.add_argument("--camera-index", type=int, default=0,
                       help="摄像头索引 (默认: 0)")
    parser.add_argument("--frame-width", type=int, default=1280,
                       help="帧宽度 (默认: 1280)")
    parser.add_argument("--frame-height", type=int, default=720,
                       help="帧高度 (默认: 720)")
    parser.add_argument("--detection-interval", type=int, default=10,
                       help="检测间隔帧数 (默认: 10)")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                       help="YOLO模型路径 (默认: yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="检测置信度阈值 (默认: 0.5)")
    parser.add_argument("--no-preprocess", action="store_true",
                       help="禁用预处理")
    
    # 交互式选择
    parser.add_argument("--interactive", action="store_true",
                       help="交互式选择摄像头")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("车牌识别 - 实时摄像头检测")
    print("=" * 60)
    
    # 交互式选择摄像头
    if args.interactive:
        camera_index, camera_backend = select_camera_interactive()
        if camera_index is None:
            print("退出程序")
            return
    else:
        camera_index = args.camera_index
        camera_backend = cv2.CAP_ANY
    
    # 创建实时检测器
    detector = RealTimeLicensePlateDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        use_preprocessing=not args.no_preprocess
    )
    
    # 启动摄像头
    detector.start_camera(
        camera_index=camera_index,
        camera_backend=camera_backend,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        detection_interval=args.detection_interval
    )


if __name__ == "__main__":
    main()