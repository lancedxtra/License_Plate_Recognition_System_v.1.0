import cv2
import numpy as np
import argparse
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 导入三个模块
from license_plate_detection import LicensePlateDetector
from license_plate_preprocessor import LicensePlatePreprocessor
from license_plate_ocr_engine import get_license_plate_info
try:
    from video_processor import VideoLicensePlateProcessor, create_video_processor_from_system
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"警告: 视频处理器模块不可用: {e}")
    VIDEO_PROCESSOR_AVAILABLE = False

try:
    from camera_manager import (
        list_all_cameras,
        select_camera_interactive,
        get_camera_name,
        get_camera_info,
        print_camera_info,
        find_best_camera,
        CameraManager
    )
    CAMERA_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"警告: 摄像头管理器模块不可用: {e}")
    print("摄像头检测功能将受限")
    CAMERA_MANAGER_AVAILABLE = False

class LicensePlateSystem:
    """
    车牌识别系统 - 整合检测、矫正、预处理和识别（含颜色检测）
    """
    
    def __init__(self, 
                 detection_model_path: str = 'yolov8s.pt',
                 detection_conf_threshold: float = 0.5,
                 use_preprocessing: bool = True):
        """
        初始化车牌识别系统
        
        Args:
            detection_model_path: YOLO检测模型路径
            detection_conf_threshold: 检测置信度阈值
            use_preprocessing: 是否使用预处理
        """
        print("=" * 60)
        print("初始化车牌识别系统...")
        print("=" * 60)
        
        # 初始化三个模块
        print("1. 加载车牌检测器...")
        self.detector = LicensePlateDetector(
            model_path=detection_model_path,
            conf_threshold=detection_conf_threshold
        )
        
        print("2. 加载车牌预处理器...")
        self.preprocessor = LicensePlatePreprocessor(
            target_size=(640, 480)
        )
        
        self.use_preprocessing = use_preprocessing
        
        print("✓ 系统初始化完成")
        print(f"  预处理: {'启用' if use_preprocessing else '禁用'}")
        print()
    
    def process_single_plate(self, original_image: np.ndarray, 
                            plate_info: Dict, 
                            output_dir: str,
                            plate_index: int,
                            save_results: bool = True) -> Dict:
        """
        处理单个车牌
        
        Args:
            original_image: 原始图像
            plate_info: 车牌信息
            output_dir: 输出目录
            plate_index: 车牌索引
            save_results: 是否保存结果
            
        Returns:
            处理结果字典
        """
        print(f"\n处理车牌 {plate_index}:")
        print(f"  检测置信度: {plate_info['confidence']:.3f}")
        print(f"  位置: {plate_info['bbox']}")
        
        # 获取矫正后的车牌图像
        rectified_image = plate_info['rectified']
        
        if rectified_image is None or rectified_image.size == 0:
            print(f"  警告: 车牌 {plate_index} 图像为空，跳过")
            return None
        
        # 1. 预处理（增强图像质量）
        preprocessed_image = rectified_image
        if self.use_preprocessing:
            print("  步骤1: 预处理图像...")
            try:
                # 使用预处理器处理图像
                preprocessed_image, preprocess_info = self.preprocessor.preprocess_with_color_recovery(
                    rectified_image,
                    detect_plate_region=True
                )
                
                # 保存预处理前后的对比
                if save_results:
                    self._save_comparison(
                        rectified_image, 
                        preprocessed_image, 
                        output_dir, 
                        f"plate_{plate_index}_preprocess"
                    )
                
            except Exception as e:
                print(f"    预处理失败: {e}")
                import traceback
                traceback.print_exc()
                preprocessed_image = rectified_image
        
        # 2. 保存预处理后的图像用于OCR
        temp_plate_path = None
        if save_results:
            temp_plate_path = f"{output_dir}/plate_{plate_index}_for_ocr.jpg"
            cv2.imwrite(temp_plate_path, preprocessed_image)
        
        # 3. OCR识别车牌信息（包含颜色检测）
        print("  步骤2: OCR识别车牌（含颜色检测）...")
        ocr_start = time.time()
        
        # 使用预处理后的图像进行识别
        if temp_plate_path:
            ocr_input_path = temp_plate_path
        else:
            # 临时保存图像用于OCR
            temp_path = f"temp_plate_{plate_index}.jpg"
            cv2.imwrite(temp_path, preprocessed_image)
            ocr_input_path = temp_path
        
        # 调用OCR引擎（包含颜色检测）
        ocr_result = get_license_plate_info(ocr_input_path)
        
        ocr_time = time.time() - ocr_start
        
        # 4. 处理OCR结果
        plate_text = "未知"
        ocr_confidence = 0.0
        plate_type = "未知"
        
        if ocr_result:
            plate_text, ocr_confidence, plate_type = ocr_result
            print(f"  ✓ 识别成功:")
            print(f"    车牌号码: {plate_text}")
            print(f"    车牌类型: {plate_type}")
            print(f"    识别置信度: {ocr_confidence:.3f}")
            print(f"    识别耗时: {ocr_time:.2f}s")
        else:
            print(f"  ✗ 识别失败")
        
        # 5. 在原图上绘制结果
        annotated_image = self._annotate_plate(
            original_image.copy(),
            plate_info['bbox'],
            plate_text,
            plate_info['confidence'],
            ocr_confidence,
            plate_type
        )
        
        # 6. 准备结果
        result = {
            'plate_id': plate_index,
            'detection_confidence': float(plate_info['confidence']),
            'bbox': plate_info['bbox'],
            'plate_text': plate_text,
            'ocr_confidence': float(ocr_confidence),
            'plate_type': plate_type,
            'rectified_image': rectified_image,
            'preprocessed_image': preprocessed_image,
            'annotated_image': annotated_image,
            'ocr_time': float(ocr_time),
        }
        
        # 7. 保存单个车牌结果
        if save_results:
            self._save_single_result(result, output_dir, plate_index)
        
        return result
    
    def process_image(self, image_path: str, 
                     save_results: bool = True,
                     output_dir: str = "results") -> List[Dict]:
        """
        处理单张图片，返回所有车牌信息
        
        Args:
            image_path: 图片路径
            save_results: 是否保存结果
            output_dir: 输出目录
            
        Returns:
            车牌信息列表
        """
        print(f"处理图片: {image_path}")
        print("-" * 60)
        
        # 读取原始图片
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"错误：无法读取图片 {image_path}")
            return []
        
        # 1. 检测并矫正车牌
        print("步骤1: 检测并矫正车牌...")
        start_time = time.time()
        
        plates_info = self.detector.detect_all_and_rectify(image_path)
        
        if not plates_info:
            print("未检测到车牌")
            return []
        
        detection_time = time.time() - start_time
        print(f"✓ 检测到 {len(plates_info)} 个车牌 (耗时: {detection_time:.2f}s)")
        
        # 创建输出目录
        if save_results:
            Path(output_dir).mkdir(exist_ok=True)
        
        # 处理每个检测到的车牌
        all_results = []
        
        for i, plate_info in enumerate(plates_info):
            result = self.process_single_plate(
                original_image=original_image,
                plate_info=plate_info,
                output_dir=output_dir,
                plate_index=i+1,
                save_results=save_results
            )
            
            if result:
                result['detection_time'] = float(detection_time)
                result['total_time'] = float(detection_time + result['ocr_time'])
                all_results.append(result)
        
        # 8. 保存包含所有车牌的原图
        if save_results and all_results:
            # 使用最后一个车牌的标注图像
            final_annotated = all_results[-1]['annotated_image']
            
            final_path = f"{output_dir}/final_annotated.jpg"
            cv2.imwrite(final_path, final_annotated)
            print(f"\n✓ 最终标注图片已保存: {final_path}")
        
        # 9. 清理临时文件
        self._cleanup_temp_files()
        
        # 10. 打印汇总结果
        self._print_summary(all_results)
        
        # 11. 保存JSON格式的完整结果
        if save_results:
            self._save_json_results(all_results, output_dir)
        
        return all_results
    
    def _annotate_plate(self, image: np.ndarray, bbox: Tuple, 
                       plate_text: str, det_conf: float,
                       ocr_conf: float, plate_type: str) -> np.ndarray:
        """
        在原图上标注车牌信息
        """
        x1, y1, x2, y2 = bbox
        
        # 根据车牌类型选择颜色
        color_map = {
            '蓝牌': (255, 0, 0),      # 蓝色
            '黄牌': (0, 255, 255),    # 黄色
            '新能源绿牌': (0, 255, 0), # 绿色
            '白牌': (255, 255, 255),  # 白色
            '黑牌': (0, 0, 0),        # 黑色
            '白牌 (警用)': (255, 255, 255),  # 白色
        }
        
        if plate_type in color_map:
            color = color_map[plate_type]
        else:
            color = (0, 255, 0) if plate_text != "未知" else (0, 0, 255)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # 准备文本信息
        lines = []
        if plate_text != "未知":
            lines.append(f"车牌: {plate_text}")
        
        lines.append(f"类型: {plate_type}")
        lines.append(f"检测: {det_conf:.2f} 识别: {ocr_conf:.2f}")
        
        # 计算文本位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # 计算总文本高度
        line_heights = []
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            line_heights.append(text_height)
        
        total_height = sum(line_heights) + 10 * len(lines)
        max_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines])
        
        # 文本背景位置（在车牌上方）
        bg_x1 = x1
        bg_y1 = max(0, y1 - total_height - 5)
        bg_x2 = x1 + max_width + 20
        bg_y2 = y1 - 5
        
        # 如果上方空间不足，放在下方
        if bg_y1 < 0:
            bg_y1 = y2 + 5
            bg_y2 = bg_y1 + total_height
        
        # 绘制文本背景
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        
        # 绘制文本
        y_offset = bg_y1 + line_heights[0] + 5
        text_color = (255, 255, 255) if plate_type in ['黄牌', '白牌', '白牌 (警用)'] else (0, 255, 0)
        
        for i, line in enumerate(lines):
            cv2.putText(image, line, (bg_x1 + 10, y_offset), 
                       font, font_scale, text_color, thickness)
            y_offset += line_heights[i] + 10
        
        return image
    
    def _save_comparison(self, before: np.ndarray, after: np.ndarray, 
                        output_dir: str, name: str):
        """保存处理前后对比图"""
        if before is None or after is None:
            return
        
        h1, w1 = before.shape[:2]
        h2, w2 = after.shape[:2]
        
        max_height = max(h1, h2)
        
        # 调整大小
        if h1 != max_height:
            scale1 = max_height / h1
            new_w1 = int(w1 * scale1)
            resized_before = cv2.resize(before, (new_w1, max_height))
        else:
            resized_before = before
            new_w1 = w1
            
        if h2 != max_height:
            scale2 = max_height / h2
            new_w2 = int(w2 * scale2)
            resized_after = cv2.resize(after, (new_w2, max_height))
        else:
            resized_after = after
            new_w2 = w2
        
        # 并排显示
        combined = np.hstack((resized_before, resized_after))
        
        # 添加标签
        cv2.putText(combined, "处理前", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "处理后", (new_w1 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        save_path = f"{output_dir}/{name}_comparison.jpg"
        cv2.imwrite(save_path, combined)
    
    def _save_single_result(self, result: Dict, output_dir: str, plate_id: int):
        """保存单个车牌结果"""
        base_path = f"{output_dir}/plate_{plate_id}"
        
        # 保存矫正后的车牌
        if result['rectified_image'] is not None:
            cv2.imwrite(f"{base_path}_rectified.jpg", result['rectified_image'])
        
        # 保存预处理后的车牌
        if result['preprocessed_image'] is not None:
            cv2.imwrite(f"{base_path}_preprocessed.jpg", result['preprocessed_image'])
        
        # 保存标注图
        if result['annotated_image'] is not None:
            cv2.imwrite(f"{base_path}_annotated.jpg", result['annotated_image'])
        
        # 保存文本结果
        with open(f"{base_path}_info.txt", "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"车牌 {plate_id} 识别结果\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("【基本信息】\n")
            f.write(f"车牌号码: {result['plate_text']}\n")
            f.write(f"车牌类型: {result['plate_type']}\n")
            f.write(f"检测置信度: {result['detection_confidence']:.4f}\n")
            f.write(f"OCR置信度: {result['ocr_confidence']:.4f}\n")
            f.write(f"位置坐标: {result['bbox']}\n\n")
            
            f.write("【时间统计】\n")
            if 'detection_time' in result:
                f.write(f"检测耗时: {result['detection_time']:.4f}s\n")
            f.write(f"识别耗时: {result['ocr_time']:.4f}s\n")
            if 'total_time' in result:
                f.write(f"总耗时: {result['total_time']:.4f}s\n\n")
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        import glob
        temp_files = glob.glob("temp_plate_*.jpg") + glob.glob("temp_frame_*.jpg")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
    
    def _save_json_results(self, results: List[Dict], output_dir: str):
        """保存JSON格式的完整结果"""
        serializable_results = []
        
        for result in results:
            # 基本信息的可序列化版本
            serializable_result = {
                'plate_id': result['plate_id'],
                'plate_text': result['plate_text'],
                'plate_type': result['plate_type'],
                'detection_confidence': result['detection_confidence'],
                'ocr_confidence': result['ocr_confidence'],
                'bbox': result['bbox'],
            }
            
            # 添加时间信息
            if 'detection_time' in result:
                serializable_result['detection_time'] = result['detection_time']
            if 'ocr_time' in result:
                serializable_result['ocr_time'] = result['ocr_time']
            if 'total_time' in result:
                serializable_result['total_time'] = result['total_time']
            
            serializable_results.append(serializable_result)
        
        # 保存JSON文件
        json_path = f"{output_dir}/results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ JSON结果已保存: {json_path}")
    
    def _print_summary(self, results: List[Dict]):
        """打印汇总结果"""
        if not results:
            return
        
        print("\n" + "=" * 60)
        print("车牌识别汇总结果")
        print("=" * 60)
        
        total_detected = len(results)
        total_recognized = sum(1 for r in results if r['plate_text'] != "未知")
        
        print(f"检测到车牌总数: {total_detected}")
        print(f"成功识别车牌数: {total_recognized}")
        print(f"识别成功率: {total_recognized/total_detected*100:.1f}%")
        
        # 颜色分布统计
        color_distribution = {}
        for result in results:
            color = result['plate_type']
            if color not in color_distribution:
                color_distribution[color] = 0
            color_distribution[color] += 1
        
        print("\n车牌颜色分布:")
        for color, count in color_distribution.items():
            percentage = count / total_detected * 100
            print(f"  {color}: {count}个 ({percentage:.1f}%)")
        
        print("\n各车牌详细结果:")
        print("-" * 60)
        for result in results:
            status = "✓" if result['plate_text'] != "未知" else "✗"
            print(f"{status} 车牌 {result['plate_id']}:")
            print(f"  号码: {result['plate_text']}")
            print(f"  类型: {result['plate_type']}")
            print(f"  检测置信度: {result['detection_confidence']:.4f}")
            print(f"  OCR置信度: {result['ocr_confidence']:.4f}")
            print()
        
        # 时间统计
        if 'total_time' in results[0]:
            total_time = sum(r['total_time'] for r in results)
            avg_time_per_plate = total_time / total_detected if total_detected > 0 else 0
            
            print(f"时间统计:")
            print(f"  总处理时间: {total_time:.4f}s")
            print(f"  平均每个车牌: {avg_time_per_plate:.4f}s")
        
        print("=" * 60)
        
    def start_camera_detection(self, 
                              camera_index: int = 0,
                              frame_width: int = 1280,
                              frame_height: int = 720,
                              fps: int = 30,
                              detection_interval: int = 10,
                              output_dir: str = "camera_results"):
        """
        启动摄像头实时检测
        
        Args:
            camera_index: 摄像头索引
            frame_width: 帧宽度
            frame_height: 帧高度
            fps: 帧率
            detection_interval: 检测间隔帧数
            output_dir: 输出目录
        """
        print("=" * 60)
        print("启动摄像头实时检测模式")
        print("=" * 60)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {camera_index}")
            return False
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 获取实际参数
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"摄像头参数: {actual_width}x{actual_height} @ {actual_fps}fps")
        print(f"检测间隔: 每{detection_interval}帧检测一次")
        print("\n控制说明:")
        print("  Q: 退出")
        print("  S: 保存当前帧")
        print("  P: 暂停/继续检测")
        print("  C: 清空检测结果")
        print("=" * 60)
        
        frame_count = 0
        detection_count = 0
        is_paused = False
        last_detections = []
        start_time = time.time()
        
        try:
            while True:
                if not is_paused:
                    # 读取帧
                    ret, frame = cap.read()
                    if not ret:
                        print("摄像头读取失败")
                        break
                    
                    frame_count += 1
                    display_frame = frame.copy()
                    
                    # 检测车牌
                    if frame_count % detection_interval == 0:
                        detection_count += 1
                        
                        # 保存临时文件用于检测
                        temp_path = f"temp_camera_frame_{detection_count}.jpg"
                        cv2.imwrite(temp_path, frame)
                        
                        try:
                            # 检测车牌
                            plates_info = self.detector.detect_all_and_rectify(temp_path)
                            
                            if plates_info:
                                for i, plate_info in enumerate(plates_info):
                                    # 处理检测到的车牌
                                    result = self._process_camera_detection(frame, plate_info, i)
                                    last_detections.append({
                                        'result': result,
                                        'frame': frame.copy(),
                                        'timestamp': time.time()
                                    })
                                    
                                    # 限制保存的数量
                                    if len(last_detections) > 10:
                                        last_detections.pop(0)
                                    
                                    # 显示结果
                                    if result.get('plate_text') != "未知":
                                        print(f"检测到车牌: {result['plate_text']} ({result.get('plate_type', '未知')})")
                        
                        except Exception as e:
                            print(f"检测出错: {e}")
                        
                        # 清理临时文件
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    
                    # 显示最近的检测结果
                    for detection in last_detections:
                        if time.time() - detection['timestamp'] < 5.0:  # 只显示5秒内的结果
                            display_frame = self._annotate_camera_frame(display_frame, detection['result'])
                    
                    # 计算并显示FPS
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # 添加统计信息
                    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"帧数: {frame_count}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"检测次数: {detection_count}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if is_paused:
                        cv2.putText(display_frame, "已暂停", (display_frame.shape[1]//2-50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                else:
                    # 暂停时显示最后一帧
                    if 'display_frame' in locals():
                        cv2.putText(display_frame, "已暂停", (display_frame.shape[1]//2-50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # 显示图像
                cv2.imshow('车牌识别 - 实时检测', display_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 退出
                    break
                elif key == ord('s'):  # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = output_path / f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"已保存截图: {filename}")
                elif key == ord('p'):  # 暂停/继续
                    is_paused = not is_paused
                    print("已暂停" if is_paused else "已继续")
                elif key == ord('c'):  # 清空检测结果
                    last_detections.clear()
                    print("已清空检测结果")
        
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"运行时出错: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 保存统计信息
            self._save_camera_statistics(frame_count, detection_count, elapsed_time, output_path)

    def _process_camera_detection(self, frame: np.ndarray, plate_info: Dict, plate_index: int) -> Dict:
        """处理摄像头检测到的车牌"""
        result = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'detection_confidence': plate_info['confidence'],
            'bbox': plate_info['bbox'],
            'plate_index': plate_index,
        }
        
        # 获取矫正后的车牌图像
        rectified_image = plate_info['rectified']
        if rectified_image is not None and rectified_image.size > 0:
            # 预处理
            try:
                preprocessed_image = rectified_image
                if self.use_preprocessing:
                    preprocessed_image, _ = self.preprocessor.preprocess_with_color_recovery(
                        rectified_image,
                        detect_plate_region=True
                    )
                
                # 保存临时文件用于OCR
                temp_path = f"temp_plate_{int(time.time())}_{plate_index}.jpg"
                cv2.imwrite(temp_path, preprocessed_image)
                
                # OCR识别
                ocr_result = get_license_plate_info(temp_path)
                
                if ocr_result:
                    plate_text, ocr_confidence, plate_type = ocr_result
                    result.update({
                        'plate_text': plate_text,
                        'ocr_confidence': ocr_confidence,
                        'plate_type': plate_type,
                        'ocr_success': True,
                    })
                else:
                    result['ocr_success'] = False
                
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
                
            except Exception as e:
                result['ocr_success'] = False
                result['error'] = str(e)
        else:
            result['ocr_success'] = False
        
        return result

    def _annotate_camera_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """在摄像头帧上标注检测结果"""
        if 'bbox' not in result:
            return frame
        
        x1, y1, x2, y2 = result['bbox']
        
        # 根据OCR结果选择颜色
        if result.get('ocr_success', False):
            color = (0, 255, 0)  # 绿色
        else:
            color = (0, 0, 255)  # 红色
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 添加文本
        text_lines = []
        if 'plate_text' in result and result['plate_text'] != "未知":
            text_lines.append(f"车牌: {result['plate_text']}")
            if 'plate_type' in result:
                text_lines.append(f"类型: {result['plate_type']}")
            if 'ocr_confidence' in result:
                text_lines.append(f"置信度: {result['ocr_confidence']:.2f}")
        
        text_lines.append(f"检测: {result['detection_confidence']:.2f}")
        
        # 计算文本位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # 计算总高度
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
        for line in text_lines:
            cv2.putText(frame, line, (bg_x1 + 5, y_offset), 
                       font, font_scale, (255, 255, 255), thickness)
            y_offset += 20
        
        return frame
    
    def _save_camera_statistics(self, frame_count, detection_count, elapsed_time, output_path):
        """保存摄像头统计信息"""
        stats_file = output_path / "statistics.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("车牌识别摄像头检测统计\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"统计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总运行时间: {elapsed_time:.2f}秒\n")
            f.write(f"总帧数: {frame_count}\n")
            f.write(f"检测次数: {detection_count}\n")
            
            if elapsed_time > 0:
                avg_fps = frame_count / elapsed_time
                f.write(f"平均FPS: {avg_fps:.2f}\n")
            
            f.write("\n系统配置:\n")
            f.write(f"预处理: {'启用' if self.use_preprocessing else '禁用'}\n")
            f.write(f"检测置信度阈值: {self.detector.conf_threshold}\n")
        
        print(f"统计信息已保存: {stats_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="车牌识别系统（支持摄像头实时检测）")
    
    # 输入源选择
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", type=str, help="处理单张图片")
    input_group.add_argument("--video", type=str, help="处理视频文件")
    input_group.add_argument("--camera", action="store_true", help="启动摄像头实时检测")
    input_group.add_argument("--batch", type=str, help="批量处理图片目录")
    
    # 摄像头选择
    parser.add_argument("--list-cameras", action="store_true",
                       help="列出所有可用摄像头")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式选择摄像头")
    parser.add_argument("--camera-info", type=int, 
                       help="显示摄像头详细信息")
    parser.add_argument("--test-camera", type=int,
                       help="测试指定摄像头")
    parser.add_argument("--find-best-camera", action="store_true",
                       help="寻找最佳摄像头")
    
    # 视频参数
    parser.add_argument("--video-start", type=float, default=0, 
                       help="视频开始时间（秒）")
    parser.add_argument("--video-duration", type=float, default=0, 
                       help="视频处理时长（秒），0表示整个视频")
    parser.add_argument("--video-output-fps", type=float, default=0, 
                       help="输出视频帧率，0表示与原视频相同")
    parser.add_argument("--max-frames", type=int, default=0, 
                       help="最大处理帧数，0表示无限制")
    parser.add_argument("--no-save", action="store_true", 
                       help="不保存处理结果（仅显示）")
    
    # 摄像头参数
    parser.add_argument("--camera-index", type=int, default=0, 
                       help="摄像头索引（默认: 0）")
    parser.add_argument("--frame-width", type=int, default=1280, 
                       help="帧宽度（默认: 1280）")
    parser.add_argument("--frame-height", type=int, default=720, 
                       help="帧高度（默认: 720）")
    parser.add_argument("--fps", type=int, default=30, 
                       help="帧率（默认: 30）")
    parser.add_argument("--detection-interval", type=int, default=10, 
                       help="检测间隔帧数（默认: 10）")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="yolov8s.pt", 
                       help="YOLO模型路径（默认: yolov8s.pt）")
    parser.add_argument("--conf", type=float, default=0.5, 
                       help="检测置信度阈值（默认: 0.5）")
    parser.add_argument("--iou", type=float, default=0.45, 
                       help="NMS IoU阈值（默认: 0.45）")
    
    # 处理参数
    parser.add_argument("--no-preprocess", action="store_true", 
                       help="禁用预处理")
    parser.add_argument("--save-all", action="store_true", 
                       help="保存所有中间结果")
    parser.add_argument("--no-display", action="store_true", 
                       help="不显示实时画面")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="results", 
                       help="输出目录（默认: results）")
    parser.add_argument("--output-format", type=str, default="jpg", 
                       choices=["jpg", "png", "bmp"], help="输出图片格式")
    parser.add_argument("--save-json", action="store_true", 
                       help="保存JSON格式结果")
    parser.add_argument("--save-txt", action="store_true", 
                       help="保存TXT格式结果")
    
    # 性能参数
    parser.add_argument("--device", type=str, default="cpu", 
                       choices=["cpu", "cuda", "mps"], 
                       help="运行设备（默认: cpu）")
    parser.add_argument("--workers", type=int, default=4, 
                       help="数据加载线程数（默认: 4）")
    parser.add_argument("--half", action="store_true", 
                       help="使用半精度推理（FP16）")
    
    # 调试参数
    parser.add_argument("--debug", action="store_true", 
                       help="启用调试模式")
    parser.add_argument("--verbose", action="store_true", 
                       help="显示详细输出")
    parser.add_argument("--version", action="version", 
                       version="车牌识别系统 v1.0.0")
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("=" * 60)
    print("车牌识别系统 v1.0.0")
    print("=" * 60)

    # 处理摄像头相关参数
    if args.list_cameras:
        if CAMERA_MANAGER_AVAILABLE:
            list_all_cameras()
        else:
            print("摄像头管理器模块不可用")
        return
    
    if args.camera_info is not None:
        if CAMERA_MANAGER_AVAILABLE:
            info = get_camera_info(args.camera_info)
            print_camera_info(info)
        else:
            print("摄像头管理器模块不可用")
        return
    
    if args.test_camera is not None:
        if CAMERA_MANAGER_AVAILABLE:
            from camera_manager import test_camera
            test_camera(args.test_camera)
        else:
            print("摄像头管理器模块不可用")
        return
    
    if args.find_best_camera:
        if CAMERA_MANAGER_AVAILABLE:
            best_camera = find_best_camera()
            if best_camera:
                print(f"最佳摄像头: 索引 {best_camera['index']}")
                print(f"分辨率: {best_camera['width']}x{best_camera['height']}")
                print(f"帧率: {best_camera['fps']:.1f}fps")
            else:
                print("未找到摄像头")
        else:
            print("摄像头管理器模块不可用")
        return

    # 检查参数组合
    if not any([args.image, args.video, args.camera, args.batch]):
        run_interactive_menu(args)
        return
    
    # 创建系统
    try:
        system = LicensePlateSystem(
            detection_model_path=args.model,
            detection_conf_threshold=args.conf,
            use_preprocessing=not args.no_preprocess
        )
    except Exception as e:
        print(f"系统初始化失败: {e}")
        print("请检查:")
        print("  1. 模型文件是否存在")
        print("  2. 依赖库是否安装")
        print("  3. 是否正确配置了CUDA（如需GPU加速）")
        return
    
    # 根据输入源处理
    if args.image:
        # 处理单张图片
        process_image_mode(system, args)
    
    elif args.camera:
        # 处理摄像头实时检测
        process_camera_mode(system, args)
    
    elif args.video:
        # 处理视频文件
        process_video_mode(system, args)
    
    elif args.batch:
        # 批量处理图片
        process_batch_mode(system, args)


def run_interactive_menu(args):
    """运行交互式菜单（可重复选择）"""
    while True:
        print("\n" + "=" * 60)
        print("车牌识别系统 - 交互模式")
        print("=" * 60)
        print("请选择模式:")
        print("  1. 处理单张图片")
        print("  2. 处理视频文件")
        print("  3. 摄像头实时检测")
        print("  4. 批量处理图片目录")
        print("  5. 摄像头管理")
        print("  6. 运行系统测试")
        print("  0. 退出")
        print("  M. 返回主菜单（重新选择模式）")
        print("  H. 显示帮助信息")
        print("=" * 60)
        
        choice = input("\n请输入选择 (0-6, M, H): ").strip().lower()
        
        if choice == "0":
            print("退出系统")
            break
        
        elif choice == "1":
            handle_image_mode(args)
            input("\n按 Enter 键返回菜单...")
        
        elif choice == "2":
            handle_video_mode(args)
            input("\n按 Enter 键返回菜单...")
        
        elif choice == "3":
            handle_camera_mode(args)
            input("\n按 Enter 键返回菜单...")
        
        elif choice == "4":
            handle_batch_mode(args)
            input("\n按 Enter 键返回菜单...")
        
        elif choice == "5":
            handle_camera_management_mode()
            continue  # 继续显示摄像头管理子菜单
        
        elif choice == "6":
            run_tests()
            input("\n按 Enter 键返回菜单...")
        
        elif choice == "m":
            continue  # 继续循环，显示主菜单
        
        elif choice == "h":
            print_help_info()
            input("\n按 Enter 键返回菜单...")
        
        else:
            print("无效选择，请重试")
            input("\n按 Enter 键返回菜单...")


def handle_camera_management_mode():
    """摄像头管理子菜单"""
    while True:
        print("\n" + "=" * 60)
        print("摄像头管理")
        print("=" * 60)
        print("请选择功能:")
        print("  1. 列出所有可用摄像头")
        print("  2. 查看摄像头详细信息")
        print("  3. 测试摄像头")
        print("  4. 交互式选择摄像头")
        print("  5. 寻找最佳摄像头")
        print("  6. 摄像头实时预览")
        print("  0. 返回主菜单")
        print("  B. 返回上一层（主菜单）")
        print("=" * 60)
        
        sub_choice = input("\n请输入选择 (0-6, B): ").strip().lower()
        
        if sub_choice == "0" or sub_choice == "b":
            print("返回主菜单")
            break
        
        elif sub_choice == "1":
            if CAMERA_MANAGER_AVAILABLE:
                list_all_cameras()
            else:
                print("摄像头管理器模块不可用")
            input("\n按 Enter 键继续...")
        
        elif sub_choice == "2":
            handle_camera_info_mode()
        
        elif sub_choice == "3":
            handle_test_camera_mode()
        
        elif sub_choice == "4":
            handle_interactive_camera_selection()
        
        elif sub_choice == "5":
            handle_find_best_camera_mode()
            input("\n按 Enter 键继续...")
        
        elif sub_choice == "6":
            handle_camera_preview_mode()
        
        else:
            print("无效选择，请重试")
            input("\n按 Enter 键继续...")


def handle_image_mode(args):
    """处理图片模式"""
    print("\n=== 图片处理模式 ===")
    
    while True:
        image_path = input("请输入图片路径 (或输入 'back' 返回): ").strip()
        
        if image_path.lower() == 'back':
            print("返回主菜单")
            break
        
        if not os.path.exists(image_path):
            print(f"错误：图片不存在 {image_path}")
            print("请检查路径是否正确")
            continue
        
        # 询问参数
        use_preprocess = input("启用预处理？(y/n, 默认y): ").strip().lower()
        output_dir = input("输出目录 (默认: results): ").strip()
        
        # 设置参数
        args.image = image_path
        args.no_preprocess = (use_preprocess == 'n')
        args.output_dir = output_dir if output_dir else "results"
        
        # 创建系统
        try:
            system = LicensePlateSystem(
                detection_model_path=args.model,
                detection_conf_threshold=args.conf,
                use_preprocessing=not args.no_preprocess
            )
            
            # 处理图片
            process_image_mode(system, args)
            
        except Exception as e:
            print(f"处理失败: {e}")
        
        # 询问是否继续处理其他图片
        another = input("\n是否处理另一张图片？(y/n): ").strip().lower()
        if another != 'y':
            break


def handle_video_mode(args):
    """处理视频模式"""
    print("\n=== 视频处理模式 ===")
    
    while True:
        video_path = input("请输入视频文件路径 (或输入 'back' 返回): ").strip()
        
        if video_path.lower() == 'back':
            print("返回主菜单")
            break
        
        if not os.path.exists(video_path):
            print(f"错误：视频文件不存在 {video_path}")
            print("请检查路径是否正确")
            continue
        
        # 询问参数
        print("\n视频处理参数:")
        start_time = input("开始时间(秒，默认0): ").strip()
        duration = input("处理时长(秒，默认整个视频): ").strip()
        detection_interval = input("检测间隔帧数(默认10): ").strip()
        output_dir = input("输出目录 (默认: results/video): ").strip()
        
        # 设置参数
        args.video = video_path
        if start_time:
            args.video_start = float(start_time)
        if duration:
            args.video_duration = float(duration)
        if detection_interval:
            args.detection_interval = int(detection_interval)
        args.output_dir = output_dir if output_dir else "results/video"
        
        # 创建系统
        try:
            system = LicensePlateSystem(
                detection_model_path=args.model,
                detection_conf_threshold=args.conf,
                use_preprocessing=not args.no_preprocess
            )
            
            # 处理视频
            process_video_mode(system, args)
            
        except Exception as e:
            print(f"处理失败: {e}")
        
        # 询问是否继续处理其他视频
        another = input("\n是否处理另一个视频？(y/n): ").strip().lower()
        if another != 'y':
            break


def handle_camera_mode(args):
    """处理摄像头模式（实时车牌检测）"""
    print("\n=== 摄像头实时检测模式 ===")
    
    # 交互式选择摄像头
    if CAMERA_MANAGER_AVAILABLE:
        use_interactive = input("交互式选择摄像头？(y/n): ").strip().lower()
        if use_interactive == 'y':
            camera_info = select_camera_interactive()
            if camera_info is None:
                print("摄像头选择取消")
                return
            args.camera_index = camera_info['index']
            if 'width' in camera_info:
                args.frame_width = camera_info['width']
            if 'height' in camera_info:
                args.frame_height = camera_info['height']
            if 'fps' in camera_info:
                args.fps = int(camera_info['fps'])
            if 'interval' in camera_info:
                args.detection_interval = camera_info['interval']
        else:
            # 手动输入参数
            camera_idx = input("摄像头索引(默认0): ").strip()
            args.camera_index = int(camera_idx) if camera_idx else 0
    else:
        print("摄像头管理器模块不可用，使用默认参数")
    
    # 询问其他参数
    print("\n摄像头参数设置:")
    width = input("帧宽度(默认1280): ").strip()
    height = input("帧高度(默认720): ").strip()
    fps = input("帧率(默认30): ").strip()
    interval = input("检测间隔帧数(默认10): ").strip()
    output_dir = input("输出目录 (默认: results/camera): ").strip()
    
    # 设置参数
    if width:
        args.frame_width = int(width)
    if height:
        args.frame_height = int(height)
    if fps:
        args.fps = int(fps)
    if interval:
        args.detection_interval = int(interval)
    args.output_dir = output_dir if output_dir else "results/camera"
    args.camera = True
    
    # 创建系统
    try:
        system = LicensePlateSystem(
            detection_model_path=args.model,
            detection_conf_threshold=args.conf,
            use_preprocessing=not args.no_preprocess
        )
        
        # 处理摄像头
        process_camera_mode(system, args)
        
    except Exception as e:
        print(f"处理失败: {e}")


def handle_batch_mode(args):
    """处理批量模式"""
    print("\n=== 批量处理模式 ===")
    
    while True:
        batch_dir = input("请输入图片目录路径 (或输入 'back' 返回): ").strip()
        
        if batch_dir.lower() == 'back':
            print("返回主菜单")
            break
        
        if not os.path.exists(batch_dir):
            print(f"错误：目录不存在 {batch_dir}")
            print("请检查路径是否正确")
            continue
        
        # 询问参数
        output_dir = input("输出目录 (默认: results/batch): ").strip()
        
        # 设置参数
        args.batch = batch_dir
        args.output_dir = output_dir if output_dir else "results/batch"
        
        # 创建系统
        try:
            system = LicensePlateSystem(
                detection_model_path=args.model,
                detection_conf_threshold=args.conf,
                use_preprocessing=not args.no_preprocess
            )
            
            # 批量处理
            process_batch_mode(system, args)
            
        except Exception as e:
            print(f"处理失败: {e}")
        
        # 询问是否继续处理其他目录
        another = input("\n是否处理另一个目录？(y/n): ").strip().lower()
        if another != 'y':
            break


def handle_camera_info_mode():
    """处理摄像头信息查询模式"""
    print("\n=== 摄像头信息查询 ===")
    
    if not CAMERA_MANAGER_AVAILABLE:
        print("摄像头管理器模块不可用")
        return
    
    while True:
        camera_idx = input("请输入摄像头索引 (或输入 'list' 列出所有, 'back' 返回): ").strip().lower()
        
        if camera_idx == 'back':
            print("返回上一级")
            break
        
        elif camera_idx == 'list':
            list_all_cameras()
            continue
        
        try:
            idx = int(camera_idx)
            info = get_camera_info(idx)
            print_camera_info(info)
            
            # 询问是否测试此摄像头
            test = input("\n是否测试此摄像头？(y/n): ").strip().lower()
            if test == 'y':
                if CAMERA_MANAGER_AVAILABLE:
                    from camera_manager import test_camera
                    test_camera(idx)
        except ValueError:
            print("请输入有效的数字索引")


def handle_test_camera_mode():
    """处理摄像头测试模式"""
    print("\n=== 摄像头测试 ===")
    
    if not CAMERA_MANAGER_AVAILABLE:
        print("摄像头管理器模块不可用")
        return
    
    while True:
        camera_idx = input("请输入摄像头索引 (或输入 'back' 返回): ").strip().lower()
        
        if camera_idx == 'back':
            print("返回上一级")
            break
        
        try:
            idx = int(camera_idx)
            from camera_manager import test_camera
            test_camera(idx)
        except ValueError:
            print("请输入有效的数字索引")


def handle_interactive_camera_selection():
    """处理交互式摄像头选择"""
    print("\n=== 交互式摄像头选择 ===")
    
    if not CAMERA_MANAGER_AVAILABLE:
        print("摄像头管理器模块不可用")
        return
    
    camera_info = select_camera_interactive()
    if camera_info is not None:
        print(f"\n已选择摄像头:")
        print(f"  索引: {camera_info['index']}")
        print(f"  名称: {camera_info.get('name', '未知')}")
        if 'width' in camera_info and 'height' in camera_info:
            print(f"  分辨率: {camera_info['width']}x{camera_info['height']}")
        if 'fps' in camera_info:
            print(f"  帧率: {camera_info['fps']:.1f}fps")
        
        # 询问是否立即开始车牌检测
        start_detection = input("\n是否立即开始车牌检测？(y/n): ").strip().lower()
        if start_detection == 'y':
            # 创建临时参数对象
            class Args:
                pass
            args = Args()
            args.camera_index = camera_info['index']
            args.frame_width = camera_info.get('width', 1280)
            args.frame_height = camera_info.get('height', 720)
            args.fps = int(camera_info.get('fps', 30))
            args.detection_interval = camera_info.get('interval', 10)
            args.output_dir = "results/camera"
            args.model = "yolov8s.pt"
            args.conf = 0.5
            args.no_preprocess = False
            
            # 创建系统并开始检测
            try:
                system = LicensePlateSystem(
                    detection_model_path=args.model,
                    detection_conf_threshold=args.conf,
                    use_preprocessing=not args.no_preprocess
                )
                process_camera_mode(system, args)
            except Exception as e:
                print(f"启动车牌检测失败: {e}")


def handle_find_best_camera_mode():
    """处理寻找最佳摄像头模式"""
    print("\n=== 寻找最佳摄像头 ===")
    
    if not CAMERA_MANAGER_AVAILABLE:
        print("摄像头管理器模块不可用")
        return
    
    best_camera = find_best_camera()
    if best_camera:
        print(f"最佳摄像头: 索引 {best_camera['index']}")
        print(f"名称: {best_camera.get('name', '未知')}")
        print(f"分辨率: {best_camera['width']}x{best_camera['height']}")
        print(f"帧率: {best_camera['fps']:.1f}fps")
        
        # 询问是否测试此摄像头
        test = input("\n是否测试此摄像头？(y/n): ").strip().lower()
        if test == 'y':
            from camera_manager import test_camera
            test_camera(best_camera['index'])
    else:
        print("未找到摄像头")


def handle_camera_preview_mode():
    """摄像头实时预览模式"""
    print("\n=== 摄像头实时预览 ===")
    
    if not CAMERA_MANAGER_AVAILABLE:
        print("摄像头管理器模块不可用")
        # 尝试使用默认方法
        camera_idx = input("请输入摄像头索引 (默认0): ").strip()
        camera_idx = int(camera_idx) if camera_idx else 0
        test_single_camera(camera_idx)
        return
    
    # 先列出所有摄像头
    list_all_cameras()
    
    # 选择摄像头
    camera_idx = input("\n请输入要预览的摄像头索引 (或输入 'back' 返回): ").strip().lower()
    
    if camera_idx == 'back':
        print("返回上一级")
        return
    
    try:
        idx = int(camera_idx)
        from camera_manager import test_camera
        test_camera(idx)
    except ValueError:
        print("请输入有效的数字索引")


def print_help_info():
    """打印帮助信息"""
    print("\n" + "=" * 60)
    print("帮助信息")
    print("=" * 60)
    print("系统功能:")
    print("  1. 图片处理 - 识别单张图片中的车牌")
    print("  2. 视频处理 - 识别视频文件中的车牌")
    print("  3. 实时检测 - 通过摄像头实时检测车牌")
    print("  4. 批量处理 - 处理目录中的所有图片")
    print("  5. 摄像头管理 - 查看和管理摄像头设备")
    print("  6. 系统测试 - 运行系统诊断和测试")
    print()
    print("摄像头管理功能:")
    print("  1. 列出所有可用摄像头")
    print("  2. 查看摄像头详细信息")
    print("  3. 测试摄像头")
    print("  4. 交互式选择摄像头")
    print("  5. 寻找最佳摄像头")
    print("  6. 摄像头实时预览")
    print()
    print("常用命令:")
    print("  python main.py --image test.jpg")
    print("  python main.py --video test.mp4")
    print("  python main.py --camera")
    print("  python main.py --batch images/")
    print()
    print("更多选项使用: python main.py --help")
    print("=" * 60)


def process_image_mode(system, args):
    """处理图片模式"""
    print(f"处理图片: {args.image}")
    
    if not os.path.exists(args.image):
        print(f"错误：图片不存在 {args.image}")
        return
    
    try:
        results = system.process_image(
            image_path=args.image,
            save_results=True,
            output_dir=args.output_dir
        )
        
        if results:
            print(f"\n✓ 处理完成！检测到 {len(results)} 个车牌")
            
            # 显示结果
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['plate_text']} ({result['plate_type']}) "
                      f"置信度: {result['ocr_confidence']:.2f}")
        else:
            print("未检测到车牌")
            
    except Exception as e:
        print(f"处理图片时出错: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


def process_camera_mode(system, args):
    """处理摄像头模式"""
    print("启动摄像头实时检测...")
    
    try:
        # 检查摄像头可用性
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {args.camera_index}")
            
            # 尝试自动检测可用摄像头
            print("尝试自动检测可用摄像头...")
            for i in range(3):
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    print(f"找到可用摄像头: 索引 {i}")
                    args.camera_index = i
                    test_cap.release()
                    cap = cv2.VideoCapture(i)
                    break
                test_cap.release()
            
            if not cap.isOpened():
                print("未找到可用摄像头")
                return
        
        cap.release()
        
        # 使用实时检测器
        print("=" * 60)
        print("摄像头参数:")
        print(f"  索引: {args.camera_index}")
        print(f"  分辨率: {args.frame_width}x{args.frame_height}")
        print(f"  帧率: {args.fps}fps")
        print(f"  检测间隔: 每{args.detection_interval}帧检测一次")
        print("=" * 60)
        
        # 创建输出目录
        camera_output_dir = os.path.join(args.output_dir, "camera")
        os.makedirs(camera_output_dir, exist_ok=True)
        
        # 启动摄像头检测
        system.start_camera_detection(
            camera_index=args.camera_index,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            fps=args.fps,
            detection_interval=args.detection_interval,
            output_dir=camera_output_dir
        )
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"摄像头检测时出错: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


def test_single_camera(camera_index, test_duration=5):
    """测试单个摄像头"""
    print(f"\n测试摄像头 {camera_index}...")
    print("按 'q' 键退出测试")
    
    cap = None
    try:
        # 尝试不同的后端
        backends_to_try = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
        
        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    print(f"  使用后端: {backend}")
                    break
            except:
                continue
        
        if not cap or not cap.isOpened():
            print("  无法打开摄像头")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 获取实际参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  实际参数: {width}x{height} @ {fps}fps")
        print("  正在显示摄像头画面...")
        
        import time
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                print("  无法读取帧")
                break
            
            frame_count += 1
            
            # 显示帧
            cv2.putText(frame, f"Camera {camera_index}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{width}x{height}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f'Camera Test - Index {camera_index}', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            actual_fps = frame_count / elapsed_time
            print(f"  实际帧率: {actual_fps:.1f} fps")
            print(f"  总帧数: {frame_count}")
        
        print("  测试完成")
        
    except Exception as e:
        print(f"  测试出错: {e}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


def process_video_mode(system, args):
    """处理视频模式"""
    print(f"处理视频: {args.video}")
    
    if not os.path.exists(args.video):
        print(f"错误：视频文件不存在 {args.video}")
        return
    
    if not VIDEO_PROCESSOR_AVAILABLE:
        print("错误：视频处理器模块未找到，请确保 video_processor.py 存在")
        print("您可以继续使用其他功能，或者修复导入问题")
        response = input("是否继续？(y/n): ").strip().lower()
        if response != 'y':
            return
    
    try:
        # 创建视频处理器
        if VIDEO_PROCESSOR_AVAILABLE:
            video_processor = VideoLicensePlateProcessor.from_system(system)
        else:
            # 使用简单的视频处理替代方案
            video_processor = None
            print("警告：使用简化视频处理模式")
            
            # 创建输出目录
            video_output_dir = os.path.join(args.output_dir, "video")
            os.makedirs(video_output_dir, exist_ok=True)
            
            # 打开视频文件
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                print(f"无法打开视频文件: {args.video}")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"视频信息: {total_frames}帧, {fps}fps")
            
            # 处理视频
            frame_idx = 0
            detection_count = 0
            unique_plates = set()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # 每30帧处理一次（或自定义间隔）
                if frame_idx % (args.detection_interval if hasattr(args, 'detection_interval') else 30) == 0:
                    # 保存临时帧
                    temp_path = f"temp_frame_{frame_idx}.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # 检测车牌
                    plates_info = system.detector.detect_all_and_rectify(temp_path)
                    
                    if plates_info:
                        for plate_info in plates_info:
                            detection_count += 1
                            
                            # 处理车牌
                            result = system._process_camera_detection(frame, plate_info, detection_count)
                            
                            if result.get('plate_text') != "未知":
                                unique_plates.add(result['plate_text'])
                                
                                # 保存结果帧
                                if not hasattr(args, 'no_save') or not args.no_save:
                                    result_frame = system._annotate_camera_frame(frame.copy(), result)
                                    cv2.imwrite(f"{video_output_dir}/frame_{frame_idx}_plate_{detection_count}.jpg", result_frame)
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # 显示进度
                if frame_idx % 100 == 0:
                    print(f"已处理 {frame_idx}/{total_frames} 帧")
            
            cap.release()
            
            print(f"\n视频处理完成:")
            print(f"  总帧数: {total_frames}")
            print(f"  检测到车牌数: {detection_count}")
            print(f"  唯一车牌数: {len(unique_plates)}")
            print(f"  输出目录: {video_output_dir}")
            
            # 保存汇总结果
            with open(f"{video_output_dir}/summary.txt", "w", encoding="utf-8") as f:
                f.write(f"视频文件: {args.video}\n")
                f.write(f"总帧数: {total_frames}\n")
                f.write(f"检测到车牌数: {detection_count}\n")
                f.write(f"唯一车牌数: {len(unique_plates)}\n")
                f.write(f"检测到的车牌: {', '.join(unique_plates)}\n")
            
            return
        
        # 使用视频处理器
        result = video_processor.process_video_file(
            video_path=args.video,
            output_dir=args.output_dir,
            detection_interval=args.detection_interval if hasattr(args, 'detection_interval') else 10,
            save_results=not args.no_save if hasattr(args, 'no_save') else True,
            save_frames=args.save_all if hasattr(args, 'save_all') else False,
            save_json=args.save_json if hasattr(args, 'save_json') else True,
            display=not args.no_display if hasattr(args, 'no_display') else True,
            start_time=args.video_start,
            duration=args.video_duration,
            output_fps=args.video_output_fps,
            show_progress=True,
            max_frames=args.max_frames
        )
        
        if result.get('success', False):
            print(f"\n✓ 视频处理完成！")
            print(f"  总帧数: {result['total_frames']}")
            print(f"  处理帧数: {result['processed_frames']}")
            print(f"  检测到车牌数: {result['detection_count']}")
            print(f"  唯一车牌数: {result['unique_plates']}")
            print(f"  输出目录: {result['output_dir']}")
            
            # 显示检测到的车牌
            if result['detections']:
                print(f"\n检测到的车牌:")
                for detection in result['detections'][:10]:  # 只显示前10个
                    plate_text = detection.get('plate_text', '未知')
                    plate_type = detection.get('plate_type', '未知')
                    frame_idx = detection.get('frame_index', 0)
                    conf = detection.get('ocr_confidence', 0)
                    print(f"  第{frame_idx}帧: {plate_text} ({plate_type}) 置信度: {conf:.2f}")
                
                if len(result['detections']) > 10:
                    print(f"  ... 还有{len(result['detections']) - 10}个检测结果")
        else:
            print(f"\n✗ 视频处理失败: {result.get('error', '未知错误')}")
        
    except Exception as e:
        print(f"处理视频时出错: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


def process_batch_mode(system, args):
    """批量处理模式"""
    print(f"批量处理目录: {args.batch}")
    
    if not os.path.exists(args.batch):
        print(f"错误：目录不存在 {args.batch}")
        return
    
    # 收集所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(args.batch).glob(f"*{ext}"))
        image_files.extend(Path(args.batch).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 创建输出目录
    batch_output_dir = os.path.join(args.output_dir, "batch")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # 批量处理
    total_plates = 0
    success_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {image_file.name}")
        
        try:
            # 为每张图片创建子目录
            image_output_dir = os.path.join(batch_output_dir, image_file.stem)
            os.makedirs(image_output_dir, exist_ok=True)
            
            # 处理图片
            results = system.process_image(
                image_path=str(image_file),
                save_results=True,
                output_dir=image_output_dir
            )
            
            if results:
                success_count += 1
                total_plates += len(results)
                print(f"  ✓ 检测到 {len(results)} 个车牌")
            else:
                print(f"  ✗ 未检测到车牌")
                
        except Exception as e:
            print(f"  处理失败: {e}")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("批量处理完成")
    print("=" * 60)
    print(f"处理图片数: {len(image_files)}")
    print(f"成功检测图片数: {success_count}")
    print(f"检测到车牌总数: {total_plates}")
    print(f"输出目录: {batch_output_dir}")


def run_tests():
    """运行系统测试"""
    print("运行系统测试...")
    
    # 检查依赖
    print("\n1. 检查依赖库:")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"    设备: CPU")
    except:
        print("  ✗ PyTorch: 未安装")
    
    try:
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except:
        print("  ✗ OpenCV: 未安装")
    
    # 检查模型文件
    print("\n2. 检查模型文件:")
    model_files = ["yolov8s.pt", "yolov8n.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024*1024)  # 转换为MB
            print(f"  ✓ {model_file}: 存在 ({file_size:.1f} MB)")
            found_models.append(model_file)
        else:
            print(f"  ✗ {model_file}: 不存在")
    
    if found_models:
        print(f"    找到 {len(found_models)} 个模型文件")
    else:
        print("    警告: 未找到任何模型文件")
        print("    请从以下链接下载:")
        print("    https://github.com/ultralytics/ultralytics")
    
    # 检查摄像头 - 改进版本
    print("\n3. 检查摄像头:")
    available_cameras = []
    max_check_index = 10  # 最多检查10个摄像头索引
    
    # 定义要尝试的后端列表
    backends = [
        cv2.CAP_DSHOW,    # DirectShow (Windows)
        cv2.CAP_MSMF,     # Microsoft Media Foundation (Windows)
        cv2.CAP_V4L2,     # Video4Linux (Linux)
        cv2.CAP_ANY,      # 自动选择
    ]
    
    backend_names = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_ANY: "AUTO"
    }
    
    print("  正在扫描摄像头...")
    
    for i in range(max_check_index):
        camera_found = False
        best_backend = None
        camera_info = None
        
        # 尝试不同的后端
        for backend in backends:
            try:
                # 使用try-except避免崩溃
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # 尝试读取一帧
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # 成功读取到帧
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        fps = int(cap.get(cv2.CAP_PROP_FPS) or 0)
                        
                        # 尝试多次读取以获取更准确的fps
                        if fps == 0:
                            import time
                            start_time = time.time()
                            frame_count = 0
                            for _ in range(30):  # 读取30帧计算fps
                                ret, _ = cap.read()
                                if ret:
                                    frame_count += 1
                            end_time = time.time()
                            if end_time - start_time > 0:
                                fps = int(frame_count / (end_time - start_time))
                        
                        camera_info = {
                            'index': i,
                            'backend': backend,
                            'backend_name': backend_names.get(backend, str(backend)),
                            'width': width if width > 0 else "未知",
                            'height': height if height > 0 else "未知",
                            'fps': fps if fps > 0 else "未知",
                            'working': True
                        }
                        
                        best_backend = backend
                        camera_found = True
                        
                        # 释放摄像头
                        cap.release()
                        break  # 找到可用后端就停止尝试
                    else:
                        cap.release()
            except Exception as e:
                # 忽略特定后端错误，继续尝试其他后端
                pass
        
        if camera_found and camera_info:
            # 获取摄像头名称（如果可用）
            camera_name = f"摄像头 {i}"
            try:
                # 尝试使用最后一个成功的后端再次打开以获取名称
                if best_backend:
                    cap = cv2.VideoCapture(i, best_backend)
                    if cap.isOpened():
                        # 在某些系统上可以获取设备名称
                        try:
                            # 尝试通过属性获取名称
                            name = cap.get(cv2.CAP_PROP_BACKEND_NAME)
                            if name:
                                camera_name = f"摄像头 {i} ({name})"
                        except:
                            pass
                        cap.release()
            except:
                pass
            
            print(f"  ✓ {camera_name} ({camera_info['backend_name']}) - "
                  f"{camera_info['width']}x{camera_info['height']} @ {camera_info['fps']}fps: 可用")
            
            available_cameras.append({
                'index': i,
                'name': camera_name,
                'backend': camera_info['backend_name'],
                'width': camera_info['width'],
                'height': camera_info['height'],
                'fps': camera_info['fps']
            })
        else:
            # 如果前几个都没有摄像头，提前结束
            if i > 3 and len(available_cameras) == 0:
                # 检查前4个都没有摄像头，就不再继续检查太多
                if i >= 5:
                    print(f"  扫描到索引 {i}，未发现更多摄像头")
                    break
    
    if available_cameras:
        print(f"\n    找到 {len(available_cameras)} 个可用摄像头")
        print("\n    可用摄像头列表:")
        for cam in available_cameras:
            print(f"      索引 {cam['index']}: {cam['name']}")
            print(f"        后端: {cam['backend']}")
            print(f"        分辨率: {cam['width']}x{cam['height']}")
            print(f"        帧率: {cam['fps']}fps")
        
        # 提供一个简单的摄像头测试功能
        print("\n    快速摄像头测试（输入 'q' 退出）:")
        if len(available_cameras) > 0:
            test_cam = input(f"    是否测试摄像头 {available_cameras[0]['index']}? (y/n): ").strip().lower()
            if test_cam == 'y':
                test_single_camera(available_cameras[0]['index'])
    else:
        print("    未找到可用摄像头")
        print("\n    可能的原因:")
        print("      1. 摄像头未连接或未正确安装")
        print("      2. 摄像头驱动程序未正确安装")
        print("      3. 摄像头被其他程序占用")
        print("      4. 权限问题（Linux/Mac需要权限）")
        print("\n    解决方法:")
        print("      1. 检查摄像头物理连接")
        print("      2. 重启电脑")
        print("      3. 更新摄像头驱动程序")
        print("      4. 关闭其他使用摄像头的程序")
        print("      5. 在Linux/Mac上尝试: sudo chmod 666 /dev/video*")
        
        # 检查是否有虚拟摄像头软件
        print("\n    虚拟摄像头选项:")
        print("      1. OBS Studio (可以创建虚拟摄像头)")
        print("      2. ManyCam")
        print("      3. CamTwist (Mac)")
    
    # 检查测试图片
    print("\n4. 检查测试图片:")
    test_images = ["test.jpg", "test1.jpg", "test2.jpg", "car.jpg", "license_plate.jpg", "test_plate.jpg"]
    found_images = []
    
    for test_image in test_images:
        if os.path.exists(test_image):
            file_size = os.path.getsize(test_image) / 1024  # 转换为KB
            print(f"  ✓ {test_image}: 存在 ({file_size:.1f} KB)")
            found_images.append(test_image)
        else:
            print(f"  ✗ {test_image}: 不存在")
    
    if found_images:
        print(f"    找到 {len(found_images)} 个测试图片")
        
        # 显示一些示例图片信息
        if len(found_images) > 0:
            print("\n    示例图片预览:")
            for img_file in found_images[:3]:  # 只显示前3个
                try:
                    img = cv2.imread(img_file)
                    if img is not None:
                        h, w = img.shape[:2]
                        print(f"      {img_file}: {w}x{h} 像素")
                except:
                    pass
    else:
        print("    警告: 未找到任何测试图片")
        print("    建议创建以下测试文件:")
        print("      test.jpg - 用于测试")
        print("      test_plate.jpg - 车牌测试")
    
    # 检查测试视频
    print("\n5. 检查测试视频:")
    test_videos = ["test_video.mp4", "test_video.avi", "test_video.mov", "car_video.mp4", "test.mp4", "sample.mp4"]
    found_videos = []
    
    for test_video in test_videos:
        if os.path.exists(test_video):
            file_size = os.path.getsize(test_video) / (1024*1024)  # 转换为MB
            # 尝试打开视频
            cap = cv2.VideoCapture(test_video)
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if frames > 0 and fps > 0:
                    duration = frames / fps
                    print(f"  ✓ {test_video}: 可用 ({frames}帧, {fps}fps, {duration:.1f}秒, {file_size:.1f} MB)")
                    found_videos.append(test_video)
                else:
                    print(f"  ? {test_video}: 可以打开但无法获取信息")
                cap.release()
            else:
                print(f"  ✗ {test_video}: 存在但无法打开")
        else:
            print(f"  ✗ {test_video}: 不存在")
    
    if found_videos:
        print(f"    找到 {len(found_videos)} 个测试视频")
    else:
        print("    警告: 未找到任何测试视频")
        print("    可以录制或下载一些测试视频")
    
    # 检查系统配置
    print("\n6. 系统配置:")
    
    # Python版本
    print(f"  Python版本: {sys.version.split()[0]}")
    
    # 操作系统
    import platform
    print(f"  操作系统: {platform.system()} {platform.release()}")
    
    # 内存信息
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  内存: {memory.total // (1024**3)} GB (可用: {memory.available // (1024**3)} GB)")
    except:
        print("  内存信息: 需要安装psutil库")
    
    # 磁盘空间
    try:
        disk = psutil.disk_usage('.')
        print(f"  磁盘空间: {disk.free // (1024**3)} GB 可用 / {disk.total // (1024**3)} GB 总量")
    except:
        pass
    
    # 性能测试
    print("\n7. 性能测试:")
    
    # OpenCV 基本操作测试
    try:
        import time
        # 创建一个测试图像
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试图像处理速度
        start_time = time.time()
        for _ in range(10):
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2_time = (time.time() - start_time) / 10
        print(f"  OpenCV处理速度: {cv2_time*1000:.1f} ms/图像")
    except:
        print("  OpenCV测试失败")
    
    # NumPy 测试
    try:
        start_time = time.time()
        for _ in range(100):
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            c = np.dot(a, b)
        numpy_time = (time.time() - start_time) / 100
        print(f"  NumPy计算速度: {numpy_time*1000:.1f} ms/矩阵乘法")
    except:
        print("  NumPy测试失败")
    
    # 最终总结
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    # 提供建议
    if len(available_cameras) == 0:
        print("建议:")
        print("  1. 连接摄像头或使用虚拟摄像头软件")
        print("  2. 运行命令测试摄像头: python main.py --list-cameras")
    
    if len(found_models) == 0:
        print("警告:")
        print("  未找到模型文件，系统可能无法工作")
        print("  请下载模型文件到当前目录:")
        print("  https://github.com/ultralytics/ultralytics")
    
    if len(found_images) == 0:
        print("提示:")
        print("  可以放置一些测试图片在当前目录")
        print("  或使用摄像头实时检测功能")
    
    print("\n系统准备状态:")
    status_ok = "✓" if len(found_models) > 0 else "✗"
    print(f"  {status_ok} 模型文件: {'已准备' if len(found_models) > 0 else '未准备'}")
    
    status_ok = "✓" if len(available_cameras) > 0 else "⚠"
    print(f"  {status_ok} 摄像头: {f'找到 {len(available_cameras)} 个' if available_cameras else '未找到'}")
    
    status_ok = "✓" if len(found_images) > 0 else "⚠"
    print(f"  {status_ok} 测试图片: {f'找到 {len(found_images)} 个' if found_images else '未找到'}")
    
    print("\n可以运行以下命令开始:")
    if len(found_images) > 0:
        print(f"  python main.py --image {found_images[0]}")
    if len(available_cameras) > 0:
        print(f"  python main.py --camera --camera-index {available_cameras[0]['index']}")
    print("  python main.py  # 进入交互模式")
    print("=" * 60)


if __name__ == "__main__":
    main()