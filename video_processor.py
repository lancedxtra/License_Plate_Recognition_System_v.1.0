# video_processor.py
import cv2
import numpy as np
import time
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import traceback

# 导入三个核心模块
try:
    from license_plate_detection import LicensePlateDetector
    from license_plate_preprocessor import LicensePlatePreprocessor
    from license_plate_ocr_engine import get_license_plate_info
    
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入模块: {e}")
    MODULES_AVAILABLE = False


class VideoLicensePlateProcessor:
    """视频车牌处理器 - 处理视频文件中的车牌识别"""
    
    def __init__(self, 
                 detector_config: Dict = None,
                 preprocessor_config: Dict = None,
                 conf_threshold: float = 0.5,
                 use_preprocessing: bool = True):
        """
        初始化视频处理器
        
        Args:
            detector_config: 检测器配置
            preprocessor_config: 预处理器配置
            conf_threshold: 检测置信度阈值
            use_preprocessing: 是否使用预处理
        """
        print("初始化视频车牌处理器...")
        
        if not MODULES_AVAILABLE:
            raise ImportError("必要的模块未找到，请确保 license_plate_detection.py, "
                            "license_plate_preprocessor.py, license_plate_ocr_engine.py 存在")
        
        # 使用提供的配置或默认配置
        detector_config = detector_config or {}
        preprocessor_config = preprocessor_config or {}
        
        # 初始化检测器
        print("  1. 加载车牌检测器...")
        self.detector = LicensePlateDetector(
            model_path=detector_config.get('model_path', 'yolov8s.pt'),
            conf_threshold=conf_threshold
        )
        
        # 初始化预处理器
        print("  2. 加载车牌预处理器...")
        self.preprocessor = LicensePlatePreprocessor(
            target_size=preprocessor_config.get('target_size', (640, 480))
        )
        
        self.conf_threshold = conf_threshold
        self.use_preprocessing = use_preprocessing
        
        # 检测统计
        self.total_frames = 0
        self.processed_frames = 0
        self.frame_count = 0
        self.all_detections = []
        self.detection_history = {}  # 车牌追踪历史
        self.plate_counter = 0
        
        print("✓ 视频处理器初始化完成")
    
    @classmethod
    def from_system(cls, system):
        """
        从LicensePlateSystem创建视频处理器
        
        Args:
            system: LicensePlateSystem实例
            
        Returns:
            VideoLicensePlateProcessor实例
        """
        return cls(
            detector_config={'model_path': 'yolov8s.pt'},
            preprocessor_config={'target_size': (640, 480)},
            conf_threshold=system.detector.conf_threshold,
            use_preprocessing=system.use_preprocessing
        )
    
    def process_video_file(self,
                          video_path: str,
                          output_dir: str = "results",
                          detection_interval: int = 10,
                          save_results: bool = True,
                          save_frames: bool = False,
                          save_json: bool = True,
                          display: bool = True,
                          start_time: float = 0,
                          duration: float = 0,
                          output_fps: int = 0,
                          show_progress: bool = True,
                          max_frames: int = 0) -> Dict:
        """
        处理视频文件
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            detection_interval: 检测间隔帧数
            save_results: 是否保存结果
            save_frames: 是否保存中间帧
            save_json: 是否保存JSON结果
            display: 是否显示处理过程
            start_time: 开始时间（秒）
            duration: 处理时长（秒，0表示全部）
            output_fps: 输出视频帧率（0表示保持原样）
            show_progress: 是否显示进度
            max_frames: 最大处理帧数（0表示无限制）
            
        Returns:
            处理结果字典
        """
        print(f"开始处理视频: {video_path}")
        print("-" * 60)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return {"error": f"无法打开视频文件 {video_path}", "success": False}
        
        # 获取视频信息
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = original_fps if original_fps > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算开始帧和结束帧
        start_frame = int(start_time * fps)
        if duration > 0:
            end_frame = min(self.total_frames, start_frame + int(duration * fps))
        else:
            end_frame = self.total_frames
        
        # 应用最大帧数限制
        if max_frames > 0:
            end_frame = min(end_frame, start_frame + max_frames)
        
        print(f"视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps:.1f}fps")
        print(f"  总帧数: {self.total_frames}")
        print(f"  处理范围: 第{start_frame}帧到第{end_frame}帧 (共{end_frame-start_frame}帧)")
        print(f"  检测间隔: 每{detection_interval}帧检测一次")
        
        # 创建输出目录
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(video_path).stem
            self.output_dir = Path(output_dir) / "video" / f"{video_name}_{timestamp}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.frames_dir = self.output_dir / "frames"
            self.plates_dir = self.output_dir / "plates"
            self.frames_dir.mkdir(exist_ok=True)
            self.plates_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None
        
        # 创建视频写入器（用于保存处理后的视频）
        out = None
        if save_results:
            output_video_path = self.output_dir / "output_video.mp4"
            output_fps_val = output_fps if output_fps > 0 else fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, output_fps_val, (width, height))
        
        # 跳转到开始帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 处理每一帧
        start_process_time = time.time()
        frame_index = start_frame
        
        try:
            while frame_index < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理当前帧
                processed_frame, frame_detections = self._process_frame(
                    frame, frame_index, fps, detection_interval
                )
                
                # 保存处理后的帧
                if save_results and out is not None:
                    out.write(processed_frame)
                    
                    # 保存关键帧
                    if save_frames and frame_detections:
                        frame_filename = self.frames_dir / f"frame_{frame_index:06d}.jpg"
                        cv2.imwrite(str(frame_filename), processed_frame)
                
                # 显示处理过程
                if display:
                    # 显示进度条
                    if show_progress:
                        progress_text = f"帧: {frame_index}/{end_frame}"
                        cv2.putText(processed_frame, progress_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow('视频处理 - 车牌识别', processed_frame)
                    
                    # 控制播放速度
                    delay = max(1, int(1000 / fps) - 10)
                    key = cv2.waitKey(delay) & 0xFF
                    
                    if key == ord('q'):  # 退出
                        print("用户中断")
                        break
                    elif key == ord('p'):  # 暂停
                        print("暂停播放，按任意键继续...")
                        cv2.waitKey(0)
                    elif key == ord('s'):  # 单步
                        print("单步模式，按任意键继续下一帧...")
                        cv2.waitKey(0)
                    elif key == ord('f'):  # 快进10帧
                        for _ in range(10):
                            ret, _ = cap.read()
                            frame_index += 1
                            if not ret:
                                break
                
                # 显示进度
                if show_progress and frame_index % 100 == 0:
                    elapsed = time.time() - start_process_time
                    progress = (frame_index - start_frame) / (end_frame - start_frame) * 100
                    estimated_total = elapsed / progress * 100 if progress > 0 else 0
                    remaining = estimated_total - elapsed if estimated_total > elapsed else 0
                    
                    print(f"进度: {frame_index}/{end_frame} ({progress:.1f}%) | "
                          f"已处理: {elapsed:.1f}s | 剩余: {remaining:.1f}s | "
                          f"检测到: {len(self.all_detections)}个车牌")
                
                frame_index += 1
        
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"处理视频时出错: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            if out is not None:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # 保存统计信息和结果
            if save_results and self.output_dir:
                elapsed_time = time.time() - start_process_time
                self._save_statistics(video_path, elapsed_time, fps, width, height, start_frame, end_frame)
                if save_json:
                    self._save_detections_json()
            
            # 打印汇总
            self._print_summary(video_path, start_process_time)
        
        # 返回结果
        result = {
            "success": True,
            "video_path": video_path,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "detection_count": len(self.all_detections),
            "unique_plates": len(self.detection_history),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "detections": self.all_detections[:100] if len(self.all_detections) > 100 else self.all_detections,  # 限制返回数量
            "frame_rate": fps,
            "resolution": f"{width}x{height}"
        }
        
        return result
    
    def _process_frame(self, frame: np.ndarray, frame_index: int, fps: float, 
                      detection_interval: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        处理单帧
        
        Returns:
            processed_frame: 处理后的帧
            detections: 当前帧的检测结果
        """
        processed_frame = frame.copy()
        detections = []
        
        # 每隔一定帧数进行检测（减少计算量）
        if frame_index % detection_interval == 0:
            self.processed_frames += 1
            
            try:
                # 保存临时文件用于检测
                temp_path = f"temp_frame_{frame_index}_{int(time.time())}.jpg"
                cv2.imwrite(temp_path, frame)
                
                # 检测车牌
                plates_info = self.detector.detect_all_and_rectify(temp_path)
                
                for i, plate_info in enumerate(plates_info):
                    # 过滤低置信度检测
                    if plate_info['confidence'] < self.conf_threshold:
                        continue
                    
                    # 识别车牌文字
                    plate_result = self._recognize_plate(plate_info, frame_index, i)
                    
                    # 添加到检测历史（用于追踪）
                    plate_id = self._track_plate(plate_result, frame_index)
                    
                    # 记录检测结果
                    detection_record = {
                        'frame_index': frame_index,
                        'timestamp': frame_index / fps,
                        'plate_id': plate_id,
                        **plate_result
                    }
                    
                    detections.append(detection_record)
                    self.all_detections.append(detection_record)
                    
                    # 在帧上标注
                    processed_frame = self._annotate_video_frame(
                        processed_frame, plate_info, plate_result, plate_id
                    )
                    
                    # 保存车牌图像
                    if self.output_dir and plate_result.get('ocr_success', False):
                        plate_text = plate_result.get('plate_text', 'unknown').replace('/', '_')
                        plate_filename = self.plates_dir / f"frame_{frame_index:06d}_plate_{i}_{plate_text}.jpg"
                        cv2.imwrite(str(plate_filename), plate_info['rectified'])
                
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"帧 {frame_index} 检测出错: {e}")
        
        return processed_frame, detections
    
    def _recognize_plate(self, plate_info: Dict, frame_index: int, plate_index: int) -> Dict:
        """识别车牌文字"""
        result = {
            'frame_index': frame_index,
            'plate_index': plate_index,
            'detection_confidence': plate_info['confidence'],
            'bbox': plate_info['bbox'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'ocr_success': False
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
            temp_path = f"temp_plate_{frame_index}_{plate_index}_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, preprocessed_image)
            
            # OCR识别
            ocr_result = get_license_plate_info(temp_path)
            
            if ocr_result:
                plate_text, ocr_confidence, plate_type = ocr_result
                result.update({
                    'plate_text': plate_text,
                    'ocr_confidence': ocr_confidence,
                    'plate_type': plate_type,
                    'ocr_success': True
                })
            
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
                
        except Exception as e:
            print(f"车牌识别出错: {e}")
            result['error'] = str(e)
        
        return result
    
    def _track_plate(self, plate_result: Dict, frame_index: int) -> str:
        """简单车牌追踪（基于位置和文字）"""
        if not plate_result.get('ocr_success', False):
            return f"unknown_{frame_index}"
        
        plate_text = plate_result['plate_text']
        if plate_text == "未知":
            return f"unknown_{frame_index}"
        
        bbox = plate_result['bbox']
        
        # 计算车牌中心点
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 查找最近的历史检测
        closest_plate_id = None
        min_distance = float('inf')
        
        for plate_id, history in self.detection_history.items():
            last_detection = history['last_detection']
            last_bbox = last_detection['bbox']
            last_x = (last_bbox[0] + last_bbox[2]) / 2
            last_y = (last_bbox[1] + last_bbox[3]) / 2
            
            # 计算欧氏距离
            distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            
            # 如果距离较小且文字相同，可能是同一个车牌
            if distance < 100 and last_detection.get('plate_text') == plate_text:
                if distance < min_distance:
                    min_distance = distance
                    closest_plate_id = plate_id
        
        # 如果是新车牌
        if closest_plate_id is None:
            self.plate_counter += 1
            plate_id = f"plate_{self.plate_counter:03d}"
            self.detection_history[plate_id] = {
                'first_frame': frame_index,
                'last_frame': frame_index,
                'detection_count': 1,
                'last_detection': plate_result,
                'all_texts': [plate_text] if plate_text != "未知" else []
            }
        else:
            # 更新现有车牌
            plate_id = closest_plate_id
            history = self.detection_history[plate_id]
            history['last_frame'] = frame_index
            history['detection_count'] += 1
            history['last_detection'] = plate_result
            
            if plate_text not in history['all_texts']:
                history['all_texts'].append(plate_text)
        
        return plate_id
    
    def _annotate_video_frame(self, frame: np.ndarray, plate_info: Dict, 
                             plate_result: Dict, plate_id: str) -> np.ndarray:
        """在视频帧上标注检测结果"""
        x1, y1, x2, y2 = plate_info['bbox']
        
        # 根据识别结果选择颜色
        if plate_result.get('ocr_success', False):
            # 根据车牌类型选择颜色
            plate_type = plate_result.get('plate_type', '')
            color_map = {
                '蓝牌': (255, 0, 0),      # 蓝色
                '黄牌': (0, 255, 255),    # 黄色
                '新能源绿牌': (0, 255, 0), # 绿色
                '绿牌': (0, 255, 0),      # 绿色
                '白牌': (255, 255, 255),  # 白色
                '黑牌': (0, 0, 0),        # 黑色
                '白牌 (警用)': (255, 255, 255),  # 白色
            }
            color = color_map.get(plate_type, (0, 255, 0))
            text = f"{plate_id}: {plate_result.get('plate_text', '')} ({plate_type})"
        else:
            color = (0, 0, 255)  # 红色 - 仅检测到
            text = f"{plate_id}: 检测"
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制车牌ID和文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 文本背景
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        bg_x2 = x1 + text_size[0] + 10
        bg_y1 = max(0, y1 - text_size[1] - 10)
        
        cv2.rectangle(frame, (x1, bg_y1), (bg_x2, y1), color, -1)
        
        # 文本颜色（确保在深色背景上可见）
        if plate_type in ['白牌', '白牌 (警用)']:
            text_color = (0, 0, 0)  # 白牌用黑色文字
        else:
            text_color = (255, 255, 255)  # 其他用白色文字
            
        cv2.putText(frame, text, (x1 + 5, y1 - 5), 
                   font, font_scale, text_color, thickness)
        
        # 添加置信度（如果可用）
        if plate_result.get('ocr_confidence'):
            conf_text = f"置信度: {plate_result['ocr_confidence']:.2f}"
            conf_size = cv2.getTextSize(conf_text, font, 0.5, 1)[0]
            cv2.putText(frame, conf_text, (x1, y2 + conf_size[1] + 5), 
                       font, 0.5, color, 1)
        
        # 添加检测置信度
        det_conf_text = f"检测: {plate_result.get('detection_confidence', 0):.2f}"
        cv2.putText(frame, det_conf_text, (x1, y2 + conf_size[1] * 2 + 10), 
                   font, 0.5, color, 1)
        
        return frame
    
    def _save_statistics(self, video_path: str, elapsed_time: float, fps: float, 
                        width: int, height: int, start_frame: int, end_frame: int):
        """保存统计信息"""
        stats_file = self.output_dir / "statistics.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("视频车牌识别统计\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"视频文件: {video_path}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"视频信息: {width}x{height} @ {fps:.1f}fps\n")
            f.write(f"总帧数: {self.total_frames}\n")
            f.write(f"处理范围: 第{start_frame}帧到第{end_frame}帧\n")
            f.write(f"处理帧数: {self.processed_frames}\n")
            f.write(f"检测间隔: 每10帧检测一次\n")
            f.write(f"检测到车牌总数: {len(self.all_detections)}\n")
            f.write(f"唯一车牌数: {len(self.detection_history)}\n")
            f.write(f"处理耗时: {elapsed_time:.2f}秒\n")
            if elapsed_time > 0:
                f.write(f"处理速度: {self.processed_frames/elapsed_time:.1f} fps\n")
            
            # 车牌统计
            if self.detection_history:
                f.write("\n车牌追踪统计:\n")
                f.write("-" * 40 + "\n")
                
                for plate_id, history in self.detection_history.items():
                    first_frame = history['first_frame']
                    last_frame = history['last_frame']
                    frame_duration = last_frame - first_frame
                    time_duration = frame_duration / fps
                    
                    f.write(f"\n车牌 {plate_id}:\n")
                    f.write(f"  首次出现: 第{first_frame}帧 ({first_frame/fps:.1f}秒)\n")
                    f.write(f"  最后出现: 第{last_frame}帧 ({last_frame/fps:.1f}秒)\n")
                    f.write(f"  持续时间: {time_duration:.1f}秒\n")
                    f.write(f"  检测次数: {history['detection_count']}\n")
                    
                    if history['all_texts']:
                        texts = ", ".join(set(history['all_texts']))
                        f.write(f"  识别文本: {texts}\n")
            
            # 检测结果汇总
            f.write("\n\n检测结果汇总:\n")
            f.write("=" * 60 + "\n")
            
            success_count = sum(1 for d in self.all_detections if d.get('ocr_success', False))
            f.write(f"成功识别次数: {success_count}/{len(self.all_detections)}\n")
            
            # 车牌类型分布
            type_distribution = {}
            for detection in self.all_detections:
                plate_type = detection.get('plate_type', '未知')
                type_distribution[plate_type] = type_distribution.get(plate_type, 0) + 1
            
            f.write("\n车牌类型分布:\n")
            for plate_type, count in type_distribution.items():
                percentage = count / len(self.all_detections) * 100 if len(self.all_detections) > 0 else 0
                f.write(f"  {plate_type}: {count}次 ({percentage:.1f}%)\n")
        
        print(f"统计信息已保存: {stats_file}")
    
    def _save_detections_json(self):
        """保存检测结果到JSON文件"""
        # 转换为可序列化的格式
        serializable_detections = []
        
        for detection in self.all_detections:
            serializable = {
                'frame_index': detection['frame_index'],
                'timestamp_seconds': detection['timestamp'],
                'plate_id': detection['plate_id'],
                'detection_confidence': detection['detection_confidence'],
                'bbox': detection['bbox']
            }
            
            if detection.get('ocr_success', False):
                serializable.update({
                    'plate_text': detection.get('plate_text', ''),
                    'plate_type': detection.get('plate_type', '未知'),
                    'ocr_confidence': detection.get('ocr_confidence', 0),
                    'ocr_success': True
                })
            
            serializable_detections.append(serializable)
        
        # 保存到文件
        json_file = self.output_dir / "detections.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_detections, f, ensure_ascii=False, indent=2)
        
        print(f"检测结果JSON已保存: {json_file}")
    
    def _print_summary(self, video_path: str, start_time: float):
        """打印处理摘要"""
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("视频处理完成摘要")
        print("=" * 60)
        print(f"视频文件: {Path(video_path).name}")
        print(f"总处理时间: {elapsed_time:.2f}秒")
        print(f"总帧数: {self.total_frames}")
        print(f"处理帧数: {self.processed_frames}")
        print(f"检测到车牌总数: {len(self.all_detections)}")
        print(f"唯一车牌数: {len(self.detection_history)}")
        if elapsed_time > 0:
            print(f"处理速度: {self.processed_frames/elapsed_time:.1f} fps")
        
        if self.detection_history:
            print(f"\n追踪到 {len(self.detection_history)} 个不同车牌:")
            for plate_id, history in self.detection_history.items():
                texts = set(history['all_texts'])
                text_str = ", ".join(texts) if texts else "未识别"
                print(f"  {plate_id}: 出现{history['detection_count']}次, "
                      f"识别结果: {text_str}")
        
        if self.output_dir:
            print(f"\n输出目录: {self.output_dir}")
        print("=" * 60)


def create_video_processor_from_system(system):
    """
    从LicensePlateSystem创建视频处理器
    
    Args:
        system: LicensePlateSystem实例
        
    Returns:
        VideoLicensePlateProcessor实例
    """
    return VideoLicensePlateProcessor.from_system(system)


def test_video_processor():
    """测试视频处理器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试视频处理器")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="模型路径")
    parser.add_argument("--output-dir", type=str, default="test_results", help="输出目录")
    parser.add_argument("--interval", type=int, default=10, help="检测间隔")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("测试视频处理器")
    print("=" * 60)
    
    try:
        # 创建视频处理器
        processor = VideoLicensePlateProcessor(
            detector_config={'model_path': args.model},
            preprocessor_config={'target_size': (640, 480)},
            conf_threshold=args.conf,
            use_preprocessing=True
        )
        
        # 处理视频
        result = processor.process_video_file(
            video_path=args.video,
            output_dir=args.output_dir,
            detection_interval=args.interval,
            save_results=True,
            save_frames=False,
            save_json=True,
            display=True,
            show_progress=True
        )
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
        if result.get('success', False):
            print(f"视频文件: {args.video}")
            print(f"总帧数: {result['total_frames']}")
            print(f"处理帧数: {result['processed_frames']}")
            print(f"检测到车牌数: {result['detection_count']}")
            print(f"唯一车牌数: {result['unique_plates']}")
            print(f"输出目录: {result['output_dir']}")
            
            # 显示检测到的车牌
            if result['detections']:
                print(f"\n检测到的车牌 (前10个):")
                for detection in result['detections'][:10]:
                    plate_text = detection.get('plate_text', '未知')
                    plate_type = detection.get('plate_type', '未知')
                    frame_idx = detection.get('frame_index', 0)
                    conf = detection.get('ocr_confidence', 0)
                    print(f"  第{frame_idx}帧: {plate_text} ({plate_type}) 置信度: {conf:.2f}")
        else:
            print(f"处理失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"测试过程中出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_video_processor()