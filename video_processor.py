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
from PIL import Image, ImageDraw, ImageFont  # [修改] 引入PIL

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
        """
        print("初始化视频车牌处理器...")
        
        if not MODULES_AVAILABLE:
            raise ImportError("必要的模块未找到...")
        
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

        # [修改] 初始化字体
        self.font = None
        self.font_path = None
        self._init_font()
        
        print("✓ 视频处理器初始化完成")

    # [新增] 初始化字体方法
    def _init_font(self):
        font_paths = [
            "simhei.ttf", "msyh.ttf", "font.ttf",
            "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttf",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font_path = path
                    self.font = ImageFont.truetype(path, 20, encoding="utf-8")
                    return
                except: continue
        self.font = ImageFont.load_default()

    # [新增] 绘制中文方法
    def draw_chinese_text(self, img, text, position, text_color, text_size=20):
        if (isinstance(img, np.ndarray)):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            font = self.font
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, text_size, encoding="utf-8")
                
            draw.text(position, text, font=font, fill=text_color, stroke_width=0)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
    
    @classmethod
    def from_system(cls, system):
        """从LicensePlateSystem创建视频处理器"""
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
        """处理视频文件 (优化：视觉暂留 + 统计合并)"""
        print(f"开始处理视频: {video_path}")
        print("-" * 60)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return {"error": f"无法打开视频文件 {video_path}", "success": False}
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = original_fps if original_fps > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        if duration > 0:
            end_frame = min(self.total_frames, start_frame + int(duration * fps))
        else:
            end_frame = self.total_frames
        
        if max_frames > 0:
            end_frame = min(end_frame, start_frame + max_frames)
        
        print(f"视频信息: {width}x{height} @ {fps:.1f}fps, 总帧数: {self.total_frames}")
        
        # 初始化输出目录
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
        
        out = None
        if save_results:
            output_video_path = self.output_dir / "output_video.mp4"
            output_fps_val = output_fps if output_fps > 0 else fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, output_fps_val, (width, height))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        start_process_time = time.time()
        frame_index = start_frame
        
        # === 核心修改：用于视觉暂留的列表 ===
        # 存储格式: [{'plate_info': ..., 'plate_result': ..., 'plate_id': ...}, ...]
        current_draw_list = [] 
        # 记录上次检测到车牌的帧，用于超时清除
        last_detection_frame = -1
        
        try:
            while frame_index < end_frame:
                ret, frame = cap.read()
                if not ret: break
                
                # 1. 执行处理（检测）
                # 注意：这里不再返回处理后的帧，而是返回检测数据
                detections_data, frame_detections = self._process_frame_data(
                    frame, frame_index, fps, detection_interval
                )
                
                # 2. 更新绘图列表 (视觉暂留逻辑)
                if frame_index % detection_interval == 0:
                    # 如果是检测帧
                    if detections_data:
                        # 如果检测到了车牌，更新绘图列表
                        current_draw_list = detections_data
                        last_detection_frame = frame_index
                    else:
                        # 如果没检测到，不要立刻清空，给一点“宽限期” (比如10帧)
                        # 如果超过宽限期还没检测到，才清空，防止框子一直卡在屏幕上
                        if frame_index - last_detection_frame > 10: 
                            current_draw_list = []
                
                # 3. 统一绘图 (每一帧都画)
                processed_frame = frame.copy()
                for item in current_draw_list:
                    processed_frame = self._annotate_video_frame(
                        processed_frame, 
                        item['plate_info'], 
                        item['plate_result'], 
                        item['plate_id']
                    )
                
                # 4. 保存与显示
                if save_results and out is not None:
                    out.write(processed_frame)
                    if save_frames and frame_detections:
                        cv2.imwrite(str(self.frames_dir / f"frame_{frame_index:06d}.jpg"), processed_frame)
                
                if display:
                    if show_progress:
                        progress_text = f"进度: {frame_index}/{end_frame}"
                        processed_frame = self.draw_chinese_text(processed_frame, progress_text, (10, 30), (0, 255, 255), 20)
                    
                    cv2.imshow('Video Processing', processed_frame)
                    
                    delay = max(1, int(1000 / fps) - 10)
                    key = cv2.waitKey(delay) & 0xFF
                    if key == ord('q'): break
                    elif key == ord('p'): cv2.waitKey(0)
                
                if show_progress and frame_index % 100 == 0:
                    elapsed = time.time() - start_process_time
                    progress = (frame_index - start_frame) / (end_frame - start_frame) * 100
                    print(f"进度: {progress:.1f}% | 耗时: {elapsed:.1f}s")
                
                frame_index += 1
        
        except KeyboardInterrupt: print("\n用户中断")
        except Exception as e:
            print(f"处理视频时出错: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            if out is not None: out.release()
            if display: cv2.destroyAllWindows()
            
            if save_results and self.output_dir:
                elapsed_time = time.time() - start_process_time
                self._save_statistics(video_path, elapsed_time, fps, width, height, start_frame, end_frame)
                if save_json: self._save_detections_json()
            
            self._print_summary(video_path, start_process_time)
        
        return {
            "success": True,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "detection_count": len(self.all_detections),
            "unique_plates": len(self.detection_history),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "detections": self.all_detections[:100] if len(self.all_detections) > 100 else self.all_detections
        }
    
    def _process_frame_data(self, frame: np.ndarray, frame_index: int, fps: float, 
                      detection_interval: int) -> Tuple[List[Dict], List[Dict]]:
        """
        处理单帧数据 (不进行绘图，只返回数据)
        返回: (visual_data_list, log_detections_list)
        """
        visual_data = [] # 用于绘图的数据
        detections = []  # 用于日志记录的数据
        
        if frame_index % detection_interval == 0:
            self.processed_frames += 1
            try:
                temp_path = f"temp_frame_{frame_index}_{int(time.time())}.jpg"
                cv2.imwrite(temp_path, frame)
                plates_info = self.detector.detect_all_and_rectify(temp_path)
                
                for i, plate_info in enumerate(plates_info):
                    if plate_info['confidence'] < self.conf_threshold: continue
                    
                    plate_result = self._recognize_plate(plate_info, frame_index, i)
                    plate_id = self._track_plate(plate_result, frame_index)
                    
                    # 1. 准备日志数据
                    detection_record = {
                        'frame_index': frame_index,
                        'timestamp': frame_index / fps,
                        'plate_id': plate_id,
                        **plate_result
                    }
                    detections.append(detection_record)
                    self.all_detections.append(detection_record)
                    
                    # 2. 准备绘图数据 (包含坐标、文字、ID)
                    visual_data.append({
                        'plate_info': plate_info,
                        'plate_result': plate_result,
                        'plate_id': plate_id
                    })
                    
                    # 保存车牌截图
                    if self.output_dir and plate_result.get('ocr_success', False):
                        plate_text = plate_result.get('plate_text', 'unknown').replace('/', '_')
                        cv2.imwrite(str(self.plates_dir / f"frame_{frame_index:06d}_plate_{i}_{plate_text}.jpg"), plate_info['rectified'])
                
                try: os.remove(temp_path)
                except: pass
            except Exception as e: print(f"帧 {frame_index} 检测出错: {e}")
        
        return visual_data, detections
    

    
    def _recognize_plate(self, plate_info: Dict, frame_index: int, plate_index: int) -> Dict:
        result = {'frame_index': frame_index, 'plate_index': plate_index,
                 'detection_confidence': plate_info['confidence'], 'bbox': plate_info['bbox'],
                 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'ocr_success': False}
        
        rectified_image = plate_info['rectified']
        if rectified_image is None or rectified_image.size == 0: return result
        
        try:
            preprocessed_image = rectified_image
            if self.use_preprocessing:
                preprocessed_image, _ = self.preprocessor.preprocess_with_color_recovery(
                    rectified_image, detect_plate_region=True)
            
            temp_path = f"temp_plate_{frame_index}_{plate_index}_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, preprocessed_image)
            ocr_result = get_license_plate_info(temp_path)
            
            if ocr_result:
                plate_text, ocr_confidence, plate_type = ocr_result
                result.update({'plate_text': plate_text, 'ocr_confidence': ocr_confidence,
                             'plate_type': plate_type, 'ocr_success': True})
            try: os.remove(temp_path)
            except: pass
        except Exception as e: result['error'] = str(e)
        return result
    
    def _track_plate(self, plate_result: Dict, frame_index: int) -> str:
        """简单车牌追踪"""
        if not plate_result.get('ocr_success', False): return f"unknown_{frame_index}"
        plate_text = plate_result['plate_text']
        if plate_text == "未知": return f"unknown_{frame_index}"
        
        bbox = plate_result['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        closest_plate_id = None
        min_distance = float('inf')
        
        for plate_id, history in self.detection_history.items():
            last_bbox = history['last_detection']['bbox']
            last_x = (last_bbox[0] + last_bbox[2]) / 2
            last_y = (last_bbox[1] + last_bbox[3]) / 2
            distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            
            if distance < 100 and history['last_detection'].get('plate_text') == plate_text:
                if distance < min_distance:
                    min_distance = distance
                    closest_plate_id = plate_id
        
        if closest_plate_id is None:
            self.plate_counter += 1
            plate_id = f"plate_{self.plate_counter:03d}"
            self.detection_history[plate_id] = {
                'first_frame': frame_index, 'last_frame': frame_index, 'detection_count': 1,
                'last_detection': plate_result, 'all_texts': [plate_text]
            }
        else:
            plate_id = closest_plate_id
            history = self.detection_history[plate_id]
            history['last_frame'] = frame_index
            history['detection_count'] += 1
            history['last_detection'] = plate_result
            if plate_text not in history['all_texts']: history['all_texts'].append(plate_text)
        
        return plate_id
    
    def _annotate_video_frame(self, frame: np.ndarray, plate_info: Dict, 
                             plate_result: Dict, plate_id: str) -> np.ndarray:
        """[修改] 在视频帧上标注检测结果 (支持中文)"""
        if not plate_info or 'bbox' not in plate_info: return frame
        x1, y1, x2, y2 = plate_info['bbox']
        
        text_lines = []
        if plate_result.get('ocr_success', False):
            plate_type = plate_result.get('plate_type', '')
            color_map = {'蓝牌': (255,0,0), '黄牌': (0,255,255), '新能源绿牌': (0,255,0)}
            color = color_map.get(plate_type, (0,255,0))
            text_lines.append(f"{plate_id}: {plate_result.get('plate_text', '')}")
            text_lines.append(f"类型: {plate_type}")
            text_lines.append(f"置信度: {plate_result.get('ocr_confidence', 0):.2f}")
        else:
            color = (0, 0, 255)
            text_lines.append(f"{plate_id}: 检测中...")
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        line_height = 25
        bg_height = len(text_lines) * line_height + 10
        bg_y1 = max(0, y1 - bg_height)
        if bg_y1 == 0: bg_y1 = y2 # 下方显示
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, bg_y1), (x1 + 220, bg_y1 + bg_height), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        y_offset = bg_y1 + 5
        for line in text_lines:
            frame = self.draw_chinese_text(frame, line, (x1 + 5, y_offset), (255, 255, 255), 20)
            y_offset += line_height
        
        return frame
    
    def _save_statistics(self, video_path, elapsed_time, fps, width, height, start_frame, end_frame):
        """保存统计信息 (优化：按车牌号去重合并)"""
        stats_file = self.output_dir / "statistics.txt"
        
        # === 1. 数据聚合逻辑 ===
        merged_plates = {} # Key: 车牌号, Value: 统计对象
        
        for plate_id, history in self.detection_history.items():
            # 获取该ID最可信的车牌号 (取历史中出现过的)
            texts = history.get('all_texts', [])
            if not texts: continue
            
            # 简单策略：取最后一个识别到的号码，或者取众数
            # 这里我们取列表里的第一个有效值作为代表
            plate_text = texts[0] 
            
            if plate_text not in merged_plates:
                merged_plates[plate_text] = {
                    'count': 0,
                    'first_frame': float('inf'),
                    'last_frame': float('-inf'),
                    'ids': []
                }
            
            # 合并数据
            data = merged_plates[plate_text]
            data['count'] += history['detection_count']
            data['first_frame'] = min(data['first_frame'], history['first_frame'])
            data['last_frame'] = max(data['last_frame'], history['last_frame'])
            data['ids'].append(plate_id)
        
        # === 2. 写入文件 ===
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"视频: {video_path}\n处理时间: {datetime.now()}\n")
            f.write(f"总帧数: {self.total_frames} | 处理: {self.processed_frames} | 耗时: {elapsed_time:.2f}s\n")
            f.write(f"检测到车牌(不去重): {len(self.all_detections)}\n")
            f.write(f"独立车牌数(去重后): {len(merged_plates)}\n")
            
            f.write("\n追踪详情 (按车牌号合并):\n")
            f.write("-" * 50 + "\n")
            
            # 按出现时间排序
            sorted_plates = sorted(merged_plates.items(), key=lambda x: x[1]['first_frame'])
            
            for text, data in sorted_plates:
                start_sec = data['first_frame'] / fps
                end_sec = data['last_frame'] / fps
                duration = end_sec - start_sec
                
                f.write(f"车牌: {text}\n")
                f.write(f"  出现时间: {start_sec:.1f}s - {end_sec:.1f}s (持续 {duration:.1f}s)\n")
                f.write(f"  检测次数: {data['count']}\n")
                # 如果你想看它关联了哪些原始ID，可以取消下面这行的注释
                # f.write(f"  关联追踪ID: {', '.join(data['ids'])}\n") 
                f.write("\n")
                

    
    def _save_detections_json(self):
        """保存JSON"""
        serializable = []
        for d in self.all_detections:
            item = {k: v for k, v in d.items() if isinstance(v, (int, float, str, list, tuple))}
            serializable.append(item)
        with open(self.output_dir / "detections.json", 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    def _print_summary(self, video_path, start_time):
        elapsed = time.time() - start_time
        print(f"\n视频处理完成: {Path(video_path).name} | 耗时: {elapsed:.2f}s")
        print(f"检测到 {len(self.detection_history)} 个唯一车牌")


def create_video_processor_from_system(system):
    return VideoLicensePlateProcessor.from_system(system)

if __name__ == "__main__":
    print("请使用 main.py --video [路径] 运行")