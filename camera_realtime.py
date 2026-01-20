# camera_realtime.py
import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageDraw, ImageFont # 引入PIL库支持中文

from license_plate_detection import LicensePlateDetector
from license_plate_preprocessor import LicensePlatePreprocessor
from license_plate_ocr_engine import get_license_plate_info


def get_all_cameras(max_test: int = 10):
    """检测所有可用的摄像头并返回详细信息"""
    cameras = []
    print("正在检测摄像头...")
    
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    
    for backend in backends:
        for i in range(max_test):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_name = f"摄像头 {i}"
                        is_duplicate = False
                        for c in cameras:
                            if c['index'] == i:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            print(f"  找到: {camera_name} - {width}x{height} @ {fps:.1f}fps")
                            cameras.append({
                                'index': i, 'backend': backend, 'name': camera_name,
                                'width': width, 'height': height, 'fps': fps, 'available': True
                            })
                    cap.release()
            except:
                pass
    return cameras


def select_camera_interactive():
    """交互式选择摄像头"""
    print("=" * 60); print("摄像头检测"); print("=" * 60)
    cameras = get_all_cameras()
    if not cameras:
        print("未找到任何摄像头！")
        return None, None
    
    print(f"\n找到 {len(cameras)} 个可用摄像头:")
    for i, cam in enumerate(cameras):
        print(f"  [{i}] {cam['name']} ({cam['width']}x{cam['height']})")
    
    while True:
        try:
            choice = input("\n请选择摄像头编号 (输入 'q' 退出): ").strip()
            if choice.lower() == 'q': return None, None
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                selected = cameras[choice_idx]
                return selected['index'], selected['backend']
        except ValueError: pass


class RealTimeLicensePlateDetector:
    """实时车牌检测器 - 优化版 (支持中文显示 + 结果锁定)"""
    
    def __init__(self, model_path: str = 'yolov8s.pt', conf_threshold: float = 0.5, use_preprocessing: bool = True):
        print("初始化实时车牌检测系统...")
        self.detector = LicensePlateDetector(model_path=model_path, conf_threshold=conf_threshold)
        self.preprocessor = LicensePlatePreprocessor(target_size=(640, 480))
        self.use_preprocessing = use_preprocessing
        
        self.frame_count = 0
        self.plate_count = 0
        self.output_dir = Path("camera_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 状态锁定变量
        self.current_plate = None
        self.missing_frames = 0
        self.lock_iou_threshold = 0.6
        
        # 字体缓存
        self.font = None
        self._init_font()
        
        print("✓ 实时检测系统初始化完成")

    def _init_font(self):
        """初始化中文字体"""
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
        print("警告: 未找到中文字体，中文显示可能异常")

    def draw_chinese_text(self, img, text, position, text_color, text_size=20):
        """使用PIL绘制中文文本"""
        if (isinstance(img, np.ndarray)):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 动态加载字体大小
            font = self.font
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, text_size, encoding="utf-8")
                
            draw.text(position, text, font=font, fill=text_color, stroke_width=0)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
    
    def start_camera(self, camera_index: int = 0, camera_backend: int = cv2.CAP_ANY, 
                    frame_width: int = 1280, frame_height: int = 720, detection_interval: int = 4):
        """启动摄像头检测"""
        print(f"\n启动摄像头 (索引: {camera_index})...")
        cap = cv2.VideoCapture(camera_index, camera_backend)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {camera_index}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        print(f"检测启动: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.1f}fps")
        print("模式：智能锁定 + 中文显示 + 自动去重")
        
        is_paused = False
        start_time = time.time()
        
        try:
            while True:
                if not is_paused:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    self.frame_count += 1
                    display_frame = frame.copy()
                    
                    # 1. 检测逻辑
                    if self.frame_count % detection_interval == 0:
                        plates_info = self._smart_detect_and_recognize(frame)
                        if plates_info:
                            self.missing_frames = 0
                            self.plate_count += 1
                        else:
                            self.missing_frames += 1
                            if self.missing_frames > 30: # 1秒左右丢失则重置
                                self.current_plate = None
                    
                    # 2. 绘制逻辑 (复用锁定结果)
                    if self.current_plate:
                        display_frame = self._annotate_detection(display_frame, self.current_plate)
                    
                    # 3. 统计信息
                    elapsed_time = time.time() - start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    display_frame = self._add_statistics_overlay(display_frame, fps)
                    
                    if is_paused:
                        display_frame = self.draw_chinese_text(display_frame, "已暂停", 
                                                             (display_frame.shape[1]//2-60, 50), (255, 0, 0), 40)

                    cv2.imshow('License Plate Recognition', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('s'):
                     if self.current_plate: self._save_plate_result(frame, self.current_plate, force=True)
                elif key == ord('p'): is_paused = not is_paused
                elif key == ord('r'): self.current_plate = None; print("已重置状态")
        
        except KeyboardInterrupt: print("\n用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return True

    def _calculate_iou(self, boxA, boxB):
        """计算IoU"""
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def _smart_detect_and_recognize(self, frame: np.ndarray) -> List[Dict]:
        """智能检测：复用旧结果或触发新OCR"""
        plates_info = []
        try:
            temp_path = self.output_dir / "temp_detect.jpg"
            cv2.imwrite(str(temp_path), frame)
            raw_detections = self.detector.detect_all_and_rectify(str(temp_path))
            try: os.remove(str(temp_path))
            except: pass
            
            if not raw_detections: return []
            
            best_detection = max(raw_detections, key=lambda x: x['confidence'])
            need_ocr = True
            
            if self.current_plate is not None:
                iou = self._calculate_iou(best_detection['bbox'], self.current_plate['bbox'])
                if iou > self.lock_iou_threshold:
                    need_ocr = False
                    self.current_plate['bbox'] = best_detection['bbox'] # 更新坐标
                    plates_info.append(self.current_plate)
            
            if need_ocr:
                plate_result = self._recognize_plate_text(best_detection, 0)
                if plate_result and 'plate_text' in plate_result:
                    best_detection['recognition'] = plate_result
                    if plate_result.get('ocr_confidence', 0) > 0.6: # 门槛
                        self.current_plate = best_detection
                        self._save_plate_result(frame, best_detection)
                        print(f"锁定: {plate_result['plate_text']}")
                    plates_info.append(best_detection)
        except Exception: pass
        return plates_info

    def _save_plate_result(self, frame, plate_info, force=False):
        """保存图片"""
        try:
            recognition = plate_info.get('recognition', {})
            text = recognition.get('plate_text', 'unknown')
            if text == "未知" and not force: return

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"plate_{text}_{timestamp}.jpg"
            save_path = self.output_dir / filename
            
            save_img = frame.copy()
            self._annotate_detection(save_img, plate_info) # 烧录文字到图片
            cv2.imwrite(str(save_path), save_img)
            print(f"  -> 已保存: {filename}")
        except: pass
    
    def _recognize_plate_text(self, plate_info: Dict, plate_index: int) -> Dict:
        """执行OCR"""
        result = {'plate_index': plate_index, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}
        rectified_image = plate_info['rectified']
        if rectified_image is None or rectified_image.size == 0: return result
        
        try:
            preprocessed_image = rectified_image
            if self.use_preprocessing:
                preprocessed_image, _ = self.preprocessor.preprocess_with_color_recovery(
                    rectified_image, detect_plate_region=True)
            
            temp_path = self.output_dir / f"temp_ocr_{self.frame_count}.jpg"
            cv2.imwrite(str(temp_path), preprocessed_image)
            ocr_result = get_license_plate_info(str(temp_path))
            
            if ocr_result:
                plate_text, ocr_confidence, plate_type = ocr_result
                result.update({'plate_text': plate_text, 'ocr_confidence': ocr_confidence,
                             'plate_type': plate_type, 'recognition_success': True})
            try: os.remove(str(temp_path))
            except: pass
        except: pass
        return result
    
    def _annotate_detection(self, frame: np.ndarray, plate_info: Dict) -> np.ndarray:
        """标注结果 (支持中文)"""
        if not plate_info or 'bbox' not in plate_info: return frame
        
        x1, y1, x2, y2 = plate_info['bbox']
        recognition = plate_info.get('recognition', {})
        
        if recognition.get('recognition_success', False):
            color = (0, 255, 0)
            text_lines = [
                f"车牌: {recognition['plate_text']}",
                f"类型: {recognition.get('plate_type','未知')}",
                f"置信度: {recognition.get('ocr_confidence',0):.2f}"
            ]
        else:
            color = (0, 0, 255)
            text_lines = ["检测中..."]
        
        # 绘制矩形框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # 绘制中文信息背景和文字
        line_height = 25
        bg_height = len(text_lines) * line_height + 10
        bg_y1 = max(0, y1 - bg_height)
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, bg_y1), (x1 + 200, y1), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 绘制中文
        y_offset = bg_y1 + 5
        for line in text_lines:
            frame = self.draw_chinese_text(frame, line, (x1 + 5, y_offset), (255, 255, 255), 20)
            y_offset += line_height
            
        return frame
    
    def _add_statistics_overlay(self, frame: np.ndarray, fps: float):
        """添加左上角统计信息 (中文)"""
        frame = self.draw_chinese_text(frame, f"FPS: {fps:.1f}", (10, 10), (0, 255, 255), 20)
        
        current_text = "无"
        if self.current_plate:
            current_text = self.current_plate['recognition'].get('plate_text', '未知')
        
        frame = self.draw_chinese_text(frame, f"当前锁定: {current_text}", (10, 40), (0, 255, 255), 20)
        frame = self.draw_chinese_text(frame, f"总计识别: {self.plate_count}", (10, 70), (0, 255, 255), 20)
        return frame

# 主函数保持兼容
def main():
    import argparse
    parser = argparse.ArgumentParser(description="车牌识别实时检测 (中文优化版)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--detection-interval", type=int, default=4)
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--no-preprocess", action="store_true")
    
    args = parser.parse_args()
    
    detector = RealTimeLicensePlateDetector(
        model_path=args.model, conf_threshold=args.conf, use_preprocessing=not args.no_preprocess)
    
    detector.start_camera(camera_index=args.camera_index, frame_width=args.frame_width, 
                         frame_height=args.frame_height, detection_interval=args.detection_interval)

if __name__ == "__main__":
    main()