import cv2
import numpy as np
from ultralytics import YOLO

class LicensePlateDetector:
    """车牌检测器（包含透视矫正）"""
    
    def __init__(self, model_path='yolov8s.pt', conf_threshold=0.5):
        """
        初始化车牌检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """初始化YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def _four_point_transform(self, image, pts):
        """
        四点透视变换
        
        Args:
            image: 原始图像
            pts: 四个点的坐标，顺序为：左上、右上、右下、左下
            
        Returns:
            透视矫正后的图像
        """
        # 解包坐标点
        rect = np.zeros((4, 2), dtype="float32")
        
        # 点按左上、右上、右下、左下排序
        # 计算各点之和与差
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上点（x+y最小）
        rect[2] = pts[np.argmax(s)]  # 右下点（x+y最大）
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上点（x-y最小）
        rect[3] = pts[np.argmax(diff)]  # 左下点（x-y最大）
        
        # 计算目标矩形的宽度和高度
        (tl, tr, br, bl) = rect
        
        # 计算新矩形的宽度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # 计算新矩形的高度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # 设置目标点坐标
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # 计算透视变换矩阵并应用变换
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def _detect_plate_corners(self, plate_image):
        """
        检测车牌四个角点
        
        Args:
            plate_image: 裁剪出的车牌图像
            
        Returns:
            四个角点坐标或None
        """
        if plate_image is None or plate_image.size == 0:
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        corners = None
        
        # 使用轮廓检测
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓的凸包
            hull = cv2.convexHull(max_contour)
            
            # 计算轮廓近似
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            
            # 如果是四边形
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
            elif len(hull) >= 4:
                # 如果是凸包，尝试找到四个角点
                # 使用最小外接矩形
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                corners = box

    def _rectify_perspective(self, original_image, bbox):
        """
        透视矫正车牌
        
        Args:
            original_image: 原始图像
            bbox: 车牌边界框 (x1, y1, x2, y2)
            
        Returns:
            矫正后的车牌图像
        """
        x1, y1, x2, y2 = bbox
        
        # 扩展边界框，确保包含整个车牌
        padding = 10
        h, w = original_image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # 提取车牌区域
        plate_region = original_image[y1:y2, x1:x2]
        
        # 检测角点
        corners = self._detect_plate_corners(plate_region)
        
        if corners is None or len(corners) != 4:
            # 如果无法检测到四个角点，返回原始裁剪图像
            return plate_region
        
        # 将角点坐标映射回原始图像
        corners_original = corners + np.array([x1, y1])
        
        # 应用透视变换
        warped = self._four_point_transform(original_image, corners_original)
        
        # 确保图像方向正确（高度大于宽度）
        h, w = warped.shape[:2]
        if w > h:
            # 旋转90度
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
        return warped
    
    def detect_and_rectify(self, image_path):
        """
        检测车牌并进行透视矫正
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple: (原始裁剪图像, 透视矫正图像) 或 (None, None)
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图片 {image_path}")
            return None, None
        
        # 执行推理
        results = self.model.predict(source=img, conf=self.conf_threshold, save=False)
        
        # 查找最佳车牌
        best_plate = None
        best_rectified = None
        best_bbox = None
        best_conf = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # 提取坐标和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # 选择置信度最高的车牌
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = (x1, y1, x2, y2)
                    
                    # 原始裁剪
                    best_plate = img[y1:y2, x1:x2]
                    
                    # 透视矫正
                    best_rectified = self._rectify_perspective(img, best_bbox)
        
        return best_plate, best_rectified
    
    def detect_all_and_rectify(self, image_path):
        """
        检测所有车牌并进行透视矫正
        
        Args:
            image_path: 图片路径
            
        Returns:
            list: 包含原始图像和矫正图像的字典列表
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图片 {image_path}")
            return []
        
        # 执行推理
        results = self.model.predict(source=img, conf=self.conf_threshold, save=False)
        
        plates_info = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for i, box in enumerate(boxes):
                # 提取坐标和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                bbox = (x1, y1, x2, y2)
                
                # 原始裁剪
                original_crop = img[y1:y2, x1:x2]
                
                # 透视矫正
                rectified = self._rectify_perspective(img, bbox)
                
                plates_info.append({
                    'index': i,
                    'confidence': conf,
                    'bbox': bbox,
                    'original_crop': original_crop,
                    'rectified': rectified,
                    'size_original': original_crop.shape,
                    'size_rectified': rectified.shape if rectified is not None else None
                })
        
        return plates_info


# ============================================
# 使用示例
# ============================================

def show_comparison(original, rectified, title="车牌透视矫正对比"):
    """显示原始和矫正后的对比"""
    if original is None or rectified is None:
        print("没有图像可显示")
        return
    
    # 调整图像大小以便并排显示
    h1, w1 = original.shape[:2]
    h2, w2 = rectified.shape[:2]
    
    # 统一高度
    max_height = max(h1, h2)
    scale1 = max_height / h1
    scale2 = max_height / h2
    
    new_w1 = int(w1 * scale1)
    new_w2 = int(w2 * scale2)
    
    resized_original = cv2.resize(original, (new_w1, max_height))
    resized_rectified = cv2.resize(rectified, (new_w2, max_height))
    
    # 并排显示
    combined = np.hstack((resized_original, resized_rectified))
    
    # 添加标签
    cv2.putText(combined, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Rectified", (new_w1 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示
    cv2.imshow(title, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_results(original, rectified, base_name="plate"):
    """保存结果"""
    if original is not None:
        cv2.imwrite(f"{base_name}_original.jpg", original)
        print(f"保存原始图像: {base_name}_original.jpg")
    
    if rectified is not None:
        cv2.imwrite(f"{base_name}_rectified.jpg", rectified)
        print(f"保存矫正图像: {base_name}_rectified.jpg")


if __name__ == "__main__":
    # 创建检测器
    detector = LicensePlateDetector(model_path='yolov8s.pt', conf_threshold=0.5)
    
    # 测试图片
    image_path = 'test1.jpg'
    
    # 检测并矫正单个车牌
    print("=== 检测并矫正单个车牌 ===")
    original_plate, rectified_plate = detector.detect_and_rectify(image_path)
    
    if original_plate is not None:
        print(f"原始车牌尺寸: {original_plate.shape}")
        if rectified_plate is not None:
            print(f"矫正车牌尺寸: {rectified_plate.shape}")
            
            # 显示对比
            show_comparison(original_plate, rectified_plate)
            
            # 保存结果
            save_results(original_plate, rectified_plate, "best_plate")
        else:
            print("无法进行透视矫正")
            
            # 只显示原始车牌
            cv2.imshow('Original Plate', original_plate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("未检测到车牌")
    
    # 检测并矫正所有车牌
    print("\n=== 检测并矫正所有车牌 ===")
    all_plates = detector.detect_all_and_rectify(image_path)
    
    if all_plates:
        print(f"检测到 {len(all_plates)} 个车牌:")
        for i, plate_info in enumerate(all_plates):
            print(f"\n车牌 {i+1}:")
            print(f"  置信度: {plate_info['confidence']:.2f}")
            print(f"  位置: {plate_info['bbox']}")
            print(f"  原始尺寸: {plate_info['size_original']}")
            print(f"  矫正尺寸: {plate_info['size_rectified']}")
            
            # 显示每个车牌的对比
            show_comparison(
                plate_info['original_crop'], 
                plate_info['rectified'],
                f"车牌 {i+1} 透视矫正对比"
            )
            
            # 保存每个车牌
            save_results(
                plate_info['original_crop'],
                plate_info['rectified'],
                f"plate_{i+1}"
            )
    else:
        print("未检测到任何车牌")