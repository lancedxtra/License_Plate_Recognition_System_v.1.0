import cv2
import numpy as np
from typing import Tuple, Optional, Union, Dict

class LicensePlatePreprocessor:
    """
    车牌图片预处理类，包含曝光、反光处理和车牌颜色恢复功能
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        """
        初始化预处理类
        
        Args:
            target_size: 目标图像尺寸 (宽, 高)
        """
        self.target_size = target_size
        
        # 车牌颜色定义 (BGR格式)
        self.plate_colors = {
            'blue': [(100, 0, 0), (255, 150, 100)],     # 蓝色车牌
            'green': [(0, 100, 0), (100, 255, 100)],    # 绿色新能源车牌
            'yellow': [(0, 150, 150), (100, 255, 255)], # 黄色车牌
            'white': [(200, 200, 200), (255, 255, 255)], # 白色车牌
            'black': [(0, 0, 0), (50, 50, 50)],         # 黑色车牌
        }
    
    def detect_plate_color(self, image: np.ndarray, plate_mask: np.ndarray = None) -> Dict:
        """
        检测车牌主要颜色
        
        Args:
            image: 输入图像
            plate_mask: 车牌区域掩码（可选）
            
        Returns:
            包含颜色信息的字典
        """
        if plate_mask is None:
            # 如果没有提供掩码，假设整个图像底部区域可能是车牌
            h, w = image.shape[:2]
            plate_mask = np.zeros((h, w), dtype=np.uint8)
            plate_mask[int(h*0.7):h, int(w*0.1):int(w*0.9)] = 255
        
        # 提取车牌区域
        if len(image.shape) == 3:
            masked_image = cv2.bitwise_and(image, image, mask=plate_mask)
        else:
            masked_image = cv2.bitwise_and(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 
                                          cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 
                                          mask=plate_mask)
        
        # 转换到HSV颜色空间进行颜色分析
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # 计算主要颜色直方图
        hist_hue = cv2.calcHist([hsv], [0], plate_mask, [180], [0, 180])
        hist_saturation = cv2.calcHist([hsv], [1], plate_mask, [256], [0, 256])
        hist_value = cv2.calcHist([hsv], [2], plate_mask, [256], [0, 256])
        
        # 找到主要色调
        main_hue = np.argmax(hist_hue)
        avg_saturation = np.mean(hsv[:, :, 1][plate_mask > 0])
        avg_value = np.mean(hsv[:, :, 2][plate_mask > 0])
        
        # 识别车牌颜色类型
        color_type = "unknown"
        
        if avg_value < 50:
            color_type = "black"
        elif avg_value > 200 and avg_saturation < 30:
            color_type = "white"
        elif 100 <= main_hue <= 130:  # 蓝色区域
            color_type = "blue"
        elif 40 <= main_hue <= 80:    # 绿色区域
            color_type = "green"
        elif 20 <= main_hue <= 40:    # 黄色区域
            color_type = "yellow"
        
        # 获取当前颜色的平均BGR值
        bgr_values = masked_image[plate_mask > 0]
        avg_bgr = np.mean(bgr_values, axis=0) if len(bgr_values) > 0 else [0, 0, 0]
        
        # 获取目标颜色
        target_bgr = self._get_target_color(color_type, avg_bgr)
        
        return {
            'color_type': color_type,
            'avg_bgr': avg_bgr,
            'target_bgr': target_bgr,
            'main_hue': main_hue,
            'avg_saturation': avg_saturation,
            'avg_value': avg_value
        }
    
    def _get_target_color(self, color_type: str, current_bgr: np.ndarray) -> np.ndarray:
        """
        根据颜色类型获取目标颜色
        
        Args:
            color_type: 颜色类型
            current_bgr: 当前平均BGR值
            
        Returns:
            目标BGR颜色
        """
        if color_type in self.plate_colors:
            # 使用预定义颜色范围的平均值
            lower, upper = self.plate_colors[color_type]
            target_color = [(l + u) // 2 for l, u in zip(lower, upper)]
        else:
            # 保持原始颜色
            target_color = current_bgr
        
        return np.array(target_color, dtype=np.float32)
    
    def restore_plate_color(self, image: np.ndarray, color_info: Dict) -> np.ndarray:
        """
        恢复车牌颜色，保持自然的颜色过渡
        
        Args:
            image: 输入图像
            color_info: 颜色信息字典
            
        Returns:
            颜色恢复后的图像
        """
        result = image.copy().astype(np.float32)
        
        color_type = color_info['color_type']
        avg_bgr = color_info['avg_bgr']
        target_bgr = color_info['target_bgr']
        
        if color_type == "unknown":
            return image.astype(np.uint8)
        
        # 计算颜色校正因子
        color_correction = np.zeros_like(result)
        
        # 分离颜色通道
        b_channel = result[:, :, 0]
        g_channel = result[:, :, 1]
        r_channel = result[:, :, 2]
        
        # 计算当前颜色与目标颜色的比例
        if avg_bgr[0] > 1:
            b_factor = target_bgr[0] / avg_bgr[0]
        else:
            b_factor = 1.0
            
        if avg_bgr[1] > 1:
            g_factor = target_bgr[1] / avg_bgr[1]
        else:
            g_factor = 1.0
            
        if avg_bgr[2] > 1:
            r_factor = target_bgr[2] / avg_bgr[2]
        else:
            r_factor = 1.0
        
        # 应用颜色校正（使用平滑过渡）
        b_corrected = np.clip(b_channel * b_factor, 0, 255)
        g_corrected = np.clip(g_channel * g_factor, 0, 255)
        r_corrected = np.clip(r_channel * r_factor, 0, 255)
        
        # 合并通道
        result = np.stack([b_corrected, g_corrected, r_corrected], axis=2)
        
        return result.astype(np.uint8)
    
    def adaptive_color_preservation(self, image: np.ndarray) -> np.ndarray:
        """
        自适应颜色保持，在增强图像的同时保留原始颜色特征
        
        Args:
            image: 输入图像
            
        Returns:
            颜色保持处理后的图像
        """
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 分离LAB通道
        l, a, b = cv2.split(lab)
        
        # 只对亮度通道进行增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # 保留原始的颜色通道
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        
        # 转换回BGR
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def detect_glare_areas(self, image: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """
        检测反光/眩光区域
        
        Args:
            image: 输入图像
            threshold: 反光检测阈值（0-1）
            
        Returns:
            反光区域掩码
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        
        kernel_size = 21
        local_mean = cv2.blur(gray_norm, (kernel_size, kernel_size))
        local_std = cv2.blur(gray_norm**2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.abs(local_std - local_mean**2))
        
        bright_areas = (gray_norm > threshold)
        low_contrast_areas = (local_std < 0.1)
        
        glare_mask = np.logical_and(bright_areas, low_contrast_areas).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)
        
        return glare_mask
    
    def detect_reflection_by_saturation(self, image: np.ndarray) -> np.ndarray:
        """
        通过饱和度检测反光区域
        
        Args:
            image: 输入图像
            
        Returns:
            反光区域掩码
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        
        low_saturation = saturation < 0.3
        high_brightness = value > 0.8
        
        reflection_mask = np.logical_and(low_saturation, high_brightness).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        
        return reflection_mask
    
    def adaptive_reflection_removal(self, image: np.ndarray) -> np.ndarray:
        """
        自适应反光去除算法，保持颜色信息
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        result = image.copy()
        
        glare_mask = self.detect_glare_areas(image, threshold=0.85)
        saturation_mask = self.detect_reflection_by_saturation(image)
        
        combined_mask = cv2.bitwise_or(glare_mask, saturation_mask)
        
        if np.sum(combined_mask > 0) > 100:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask_dilated = cv2.dilate(combined_mask, kernel)
            
            mask_float = combined_mask_dilated.astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_float, (31, 31), 15)
            
            for channel in range(3):
                channel_img = result[:, :, channel].astype(np.float32)
                background = cv2.medianBlur(result[:, :, channel], 21)
                
                # 对每个颜色通道独立处理，保持颜色平衡
                corrected_channel = channel_img * (1 - mask_blurred) + background * mask_blurred
                
                result[:, :, channel] = np.clip(corrected_channel, 0, 255).astype(np.uint8)
        
        return result
    
    def detect_exposure(self, image: np.ndarray) -> Dict:
        """
        检测图片曝光情况
        
        Args:
            image: 输入图像
            
        Returns:
            包含曝光信息的字典
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        total_pixels = gray.shape[0] * gray.shape[1]
        under_exposed = np.sum(hist[:50]) / total_pixels
        over_exposed = np.sum(hist[200:]) / total_pixels
        mean_brightness = np.mean(gray)
        
        glare_mask = self.detect_glare_areas(image)
        glare_ratio = np.sum(glare_mask > 0) / total_pixels
        
        exposure_status = "normal"
        if glare_ratio > 0.1:
            exposure_status = "reflection"
        elif under_exposed > 0.3:
            exposure_status = "under_exposed"
        elif over_exposed > 0.3:
            exposure_status = "over_exposed"
        elif mean_brightness < 50:
            exposure_status = "dark"
        elif mean_brightness > 200:
            exposure_status = "bright"
            
        return {
            'status': exposure_status,
            'mean_brightness': mean_brightness,
            'under_exposed_ratio': under_exposed,
            'over_exposed_ratio': over_exposed,
            'glare_ratio': glare_ratio,
            'histogram': hist
        }
    
    def adjust_exposure_preserve_color(self, image: np.ndarray, exposure_info: Dict) -> np.ndarray:
        """
        根据曝光情况调整图像，同时保持颜色
        
        Args:
            image: 输入图像
            exposure_info: 曝光信息
            
        Returns:
            调整后的图像
        """
        result = image.copy()
        status = exposure_info['status']
        
        if status == "reflection":
            # 处理反光但保持颜色
            result = self.adaptive_reflection_removal(result)
            
        elif status == "under_exposed" or status == "dark":
            # 使用LAB空间，只增强亮度通道
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
        elif status == "over_exposed" or status == "bright":
            # 降低亮度但保持颜色
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.addWeighted(v, 0.7, np.zeros_like(v), 0, 0)
            enhanced_hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def enhance_text_contrast_color(self, image: np.ndarray, color_info: Dict) -> np.ndarray:
        """
        增强车牌文字对比度，同时保持车牌颜色
        
        Args:
            image: 输入图像
            color_info: 颜色信息
            
        Returns:
            增强后的图像
        """
        # 转换为LAB空间处理
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强亮度通道对比度
        l_normalized = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        
        # 根据车牌颜色类型调整增强策略
        color_type = color_info['color_type']
        
        if color_type in ['blue', 'green']:
            # 深色车牌，增强文字对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_normalized)
        elif color_type in ['yellow', 'white']:
            # 浅色车牌，适度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_normalized)
        else:
            l_enhanced = l_normalized
        
        # 合并LAB通道
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        
        # 转换回BGR
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def multi_scale_detail_enhancement_color(self, image: np.ndarray) -> np.ndarray:
        """
        多尺度细节增强，同时保持颜色
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        scales = [1, 2, 4]
        gaussian_pyramid = []
        
        current = image.astype(np.float32)
        for scale in scales:
            gaussian_pyramid.append(current)
            current = cv2.pyrDown(current)
        
        reconstructed = gaussian_pyramid[-1]
        for i in range(len(gaussian_pyramid)-2, -1, -1):
            upscaled = cv2.pyrUp(reconstructed)
            upscaled = cv2.resize(upscaled, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            
            detail = gaussian_pyramid[i] - upscaled
            # 增强细节但不改变颜色
            detail_enhanced = detail * 1.5
            
            reconstructed = upscaled + detail_enhanced
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    def preprocess_with_color_recovery(self, image: Union[str, np.ndarray], 
                                      detect_plate_region: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        完整的预处理流程，包含颜色恢复
        
        Args:
            image: 图片路径或图像数组
            detect_plate_region: 是否自动检测车牌区域
            
        Returns:
            处理后的图像和处理信息
        """
        # 加载图片
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法加载图片: {image}")
        else:
            img = image.copy()
        
        # 检测曝光情况
        exposure_info = self.detect_exposure(img)
        
        # 调整曝光（保持颜色）
        img = self.adjust_exposure_preserve_color(img, exposure_info)
        
        # 检测车牌颜色（如果需要）
        color_info = None
        if detect_plate_region:
            color_info = self.detect_plate_color(img)
            
            # 恢复车牌颜色
            img = self.restore_plate_color(img, color_info)
        
        # 多尺度细节增强（保持颜色）
        img = self.multi_scale_detail_enhancement_color(img)
        
        # 自适应颜色保持
        img = self.adaptive_color_preservation(img)
        
        # 增强文字对比度（考虑颜色）
        if color_info:
            img = self.enhance_text_contrast_color(img, color_info)
        
        # 调整尺寸
        img = self.resize_with_aspect_ratio(img)
        
        # 轻微锐化
        kernel = np.array([[0, -0.5, 0],
                          [-0.5, 3, -0.5],
                          [0, -0.5, 0]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img, {
            'exposure_info': exposure_info,
            'color_info': color_info,
            'final_shape': img.shape
        }
    
    def resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        保持宽高比调整图像大小
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if new_w < target_w or new_h < target_h:
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            resized = cv2.copyMakeBorder(
                resized, 
                top, bottom, left, right, 
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
        
        return resized