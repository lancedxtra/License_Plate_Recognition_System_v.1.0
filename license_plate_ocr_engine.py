import os
import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR
import re

# ==================== 初始化 ====================
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# 1. 在外部设置运行设备（替代原来的 use_gpu 参数）
if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu')  # 显式切换到 GPU
    print(f"当前使用设备: GPU (CUDA)")
else:
    paddle.set_device('cpu')
    print(f"当前使用设备: CPU")


# 2. 初始化 PaddleOCR
# 修正说明：
# 1. 移除了 'use_gpu' 参数（解决 ValueError）
# 2. 保留 'enable_mkldnn=False'（解决 CPU 下的 ConvertPirAttribute 崩溃）
# 3. 如果看见 DeprecationWarning 警告是正常的，不影响运行
_ocr = PaddleOCR(
    use_angle_cls=True,         # 是否使用方向分类器
    lang="ch",                  # 语言
    text_det_box_thresh=0.5,    # 检测阈值
    text_det_unclip_ratio=1.6,  # 检测框扩张比例
    enable_mkldnn=False         # <--- 【关键】必须关闭 CPU 加速，否则会报错崩溃
)
# ==================== 车牌汉字开头列表 ====================
CHINESE_PROVINCES = [
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑",
    "沪", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
    "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新",
    "使", "领", "学", "警", "港", "澳"  # 特殊车牌汉字
]

# ==================== 优化车牌颜色识别 (HSV + 像素统计抗干扰版) ====================
def get_plate_type_by_hsv(img_crop, text):
    """
    车牌颜色识别 - 严格遵循位数逻辑
    逻辑：字符长度为8 -> 绿牌；字符长度不为8 -> 强制排除绿牌
    """
    if img_crop is None or img_crop.size == 0: 
        return "未知"
    
    # 清理文本，确保长度计算准确
    text = str(text).upper().replace(" ", "").replace(".", "").strip()
    text_len = len(text)
    
    # --- 1. 绝对规则判断 (优先级最高) ---
    
    # 【用户指定逻辑】如果字符长度为8，直接判定为绿牌
    if text_len == 8:
        return "绿牌"
        
    # 特殊车牌前缀判断 (针对7位及以下的情况)
    if "警" in text or "应急" in text: return "白牌 (警用)"
    if "使" in text or "领" in text: return "黑牌/白牌 (使领馆)"
    if "学" in text: return "黄牌 (教练)"
    if "港" in text or "澳" in text: return "黑牌 (港澳)"

    # --- 2. 图像预处理 ---
    h, w = img_crop.shape[:2]
    # 中心裁剪：只取中间区域分析颜色，避开边框
    crop_h_start, crop_h_end = int(h * 0.25), int(h * 0.75)
    crop_w_start, crop_w_end = int(w * 0.1), int(w * 0.9)
    roi = img_crop[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    
    if roi.size == 0: return "未知"
    
    # 转 HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # --- 3. 定义精准颜色范围 ---
    # 蓝牌
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    # 黄牌
    lower_yellow = np.array([11, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    # 绿牌
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([99, 255, 255])
    
    # --- 4. 像素统计 ---
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    score_blue = cv2.countNonZero(mask_blue)
    score_yellow = cv2.countNonZero(mask_yellow)
    score_green = cv2.countNonZero(mask_green)
    
    # 计算有效彩色区域总像素 (S>40, V>40, V<230 排除极黑极白)
    mask_valid = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([180, 255, 230]))
    total_valid = cv2.countNonZero(mask_valid)
    if total_valid == 0: total_valid = 1
    
    ratio_blue = score_blue / total_valid
    ratio_yellow = score_yellow / total_valid
    ratio_green = score_green / total_valid
    
    # --- 5. 颜色竞争判定 ---
    max_score = max(ratio_blue, ratio_yellow, ratio_green)
    
    # 如果颜色特征都不明显
    if max_score < 0.2:
        # 分析亮度
        avg_v = np.mean(hsv[:,:,2])
        if avg_v < 50: return "黑牌"
        if avg_v > 200: return "白牌"
        # 默认兜底：7位通常是蓝牌
        return "蓝牌"
        
    # 颜色分支
    if ratio_green == max_score:
        # 【用户指定逻辑】HSV显示是绿色，但字符不是8位 -> 强制否定绿牌
        # 因为前面 if text_len == 8 已返回，能走到这里说明 text_len != 8
        # 回退逻辑：比较黄和蓝，谁大选谁，默认偏向蓝牌
        if ratio_yellow > ratio_blue:
            return "黄牌"
        else:
            return "蓝牌" # 蓝牌在某些光线下容易偏青，误判为绿
            
    elif ratio_yellow == max_score:
        return "黄牌"
        
    elif ratio_blue == max_score:
        return "蓝牌"
        
    return "蓝牌" # 最终兜底

# ==================== 车牌格式验证函数 ====================
def validate_license_plate_format(text):
    """
    验证车牌格式规则（严格模式）：
    1. 必须以汉字开头
    2. 汉字后的第一位必须是字母（不能是I/O）
    3. 汉字后的第二位必须是字母（不能是I/O）
    4. 数字部分不能以0开头
    5. 不能包含I/O字母（在第二、三位）
    """
    if len(text) < 3:  # 至少要有汉字+两个字母
        return False
    
    # 1. 检查第一个字符是否为汉字
    first_char = text[0]
    if first_char not in CHINESE_PROVINCES:
        return False
    
    # 2. 检查第二个字符（汉字后第一位）是否为字母且不是I/O
    second_char = text[1]
    if not second_char.isalpha() or second_char in ['I', 'O']:
        return False
    
    # 3. 检查第三个字符（汉字后第二位）是否为字母且不是I/O
    third_char = text[2]
    if not third_char.isalpha() or third_char in ['I', 'O']:
        return False
    
    # 4. 验证后续字符格式
    remaining_text = text[3:]
    
    if not remaining_text:  # 必须有后续字符
        return False
    
    # 5. 检查数字部分不能以0开头
    # 找到第一个数字
    for char in remaining_text:
        if char.isdigit():
            if char == '0':
                return False  # 数字以0开头
            break
    
    # 6. 统计数字数量
    digit_count = sum(1 for c in remaining_text if c.isdigit())
    
    if digit_count == 0:
        return False
    
    # 7. 长度检查
    total_length = len(text)
    
    # 特殊处理警车（6位）
    if total_length == 6:
        if text[0] == "警" and text[1].isalpha() and text[1] not in ['I', 'O']:
            remaining = text[2:]
            if len(remaining) == 4 and all(c.isdigit() for c in remaining):
                if remaining[0] == '0':
                    return False
                return True
    
    # 8. 普通车牌长度检查
    if total_length < 7 or total_length > 8:
        return False
    
    # 9. 验证新能源车牌
    if total_length == 8 and text[-1] in ['D', 'F']:
        middle = text[3:7]
        if middle[0] == '0':
            return False
    
    return True

# ==================== 车牌评分函数 ====================
def calculate_plate_score(text, ocr_confidence):
    """
    计算车牌得分，考虑格式匹配度和OCR置信度
    """
    score = ocr_confidence * 0.7  # OCR置信度权重70%
    
    if validate_license_plate_format(text):
        score += 0.3  # 格式正确加30%
    
    if 6 <= len(text) <= 8:
        score += 0.1
    
    if any(c.isdigit() for c in text):
        score += 0.1
    
    return min(score, 1.0)

# ==================== 字符纠正函数 (修复版) ====================
def correct_license_plate_text(text):
    """
    纠正车牌文本中的常见错误
    """
    if not text or len(text) < 2:
        return text
    
    corrected = text.upper()
    
    # 常见错误字符映射
    char_replacements = {
        'I': '1', 'O': '0',
        'Q': '0', 'U': 'V'
    }
    
    # 特殊汉字纠正 (OCR常把"警"看错)
    special_corrections = {
        "五水": "警", "言敬": "警", "敬": "警", 
        "使": "使", "领": "领", "学": "学", "港": "港", "澳": "澳"
    }
    
    for wrong, right in special_corrections.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, right)
    
    # 1. 确保第一个字符是汉字
    first_char = corrected[0]
    if first_char not in CHINESE_PROVINCES:
        if first_char in char_replacements:
            corrected = "京" + corrected[1:] # 默认容错
    
    # 2. 处理第二个字符（必须是字母）
    if len(corrected) > 1:
        second_char = corrected[1]
        # 第二位如果是0，肯定是D的误读 (如 川0 -> 川D)
        if second_char == '0':
            corrected = corrected[0] + 'D' + corrected[2:]
        elif second_char in ['I', 'O']:
            corrected = corrected[0] + char_replacements.get(second_char, second_char) + corrected[2:]

    # 3. 处理第三个字符
    # 【核心修复】这里删除了 "0->D" 的强制转换，因为第3位允许是数字0
    if len(corrected) > 2:
        third_char = corrected[2]
        
        # 只纠正明显的字母错误 (I/O)
        if third_char in ['I', 'O']:
            corrected = corrected[:2] + char_replacements[third_char] + corrected[3:]
            
        # 注意：这里我们移除了 if third_char == '0': corrected = ... 'D' 的逻辑
        # 因为在 "浙C0985" 中，第3位本身就是数字0，不应该被改成D

    # 4. 处理最后一位（针对新能源）
    if len(corrected) == 8:
         # 如果是新能源，最后一位误读成0的可能性很小，通常不需要强行纠正
         pass
    
    # 5. 清理非法字符
    corrected = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', corrected)
    
    return corrected

# ==================== 主识别函数 (适配 PaddleX v2.10 Server 格式) ====================
def get_license_plate_info(img_path):
    try:
        # --- 1. 读取图像 ---
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            full_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if full_img is None: return None
        
        # --- 2. OCR识别 ---
        # 这里的 result 是原始返回数据
        result = _ocr.ocr(img_path)
        
        if not result: return None

        # --- 3. 【核心修复】数据标准化解析 ---
        # 我们要把各种奇形怪状的返回格式，统一转成标准的 candidate 列表
        ocr_candidates = []
        
        # 获取第一层数据
        data = result[0] if (isinstance(result, list) and len(result) > 0) else result

        # >>> 针对你遇到的 PaddleX Server 格式 (rec_texts 是列表) <<<
        if isinstance(data, dict) and 'rec_texts' in data and isinstance(data['rec_texts'], list):
            texts = data['rec_texts']
            scores = data.get('rec_scores', [])
            # 坐标可能是 dt_polys 或 points
            boxes = data.get('dt_polys') if 'dt_polys' in data else data.get('points', [])
            
            # 将列表拆解（Zip）成单个对象
            for i, text in enumerate(texts):
                score = scores[i] if i < len(scores) else 0.0
                box = boxes[i] if i < len(boxes) else []
                ocr_candidates.append({'text': text, 'score': score, 'box': box})
        
        # >>> 针对旧版标准格式 [[box, (text, score)]] <<<
        elif isinstance(data, list):
            for line in data:
                if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                    ocr_candidates.append({'text': line[1][0], 'score': line[1][1], 'box': line[0]})
        
        # >>> 针对通用字典格式 (单行) <<<
        elif isinstance(data, dict) and 'text' in data:
             ocr_candidates.append(data)

        # --- 4. 遍历并筛选 ---
        plate_candidates = []
        
        print(f"📊 [DEBUG] 解析出 {len(ocr_candidates)} 个文本区域")

        for item in ocr_candidates:
            # 统一获取属性
            text = item.get('text', '')
            score = float(item.get('score', 0.0))
            coords = item.get('box', [])
            
            # 清理文本
            cleaned_text = text.replace(" ", "").replace("·", "").replace(".", "").replace("-", "").upper()
            
            print(f"  📝 识别结果: '{text}' -> 清洗后: '{cleaned_text}'")

            # 基础过滤
            if len(cleaned_text) < 5: continue
            if not any(c.isdigit() for c in cleaned_text): continue
            
            # 字符纠正
            corrected_text = correct_license_plate_text(cleaned_text)
            # 格式验证
            is_valid_format = validate_license_plate_format(corrected_text)
            # 计算得分
            final_score = calculate_plate_score(corrected_text, score)
            
            # 只有通过基础过滤的才加入候选
            plate_candidates.append({
                'corrected_text': corrected_text,
                'final_score': final_score,
                'coords': coords,
                'is_valid_format': is_valid_format
            })

        if not plate_candidates:
            return None
        
        # --- 5. 选择最佳结果 ---
        # 排序：优先格式正确，其次看分数
        plate_candidates.sort(key=lambda x: (x['is_valid_format'], x['final_score']), reverse=True)
        best_candidate = plate_candidates[0]
        
        final_text = best_candidate['corrected_text']
        final_score = best_candidate['final_score']
        
        # --- 6. 颜色检测 ---
        try:
            coords = best_candidate['coords']
            box = np.array(coords).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box)
            h_img, w_img = full_img.shape[:2]
            # 稍微扩大一点裁剪范围以免切掉边缘颜色
            plate_crop = full_img[max(0, y):min(h_img, y+h), max(0, x):min(w_img, x+w)]
            plate_type = get_plate_type_by_hsv(plate_crop, final_text)
        except:
            plate_type = "未知"
        
        return final_text, final_score, plate_type

    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

# ==================== 测试函数 ====================
def test_license_plate_format():
    """测试车牌格式验证函数"""
    test_cases = [
        # 正确格式
        ("京AB1234", True),      # 标准格式 ✓
        ("沪BD8888", True),      # 第二三位都是字母 ✓
        ("粤ZD1234", True),      # Z开头+字母 ✓
        ("使AB1234", True),      # 使馆车牌 ✓
        ("警A88888", True),      # 警车车牌 ✓
        ("京AD1234", True),      # 新能源车牌 ✓
        ("京AF1234", True),      # 新能源车牌 ✓
        
        # 测试纠正后的格式
        ("京AD123D", True),      # 纠正后：0→D ✓
        ("京AB123B", True),      # 纠正后：8→B ✓
        
        # 错误格式
        ("京A01234", False),     # 汉字后第二位不是字母 ✗
        ("京1B2345", False),     # 汉字后第一位不是字母 ✗
        ("京AB0123", False),     # 数字部分以0开头 ✗
        ("京IA1234", False),     # 包含I字母 ✗
        ("京AO1234", False),     # 包含O字母 ✗
        ("京A01234", False),     # 第二位是0 ✗
        ("京A81234", False),     # 第二位是8 ✗
        ("京1D2345", False),     # 第一位不是汉字 ✗
        ("警A0888", False),      # 警车数字以0开头 ✗
    ]
    
    # 测试纠正函数
    print("字符纠正测试:")
    print("-" * 40)
    test_corrections = [
        ("京01234", "京DD234"),   # 第二三位0→D
        ("京81234", "京BD234"),   # 第二位8→B, 第三位1→I
        ("京I0234", "京1D234"),   # I→1, 0→D
        ("京O8234", "京0B234"),   # O→0, 8→B
        ("京A0123D", "京AD123D"), # 新能源车牌纠正
    ]
    
    for original, expected in test_corrections:
        corrected = correct_license_plate_text(original)
        status = "✓" if corrected == expected else "✗"
        print(f"{status} {original:10} → {corrected:10} (期望: {expected})")
    
    print("\n车牌格式验证测试:")
    print("-" * 40)
    all_passed = True
    for text, expected in test_cases:
        result = validate_license_plate_format(text)
        status = "✓" if result == expected else "✗"
        if status == "✗":
            all_passed = False
        print(f"{status} {text:15} → {'有效' if result else '无效':8} "
              f"(期望: {'有效' if expected else '无效'})")
    
    print("-" * 40)
    print(f"测试结果: {'全部通过' if all_passed else '存在失败'}")


# ==================== 运行入口 ====================
if __name__ == "__main__":
    # --- 核心修改：自动获取 main.py 所在的文件夹路径 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接出图片的完整绝对路径
    image_path = os.path.join(current_dir, "car.jpg")
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试读取图片: {image_path}")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：依然找不到文件。请检查文件名是否真的是 'car.jpg' (注意大小写和扩展名隐藏)")
        # 调试：列出该文件夹下有哪些文件
        print(f"文件夹 {current_dir} 下的文件有: {os.listdir(current_dir)}")
    else:
        print(f"正在识别图片...")
        
        # 2. 调用识别函数
        result = get_license_plate_info(image_path)
        
        # 3. 输出结果
        if result:
            text, score, plate_type = result
            print("=" * 30)
            print(f"【识别成功】")
            print(f"车牌号码: {text}")
            print(f"车牌颜色: {plate_type}")
            print(f"置信度:   {score:.4f}")
            print("=" * 30)
        else:
            print("未能识别出有效车牌。")