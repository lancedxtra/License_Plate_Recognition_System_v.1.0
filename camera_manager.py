# camera_manager.py
"""
摄像头管理模块 - 检测和管理摄像头设备
"""

import cv2
import os
import sys
import time
from typing import List, Dict, Optional, Tuple

# ================= 原有核心功能 (保持兼容性) =================

def get_all_cameras(max_test: int = 5) -> List[Dict]:
    """检测所有可用的摄像头"""
    cameras = []
    
    # 检查常见的后端
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "自动"),
    ]
    
    print("正在检测摄像头...")
    
    for i in range(max_test):
        # 对每个索引尝试检测
        found = False
        for backend, backend_name in backends:
            if found: break
            
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_name = f"摄像头 {i}"
                        
                        cameras.append({
                            'index': i,
                            'name': camera_name,
                            'backends': [backend_name], # 保持列表结构兼容
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'backend': backend,
                            'available': True
                        })
                        print(f"  ✓ 摄像头 {i}: {width}x{height} ({backend_name})")
                        found = True
                    cap.release()
            except:
                pass
    
    return cameras


def list_all_cameras() -> None:
    """列出所有可用摄像头 (main.py 依赖)"""
    print("=" * 60)
    print("摄像头检测")
    print("=" * 60)
    
    cameras = get_all_cameras()
    
    if not cameras:
        print("未找到任何摄像头！")
        return
    
    print(f"\n找到 {len(cameras)} 个可用摄像头:")
    for i, cam in enumerate(cameras):
        print(f"\n[{i}] {cam['name']}")
        print(f"  索引: {cam['index']}")
        print(f"  分辨率: {cam['width']}x{cam['height']}")
        print(f"  帧率: {cam['fps']:.1f}fps")


def select_camera_interactive() -> Optional[Dict]:
    """交互式选择摄像头 (main.py 依赖)"""
    print("=" * 60)
    print("摄像头选择")
    print("=" * 60)
    
    cameras = get_all_cameras()
    
    if not cameras:
        print("未找到任何摄像头！")
        return None
    
    for i, cam in enumerate(cameras):
        print(f"  [{i}] {cam['name']} - {cam['width']}x{cam['height']}")
    
    while True:
        try:
            choice = input("\n请选择摄像头编号 (输入 'q' 退出): ").strip()
            if choice.lower() == 'q': return None
            
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                selected = cameras[choice_idx]
                
                # 询问是否修改参数
                modify = input("是否修改参数？(y/n): ").strip().lower()
                if modify == 'y':
                    w = input(f"宽度 (当前{selected['width']}): ").strip()
                    h = input(f"高度 (当前{selected['height']}): ").strip()
                    if w: selected['width'] = int(w)
                    if h: selected['height'] = int(h)
                    
                return selected
        except ValueError:
            print("输入无效")


def get_camera_info(camera_index: int) -> Dict:
    """获取摄像头详细信息 (main.py 依赖)"""
    cap = cv2.VideoCapture(camera_index)
    props = {}
    if cap.isOpened():
        props = {
            "宽度": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "高度": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "FPS": cap.get(cv2.CAP_PROP_FPS),
            "亮度": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "后端": cap.get(cv2.CAP_PROP_BACKEND)
        }
    cap.release()
    
    return {
        'index': camera_index,
        'available': bool(props),
        'properties': props,
        'width': props.get('宽度', 0),
        'height': props.get('高度', 0),
        'fps': props.get('FPS', 0),
        'backend_name': 'Auto'
    }


def print_camera_info(info: Dict) -> None:
    """打印摄像头信息 (main.py 依赖)"""
    print(f"摄像头 {info['index']} 信息:")
    for k, v in info.get('properties', {}).items():
        print(f"  {k}: {v}")


def find_best_camera() -> Optional[Dict]:
    """寻找最佳摄像头 (main.py 依赖)"""
    cameras = get_all_cameras()
    if not cameras: return None
    # 按分辨率排序
    cameras.sort(key=lambda x: x['width'] * x['height'], reverse=True)
    return cameras[0]


def get_camera_name(camera_index: int, backend: int = cv2.CAP_ANY) -> str:
    """获取摄像头名称 (main.py 依赖)"""
    return f"摄像头 {camera_index}"


def test_camera(camera_index: int, backend: int = cv2.CAP_ANY, duration: int = 100000) -> bool:
    """
    测试摄像头 (main.py 依赖)
    修改：增加实时预览窗口，按Q退出
    """
    print(f"测试摄像头 {camera_index}... (按 'q' 退出)")
    
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print("无法打开摄像头")
        return False
    
    # 尝试设置高分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break
            
        cv2.putText(frame, f"Cam {camera_index}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Test Camera {camera_index}", frame)
        
        # main.py 调用时可能传入较短duration用于自动测试，这里兼容一下
        # 但如果是独立运行，我们希望它是交互式的
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return True


class CameraManager:
    """摄像头管理器类 (main.py 依赖)"""
    def __init__(self):
        self.cameras = []
        self.selected_camera = None
        
    def scan_cameras(self, max_test: int = 5):
        self.cameras = get_all_cameras(max_test)
        return self.cameras
    
    def select_camera(self, camera_index: int = None):
        if not self.cameras: self.scan_cameras()
        if camera_index is None:
            self.selected_camera = select_camera_interactive()
        else:
            self.selected_camera = next((c for c in self.cameras if c['index'] == camera_index), None)
        return self.selected_camera is not None

    def get_selected_camera_info(self):
        if self.selected_camera:
            return get_camera_info(self.selected_camera['index'])
        return None


# ================= 新增：独立运行时的交互菜单 =================

def interactive_menu():
    """独立运行时显示的菜单"""
    while True:
        print("\n" + "="*40)
        print("   摄像头硬件测试工具")
        print("="*40)
        print("1. 扫描摄像头列表")
        print("2. 预览摄像头 (测试画面)")
        print("3. 查看参数详情")
        print("0. 退出")
        print("="*40)
        
        choice = input("请输入选项: ").strip()
        
        if choice == '1':
            list_all_cameras()
            
        elif choice == '2':
            idx = input("请输入摄像头索引 (默认0): ").strip()
            idx = int(idx) if idx.isdigit() else 0
            test_camera(idx) # 复用兼容函数
            
        elif choice == '3':
            idx = input("请输入摄像头索引 (默认0): ").strip()
            idx = int(idx) if idx.isdigit() else 0
            info = get_camera_info(idx)
            print_camera_info(info)
            
        elif choice == '0':
            break

def main_test():
    """入口函数"""
    import argparse
    parser = argparse.ArgumentParser(description="摄像头管理器")
    parser.add_argument("--list", action="store_true", help="列出摄像头")
    parser.add_argument("--test", type=int, help="测试指定摄像头")
    parser.add_argument("--info", type=int, help="获取摄像头信息")
    # 添加 main.py 可能用到的其他参数以防报错
    parser.add_argument("--find-best", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    
    # 逻辑判断：如果有参数，执行对应功能；如果没有参数，进入交互菜单
    if args.list:
        list_all_cameras()
    elif args.test is not None:
        test_camera(args.test)
    elif args.info is not None:
        info = get_camera_info(args.info)
        print_camera_info(info)
    elif args.interactive: # 兼容 main.py 的调用
        cm = CameraManager()
        if cm.select_camera():
            print("选择成功")
    else:
        # 没有任何参数时，进入我们新写的交互菜单
        interactive_menu()


if __name__ == "__main__":
    main_test()