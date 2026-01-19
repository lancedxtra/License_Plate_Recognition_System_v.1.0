# camera_manager.py
"""
摄像头管理模块 - 检测和管理摄像头设备
"""

import cv2
import os
from typing import List, Dict, Optional, Tuple


def get_all_cameras(max_test: int = 10) -> List[Dict]:
    """
    检测所有可用的摄像头
    
    Args:
        max_test: 最大测试摄像头数量
        
    Returns:
        摄像头信息列表
    """
    cameras = []
    
    # 检查常见的后端
    backends = [
        (cv2.CAP_ANY, "自动"),
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)"),
        (cv2.CAP_VFW, "Video for Windows"),
        (cv2.CAP_FIREWIRE, "FireWire (IEEE 1394)"),
        (cv2.CAP_QT, "QuickTime"),
        (cv2.CAP_UNICAP, "Unicap"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_OPENNI, "OpenNI"),
        (cv2.CAP_OPENNI2, "OpenNI2"),
        (cv2.CAP_IMAGES, "Images"),
        (cv2.CAP_ARAVIS, "Aravis"),
    ]
    
    print("正在检测摄像头...")
    
    for backend, backend_name in backends:
        for i in range(max_test):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_name = f"摄像头 {i}"
                    
                    # 检查是否已存在
                    existing = next((c for c in cameras if c['index'] == i), None)
                    if existing:
                        existing['backends'].append(backend_name)
                        if width * height > existing['width'] * existing['height']:
                            existing['width'] = width
                            existing['height'] = height
                            existing['fps'] = fps
                    else:
                        cameras.append({
                            'index': i,
                            'name': camera_name,
                            'backends': [backend_name],
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'backend': backend,
                            'available': True
                        })
                    print(f"  ✓ 后端 {backend_name}: 摄像头 {i} ({width}x{height} @ {fps:.1f}fps)")
                cap.release()
    
    return cameras


def list_all_cameras() -> None:
    """
    列出所有可用摄像头
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
        print("  4. 尝试使用管理员权限运行程序")
        return
    
    print(f"\n找到 {len(cameras)} 个可用摄像头:")
    print("=" * 60)
    
    for i, cam in enumerate(cameras):
        print(f"\n[{i}] {cam['name']}")
        print(f"  索引: {cam['index']}")
        print(f"  支持的后端: {', '.join(cam['backends'])}")
        print(f"  最大分辨率: {cam['width']}x{cam['height']}")
        print(f"  帧率: {cam['fps']:.1f}fps")
        print(f"  推荐后端: {cam['backends'][0]}")
    
    print("\n使用说明:")
    print("  1. 列出摄像头: python main.py --list-cameras")
    print("  2. 使用默认摄像头: python main.py --camera")
    print("  3. 使用指定摄像头: python main.py --camera --camera-index 0")
    print("  4. 交互式选择: python main.py --camera --interactive")
    print("=" * 60)


def select_camera_interactive() -> Optional[Dict]:
    """
    交互式选择摄像头
    
    Returns:
        选择的摄像头信息，或 None（用户取消）
    """
    print("=" * 60)
    print("摄像头选择")
    print("=" * 60)
    
    cameras = get_all_cameras()
    
    if not cameras:
        print("未找到任何摄像头！")
        print("请检查：")
        print("  1. 摄像头是否正确连接")
        print("  2. 摄像头驱动程序是否已安装")
        print("  3. 摄像头是否被其他程序占用")
        return None
    
    print(f"\n找到 {len(cameras)} 个可用摄像头:")
    for i, cam in enumerate(cameras):
        print(f"  [{i}] {cam['name']}")
        print(f"      分辨率: {cam['width']}x{cam['height']}")
        print(f"      帧率: {cam['fps']:.1f}fps")
        print(f"      后端: {cam['backends'][0]}")
        print()
    
    # 让用户选择
    while True:
        try:
            choice = input("\n请选择摄像头编号 (输入 'q' 退出): ").strip()
            
            if choice.lower() == 'q':
                print("用户取消选择")
                return None
            
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                selected_camera = cameras[choice_idx]
                print(f"\n已选择: {selected_camera['name']}")
                print(f"  索引: {selected_camera['index']}")
                print(f"  分辨率: {selected_camera['width']}x{selected_camera['height']}")
                print(f"  帧率: {selected_camera['fps']:.1f}fps")
                print(f"  后端: {selected_camera['backends'][0]}")
                
                # 询问是否修改参数
                modify = input("\n是否修改摄像头参数？(y/n): ").strip().lower()
                if modify == 'y':
                    width = input(f"帧宽度 (当前: {selected_camera['width']}，输入0使用默认): ").strip()
                    height = input(f"帧高度 (当前: {selected_camera['height']}，输入0使用默认): ").strip()
                    fps = input(f"帧率 (当前: {selected_camera['fps']:.1f}，输入0使用默认): ").strip()
                    interval = input(f"检测间隔帧数 (默认: 10): ").strip()
                    
                    if width and width != '0':
                        selected_camera['width'] = int(width)
                    if height and height != '0':
                        selected_camera['height'] = int(height)
                    if fps and fps != '0':
                        selected_camera['fps'] = float(fps)
                    if interval:
                        selected_camera['interval'] = int(interval)
                    else:
                        selected_camera['interval'] = 10
                
                return selected_camera
            else:
                print(f"错误: 请输入 0 到 {len(cameras)-1} 之间的数字")
        except ValueError:
            print("错误: 请输入有效的数字")
        except Exception as e:
            print(f"选择摄像头时出错: {e}")
            return None


def get_camera_name(camera_index: int, backend: int = cv2.CAP_ANY) -> str:
    """
    获取摄像头名称
    
    Args:
        camera_index: 摄像头索引
        backend: 摄像头后端
        
    Returns:
        摄像头名称
    """
    cap = cv2.VideoCapture(camera_index, backend)
    if cap.isOpened():
        try:
            # 在某些平台上，这可能返回设备名称
            backend_name = cap.getBackendName()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            return f"摄像头 {camera_index} ({backend_name}) - {width}x{height} @ {fps:.1f}fps"
        except:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            return f"摄像头 {camera_index} - {width}x{height} @ {fps:.1f}fps"
        finally:
            cap.release()
    return f"摄像头 {camera_index} (不可用)"


def test_camera(camera_index: int, backend: int = cv2.CAP_ANY, duration: int = 3) -> bool:
    """
    测试摄像头是否可用
    
    Args:
        camera_index: 摄像头索引
        backend: 摄像头后端
        duration: 测试时长（秒）
        
    Returns:
        摄像头是否可用
    """
    print(f"测试摄像头 {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print(f"摄像头 {camera_index} 无法打开")
        return False
    
    try:
        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"摄像头信息: {width}x{height} @ {fps:.1f}fps")
        
        # 测试读取帧
        print("测试帧读取...")
        frames_read = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"第{frames_read}帧读取失败")
                break
            
            frames_read += 1
            
            # 计算运行时间
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time >= duration:
                break
        
        if frames_read > 0:
            actual_fps = frames_read / elapsed_time
            print(f"✓ 摄像头测试通过: {frames_read}帧, 实际帧率: {actual_fps:.1f}fps")
            return True
        else:
            print("✗ 摄像头测试失败: 无法读取帧")
            return False
            
    except Exception as e:
        print(f"摄像头测试出错: {e}")
        return False
    finally:
        cap.release()


def get_camera_info(camera_index: int) -> Dict:
    """
    获取摄像头详细信息
    
    Args:
        camera_index: 摄像头索引
        
    Returns:
        摄像头信息字典
    """
    # 先扫描所有摄像头
    cameras = get_all_cameras()
    
    # 查找指定摄像头
    camera_info = None
    for cam in cameras:
        if cam['index'] == camera_index:
            camera_info = cam
            break
    
    if not camera_info:
        return {
            'index': camera_index,
            'available': False,
            'error': f'未找到摄像头索引 {camera_index}'
        }
    
    # 获取摄像头属性（安全的方式）
    cap = None
    properties = {}
    
    try:
        backend = camera_info.get('backend', cv2.CAP_ANY)
        cap = cv2.VideoCapture(camera_index, backend)
        
        if cap.isOpened():
            # 定义常见的摄像头属性
            prop_names = {
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
                cv2.CAP_PROP_ZOOM: "缩放",
                cv2.CAP_PROP_PAN: "平移",
                cv2.CAP_PROP_TILT: "倾斜",
                cv2.CAP_PROP_ROLL: "旋转",
                cv2.CAP_PROP_BACKLIGHT: "背光补偿",
                cv2.CAP_PROP_AUTO_WB: "自动白平衡",
                cv2.CAP_PROP_WB_TEMPERATURE: "白平衡温度",
                cv2.CAP_PROP_GAMMA: "伽马值",
                cv2.CAP_PROP_TEMPERATURE: "温度",
                cv2.CAP_PROP_SHARPNESS: "锐度",
                cv2.CAP_PROP_BACKEND: "后端",
                cv2.CAP_PROP_BUFFERSIZE: "缓冲区大小",
            }
            
            # 添加可用的属性 - 使用安全的方式
            for prop_code, prop_name in prop_names.items():
                try:
                    value = cap.get(prop_code)
                    if value is not None and value != -1:
                        properties[prop_name] = value
                except Exception:
                    # 如果属性不存在或获取失败，跳过
                    continue
            
            cap.release()
    except Exception as e:
        if cap:
            cap.release()
        print(f"获取摄像头属性时出错: {e}")
    
    # 合并信息
    result = {
        'index': camera_index,
        'available': True,
        'name': camera_info.get('name', f'摄像头 {camera_index}'),
        'backend': camera_info.get('backends', ['未知'])[0] if camera_info.get('backends') else '未知',
        'width': camera_info.get('width', 0),
        'height': camera_info.get('height', 0),
        'fps': camera_info.get('fps', 0),
        'properties': properties,
    }
    
    return result


def print_camera_info(info: Dict) -> None:
    """
    打印摄像头信息
    
    Args:
        info: 摄像头信息字典
    """
    print("=" * 60)
    print("摄像头详细信息")
    print("=" * 60)
    
    if not info['available']:
        print(f"摄像头 {info['index']} 不可用")
        return
    
    print(f"摄像头索引: {info['index']}")
    print(f"后端: {info.get('backend_name', '未知')}")
    print(f"分辨率: {info['width']}x{info['height']}")
    print(f"帧率: {info['fps']:.1f}fps")
    
    if info['properties']:
        print("\n摄像头属性:")
        for prop_name, value in info['properties'].items():
            print(f"  {prop_name}: {value:.2f}")
    
    print("=" * 60)


def find_best_camera() -> Optional[Dict]:
    """
    寻找最佳摄像头（最高分辨率）
    
    Returns:
        最佳摄像头信息
    """
    cameras = get_all_cameras()
    
    if not cameras:
        return None
    
    # 按分辨率排序
    cameras.sort(key=lambda x: x['width'] * x['height'], reverse=True)
    
    return cameras[0]


class CameraManager:
    """
    摄像头管理器
    """
    
    def __init__(self):
        """初始化摄像头管理器"""
        self.cameras = []
        self.selected_camera = None
        
    def scan_cameras(self, max_test: int = 10):
        """扫描所有摄像头"""
        self.cameras = get_all_cameras(max_test)
        return self.cameras
    
    def select_camera(self, camera_index: int = None):
        """
        选择摄像头
        
        Args:
            camera_index: 摄像头索引，如果为None则交互式选择
        """
        if not self.cameras:  # 如果没有扫描过摄像头，先扫描
            self.scan_cameras()
        
        if camera_index is None:
            self.selected_camera = select_camera_interactive()
        else:
            # 查找指定摄像头
            for cam in self.cameras:
                if cam['index'] == camera_index:
                    self.selected_camera = cam
                    break
            
            if not self.selected_camera:
                print(f"未找到摄像头索引 {camera_index}")
                # 可以尝试直接打开摄像头
                print("尝试直接打开摄像头...")
                if test_camera(camera_index):
                    self.selected_camera = {
                        'index': camera_index,
                        'name': f"摄像头 {camera_index}",
                        'backends': ["自动检测"],
                        'width': 640,
                        'height': 480,
                        'fps': 30.0,
                        'backend': cv2.CAP_ANY,
                        'available': True
                    }
                    self.cameras.append(self.selected_camera)
                else:
                    return False
        
        return self.selected_camera is not None
    
    def get_selected_camera_info(self):
        """获取选择的摄像头信息"""
        if self.selected_camera:
            return get_camera_info(
                self.selected_camera['index'],
                self.selected_camera.get('backend', cv2.CAP_ANY)
            )
        return None
    
    def test_selected_camera(self, duration: int = 3):
        """测试选择的摄像头"""
        if not self.selected_camera:
            print("未选择摄像头")
            return False
        
        return test_camera(
            self.selected_camera['index'],
            self.selected_camera.get('backend', cv2.CAP_ANY),
            duration
        )


# 测试函数
def main_test():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="摄像头管理器测试")
    parser.add_argument("--list", action="store_true", help="列出所有摄像头")
    parser.add_argument("--test", type=int, help="测试指定摄像头")
    parser.add_argument("--info", type=int, help="获取摄像头详细信息")
    parser.add_argument("--find-best", action="store_true", help="寻找最佳摄像头")
    parser.add_argument("--interactive", action="store_true", help="交互式测试")
    
    args = parser.parse_args()
    
    if args.list:
        list_all_cameras()
    elif args.test is not None:
        test_camera(args.test)
    elif args.info is not None:
        info = get_camera_info(args.info)
        print_camera_info(info)
    elif args.find_best:
        best_camera = find_best_camera()
        if best_camera:
            print(f"最佳摄像头: 索引 {best_camera['index']}")
            print(f"分辨率: {best_camera['width']}x{best_camera['height']}")
            print(f"帧率: {best_camera['fps']:.1f}fps")
        else:
            print("未找到摄像头")
    elif args.interactive:
        manager = CameraManager()
        manager.scan_cameras()
        if manager.select_camera():
            info = manager.get_selected_camera_info()
            print_camera_info(info)
            print("\n测试摄像头...")
            manager.test_selected_camera()
    else:
        print("请指定操作:")
        print("  --list: 列出所有摄像头")
        print("  --test <index>: 测试摄像头")
        print("  --info <index>: 获取摄像头信息")
        print("  --find-best: 寻找最佳摄像头")
        print("  --interactive: 交互式测试")


if __name__ == "__main__":
    main_test()